#include "Dev2PqRemesher.h"

#if ALICE2_WITH_DIRECTIONAL

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <igl/grad.h>
#include <igl/isolines.h>
#include <igl/principal_curvature.h>

#include <directional/TriMesh.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/curl_matrices.h>
#include <directional/power_field.h>
#include <directional/power_to_raw.h>
#include <directional/principal_matching.h>
#include <directional/project_curl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>

namespace alice2 {
namespace {

constexpr float kEpsilon = 1e-10f;
using EdgeKey = std::pair<int, int>;

struct TriangleInput {
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi faces;
};

EdgeKey edgeKey(int a, int b) { return a < b ? EdgeKey{a, b} : EdgeKey{b, a}; }

bool triangleInput(const MeshData& mesh, TriangleInput& input, std::string& diagnostic) {
    std::vector<std::array<int, 3>> triangles;
    for (const MeshFace& face : mesh.faces) {
        if (face.vertices.size() < 3) continue;
        for (const int vertex : face.vertices) {
            if (vertex < 0 || vertex >= static_cast<int>(mesh.vertices.size())) {
                diagnostic = "input has an invalid vertex index";
                return false;
            }
        }
        for (int corner = 1; corner + 1 < static_cast<int>(face.vertices.size()); ++corner) {
            const std::array<int, 3> triangle{face.vertices[0], face.vertices[corner], face.vertices[corner + 1]};
            const Vec3& a = mesh.vertices[triangle[0]].position;
            const Vec3& b = mesh.vertices[triangle[1]].position;
            const Vec3& c = mesh.vertices[triangle[2]].position;
            if ((b - a).cross(c - a).lengthSquared() <= kEpsilon) {
                diagnostic = "input contains a degenerate triangle";
                return false;
            }
            triangles.push_back(triangle);
        }
    }
    if (triangles.empty()) {
        diagnostic = "input has no usable faces";
        return false;
    }
    std::map<EdgeKey, int> edgeUses;
    for (const auto& triangle : triangles)
        for (int edge = 0; edge < 3; ++edge)
            if (++edgeUses[edgeKey(triangle[edge], triangle[(edge + 1) % 3])] > 2) {
                diagnostic = "input is non-manifold";
                return false;
            }

    input.vertices.resize(mesh.vertices.size(), 3);
    for (int vertex = 0; vertex < static_cast<int>(mesh.vertices.size()); ++vertex) {
        const Vec3& point = mesh.vertices[vertex].position;
        input.vertices.row(vertex) << point.x, point.y, point.z;
    }
    input.faces.resize(triangles.size(), 3);
    for (int face = 0; face < static_cast<int>(triangles.size()); ++face)
        for (int corner = 0; corner < 3; ++corner) input.faces(face, corner) = triangles[face][corner];
    return true;
}

Vec3 faceNormal(const TriangleInput& input, int face) {
    const Eigen::Vector3d a = input.vertices.row(input.faces(face, 0));
    const Eigen::Vector3d b = input.vertices.row(input.faces(face, 1));
    const Eigen::Vector3d c = input.vertices.row(input.faces(face, 2));
    const Eigen::Vector3d normal = (b - a).cross(c - a).normalized();
    return Vec3(static_cast<float>(normal.x()), static_cast<float>(normal.y()), static_cast<float>(normal.z()));
}

Eigen::SparseMatrix<double> faceMass(const TriangleInput& input) {
    const int faces = input.faces.rows();
    Eigen::SparseMatrix<double> mass(3 * faces, 3 * faces);
    for (int face = 0; face < faces; ++face) {
        const Eigen::Vector3d a = input.vertices.row(input.faces(face, 0));
        const Eigen::Vector3d b = input.vertices.row(input.faces(face, 1));
        const Eigen::Vector3d c = input.vertices.row(input.faces(face, 2));
        const double area = 0.5 * (b - a).cross(c - a).norm();
        mass.insert(face, face) = area;
        mass.insert(faces + face, faces + face) = area;
        mass.insert(2 * faces + face, 2 * faces + face) = area;
    }
    mass.makeCompressed();
    return mass;
}

bool integrateField(const TriangleInput& input, const std::vector<Vec3>& field, Eigen::VectorXd& scalar) {
    const int faceCount = input.faces.rows();
    Eigen::SparseMatrix<double> gradient;
    igl::grad(input.vertices, input.faces, gradient);
    Eigen::VectorXd target(3 * faceCount);
    for (int face = 0; face < faceCount; ++face) {
        target(face) = field[face].x;
        target(faceCount + face) = field[face].y;
        target(2 * faceCount + face) = field[face].z;
    }
    const Eigen::SparseMatrix<double> mass = faceMass(input);
    Eigen::SparseMatrix<double> system = gradient.transpose() * mass * gradient;
    system.coeffRef(0, 0) += 1.0;
    system.makeCompressed();
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(system);
    if (solver.info() != Eigen::Success) return false;
    scalar = solver.solve(gradient.transpose() * mass * target);
    return solver.info() == Eigen::Success && scalar.allFinite();
}

std::vector<Vec3> gradients(const TriangleInput& input, const Eigen::VectorXd& scalar) {
    Eigen::SparseMatrix<double> gradient;
    igl::grad(input.vertices, input.faces, gradient);
    const Eigen::VectorXd values = gradient * scalar;
    const int faceCount = input.faces.rows();
    std::vector<Vec3> result(faceCount);
    for (int face = 0; face < faceCount; ++face)
        result[face] = Vec3(static_cast<float>(values(face)),
                            static_cast<float>(values(faceCount + face)),
                            static_cast<float>(values(2 * faceCount + face)));
    return result;
}

void extractIsolines(const TriangleInput& input, const Eigen::VectorXd& scalar, float spacing,
                     std::vector<std::vector<Vec3>>& result) {
    result.clear();
    const int first = static_cast<int>(std::ceil(scalar.minCoeff() / spacing));
    const int last = static_cast<int>(std::floor(scalar.maxCoeff() / spacing));
    if (last < first || last - first > 512) return;
    Eigen::VectorXd levels(last - first + 1);
    for (int i = 0; i < levels.size(); ++i) levels(i) = (first + i) * spacing;
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi edges;
    Eigen::VectorXi edgeLevels;
    igl::isolines(input.vertices, input.faces, scalar, levels, vertices, edges, edgeLevels);
    std::vector<std::vector<int>> incident(vertices.rows());
    for (int edge = 0; edge < edges.rows(); ++edge) {
        incident[edges(edge, 0)].push_back(edge);
        incident[edges(edge, 1)].push_back(edge);
    }
    std::vector<char> used(edges.rows(), 0);
    for (int seed = 0; seed < edges.rows(); ++seed) {
        if (used[seed]) continue;
        const int level = edgeLevels(seed);
        std::vector<int> path{edges(seed, 0), edges(seed, 1)};
        used[seed] = 1;
        const auto extend = [&](bool prepend) {
            int endpoint = prepend ? path.front() : path.back();
            while (true) {
                int next = -1;
                for (const int edge : incident[endpoint])
                    if (!used[edge] && edgeLevels(edge) == level) { next = edge; break; }
                if (next < 0) break;
                used[next] = 1;
                const int vertex = edges(next, 0) == endpoint ? edges(next, 1) : edges(next, 0);
                if (prepend) path.insert(path.begin(), vertex); else path.push_back(vertex);
                endpoint = vertex;
            }
        };
        extend(false);
        extend(true);
        std::vector<Vec3> line;
        line.reserve(path.size());
        for (const int vertex : path) {
            const Eigen::Vector3d point = vertices.row(vertex);
            line.emplace_back(static_cast<float>(point.x()), static_cast<float>(point.y()), static_cast<float>(point.z()));
        }
        result.push_back(std::move(line));
    }
}

float maximumAbsolute(const Eigen::VectorXd& values) {
    return values.size() == 0 ? 0.0f : static_cast<float>(values.cwiseAbs().maxCoeff());
}

} // namespace

Dev2PqResult Dev2PqRemesher::remesh(const MeshData& mesh, const Dev2PqOptions& options) const {
    Dev2PqResult result;
    if (options.stripSpacing <= 0.0f || options.alignmentWeight <= 0.0f) {
        result.diagnostic = "spacing and alignment weight must be positive";
        return result;
    }
    TriangleInput input;
    if (!triangleInput(mesh, input, result.diagnostic)) return result;
    const int faceCount = input.faces.rows();

    Eigen::MatrixXd principalA, principalB;
    Eigen::VectorXd curvatureA, curvatureB;
    std::vector<int> badVertices;
    igl::principal_curvature(input.vertices, input.faces, principalA, principalB, curvatureA, curvatureB, badVertices, 8, true);
    if (principalA.rows() != input.vertices.rows() || principalB.rows() != input.vertices.rows()) {
        result.diagnostic = "libigl principal curvature failed";
        return result;
    }

    result.faceCentres.resize(faceCount);
    result.rawRulings.resize(faceCount);
    result.confidence.resize(faceCount, 0.0f);
    std::vector<Vec3> normals(faceCount);
    std::vector<Vec3> targetGradients(faceCount);
    for (int face = 0; face < faceCount; ++face) {
        normals[face] = faceNormal(input, face);
        const Eigen::Vector3d centre = (input.vertices.row(input.faces(face, 0)) + input.vertices.row(input.faces(face, 1)) + input.vertices.row(input.faces(face, 2))) / 3.0;
        result.faceCentres[face] = Vec3(static_cast<float>(centre.x()), static_cast<float>(centre.y()), static_cast<float>(centre.z()));
        Vec3 ruling;
        float confidence = 0.0f;
        for (int corner = 0; corner < 3; ++corner) {
            const int vertex = input.faces(face, corner);
            const float a = static_cast<float>(std::abs(curvatureA(vertex)));
            const float b = static_cast<float>(std::abs(curvatureB(vertex)));
            const Eigen::Vector3d direction = a <= b ? principalA.row(vertex) : principalB.row(vertex);
            Vec3 candidate(static_cast<float>(direction.x()), static_cast<float>(direction.y()), static_cast<float>(direction.z()));
            candidate -= normals[face] * candidate.dot(normals[face]);
            if (candidate.lengthSquared() <= kEpsilon) continue;
            candidate.normalize();
            if (ruling.lengthSquared() > kEpsilon && ruling.dot(candidate) < 0.0f) candidate = -candidate;
            ruling += candidate;
            confidence += std::abs(a - b) / (a + b + 1e-8f);
        }
        if (ruling.lengthSquared() <= kEpsilon) {
            const Eigen::Vector3d edge = input.vertices.row(input.faces(face, 1)) - input.vertices.row(input.faces(face, 0));
            ruling = Vec3(static_cast<float>(edge.x()), static_cast<float>(edge.y()), static_cast<float>(edge.z())).normalized();
        } else ruling.normalize();
        result.rawRulings[face] = ruling;
        result.confidence[face] = confidence / 3.0f;
        targetGradients[face] = normals[face].cross(ruling).normalized();
    }

    directional::TriMesh directionalMesh;
    directionalMesh.set_mesh(input.vertices, input.faces);
    directional::PCFaceTangentBundle tangentBundle;
    tangentBundle.init(directionalMesh);
    Eigen::VectorXi allFaces = Eigen::VectorXi::LinSpaced(faceCount, 0, faceCount - 1);
    Eigen::MatrixXd constraints(faceCount, 3);
    Eigen::VectorXd weights(faceCount);
    for (int face = 0; face < faceCount; ++face) {
        constraints.row(face) << targetGradients[face].x, targetGradients[face].y, targetGradients[face].z;
        weights(face) = std::max(0.01f, result.confidence[face]) * options.alignmentWeight;
    }

    directional::CartesianField powerField;
    directional::power_field(tangentBundle, allFaces, constraints, weights, 2, powerField, true);
    directional::CartesianField rawGamma;
    directional::power_to_raw(powerField, 2, rawGamma, true);
    directional::principal_matching(rawGamma);
    result.singularityCount = rawGamma.singLocalCycles.size();

    const Eigen::SparseMatrix<double> curl = directional::curl_matrix_2D<double>(directionalMesh, true, 2, 1, rawGamma.matching);
    result.maxCurlBefore = maximumAbsolute(curl * rawGamma.flatten(true));

    directional::CartesianField curlFreeGamma = rawGamma;
    if (options.useDirectionalCurlProjection) {
        directional::project_curl(rawGamma, Eigen::VectorXi(), Eigen::MatrixXd(), curlFreeGamma);
        directional::principal_matching(curlFreeGamma);
    }
    result.maxCurlAfter = maximumAbsolute(curl * curlFreeGamma.flatten(true));

    std::vector<Vec3> gradientField(faceCount);
    for (int face = 0; face < faceCount; ++face) {
        const Eigen::RowVector3d gamma = curlFreeGamma.extField.block(face, 0, 1, 3);
        gradientField[face] = Vec3(static_cast<float>(gamma.x()), static_cast<float>(gamma.y()), static_cast<float>(gamma.z()));
    }
    Eigen::VectorXd scalar;
    if (!integrateField(input, gradientField, scalar)) {
        result.diagnostic = "failed to integrate Directional's curl-projected field";
        return result;
    }
    result.scalarU.assign(scalar.data(), scalar.data() + scalar.size());
    const std::vector<Vec3> finalGradients = gradients(input, scalar);
    result.optimizedRulings.resize(faceCount);
    for (int face = 0; face < faceCount; ++face) {
        Vec3 ruling = normals[face].cross(finalGradients[face]);
        if (ruling.lengthSquared() <= kEpsilon) ruling = result.rawRulings[face];
        else ruling.normalize();
        if (ruling.dot(result.rawRulings[face]) < 0.0f) ruling = -ruling;
        result.optimizedRulings[face] = ruling;
    }
    const Eigen::RowVector3d lower = input.vertices.colwise().minCoeff();
    const Eigen::RowVector3d upper = input.vertices.colwise().maxCoeff();
    const float spacing = options.stripSpacing * std::max(static_cast<float>((upper - lower).norm()), 1e-5f);
    extractIsolines(input, scalar, spacing, result.isolines);
    result.success = true;
    result.diagnostic = "Directional power-2 field + principal matching + curl projection";
    return result;
}

} // namespace alice2

#else

namespace alice2 {
Dev2PqResult Dev2PqRemesher::remesh(const MeshData&, const Dev2PqOptions&) const {
    Dev2PqResult result;
    result.diagnostic = "Directional support is disabled at configure time";
    return result;
}
} // namespace alice2

#endif
