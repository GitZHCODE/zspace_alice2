#include "Dev2PqRemesher.h"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <igl/grad.h>
#include <igl/isolines.h>
#include <igl/planarize_quad_mesh.h>
#include <igl/principal_curvature.h>
#include <igl/quad_planarity.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <queue>
#include <set>
#include <sstream>

namespace alice2 {
namespace {
constexpr double kEpsilon = 1e-10;

using EdgeKey = std::pair<int, int>;

struct TriangleInput {
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi faces;
};

struct SurfacePoint {
    Vec3 position;
    Vec3 normal;
    int face{-1};
};

EdgeKey edgeKey(int a, int b) { return a < b ? EdgeKey{a, b} : EdgeKey{b, a}; }

bool buildTriangleInput(const MeshData& mesh, TriangleInput& input, std::string& diagnostic) {
    if (mesh.vertices.empty() || mesh.faces.empty()) {
        diagnostic = "input mesh is empty";
        return false;
    }
    std::vector<std::array<int, 3>> triangles;
    for (const MeshFace& face : mesh.faces) {
        if (face.vertices.size() < 3) continue;
        for (int id : face.vertices) {
            if (id < 0 || id >= static_cast<int>(mesh.vertices.size())) {
                diagnostic = "input mesh has an invalid vertex index";
                return false;
            }
        }
        for (int corner = 1; corner + 1 < static_cast<int>(face.vertices.size()); ++corner) {
            const std::array<int, 3> triangle{face.vertices[0], face.vertices[corner], face.vertices[corner + 1]};
            const Vec3& a = mesh.vertices[triangle[0]].position;
            const Vec3& b = mesh.vertices[triangle[1]].position;
            const Vec3& c = mesh.vertices[triangle[2]].position;
            if ((b - a).cross(c - a).lengthSquared() <= static_cast<float>(kEpsilon)) {
                diagnostic = "input mesh contains a degenerate triangle";
                return false;
            }
            triangles.push_back(triangle);
        }
    }
    if (triangles.empty()) {
        diagnostic = "input mesh has no usable faces";
        return false;
    }
    std::map<EdgeKey, int> edgeUses;
    for (const auto& triangle : triangles) {
        for (int edge = 0; edge < 3; ++edge) {
            const EdgeKey key = edgeKey(triangle[edge], triangle[(edge + 1) % 3]);
            if (++edgeUses[key] > 2) {
                diagnostic = "input mesh is non-manifold";
                return false;
            }
        }
    }
    input.vertices.resize(mesh.vertices.size(), 3);
    for (int i = 0; i < static_cast<int>(mesh.vertices.size()); ++i) {
        const Vec3& point = mesh.vertices[i].position;
        input.vertices.row(i) << point.x, point.y, point.z;
    }
    input.faces.resize(triangles.size(), 3);
    for (int i = 0; i < static_cast<int>(triangles.size()); ++i)
        for (int corner = 0; corner < 3; ++corner) input.faces(i, corner) = triangles[i][corner];
    return true;
}

std::vector<std::vector<int>> faceAdjacency(const Eigen::MatrixXi& faces) {
    std::map<EdgeKey, std::vector<int>> edgeFaces;
    for (int face = 0; face < faces.rows(); ++face)
        for (int edge = 0; edge < 3; ++edge)
            edgeFaces[edgeKey(faces(face, edge), faces(face, (edge + 1) % 3))].push_back(face);
    std::vector<std::vector<int>> adjacency(faces.rows());
    for (const auto& [edge, incident] : edgeFaces) {
        if (incident.size() != 2) continue;
        adjacency[incident[0]].push_back(incident[1]);
        adjacency[incident[1]].push_back(incident[0]);
    }
    return adjacency;
}

Vec3 faceNormal(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces, int face) {
    const Eigen::Vector3d a = vertices.row(faces(face, 0));
    const Eigen::Vector3d b = vertices.row(faces(face, 1));
    const Eigen::Vector3d c = vertices.row(faces(face, 2));
    const Eigen::Vector3d normal = (b - a).cross(c - a).normalized();
    return Vec3(static_cast<float>(normal.x()), static_cast<float>(normal.y()), static_cast<float>(normal.z()));
}

bool solveScalarField(const Eigen::MatrixXd& vertices,
                      const Eigen::MatrixXi& faces,
                      const std::vector<Vec3>& faceTarget,
                      Eigen::VectorXd& values) {
    Eigen::SparseMatrix<double> gradient;
    igl::grad(vertices, faces, gradient);
    Eigen::VectorXd target(3 * faces.rows());
    for (int face = 0; face < faces.rows(); ++face) {
        target(3 * face) = faceTarget[face].x;
        target(3 * face + 1) = faceTarget[face].y;
        target(3 * face + 2) = faceTarget[face].z;
    }
    Eigen::SparseMatrix<double> system = gradient.transpose() * gradient;
    system.coeffRef(0, 0) += 1.0; // deterministic gauge for the integrated scalar
    system.makeCompressed();
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(system);
    if (solver.info() != Eigen::Success) return false;
    values = solver.solve(gradient.transpose() * target);
    return solver.info() == Eigen::Success && values.allFinite();
}

std::vector<Vec3> evaluateFaceGradient(const Eigen::MatrixXd& vertices,
                                       const Eigen::MatrixXi& faces,
                                       const Eigen::VectorXd& scalar) {
    Eigen::SparseMatrix<double> gradient;
    igl::grad(vertices, faces, gradient);
    const Eigen::VectorXd values = gradient * scalar;
    std::vector<Vec3> result(faces.rows());
    for (int face = 0; face < faces.rows(); ++face)
        result[face] = Vec3(static_cast<float>(values(3 * face)),
                            static_cast<float>(values(3 * face + 1)),
                            static_cast<float>(values(3 * face + 2)));
    return result;
}

bool barycentric(const Vec2& point, const Vec2& a, const Vec2& b, const Vec2& c,
                 float& u, float& v, float& w) {
    const float denominator = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    if (std::abs(denominator) <= 1e-7f) return false;
    u = ((b.y - c.y) * (point.x - c.x) + (c.x - b.x) * (point.y - c.y)) / denominator;
    v = ((c.y - a.y) * (point.x - c.x) + (a.x - c.x) * (point.y - c.y)) / denominator;
    w = 1.0f - u - v;
    return u >= -1e-4f && v >= -1e-4f && w >= -1e-4f;
}

bool locateParameterPoint(const TriangleInput& input,
                          const Eigen::VectorXd& uValues,
                          const Eigen::VectorXd& vValues,
                          const Vec2& parameter,
                          SurfacePoint& sample) {
    for (int face = 0; face < input.faces.rows(); ++face) {
        const int a = input.faces(face, 0), b = input.faces(face, 1), c = input.faces(face, 2);
        float wa = 0.0f, wb = 0.0f, wc = 0.0f;
        if (!barycentric(parameter,
                         Vec2(static_cast<float>(uValues(a)), static_cast<float>(vValues(a))),
                         Vec2(static_cast<float>(uValues(b)), static_cast<float>(vValues(b))),
                         Vec2(static_cast<float>(uValues(c)), static_cast<float>(vValues(c))),
                         wa, wb, wc)) continue;
        const Eigen::Vector3d point = wa * input.vertices.row(a) + wb * input.vertices.row(b) + wc * input.vertices.row(c);
        sample.position = Vec3(static_cast<float>(point.x()), static_cast<float>(point.y()), static_cast<float>(point.z()));
        sample.normal = faceNormal(input.vertices, input.faces, face);
        sample.face = face;
        return true;
    }
    return false;
}

void buildRulingIsolines(const TriangleInput& input,
                         const Eigen::VectorXd& scalar,
                         float spacing,
                         std::vector<std::vector<Vec3>>& result) {
    const double minimum = scalar.minCoeff();
    const double maximum = scalar.maxCoeff();
    const int first = static_cast<int>(std::ceil(minimum / spacing));
    const int last = static_cast<int>(std::floor(maximum / spacing));
    if (last < first || last - first > 512) return;
    Eigen::VectorXd levels(last - first + 1);
    for (int index = 0; index < levels.size(); ++index) levels(index) = (first + index) * spacing;
    Eigen::MatrixXd lineVertices;
    Eigen::MatrixXi lineEdges;
    Eigen::VectorXi lineLevels;
    igl::isolines(input.vertices, input.faces, scalar, levels, lineVertices, lineEdges, lineLevels);
    for (int edge = 0; edge < lineEdges.rows(); ++edge) {
        const Eigen::Vector3d a = lineVertices.row(lineEdges(edge, 0));
        const Eigen::Vector3d b = lineVertices.row(lineEdges(edge, 1));
        result.push_back({Vec3(static_cast<float>(a.x()), static_cast<float>(a.y()), static_cast<float>(a.z())),
                          Vec3(static_cast<float>(b.x()), static_cast<float>(b.y()), static_cast<float>(b.z()))});
    }
}

void rebuildEdges(MeshData& mesh) {
    std::set<EdgeKey> edges;
    for (const MeshFace& face : mesh.faces) {
        for (int corner = 0; corner < static_cast<int>(face.vertices.size()); ++corner)
            edges.insert(edgeKey(face.vertices[corner], face.vertices[(corner + 1) % face.vertices.size()]));
    }
    mesh.edges.clear();
    mesh.edges.reserve(edges.size());
    for (const EdgeKey& edge : edges) mesh.edges.emplace_back(edge.first, edge.second, Color(0.08f, 0.36f, 0.12f, 1.0f));
}

void appendFallbackTriangle(MeshData& output, const TriangleInput& input, int face) {
    const int first = static_cast<int>(output.vertices.size());
    const Vec3 normal = faceNormal(input.vertices, input.faces, face);
    for (int corner = 0; corner < 3; ++corner) {
        const Eigen::Vector3d point = input.vertices.row(input.faces(face, corner));
        output.vertices.emplace_back(Vec3(static_cast<float>(point.x()), static_cast<float>(point.y()), static_cast<float>(point.z())),
                                     normal, Color(0.85f, 0.42f, 0.06f, 1.0f));
    }
    output.faces.emplace_back(std::vector<int>{first, first + 1, first + 2}, normal, Color(0.85f, 0.42f, 0.06f, 1.0f));
}
} // namespace

Dev2PqResult Dev2PqRemesher::remesh(const MeshData& mesh, const Dev2PqOptions& options) const {
    Dev2PqResult result;
    if (options.stripSpacing <= 0.0f || options.fieldIterations < 1) {
        result.diagnostic = "strip spacing must be positive and field iterations must be nonzero";
        return result;
    }

    TriangleInput input;
    if (!buildTriangleInput(mesh, input, result.diagnostic)) return result;
    const auto adjacency = faceAdjacency(input.faces);

    // Use the direction of the strongest absolute bend as the stable quantity,
    // then obtain the developable ruling as its tangent-plane perpendicular.
    std::vector<Vec3> vertexMajorDirections(input.vertices.rows());
    std::vector<float> vertexConfidence(input.vertices.rows(), 0.0f);
    if (options.curvatureEstimator == Dev2PqCurvatureEstimator::MeshObject) {
        // Match the local curvature cross field used by sketch_stress_aligned_streamlines.
        MeshObject curvatureMesh("dev2pq_curvature_input");
        curvatureMesh.setMeshData(std::make_shared<MeshData>(mesh));
        const auto curvature = curvatureMesh.principalCurvature(false);
        if (curvature.principalDirections.size() != mesh.vertices.size() ||
            curvature.k1.size() != mesh.vertices.size() ||
            curvature.k2.size() != mesh.vertices.size()) {
            result.diagnostic = "MeshObject principal-curvature estimation failed";
            return result;
        }
        for (int vertex = 0; vertex < input.vertices.rows(); ++vertex) {
            const float k1Abs = std::abs(curvature.k1[vertex]);
            const float k2Abs = std::abs(curvature.k2[vertex]);
            vertexMajorDirections[vertex] = curvature.principalDirections[vertex].normalized();
            vertexConfidence[vertex] = std::abs(k1Abs - k2Abs) / (k1Abs + k2Abs + 1e-8f);
        }
    } else {
        Eigen::MatrixXd principalA, principalB;
        Eigen::VectorXd curvatureA, curvatureB;
        std::vector<int> badVertices;
        igl::principal_curvature(input.vertices, input.faces,
                                 principalA, principalB, curvatureA, curvatureB,
                                 badVertices, 8, true);
        if (principalA.rows() != input.vertices.rows() ||
            principalB.rows() != input.vertices.rows()) {
            result.diagnostic = "libigl principal-curvature estimation failed";
            return result;
        }
        for (int vertex = 0; vertex < input.vertices.rows(); ++vertex) {
            const double curvatureAAbs = std::abs(curvatureA(vertex));
            const double curvatureBAbs = std::abs(curvatureB(vertex));
            const bool aIsMajor = curvatureAAbs >= curvatureBAbs;
            const Eigen::Vector3d direction = aIsMajor ? principalA.row(vertex) : principalB.row(vertex);
            vertexMajorDirections[vertex] = Vec3(static_cast<float>(direction.x()),
                                                  static_cast<float>(direction.y()),
                                                  static_cast<float>(direction.z())).normalized();
            vertexConfidence[vertex] = static_cast<float>(std::abs(curvatureAAbs - curvatureBAbs) /
                                                           (curvatureAAbs + curvatureBAbs + 1e-8));
        }
    }

    std::vector<Vec3> normals(input.faces.rows());
    result.faceRulings.resize(input.faces.rows());
    result.faceConfidence.resize(input.faces.rows());
    for (int face = 0; face < input.faces.rows(); ++face) {
        normals[face] = faceNormal(input.vertices, input.faces, face);
        Vec3 major;
        float confidence = 0.0f;
        for (int corner = 0; corner < 3; ++corner) {
            const int vertex = input.faces(face, corner);
            Vec3 candidate = vertexMajorDirections[vertex] - normals[face] * vertexMajorDirections[vertex].dot(normals[face]);
            if (candidate.lengthSquared() <= 1e-10f) continue;
            candidate.normalize();
            if (major.lengthSquared() > 1e-10f && major.dot(candidate) < 0.0f) candidate = -candidate;
            major += candidate;
            confidence += vertexConfidence[vertex];
        }
        Vec3 ruling;
        if (major.lengthSquared() <= 1e-10f) {
            const Eigen::Vector3d edge = input.vertices.row(input.faces(face, 1)) - input.vertices.row(input.faces(face, 0));
            ruling = Vec3(static_cast<float>(edge.x()), static_cast<float>(edge.y()), static_cast<float>(edge.z())).normalized();
        } else {
            major.normalize();
            ruling = normals[face].cross(major).normalized();
        }
        result.faceRulings[face] = ruling;
        result.faceConfidence[face] = confidence / 3.0f;
    }
    result.rawFaceRulings = result.faceRulings;
    result.rawFaceConfidence = result.faceConfidence;

    // Curvature magnitude alone is not enough: a noisy surface can report a strong
    // principal direction that has no consistent continuation. Rulings are line
    // fields, hence the absolute dot product is the sign-invariant coherence test.
    std::vector<float> coherence(input.faces.rows(), 1.0f);
    for (int face = 0; face < input.faces.rows(); ++face) {
        float agreement = 0.0f;
        int count = 0;
        for (int neighbor : adjacency[face]) {
            if (normals[face].dot(normals[neighbor]) < 0.4f) continue;
            agreement += std::abs(result.faceRulings[face].dot(result.faceRulings[neighbor]));
            ++count;
        }
        if (count > 0) coherence[face] = agreement / static_cast<float>(count);
        // Keep coherent fields, but quickly de-emphasize nearly random directions.
        const float stableFraction = std::clamp((coherence[face] - 0.35f) / 0.55f, 0.0f, 1.0f);
        result.faceConfidence[face] *= stableFraction;
    }

    // Alternating field smoothing and integration: smoothing keeps rulings coherent,
    // while each least-squares integration projects their perpendicular field onto gradients.
    for (int iteration = 0; iteration < options.fieldIterations; ++iteration) {
        std::vector<Vec3> smoothed = result.faceRulings;
        for (int face = 0; face < input.faces.rows(); ++face) {
            Vec3 average = result.faceRulings[face] * std::max(0.05f, result.faceConfidence[face]);
            float weight = std::max(0.05f, result.faceConfidence[face]);
            for (int neighbor : adjacency[face]) {
                if (normals[face].dot(normals[neighbor]) < 0.4f) continue; // retain sharp-fold freedom
                Vec3 candidate = result.faceRulings[neighbor];
                if (candidate.dot(result.faceRulings[face]) < 0.0f) candidate = -candidate;
                const float candidateWeight = std::max(0.05f, result.faceConfidence[neighbor]);
                average += candidate * candidateWeight;
                weight += candidateWeight;
            }
            average = average / weight;
            average -= normals[face] * average.dot(normals[face]);
            if (average.lengthSquared() > 1e-10f) {
                average.normalize();
                smoothed[face] = (result.faceRulings[face] * (1.0f - options.fieldSmoothing) + average * options.fieldSmoothing).normalized();
            }
        }
        result.faceRulings = std::move(smoothed);

        std::vector<Vec3> targetU(input.faces.rows());
        for (int face = 0; face < input.faces.rows(); ++face) targetU[face] = normals[face].cross(result.faceRulings[face]).normalized();
        Eigen::VectorXd scalar;
        if (!solveScalarField(input.vertices, input.faces, targetU, scalar)) {
            result.diagnostic = "failed to integrate the ruling-perpendicular field";
            return result;
        }
        const std::vector<Vec3> gradients = evaluateFaceGradient(input.vertices, input.faces, scalar);
        for (int face = 0; face < input.faces.rows(); ++face) {
            Vec3 gradient = gradients[face];
            gradient -= normals[face] * gradient.dot(normals[face]);
            if (gradient.lengthSquared() <= 1e-10f) continue;
            gradient.normalize();
            if (gradient.dot(targetU[face]) < 0.0f) gradient = -gradient;
            result.faceRulings[face] = normals[face].cross(
                (targetU[face] * (1.0f - options.fieldSmoothing) + gradient * options.fieldSmoothing).normalized()).normalized();
        }
    }

    std::vector<Vec3> targetU(input.faces.rows()), targetV(input.faces.rows());
    for (int face = 0; face < input.faces.rows(); ++face) {
        targetU[face] = normals[face].cross(result.faceRulings[face]).normalized();
        targetV[face] = result.faceRulings[face];
    }
    Eigen::VectorXd scalarU, scalarV;
    if (!solveScalarField(input.vertices, input.faces, targetU, scalarU) ||
        !solveScalarField(input.vertices, input.faces, targetV, scalarV)) {
        result.diagnostic = "failed to integrate strip coordinates";
        return result;
    }
    result.scalarU.assign(scalarU.data(), scalarU.data() + scalarU.size());
    result.scalarV.assign(scalarV.data(), scalarV.data() + scalarV.size());

    const Eigen::RowVector3d lower = input.vertices.colwise().minCoeff();
    const Eigen::RowVector3d upper = input.vertices.colwise().maxCoeff();
    const float spacing = options.stripSpacing * std::max(static_cast<float>((upper - lower).norm()), 1e-5f);
    buildRulingIsolines(input, scalarU, spacing, result.rulingIsolines);

    const float minU = static_cast<float>(scalarU.minCoeff() / spacing);
    const float maxU = static_cast<float>(scalarU.maxCoeff() / spacing);
    const float minV = static_cast<float>(scalarV.minCoeff() / spacing);
    const float maxV = static_cast<float>(scalarV.maxCoeff() / spacing);
    const int firstU = static_cast<int>(std::floor(minU));
    const int lastU = static_cast<int>(std::ceil(maxU));
    const int firstV = static_cast<int>(std::floor(minV));
    const int lastV = static_cast<int>(std::ceil(maxV));
    const long long cellCount = static_cast<long long>(lastU - firstU) * static_cast<long long>(lastV - firstV);
    if (cellCount <= 0 || cellCount > 100000) {
        result.diagnostic = "integrated parameter domain is too large to sample";
        return result;
    }

    auto output = std::make_shared<MeshData>();
    std::map<std::pair<int, int>, int> gridVertices;
    const auto appendVertex = [&](int u, int v, const SurfacePoint& sample) {
        const std::pair<int, int> key{u, v};
        const auto existing = gridVertices.find(key);
        if (existing != gridVertices.end()) return existing->second;
        const int index = static_cast<int>(output->vertices.size());
        output->vertices.emplace_back(sample.position, sample.normal, Color(0.13f, 0.60f, 0.22f, 1.0f));
        gridVertices.emplace(key, index);
        return index;
    };

    for (int v = firstV; v < lastV; ++v) {
        for (int u = firstU; u < lastU; ++u) {
            SurfacePoint corners[4];
            SurfacePoint centre;
            const Vec2 parameters[4]{{u * spacing, v * spacing}, {(u + 1) * spacing, v * spacing},
                                     {(u + 1) * spacing, (v + 1) * spacing}, {u * spacing, (v + 1) * spacing}};
            bool valid = locateParameterPoint(input, scalarU, scalarV,
                                              Vec2((u + 0.5f) * spacing, (v + 0.5f) * spacing), centre);
            for (int corner = 0; corner < 4 && valid; ++corner)
                valid = locateParameterPoint(input, scalarU, scalarV, parameters[corner], corners[corner]);
            if (!valid || centre.face < 0 || result.faceConfidence[centre.face] < options.confidenceThreshold) continue;
            int indices[4]{appendVertex(u, v, corners[0]), appendVertex(u + 1, v, corners[1]),
                           appendVertex(u + 1, v + 1, corners[2]), appendVertex(u, v + 1, corners[3])};
            const Vec3 normal = (corners[1].position - corners[0].position).cross(corners[2].position - corners[0].position);
            if (normal.dot(centre.normal) < 0.0f) std::swap(indices[1], indices[3]);
            output->faces.emplace_back(std::vector<int>{indices[0], indices[1], indices[2], indices[3]}, centre.normal,
                                       Color(0.13f, 0.60f, 0.22f, 1.0f));
            ++result.quadCount;
        }
    }

    // Flat or unreliable ruling regions remain represented by intrinsically planar source triangles.
    for (int face = 0; face < input.faces.rows(); ++face) {
        if (result.faceConfidence[face] >= options.confidenceThreshold) continue;
        appendFallbackTriangle(*output, input, face);
        ++result.planarFaceCount;
    }

    if (result.quadCount > 1) {
        Eigen::MatrixXd vertices(output->vertices.size(), 3);
        Eigen::MatrixXi quads(result.quadCount, 4);
        for (int vertex = 0; vertex < static_cast<int>(output->vertices.size()); ++vertex) {
            const Vec3& point = output->vertices[vertex].position;
            vertices.row(vertex) << point.x, point.y, point.z;
        }
        int quad = 0;
        for (const MeshFace& face : output->faces) {
            if (face.vertices.size() != 4) continue;
            for (int corner = 0; corner < 4; ++corner) quads(quad, corner) = face.vertices[corner];
            ++quad;
        }
        Eigen::MatrixXd planarized;
        igl::planarize_quad_mesh(vertices, quads, options.planarizationIterations, options.planarityTolerance, planarized);
        for (int vertex = 0; vertex < static_cast<int>(output->vertices.size()); ++vertex)
            output->vertices[vertex].position = Vec3(static_cast<float>(planarized(vertex, 0)),
                                                       static_cast<float>(planarized(vertex, 1)),
                                                       static_cast<float>(planarized(vertex, 2)));
        Eigen::VectorXd planarity;
        igl::quad_planarity(planarized, quads, planarity);
        result.maxQuadNonPlanarity = planarity.size() == 0 ? 0.0f : static_cast<float>(planarity.maxCoeff());
    }
    rebuildEdges(*output);
    output->calculateNormals();
    output->triangulationDirty = true;
    result.mesh = std::move(output);
    result.success = true;
    std::ostringstream message;
    message << "Dev2PQ prototype | quads " << result.quadCount << " | planar faces " << result.planarFaceCount
            << " | max quad residual " << result.maxQuadNonPlanarity;
    result.diagnostic = message.str();
    return result;
}

} // namespace alice2
