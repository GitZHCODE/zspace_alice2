#include "Dev2PqRemesher.h"

#if ALICE2_WITH_DIRECTIONAL

#include <Eigen/Core>
#include <igl/principal_curvature.h>

#include <directional/TriMesh.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/curl_matrices.h>
#include <directional/integrate.h>
#include <directional/isolines.h>
#include <directional/power_field.h>
#include <directional/power_to_raw.h>
#include <directional/principal_matching.h>
#include <directional/project_curl.h>
#include <directional/setup_integration.h>

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

void extractDirectionalIsolines(const directional::TriMesh& cutMesh, const Eigen::MatrixXd& functions,
                                std::vector<std::vector<Vec3>>& result) {
    result.clear();
    if (functions.rows() != cutMesh.V.rows() || functions.cols() == 0) return;

    // A p=2 field is sign-symmetric, so one branch represents the same set of
    // unoriented strip lines as its negative.  The cut mesh carries the seam
    // copies needed to keep that branch single-valued during extraction.
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi edges;
    Eigen::MatrixXi originalEdges;
    Eigen::MatrixXd normals;
    directional::isolines(cutMesh.V, cutMesh.F, functions.col(0), 100, vertices, edges, originalEdges, normals);
    result.reserve(edges.rows());
    for (int edge = 0; edge < edges.rows(); ++edge) {
        const Eigen::Vector3d a = vertices.row(edges(edge, 0));
        const Eigen::Vector3d b = vertices.row(edges(edge, 1));
        if ((a - b).squaredNorm() <= 1e-20) continue;
        result.push_back({Vec3(static_cast<float>(a.x()), static_cast<float>(a.y()), static_cast<float>(a.z())),
                          Vec3(static_cast<float>(b.x()), static_cast<float>(b.y()), static_cast<float>(b.z()))});
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
    }
    const Eigen::SparseMatrix<double> projectedCurl =
        directional::curl_matrix_2D<double>(directionalMesh, true, 2, 1, curlFreeGamma.matching);
    result.maxCurlAfter = maximumAbsolute(projectedCurl * curlFreeGamma.flatten(true));

    result.optimizedRulings.resize(faceCount);
    for (int face = 0; face < faceCount; ++face) {
        const Eigen::RowVector3d gamma = curlFreeGamma.extField.block(face, 0, 1, 3);
        Vec3 ruling = normals[face].cross(
            Vec3(static_cast<float>(gamma.x()), static_cast<float>(gamma.y()), static_cast<float>(gamma.z())));
        if (ruling.lengthSquared() <= kEpsilon) ruling = result.rawRulings[face];
        else ruling.normalize();
        if (ruling.dot(result.rawRulings[face]) < 0.0f) ruling = -ruling;
        result.optimizedRulings[face] = ruling;
    }

    // Directional's integration preserves the p=2 matching and cuts only the
    // necessary seam graph.  A normal Poisson solve on the original vertices
    // cannot represent this branched function and produces false connections.
    directional::IntegrationData integration(2);
    integration.lengthRatio = options.stripSpacing;
    integration.integralSeamless = true;
    integration.roundSeams = false;
    directional::TriMesh cutMesh;
    directional::CartesianField combedGamma;
    directional::setup_integration(curlFreeGamma, integration, cutMesh, combedGamma);
    Eigen::MatrixXd functions;
    Eigen::MatrixXd cornerFunctions;
    if (!directional::integrate(combedGamma, integration, cutMesh, functions, cornerFunctions) ||
        functions.rows() == 0 || !functions.allFinite()) {
        result.diagnostic = "Directional seamless integration failed";
        return result;
    }
    result.scalarU.assign(functions.col(0).data(), functions.col(0).data() + functions.rows());
    extractDirectionalIsolines(cutMesh, functions, result.isolines);
    result.success = true;
    result.diagnostic = "Directional p=2 field + matching-aware curl projection + seamless integration";
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
