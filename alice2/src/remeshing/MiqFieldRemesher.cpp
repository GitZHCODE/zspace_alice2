#include "MiqFieldRemesher.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <queue>
#include <set>
#include <sstream>

#ifndef ALICE2_WITH_MIQ
#define ALICE2_WITH_MIQ 0
#endif

#if ALICE2_WITH_MIQ
#include <Eigen/Core>
#include <igl/copyleft/comiso/miq.h>
#endif

namespace alice2 {
namespace {
constexpr float kEpsilon = 1e-6f;
constexpr float kTangentTolerance = 1e-3f;

struct EdgeUse { int face; int start; int end; };
using EdgeMap = std::map<std::pair<int, int>, std::vector<EdgeUse>>;

std::pair<int, int> sortedEdge(int a, int b) { return a < b ? std::make_pair(a, b) : std::make_pair(b, a); }
bool validVertex(const MeshData& mesh, int id) { return id >= 0 && id < static_cast<int>(mesh.vertices.size()); }
float uvComponent(const Vec2& uv, int axis) { return axis == 0 ? uv.x : uv.y; }

EdgeMap collectEdges(const MeshData& mesh) {
    EdgeMap edges;
    for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
        const auto& face = mesh.faces[fi];
        for (int i = 0; i < 3; ++i) {
            const int a = face.vertices[i];
            const int b = face.vertices[(i + 1) % 3];
            edges[sortedEdge(a, b)].push_back({fi, a, b});
        }
    }
    return edges;
}

bool validateTopology(const MeshData& mesh, std::string& message) {
    if (mesh.vertices.empty() || mesh.faces.empty()) { message = "input mesh is empty"; return false; }
    for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
        const auto& face = mesh.faces[fi];
        if (face.vertices.size() != 3) { message = "MIQ requires triangle-only input (face " + std::to_string(fi) + ")"; return false; }
        for (int id : face.vertices) {
            if (!validVertex(mesh, id)) { message = "face " + std::to_string(fi) + " references an invalid vertex"; return false; }
        }
        if (mesh.calculateFaceNormal(face).lengthSquared() <= kEpsilon * kEpsilon) {
            message = "face " + std::to_string(fi) + " is degenerate"; return false;
        }
    }

    const EdgeMap edges = collectEdges(mesh);
    std::vector<std::vector<int>> faceNeighbors(mesh.faces.size());
    std::map<int, std::vector<int>> boundary;
    for (const auto& [edge, uses] : edges) {
        if (uses.size() > 2) { message = "input mesh is non-manifold"; return false; }
        if (uses.size() == 2) {
            if (uses[0].start == uses[1].start && uses[0].end == uses[1].end) {
                message = "input mesh has inconsistent face winding"; return false;
            }
            faceNeighbors[uses[0].face].push_back(uses[1].face);
            faceNeighbors[uses[1].face].push_back(uses[0].face);
        } else if (uses.size() == 1) {
            boundary[edge.first].push_back(edge.second);
            boundary[edge.second].push_back(edge.first);
        }
    }
    if (boundary.empty()) { message = "MIQ spike supports one-boundary patches only; the input mesh is closed"; return false; }
    for (const auto& [vertex, neighbors] : boundary) {
        if (neighbors.size() != 2) { message = "input mesh boundary is not a manifold loop"; return false; }
    }
    std::set<int> visitedBoundary;
    std::queue<int> boundaryQueue;
    boundaryQueue.push(boundary.begin()->first);
    visitedBoundary.insert(boundary.begin()->first);
    while (!boundaryQueue.empty()) {
        const int current = boundaryQueue.front(); boundaryQueue.pop();
        for (int next : boundary[current]) if (visitedBoundary.insert(next).second) boundaryQueue.push(next);
    }
    if (visitedBoundary.size() != boundary.size()) { message = "MIQ spike supports exactly one boundary loop"; return false; }

    std::vector<char> visitedFaces(mesh.faces.size(), 0);
    std::queue<int> faceQueue;
    faceQueue.push(0); visitedFaces[0] = 1;
    int count = 0;
    while (!faceQueue.empty()) {
        const int current = faceQueue.front(); faceQueue.pop(); ++count;
        for (int next : faceNeighbors[current]) if (!visitedFaces[next]) { visitedFaces[next] = 1; faceQueue.push(next); }
    }
    if (count != static_cast<int>(mesh.faces.size())) { message = "input mesh must be one connected surface patch"; return false; }
    return true;
}

bool validateField(const MeshData& mesh, const TensorField& field, std::string& message) {
    if (field.size() != mesh.faces.size()) { message = "field must contain exactly one tensor per input face"; return false; }
    for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
        const auto& tensor = field[fi];
        const float majorLength = tensor.majorDirection.length();
        const float minorLength = tensor.minorDirection.length();
        if (majorLength <= kEpsilon || minorLength <= kEpsilon) { message = "field contains a zero direction on face " + std::to_string(fi); return false; }
        const Vec3 normal = mesh.calculateFaceNormal(mesh.faces[fi]).normalized();
        const Vec3 major = tensor.majorDirection / majorLength;
        const Vec3 minor = tensor.minorDirection / minorLength;
        if (std::abs(major.dot(normal)) > kTangentTolerance || std::abs(minor.dot(normal)) > kTangentTolerance) {
            message = "field directions must be tangent to face " + std::to_string(fi); return false;
        }
        if (std::abs(major.dot(minor)) > kTangentTolerance) {
            message = "field directions must be orthogonal on face " + std::to_string(fi); return false;
        }
    }
    return true;
}

void appendIsoSegments(const Vec3 p[3], const Vec2 uv[3], int axis, std::vector<std::vector<Vec3>>& lines) {
    float minimum = uvComponent(uv[0], axis), maximum = minimum;
    for (int i = 1; i < 3; ++i) {
        minimum = std::min(minimum, uvComponent(uv[i], axis));
        maximum = std::max(maximum, uvComponent(uv[i], axis));
    }
    for (int level = static_cast<int>(std::ceil(minimum - 1e-5f)); level <= static_cast<int>(std::floor(maximum + 1e-5f)); ++level) {
        std::vector<Vec3> hits;
        for (int edge = 0; edge < 3; ++edge) {
            const int a = edge, b = (edge + 1) % 3;
            const float av = uvComponent(uv[a], axis) - static_cast<float>(level);
            const float bv = uvComponent(uv[b], axis) - static_cast<float>(level);
            if (std::abs(av) <= 1e-5f && std::abs(bv) <= 1e-5f) continue;
            if ((av < -1e-5f && bv < -1e-5f) || (av > 1e-5f && bv > 1e-5f)) continue;
            const float denominator = av - bv;
            if (std::abs(denominator) <= kEpsilon) continue;
            const Vec3 point = p[a] + (p[b] - p[a]) * std::clamp(av / denominator, 0.0f, 1.0f);
            bool duplicate = false;
            for (const Vec3& hit : hits) if ((point - hit).lengthSquared() < 1e-10f) { duplicate = true; break; }
            if (!duplicate) hits.push_back(point);
        }
        if (hits.size() == 2) lines.push_back({hits[0], hits[1]});
    }
}

void buildGridLines(const MeshData& mesh, MiqRemeshResult& result) {
    for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
        if (fi >= static_cast<int>(result.uvFaces.size())) break;
        Vec3 p[3]; Vec2 uv[3]; bool valid = true;
        for (int corner = 0; corner < 3; ++corner) {
            const int vertexId = mesh.faces[fi].vertices[corner];
            const int uvId = result.uvFaces[fi][corner];
            if (!validVertex(mesh, vertexId) || uvId < 0 || uvId >= static_cast<int>(result.uv.size())) { valid = false; break; }
            p[corner] = mesh.vertices[vertexId].position;
            uv[corner] = result.uv[uvId];
        }
        if (!valid) continue;
        appendIsoSegments(p, uv, 0, result.gridLines.u);
        appendIsoSegments(p, uv, 1, result.gridLines.v);
    }
}
} // namespace

MiqRemeshResult MiqFieldRemesher::parameterize(const MeshData& mesh, const TensorField& field, const MiqRemeshOptions& options) const {
    MiqRemeshResult result;
    if (options.targetSpacing <= kEpsilon) { result.diagnostic = "target spacing must be positive"; return result; }
    if (!validateTopology(mesh, result.diagnostic) || !validateField(mesh, field, result.diagnostic)) return result;
#if !ALICE2_WITH_MIQ
    result.diagnostic = "MIQ support is disabled; configure with -DALICE2_ENABLE_MIQ_REMESH=ON";
    return result;
#else
    Eigen::MatrixXd V(mesh.vertices.size(), 3), PD1(mesh.faces.size(), 3), PD2(mesh.faces.size(), 3);
    Eigen::MatrixXi F(mesh.faces.size(), 3);
    Vec3 minBounds = mesh.vertices.front().position, maxBounds = minBounds;
    for (int vi = 0; vi < static_cast<int>(mesh.vertices.size()); ++vi) {
        const Vec3& p = mesh.vertices[vi].position;
        V.row(vi) << p.x, p.y, p.z;
        minBounds.x = std::min(minBounds.x, p.x); minBounds.y = std::min(minBounds.y, p.y); minBounds.z = std::min(minBounds.z, p.z);
        maxBounds.x = std::max(maxBounds.x, p.x); maxBounds.y = std::max(maxBounds.y, p.y); maxBounds.z = std::max(maxBounds.z, p.z);
    }
    for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
        for (int corner = 0; corner < 3; ++corner) F(fi, corner) = mesh.faces[fi].vertices[corner];
        const Vec3 major = field[fi].majorDirection.normalized(), minor = field[fi].minorDirection.normalized();
        PD1.row(fi) << major.x, major.y, major.z;
        PD2.row(fi) << minor.x, minor.y, minor.z;
    }
    Eigen::MatrixXd UV;
    Eigen::MatrixXi FUV;
    const float diagonal = std::max((maxBounds - minBounds).length(), kEpsilon);
    try {
        igl::copyleft::comiso::miq(V, F, PD1, PD2, UV, FUV,
                                   static_cast<double>(diagonal / options.targetSpacing), 5.0, true,
                                   options.stiffnessIterations, options.localIterations, true, true);
    } catch (const std::exception& error) {
        result.diagnostic = std::string("MIQ solve failed: ") + error.what(); return result;
    }
    if (UV.rows() == 0 || FUV.rows() != F.rows() || FUV.cols() != 3) { result.diagnostic = "MIQ returned a malformed UV parameterization"; return result; }
    result.uv.reserve(UV.rows());
    for (int i = 0; i < UV.rows(); ++i) result.uv.emplace_back(static_cast<float>(UV(i, 0)), static_cast<float>(UV(i, 1)));
    std::set<int> usedUv;
    for (int fi = 0; fi < FUV.rows(); ++fi) {
        const std::array<int, 3> face{FUV(fi, 0), FUV(fi, 1), FUV(fi, 2)};
        for (int id : face) if (id < 0 || id >= UV.rows()) { result.diagnostic = "MIQ returned an invalid UV face index"; return result; }
        usedUv.insert(face[0]); usedUv.insert(face[1]); usedUv.insert(face[2]); result.uvFaces.push_back(face);
    }
    result.seamVertexCount = std::max(0, static_cast<int>(usedUv.size()) - static_cast<int>(mesh.vertices.size()));
    buildGridLines(mesh, result);
    result.success = true;
    std::ostringstream summary;
    summary << "MIQ UV " << result.uv.size() << " | seams " << result.seamVertexCount << " | u/v segments " << result.gridLines.u.size() << "/" << result.gridLines.v.size();
    result.diagnostic = summary.str();
    return result;
#endif
}
} // namespace alice2
