#include "DevelopableRibbon.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>

namespace alice2 {
namespace {

constexpr float kEpsilon = 1e-8f;
constexpr double kPi = 3.14159265358979323846;

struct Edge {
    int a = -1;
    int b = -1;

    Edge() = default;
    Edge(int first, int second) : a(std::min(first, second)), b(std::max(first, second)) {}

    bool operator<(const Edge& other) const {
        return a != other.a ? a < other.a : b < other.b;
    }

    bool contains(int vertex) const { return a == vertex || b == vertex; }
};

using EdgeFaces = std::map<Edge, std::vector<int>>;

std::array<Edge, 4> faceEdges(const std::array<int, 4>& face) {
    return {Edge(face[0], face[1]), Edge(face[1], face[2]),
            Edge(face[2], face[3]), Edge(face[3], face[0])};
}

bool hasEdge(const std::array<int, 4>& face, int a, int b) {
    const Edge target(a, b);
    for (const Edge& edge : faceEdges(face)) {
        if (edge.a == target.a && edge.b == target.b) return true;
    }
    return false;
}

int otherVertex(const Edge& edge, int vertex) {
    return edge.a == vertex ? edge.b : (edge.b == vertex ? edge.a : -1);
}

float clampUnit(float value) {
    return std::max(-1.0f, std::min(1.0f, value));
}

double wrapAngle(double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

Vec3 faceNormal(const QuadRibbon& ribbon, int faceIndex) {
    const std::array<int, 4>& f = ribbon.faces[faceIndex];
    const Vec3& a = ribbon.vertices[f[0]];
    const Vec3& b = ribbon.vertices[f[1]];
    const Vec3& d = ribbon.vertices[f[3]];
    return (b - a).cross(d - a).normalized();
}

float facePlanarityError(const QuadRibbon& ribbon, const std::array<int, 4>& f) {
    const Vec3& a = ribbon.vertices[f[0]];
    const Vec3& b = ribbon.vertices[f[1]];
    const Vec3& c = ribbon.vertices[f[2]];
    const Vec3& d = ribbon.vertices[f[3]];
    const Vec3 cross = (b - a).cross(d - a);
    const float denominator = cross.length() * (c - a).length() + kEpsilon;
    return std::abs(cross.dot(c - a)) / denominator;
}

void setDiagnostic(std::string* diagnostic, const std::string& message) {
    if (diagnostic) *diagnostic = message;
}

std::pair<double, double> meanAndDeviation(const std::vector<RibbonSignature>& signatures, bool bendChannel) {
    std::vector<double> bends;
    for (const RibbonSignature& signature : signatures) {
        const std::vector<double>& values = bendChannel ? signature.bend : signature.rulingAngle;
        bends.insert(bends.end(), values.begin(), values.end());
    }
    if (bends.empty()) return {0.0, 1.0};
    const double mean = std::accumulate(bends.begin(), bends.end(), 0.0) / bends.size();
    double variance = 0.0;
    for (double value : bends) variance += (value - mean) * (value - mean);
    return {mean, std::max(std::sqrt(variance / bends.size()), 1e-12)};
}

double normalisedSignatureDistance(const RibbonSignature& a,
                                   const RibbonSignature& b,
                                   double bendDeviation,
                                   double angleDeviation,
                                   double bendWeight,
                                   double rulingWeight,
                                   bool reverseB) {
    if (a.bend.size() != b.bend.size() || a.rulingAngle.size() != b.rulingAngle.size() ||
        a.bend.size() != a.rulingAngle.size() || a.bend.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    double bendSum = 0.0;
    double rulingSum = 0.0;
    for (size_t i = 0; i < a.bend.size(); ++i) {
        const size_t bi = reverseB ? a.bend.size() - 1 - i : i;
        const double otherBend = reverseB ? -b.bend[bi] : b.bend[bi];
        const double otherAngle = reverseB ? wrapAngle(kPi - b.rulingAngle[bi]) : b.rulingAngle[bi];
        const double bendDifference = (a.bend[i] - otherBend) / bendDeviation;
        const double angleDifference = wrapAngle(a.rulingAngle[i] - otherAngle) / angleDeviation;
        bendSum += bendDifference * bendDifference;
        rulingSum += angleDifference * angleDifference;
    }
    return std::sqrt(std::max(0.0, bendWeight) * bendSum / a.bend.size() +
                     std::max(0.0, rulingWeight) * rulingSum / a.bend.size());
}

bool sequentialPlanarize(QuadRibbon& ribbon) {
    bool projectedAnyFace = false;
    for (size_t i = 0; i < ribbon.faces.size(); ++i) {
        const int p0 = ribbon.railP[i];
        const int p1 = ribbon.railP[i + 1];
        const int q0 = ribbon.railQ[i];
        const int q1 = ribbon.railQ[i + 1];
        const Vec3 normal = (ribbon.vertices[p1] - ribbon.vertices[p0]).cross(ribbon.vertices[q0] - ribbon.vertices[p0]);
        const float normalLengthSquared = normal.lengthSquared();
        if (normalLengthSquared <= kEpsilon) continue;
        const float signedDistance = normal.dot(ribbon.vertices[q1] - ribbon.vertices[p0]) / normalLengthSquared;
        ribbon.vertices[q1] -= normal * signedDistance;
        projectedAnyFace = true;
    }
    return projectedAnyFace;
}

} // namespace

bool orderRibbon(const MeshData& mesh, QuadRibbon& ribbon, std::string* diagnostic) {
    ribbon = QuadRibbon{};
    if (mesh.faces.empty()) {
        setDiagnostic(diagnostic, "Ribbon mesh contains no faces.");
        return false;
    }

    std::vector<std::array<int, 4>> inputFaces;
    inputFaces.reserve(mesh.faces.size());
    const int vertexCount = static_cast<int>(mesh.vertices.size());
    for (size_t i = 0; i < mesh.faces.size(); ++i) {
        const std::vector<int>& vertices = mesh.faces[i].vertices;
        if (vertices.size() != 4) {
            setDiagnostic(diagnostic, "Face " + std::to_string(i) + " is not a quad.");
            return false;
        }
        std::array<int, 4> face{vertices[0], vertices[1], vertices[2], vertices[3]};
        for (int vertex : face) {
            if (vertex < 0 || vertex >= vertexCount) {
                setDiagnostic(diagnostic, "Face " + std::to_string(i) + " has an invalid vertex index.");
                return false;
            }
        }
        inputFaces.push_back(face);
    }

    EdgeFaces edgeFaces;
    for (int fi = 0; fi < static_cast<int>(inputFaces.size()); ++fi) {
        for (const Edge& edge : faceEdges(inputFaces[fi])) edgeFaces[edge].push_back(fi);
    }

    std::vector<std::vector<int>> neighbours(inputFaces.size());
    std::map<std::pair<int, int>, Edge> sharedEdges;
    for (const auto& [edge, incident] : edgeFaces) {
        if (incident.size() > 2) {
            setDiagnostic(diagnostic, "Ribbon is non-manifold: an edge belongs to more than two faces.");
            return false;
        }
        if (incident.size() == 2) {
            const int a = incident[0];
            const int b = incident[1];
            neighbours[a].push_back(b);
            neighbours[b].push_back(a);
            sharedEdges[{std::min(a, b), std::max(a, b)}] = edge;
        }
    }

    std::vector<int> ends;
    for (int fi = 0; fi < static_cast<int>(neighbours.size()); ++fi) {
        if (neighbours[fi].size() == 1) ends.push_back(fi);
        else if (neighbours[fi].size() != 2 && inputFaces.size() != 1) {
            setDiagnostic(diagnostic, "Face adjacency is not an open chain.");
            return false;
        }
    }
    if (inputFaces.size() > 1 && ends.size() != 2) {
        setDiagnostic(diagnostic, "A valid open ribbon must have exactly two end faces.");
        return false;
    }

    std::vector<int> orderedFaces;
    orderedFaces.reserve(inputFaces.size());
    int previous = -1;
    int current = inputFaces.size() == 1 ? 0 : ends.front();
    while (current != -1) {
        orderedFaces.push_back(current);
        int next = -1;
        for (int neighbour : neighbours[current]) {
            if (neighbour != previous) {
                next = neighbour;
                break;
            }
        }
        previous = current;
        current = next;
    }
    if (orderedFaces.size() != inputFaces.size()) {
        setDiagnostic(diagnostic, "Ribbon faces are disconnected.");
        return false;
    }

    std::vector<Edge> stations;
    if (inputFaces.size() == 1) {
        const std::array<int, 4>& f = inputFaces.front();
        ribbon.railP = {f[0], f[1]};
        ribbon.railQ = {f[3], f[2]};
    } else {
        stations.reserve(inputFaces.size() + 1);
        const Edge firstShared = sharedEdges[{std::min(orderedFaces[0], orderedFaces[1]),
                                              std::max(orderedFaces[0], orderedFaces[1])}];
        Edge firstEnd;
        bool foundFirstEnd = false;
        for (const Edge& edge : faceEdges(inputFaces[orderedFaces[0]])) {
            if (!edge.contains(firstShared.a) && !edge.contains(firstShared.b)) {
                firstEnd = edge;
                foundFirstEnd = true;
                break;
            }
        }
        if (!foundFirstEnd) {
            setDiagnostic(diagnostic, "Could not identify the first transverse boundary edge.");
            return false;
        }
        stations.push_back(firstEnd);
        for (size_t i = 0; i + 1 < orderedFaces.size(); ++i) {
            stations.push_back(sharedEdges[{std::min(orderedFaces[i], orderedFaces[i + 1]),
                                            std::max(orderedFaces[i], orderedFaces[i + 1])}]);
        }
        const Edge lastShared = stations.back();
        Edge lastEnd;
        bool foundLastEnd = false;
        for (const Edge& edge : faceEdges(inputFaces[orderedFaces.back()])) {
            if (!edge.contains(lastShared.a) && !edge.contains(lastShared.b)) {
                lastEnd = edge;
                foundLastEnd = true;
                break;
            }
        }
        if (!foundLastEnd) {
            setDiagnostic(diagnostic, "Could not identify the last transverse boundary edge.");
            return false;
        }
        stations.push_back(lastEnd);

        ribbon.railP.reserve(stations.size());
        ribbon.railQ.reserve(stations.size());
        ribbon.railP.push_back(stations[0].a);
        ribbon.railQ.push_back(stations[0].b);
        for (size_t i = 0; i + 1 < stations.size(); ++i) {
            const std::array<int, 4>& face = inputFaces[orderedFaces[i]];
            const int p = ribbon.railP.back();
            int nextP = -1;
            for (int candidate : {stations[i + 1].a, stations[i + 1].b}) {
                if (hasEdge(face, p, candidate)) {
                    nextP = candidate;
                    break;
                }
            }
            if (nextP == -1) {
                setDiagnostic(diagnostic, "Could not propagate a consistent rail through face " + std::to_string(orderedFaces[i]) + ".");
                return false;
            }
            const int nextQ = otherVertex(stations[i + 1], nextP);
            if (nextQ == -1 || !hasEdge(face, ribbon.railQ.back(), nextQ)) {
                setDiagnostic(diagnostic, "Input quads do not form a one-cell-wide ribbon.");
                return false;
            }
            ribbon.railP.push_back(nextP);
            ribbon.railQ.push_back(nextQ);
        }
    }

    ribbon.vertices.reserve(mesh.vertices.size());
    for (const MeshVertex& vertex : mesh.vertices) ribbon.vertices.push_back(vertex.position);
    ribbon.faces.reserve(inputFaces.size());
    for (size_t i = 0; i < inputFaces.size(); ++i) {
        ribbon.faces.push_back({ribbon.railP[i], ribbon.railP[i + 1], ribbon.railQ[i + 1], ribbon.railQ[i]});
        ribbon.sourceFaceIndices.push_back(orderedFaces[i]);
    }
    setDiagnostic(diagnostic, "Ribbon ordered: " + std::to_string(ribbon.faces.size()) + " faces, " +
                                std::to_string(ribbon.railP.size()) + " stations.");
    return true;
}

float maxRibbonPlanarityError(const QuadRibbon& ribbon) {
    float maximum = 0.0f;
    for (const std::array<int, 4>& face : ribbon.faces) maximum = std::max(maximum, facePlanarityError(ribbon, face));
    return maximum;
}

RibbonPlanarizationResult planarizeRibbon(QuadRibbon& ribbon,
                                          int maxIterations,
                                          float tolerance,
                                          float originalWeight) {
    RibbonPlanarizationResult result;
    if (ribbon.vertices.empty() || ribbon.faces.empty()) return result;
    maxIterations = std::max(0, maxIterations);
    tolerance = std::max(0.0f, tolerance);
    originalWeight = std::max(0.0f, originalWeight);
    const std::vector<Vec3> original = ribbon.vertices;
    result.maxPlanarityError = maxRibbonPlanarityError(ribbon);

    while (result.iterations < maxIterations && result.maxPlanarityError > tolerance) {
        std::vector<Vec3> projectedSum(ribbon.vertices.size(), Vec3{});
        std::vector<float> projectedCount(ribbon.vertices.size(), 0.0f);
        for (const std::array<int, 4>& face : ribbon.faces) {
            Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
            for (int id : face) {
                const Vec3& p = ribbon.vertices[id];
                centroid += Eigen::Vector3d(p.x, p.y, p.z);
            }
            centroid /= 4.0;
            Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
            for (int id : face) {
                const Vec3& p = ribbon.vertices[id];
                const Eigen::Vector3d delta(p.x - centroid.x(), p.y - centroid.y(), p.z - centroid.z());
                covariance += delta * delta.transpose();
            }
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
            if (solver.info() != Eigen::Success) continue;
            const Eigen::Vector3d normal = solver.eigenvectors().col(0);
            for (int id : face) {
                const Vec3& p = ribbon.vertices[id];
                Eigen::Vector3d point(p.x, p.y, p.z);
                point -= normal * normal.dot(point - centroid);
                projectedSum[id] += Vec3(static_cast<float>(point.x()), static_cast<float>(point.y()), static_cast<float>(point.z()));
                projectedCount[id] += 1.0f;
            }
        }
        for (size_t id = 0; id < ribbon.vertices.size(); ++id) {
            if (projectedCount[id] <= 0.0f) continue;
            ribbon.vertices[id] = (projectedSum[id] + original[id] * originalWeight) /
                                  (projectedCount[id] + originalWeight);
        }
        ++result.iterations;
        result.maxPlanarityError = maxRibbonPlanarityError(ribbon);
    }
    if (result.maxPlanarityError > tolerance && sequentialPlanarize(ribbon)) {
        result.usedSequentialFallback = true;
        result.maxPlanarityError = maxRibbonPlanarityError(ribbon);
    }
    result.converged = result.maxPlanarityError <= tolerance;
    return result;
}

QuadRibbon offsetRibbonAlongVertexNormals(const QuadRibbon& ribbon, float offset) {
    QuadRibbon result = ribbon;
    std::vector<Vec3> normalSums(ribbon.vertices.size(), Vec3{});
    for (const std::array<int, 4>& face : ribbon.faces) {
        const Vec3& a = ribbon.vertices[face[0]];
        const Vec3& b = ribbon.vertices[face[1]];
        const Vec3& d = ribbon.vertices[face[3]];
        const Vec3 normal = (b - a).cross(d - a).normalized();
        for (int vertex : face) {
            if (vertex >= 0 && vertex < static_cast<int>(normalSums.size())) normalSums[vertex] += normal;
        }
    }
    for (size_t vertex = 0; vertex < result.vertices.size(); ++vertex) {
        if (normalSums[vertex].lengthSquared() > kEpsilon) result.vertices[vertex] += normalSums[vertex].normalized() * offset;
    }
    return result;
}

std::vector<RibbonSignature> buildRibbonSignatures(const QuadRibbon& ribbon,
                                                    int facesPerStrip,
                                                    int stride) {
    std::vector<RibbonSignature> signatures;
    if (facesPerStrip < 2 || facesPerStrip > static_cast<int>(ribbon.faces.size())) return signatures;
    if (stride <= 0) stride = facesPerStrip;

    std::vector<Vec3> normals(ribbon.faces.size());
    for (int i = 0; i < static_cast<int>(ribbon.faces.size()); ++i) normals[i] = faceNormal(ribbon, i);
    for (int start = 0; start + facesPerStrip <= static_cast<int>(ribbon.faces.size()); start += stride) {
        RibbonSignature signature;
        signature.startFace = start;
        signature.faceCount = facesPerStrip;
        signature.bend.reserve(facesPerStrip - 1);
        signature.rulingAngle.reserve(facesPerStrip - 1);
        for (int localStation = 1; localStation < facesPerStrip; ++localStation) {
            const int station = start + localStation;
            const Vec3 ruling = (ribbon.vertices[ribbon.railQ[station]] - ribbon.vertices[ribbon.railP[station]]).normalized();
            const Vec3 previousNormal = normals[station - 1];
            const Vec3 nextNormal = normals[station];
            const double bend = std::atan2(ruling.dot(previousNormal.cross(nextNormal)),
                                           clampUnit(previousNormal.dot(nextNormal)));

            const Vec3 previousCentre = (ribbon.vertices[ribbon.railP[station - 1]] + ribbon.vertices[ribbon.railQ[station - 1]]) * 0.5f;
            const Vec3 nextCentre = (ribbon.vertices[ribbon.railP[station + 1]] + ribbon.vertices[ribbon.railQ[station + 1]]) * 0.5f;
            const Vec3 tangent = (nextCentre - previousCentre).normalized();
            Vec3 averageNormal = previousNormal + nextNormal;
            if (averageNormal.lengthSquared() <= kEpsilon) averageNormal = previousNormal;
            else averageNormal.normalize();
            const double beta = std::atan2(averageNormal.dot(tangent.cross(ruling)), tangent.dot(ruling));
            signature.bend.push_back(bend);
            signature.rulingAngle.push_back(beta);
        }
        signatures.push_back(std::move(signature));
    }
    return signatures;
}

double ribbonSignatureDistance(const RibbonSignature& a,
                               const RibbonSignature& b,
                               double bendWeight,
                               double rulingWeight,
                               bool reverseB) {
    if (a.bend.size() != b.bend.size() || a.rulingAngle.size() != b.rulingAngle.size() ||
        a.bend.size() != a.rulingAngle.size() || a.bend.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    const size_t count = a.bend.size();
    double bendSum = 0.0;
    double rulingSum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const size_t bi = reverseB ? count - 1 - i : i;
        const double otherBend = reverseB ? -b.bend[bi] : b.bend[bi];
        const double otherAngle = reverseB ? wrapAngle(kPi - b.rulingAngle[bi]) : b.rulingAngle[bi];
        const double bendDifference = a.bend[i] - otherBend;
        const double angleDifference = wrapAngle(a.rulingAngle[i] - otherAngle);
        bendSum += bendDifference * bendDifference;
        rulingSum += angleDifference * angleDifference;
    }
    return std::sqrt(std::max(0.0, bendWeight) * bendSum / count +
                     std::max(0.0, rulingWeight) * rulingSum / count);
}

std::vector<RibbonMatch> findSimilarRibbonStrips(const std::vector<RibbonSignature>& signatures,
                                                  int topK,
                                                  double bendWeight,
                                                  double rulingWeight) {
    std::vector<RibbonMatch> matches;
    if (topK <= 0 || signatures.size() < 2) return matches;
    const double bendDeviation = meanAndDeviation(signatures, true).second;
    const double angleDeviation = meanAndDeviation(signatures, false).second;
    for (int a = 0; a < static_cast<int>(signatures.size()); ++a) {
        for (int b = a + 1; b < static_cast<int>(signatures.size()); ++b) {
            const double forward = normalisedSignatureDistance(signatures[a], signatures[b], bendDeviation, angleDeviation,
                                                               bendWeight, rulingWeight, false);
            const double reversed = normalisedSignatureDistance(signatures[a], signatures[b], bendDeviation, angleDeviation,
                                                                bendWeight, rulingWeight, true);
            if (!std::isfinite(forward) && !std::isfinite(reversed)) continue;
            matches.push_back({a, b, std::min(forward, reversed), reversed < forward});
        }
    }
    std::sort(matches.begin(), matches.end(), [](const RibbonMatch& left, const RibbonMatch& right) {
        return left.distance < right.distance;
    });
    if (matches.size() > static_cast<size_t>(topK)) matches.resize(static_cast<size_t>(topK));
    return matches;
}

} // namespace alice2
