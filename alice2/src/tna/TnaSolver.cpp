#include "TnaSolver.h"

#include "../computeGeom/ComputeMesh.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <sstream>
#include <unordered_set>

namespace alice2 {
namespace {

bool appendBoundaryFace(MeshData& mesh, const std::vector<int>& group) {
    // Adjacent supports define an existing boundary edge, not a valid n-gon.
    if (group.size() < 3) return false;

    MeshFace face;
    face.vertices = group;
    face.normal = mesh.calculateFaceNormal(face);
    face.color = Color(0.78f, 0.84f, 1.0f, 1.0f);
    mesh.faces.push_back(std::move(face));
    return true;
}

std::vector<int> walkBoundaryLoop(const std::shared_ptr<HeMeshHalfedge>& first) {
    std::vector<int> vertices;
    if (!first) return vertices;

    auto current = first;
    do {
        const auto start = current->getStartVertex();
        if (!start) return {};
        vertices.push_back(start->getId());
        current = current->getNext();
    } while (current && current != first);

    return current == first ? vertices : std::vector<int>{};
}

Vec3 faceCentroid(const MeshData& mesh, const MeshFace& face) {
    Vec3 centroid(0.0f, 0.0f, 0.0f);
    if (face.vertices.empty()) return centroid;
    for (const int vertex : face.vertices) centroid = centroid + mesh.vertices[vertex].position;
    return centroid * (1.0f / static_cast<float>(face.vertices.size()));
}

float reciprocalAngleDegrees(const Vec3& formVector, const Vec3& forceVector) {
    const float formLength = formVector.length();
    const float forceLength = forceVector.length();
    if (formLength <= 1e-6f || forceLength <= 1e-6f) return 0.0f;
    const float cosine = std::clamp(std::abs(formVector.dot(forceVector)) /
                                        (formLength * forceLength),
                                    0.0f, 1.0f);
    return std::acos(cosine) * 180.0f / std::numbers::pi_v<float>;
}

} // namespace

TnaFormDiagram TnaSolver::makeFormDiagram(const MeshData& input,
                                          const std::vector<int>& requestedSupports) const {
    TnaFormDiagram result;
    if (input.vertices.empty() || input.faces.empty()) {
        result.diagnostic = "TNA form diagram needs a mesh with vertices and faces";
        return result;
    }

    ComputeMesh topology("tna_form_topology", input, true);
    std::vector<std::shared_ptr<HeMeshHalfedge>> boundaryStarts;
    std::unordered_set<int> boundaryVertexIds;
    std::unordered_set<int> visitedBoundaryHalfedges;
    for (const auto& halfedge : topology.getHalfedges()) {
        if (!halfedge || !halfedge->onBoundary() ||
            visitedBoundaryHalfedges.contains(halfedge->getId())) continue;

        const std::vector<int> loop = walkBoundaryLoop(halfedge);
        if (loop.empty()) {
            result.diagnostic = "Could not walk a closed boundary loop; check mesh face winding";
            return result;
        }

        auto current = halfedge;
        do {
            visitedBoundaryHalfedges.insert(current->getId());
            current = current->getNext();
        } while (current && current != halfedge);

        boundaryStarts.push_back(halfedge);
        boundaryVertexIds.insert(loop.begin(), loop.end());
    }

    if (boundaryStarts.empty()) {
        result.diagnostic = "TNA form diagram needs at least one boundary loop";
        return result;
    }

    std::unordered_set<int> supportSet;
    if (requestedSupports.empty()) {
        // Ordinary OBJ files have no vertex-colour support tags.
        supportSet = boundaryVertexIds;
    } else {
        for (const int vertex : requestedSupports) {
            if (vertex >= 0 && vertex < static_cast<int>(input.vertices.size()) &&
                boundaryVertexIds.contains(vertex)) {
                supportSet.insert(vertex);
            }
        }
        if (supportSet.empty()) {
            result.diagnostic = "Explicit support vertices must lie on a mesh boundary";
            return result;
        }
    }
    result.supportVertices.assign(supportSet.begin(), supportSet.end());
    std::sort(result.supportVertices.begin(), result.supportVertices.end());

    result.mesh = input;
    for (const auto& start : boundaryStarts) {
        const std::vector<int> loop = walkBoundaryLoop(start);
        std::vector<int> supportPositions;
        for (int i = 0; i < static_cast<int>(loop.size()); ++i) {
            if (supportSet.contains(loop[i])) supportPositions.push_back(i);
        }

        if (supportPositions.size() < 2) {
            // An opening with no supports still needs one exterior face for
            // the later dual construction.
            result.boundaryGroups.push_back(loop);
            if (appendBoundaryFace(result.mesh, loop)) ++result.appendedBoundaryFaces;
            else ++result.skippedDegenerateGroups;
            continue;
        }

        for (int groupIndex = 0; groupIndex < static_cast<int>(supportPositions.size()); ++groupIndex) {
            const int first = supportPositions[groupIndex];
            const int last = supportPositions[(groupIndex + 1) % supportPositions.size()];
            std::vector<int> group;
            for (int i = first;; i = (i + 1) % static_cast<int>(loop.size())) {
                group.push_back(loop[i]);
                if (i == last) break;
            }
            result.boundaryGroups.push_back(group);
            if (appendBoundaryFace(result.mesh, group)) ++result.appendedBoundaryFaces;
            else ++result.skippedDegenerateGroups;
        }
    }

    // Recreate the edge list so the support-to-support closing edge of each
    // exterior n-gon is visible and available to the next topology stage.
    MeshObject formObject("tna_form_diagram");
    formObject.setMeshData(std::make_shared<MeshData>(result.mesh));
    formObject.generateEdgesFromFaces();
    formObject.getMeshData()->calculateNormals();
    formObject.getMeshData()->triangulationDirty = true;
    result.mesh = *formObject.getMeshData();

    std::ostringstream status;
    status << "Form diagram: " << result.mesh.vertices.size() << " vertices, "
           << result.mesh.faces.size() << " faces, " << result.supportVertices.size()
           << " supports, " << result.boundaryGroups.size() << " boundary groups, "
           << result.appendedBoundaryFaces << " exterior n-gons";
    if (result.skippedDegenerateGroups > 0) {
        status << " (" << result.skippedDegenerateGroups << " adjacent-support groups kept as edges)";
    }
    result.diagnostic = status.str();
    result.success = true;
    return result;
}

TnaForceDiagram TnaSolver::makeForceDiagram(const MeshData& formDiagram) const {
    TnaForceDiagram result;
    if (formDiagram.vertices.empty() || formDiagram.faces.empty()) {
        result.diagnostic = "TNA force diagram needs a form mesh with vertices and faces";
        return result;
    }

    ComputeMesh topology("tna_force_topology", formDiagram, true);
    if (topology.getFaces().size() != formDiagram.faces.size()) {
        result.diagnostic = "Could not construct form half-edge topology for the force dual";
        return result;
    }

    // A dual vertex sits at every form-face centroid, including the exterior
    // n-gons created in stage 1.
    result.mesh.vertices.reserve(formDiagram.faces.size());
    result.forceVertexFormFaces.reserve(formDiagram.faces.size());
    for (const auto& face : topology.getFaces()) {
        const int formFace = face->getId();
        if (formFace < 0 || formFace >= static_cast<int>(formDiagram.faces.size())) continue;
        result.mesh.vertices.emplace_back(faceCentroid(formDiagram, formDiagram.faces[formFace]),
                                          Vec3(0.0f, 0.0f, 1.0f),
                                          Color(0.86f, 0.18f, 0.05f, 1.0f));
        result.forceVertexFormFaces.push_back(formFace);
    }

    // Every form edge shared by exactly two faces becomes a dual force edge.
    // The closing chord of an exterior n-gon has only one incident face and
    // is intentionally absent from this straightforward dual.
    for (const auto& edge : topology.getEdges()) {
        const std::vector<std::shared_ptr<HeMeshFace>> faces = edge->getFaces();
        if (faces.size() != 2) continue;
        const int firstFace = faces[0]->getId();
        const int secondFace = faces[1]->getId();
        if (firstFace < 0 || secondFace < 0 ||
            firstFace >= static_cast<int>(result.mesh.vertices.size()) ||
            secondFace >= static_cast<int>(result.mesh.vertices.size())) continue;

        result.mesh.edges.emplace_back(firstFace, secondFace, Color(0.86f, 0.18f, 0.05f, 1.0f));
        result.forceEdgeFormEdges.push_back(edge->getId());

        const auto [formA, formB] = edge->getVertices();
        if (!formA || !formB) {
            result.edgeAnglesDegrees.push_back(0.0f);
            continue;
        }
        const Vec3 formVector = formB->getPosition() - formA->getPosition();
        const Vec3 forceVector = result.mesh.vertices[secondFace].position -
                                 result.mesh.vertices[firstFace].position;
        result.edgeAnglesDegrees.push_back(reciprocalAngleDegrees(formVector, forceVector));
    }

    result.mesh.triangulationDirty = true;
    std::ostringstream status;
    status << "Force diagram: " << result.mesh.vertices.size() << " dual vertices, "
           << result.mesh.edges.size() << " dual edges";
    result.diagnostic = status.str();
    result.success = !result.mesh.vertices.empty();
    return result;
}

} // namespace alice2
