// #define __MAIN__
#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>
#include <computeGeom/ComputeMesh.h>

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

using namespace alice2;

class DevelopableRibbonSketch : public ISketch {
public:
    std::string getName() const override { return "Developable Ribbon"; }
    std::string getDescription() const override { return "Traverses stereotomy face blocks and displays them as thin solids"; }
    std::string getAuthor() const override { return "alice2 User"; }

    void setup() override {
        scene().setBackgroundColor(Color(1.0f, 1.0f, 1.0f));
        scene().setShowGrid(false);
        scene().setShowAxes(true);
        scene().setAxesLength(0.2f);
        m_ui = std::make_unique<SimpleUI>(input());
        m_ui->setTheme(SimpleUI::UITheme::Dark);
        m_ui->addSlider("Faces per strip", Vec2{10.0f, 52.0f}, 210.0f, 2.0f,
                        static_cast<float>(kMaxFacesPerStrip), m_facesPerStripSlider);
        m_ui->addSlider("Thickness (vertex normals)", Vec2{10.0f, 80.0f}, 210.0f, 0.0f, 0.10f, m_stripThickness);
        m_ui->addSlider("Max interface cost", Vec2{10.0f, 108.0f}, 210.0f, 0.0f, 5.0f, m_stackBreakThreshold);
        reload();
    }

    void update(float) override {
        const int requestedFaceCount = std::clamp(static_cast<int>(std::lround(m_facesPerStripSlider)), 2, kMaxFacesPerStrip);
        bool signaturesChanged = false;
        if (m_valid && requestedFaceCount != m_facesPerStrip) {
            m_facesPerStrip = requestedFaceCount;
            signaturesChanged = true;
        }
        if (m_valid && std::abs(m_stripThickness - m_lastStackThickness) > 1e-6f) signaturesChanged = true;
        if (signaturesChanged) refreshSignatures();
        if (m_valid && std::abs(m_stackBreakThreshold - m_lastStackBreakThreshold) > 1e-6f) {
            m_lastStackBreakThreshold = m_stackBreakThreshold;
            if (m_showStack) rebuildStackVisualisation();
            printStackPartition();
        }
    }

    void cleanup() override {
        clearRibbonBlockVisualisation();
        clearStackVisualisation();
    }

    void draw(Renderer& renderer, Camera&) override {
        renderer.setColor(Color(0.0f, 0.0f, 0.0f, 1.0f));
        renderer.drawString("r reload | p solve planar faces | s toggle stack | i ruling-ray check | k toolpath-safe stack | o toggle original wire | [ / ] strip faces", 10.0f, 30.0f);
        if (m_ui) m_ui->draw(renderer);
        renderer.setColor(Color(0.0f, 0.0f, 0.0f, 1.0f));
        renderer.drawString(m_status, 10.0f, 120.0f);
        if (m_showOriginal && m_original) drawEdges(renderer, *m_original->getMeshData(), Color(0.62f, 0.62f, 0.62f, 1.0f), 1.0f);
        if (m_valid) {
            drawRulings(renderer);
            drawStripIndices(renderer);
        }
        if (m_showStack) {
            drawStackAnnotations(renderer);
            drawStackBounds(renderer);
            if (m_showRulingRayCheck) drawRulingRayHits(renderer);
        }
    }

    bool onKeyPress(unsigned char key, int, int) override {
        switch (key) {
            case 'r': case 'R': reload(); return true;
            case 'p': case 'P': planarise(); return true;
            case 's': case 'S': toggleStackVisualisation(); return true;
            case 'i': case 'I': toggleRulingRayCheck(); return true;
            case 'k': case 'K': rebuildToolpathSafeStack(); return true;
            case 'o': case 'O': m_showOriginal = !m_showOriginal; return true;
            case '[': m_facesPerStrip = std::max(2, m_facesPerStrip - 1); m_facesPerStripSlider = static_cast<float>(m_facesPerStrip); refreshSignatures(); return true;
            case ']': m_facesPerStrip = std::min(kMaxFacesPerStrip, m_facesPerStrip + 1); m_facesPerStripSlider = static_cast<float>(m_facesPerStrip); refreshSignatures(); return true;
            default: return false;
        }
    }

    bool onMousePress(int button, int state, int x, int y) override {
        return m_ui && m_ui->onMousePress(button, state, x, y);
    }

    bool onMouseMove(int x, int y) override {
        return m_ui && m_ui->onMouseMove(x, y);
    }

private:
    static constexpr int kMaxFacesPerStrip = 36;

    struct FaceBlock {
        std::vector<int> sourceFaces;
        std::vector<std::pair<int, int>> walkEdges;
    };

    static bool extractFaceBlocks(const ComputeMesh& mesh, int facesPerBlock,
                                  std::vector<FaceBlock>& blocks, std::string* diagnostic) {
        const std::shared_ptr<MeshData> data = mesh.getMeshData();
        if (!data || data->faces.empty()) {
            if (diagnostic) *diagnostic = "Stereotomy mesh contains no faces.";
            return false;
        }
        blocks.clear();
        const auto& vertices = mesh.getVertices();
        std::vector<std::shared_ptr<HeMeshVertex>> starts;
        for (const auto& vertex : vertices) {
            if (!vertex || vertex->getValency() != 3) continue;
            int valencyTwo = 0;
            int valencyFour = 0;
            for (const auto& neighbour : vertex->getConnectedVertices()) {
                if (!neighbour) continue;
                valencyTwo += neighbour->getValency() == 2;
                valencyFour += neighbour->getValency() == 4;
            }
            if (valencyTwo == 2 && valencyFour == 1) starts.push_back(vertex);
        }
        if (starts.empty()) {
            if (diagnostic) *diagnostic = "No valency-3 centre-graph endpoint with two valency-2 neighbours was found.";
            return false;
        }

        std::vector<bool> assigned(data->faces.size(), false);
        int assignedCount = 0;
        auto flush = [&](FaceBlock& pending) {
            if (!pending.sourceFaces.empty()) {
                blocks.push_back(std::move(pending));
                pending = FaceBlock{};
            }
        };
        auto walkFrom = [&](const std::shared_ptr<HeMeshVertex>& start) -> bool {
            std::shared_ptr<HeMeshHalfedge> startHalfedge;
            for (const auto& halfedge : start->getHalfedges()) {
                if (halfedge && halfedge->getVertex() && halfedge->getVertex()->getValency() == 4) {
                    startHalfedge = halfedge;
                    break;
                }
            }
            if (!startHalfedge) return false;

            FaceBlock pending;
            std::shared_ptr<HeMeshHalfedge> current = startHalfedge;
            bool firstStep = true;
            const int guardLimit = std::max(1, static_cast<int>(mesh.getHalfedges().size()) * 2);
            for (int step = 0; step < guardLimit; ++step) {
                if (!firstStep && current == startHalfedge) {
                    flush(pending);
                    return true;
                }
                firstStep = false;
                if (!current || !current->getFace() || !current->getStartVertex() || !current->getVertex()) return false;
                const int face = current->getFace()->getId();
                if (face < 0 || face >= static_cast<int>(assigned.size())) return false;
                if (!assigned[face]) {
                    assigned[face] = true;
                    ++assignedCount;
                    pending.sourceFaces.push_back(face);
                    pending.walkEdges.push_back({current->getStartVertex()->getId(), current->getVertex()->getId()});
                    if (static_cast<int>(pending.sourceFaces.size()) == facesPerBlock) flush(pending);
                }

                if (current->getVertex()->getValency() == 3) {
                    flush(pending); // Turning always resets the N-face count.
                    current = current->getSymmetry();
                    if (!current || current->onBoundary()) return false;
                    continue;
                }
                const auto next = current->getNext();
                const auto twin = next ? next->getSymmetry() : nullptr;
                current = twin ? twin->getNext() : nullptr; // next -> twin -> next
                if (!current || current->onBoundary()) return false;
            }
            return false;
        };

        for (const auto& start : starts) {
            if (assignedCount == static_cast<int>(data->faces.size())) break;
            if (!walkFrom(start)) {
                if (diagnostic) *diagnostic = "Centre-graph next/twin/next walk did not close.";
                return false;
            }
        }
        if (assignedCount != static_cast<int>(data->faces.size())) {
            if (diagnostic) *diagnostic = "Centre-graph walks assigned " + std::to_string(assignedCount) + " of " +
                                          std::to_string(data->faces.size()) + " faces.";
            return false;
        }
        if (diagnostic) *diagnostic = "Stereotomy walk: " + std::to_string(blocks.size()) + " blocks from " +
                                      std::to_string(data->faces.size()) + " faces.";
        return true;
    }

    static Vec3 faceCentre(const MeshData& mesh, int faceIndex, const std::vector<Vec3>& positions) {
        Vec3 centre;
        const std::vector<int>& face = mesh.faces[faceIndex].vertices;
        for (int vertex : face) centre += positions[vertex];
        return face.empty() ? centre : centre / static_cast<float>(face.size());
    }

    static Vec3 faceNormal(const MeshData& mesh, int faceIndex, const std::vector<Vec3>& positions) {
        const std::vector<int>& face = mesh.faces[faceIndex].vertices;
        if (face.size() < 3) return Vec3{};
        return (positions[face[1]] - positions[face[0]]).cross(positions[face[2]] - positions[face[0]]).normalized();
    }

    // The traversal edge is not necessarily a ruling.  A ribbon ruling is
    // the transverse edge shared by two consecutive quad faces in its
    // ordered face walk.
    static bool sharedFaceEdge(const MeshData& mesh, int firstFace, int secondFace, std::pair<int, int>& edge) {
        if (firstFace < 0 || secondFace < 0 || firstFace >= static_cast<int>(mesh.faces.size()) ||
            secondFace >= static_cast<int>(mesh.faces.size())) return false;
        const std::vector<int>& first = mesh.faces[firstFace].vertices;
        const std::vector<int>& second = mesh.faces[secondFace].vertices;
        if (first.size() < 2 || second.size() < 2) return false;
        for (size_t firstVertex = 0; firstVertex < first.size(); ++firstVertex) {
            const int a = first[firstVertex];
            const int b = first[(firstVertex + 1) % first.size()];
            for (size_t secondVertex = 0; secondVertex < second.size(); ++secondVertex) {
                const int c = second[secondVertex];
                const int d = second[(secondVertex + 1) % second.size()];
                if ((a == c && b == d) || (a == d && b == c)) {
                    edge = {a, b};
                    return true;
                }
            }
        }
        return false;
    }

    static bool oppositeQuadEdge(const MeshData& mesh, int faceIndex, const std::pair<int, int>& edge,
                                 std::pair<int, int>& opposite) {
        if (faceIndex < 0 || faceIndex >= static_cast<int>(mesh.faces.size())) return false;
        const std::vector<int>& face = mesh.faces[faceIndex].vertices;
        if (face.size() != 4) return false;
        for (size_t vertex = 0; vertex < face.size(); ++vertex) {
            const int a = face[vertex];
            const int b = face[(vertex + 1) % face.size()];
            if (!((a == edge.first && b == edge.second) || (a == edge.second && b == edge.first))) continue;
            opposite = {face[(vertex + 2) % face.size()], face[(vertex + 3) % face.size()]};
            return true;
        }
        return false;
    }

    static std::vector<std::pair<int, int>> blockRulingEdges(const MeshData& mesh, const FaceBlock& block) {
        std::vector<std::pair<int, int>> result;
        if (block.sourceFaces.size() < 2) return result;
        std::vector<std::pair<int, int>> interior;
        interior.reserve(block.sourceFaces.size() - 1);
        for (size_t face = 1; face < block.sourceFaces.size(); ++face) {
            std::pair<int, int> edge;
            if (sharedFaceEdge(mesh, block.sourceFaces[face - 1], block.sourceFaces[face], edge)) {
                interior.push_back(edge);
            }
        }
        if (interior.empty()) return result;
        std::pair<int, int> startBoundary;
        if (oppositeQuadEdge(mesh, block.sourceFaces.front(), interior.front(), startBoundary)) {
            result.push_back(startBoundary);
        }
        result.insert(result.end(), interior.begin(), interior.end());
        std::pair<int, int> endBoundary;
        if (oppositeQuadEdge(mesh, block.sourceFaces.back(), interior.back(), endBoundary)) {
            result.push_back(endBoundary);
        }
        return result;
    }

    // Each solid must use normals averaged over its own faces.  Averaging over
    // the whole stereotomy mesh lets adjacent, unrelated strips tilt a lower
    // skin at shared vertices and corrupts both the solid and its signature.
    static std::vector<Vec3> offsetBlockAlongVertexNormals(const MeshData& mesh, const FaceBlock& block,
                                                            const std::vector<Vec3>& topPositions, float thickness) {
        std::vector<Vec3> result = topPositions;
        std::vector<Vec3> normalSums(topPositions.size(), Vec3{});
        for (int faceIndex : block.sourceFaces) {
            if (faceIndex < 0 || faceIndex >= static_cast<int>(mesh.faces.size())) continue;
            const Vec3 normal = faceNormal(mesh, faceIndex, topPositions);
            for (int vertex : mesh.faces[faceIndex].vertices) {
                if (vertex >= 0 && vertex < static_cast<int>(normalSums.size())) normalSums[vertex] += normal;
            }
        }
        for (size_t vertex = 0; vertex < result.size(); ++vertex) {
            if (normalSums[vertex].lengthSquared() > 1e-8f) {
                result[vertex] -= normalSums[vertex].normalized() * thickness;
            }
        }
        return result;
    }

    // Absolute distance of the fourth quad vertex from the plane through the
    // first three.  The stacking descriptor has one normal per face, so this
    // is the relevant precondition to report before comparing strips.
    static std::pair<double, double> quadPlanarityDistances(const MeshData& mesh,
                                                            const std::vector<Vec3>& positions) {
        double sum = 0.0;
        double maximum = 0.0;
        int quadCount = 0;
        for (const MeshFace& face : mesh.faces) {
            if (face.vertices.size() != 4) continue;
            const int a = face.vertices[0];
            const int b = face.vertices[1];
            const int c = face.vertices[2];
            const int d = face.vertices[3];
            if (a < 0 || b < 0 || c < 0 || d < 0 || a >= static_cast<int>(positions.size()) ||
                b >= static_cast<int>(positions.size()) || c >= static_cast<int>(positions.size()) ||
                d >= static_cast<int>(positions.size())) continue;
            const Vec3 normal = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
            const float normalLength = normal.length();
            if (normalLength <= 1e-8f) continue;
            const double distance = std::abs(static_cast<double>(normal.dot(positions[d] - positions[a])) / normalLength);
            sum += distance;
            maximum = std::max(maximum, distance);
            ++quadCount;
        }
        return {quadCount == 0 ? 0.0 : sum / quadCount, maximum};
    }

    static RibbonSignature buildBlockSignature(const MeshData& mesh, const FaceBlock& block, int blockIndex,
                                               const std::vector<Vec3>& positions) {
        RibbonSignature signature;
        signature.startFace = blockIndex; // Maps signature indices back to face blocks.
        signature.faceCount = static_cast<int>(block.sourceFaces.size());
        if (block.sourceFaces.size() < 2 || block.walkEdges.size() != block.sourceFaces.size()) return signature;
        std::vector<Vec3> normals;
        normals.reserve(block.sourceFaces.size());
        for (int face : block.sourceFaces) normals.push_back(faceNormal(mesh, face, positions));
        double accumulatedLength = 0.0;
        for (int station = 1; station < static_cast<int>(block.sourceFaces.size()); ++station) {
            const auto [edgeStart, edgeEnd] = block.walkEdges[station];
            const Vec3 ruling = (positions[edgeEnd] - positions[edgeStart]).normalized();
            const Vec3& previousNormal = normals[station - 1];
            const Vec3& nextNormal = normals[station];
            const Vec3 previousCentre = faceCentre(mesh, block.sourceFaces[station - 1], positions);
            const Vec3 currentCentre = faceCentre(mesh, block.sourceFaces[station], positions);
            const int nextFace = std::min(station + 1, static_cast<int>(block.sourceFaces.size()) - 1);
            const Vec3 nextCentre = faceCentre(mesh, block.sourceFaces[nextFace], positions);
            const double stationLength = std::max(1e-8, static_cast<double>((currentCentre - previousCentre).length()));
            const double dihedral = std::atan2(ruling.dot(previousNormal.cross(nextNormal)),
                                                std::clamp(static_cast<double>(previousNormal.dot(nextNormal)), -1.0, 1.0));
            const Vec3 tangent = (nextCentre - previousCentre).normalized();
            Vec3 averageNormal = previousNormal + nextNormal;
            if (averageNormal.lengthSquared() <= 1e-8f) averageNormal = previousNormal;
            else averageNormal.normalize();
            // A dihedral is an angle, not curvature.  Divide by the dual
            // longitudinal spacing before using it in the curvature tensor.
            signature.bend.push_back(dihedral / stationLength);
            signature.rulingAngle.push_back(std::atan2(averageNormal.dot(tangent.cross(ruling)), tangent.dot(ruling)));
            signature.rulingLength.push_back((positions[edgeEnd] - positions[edgeStart]).length());
            signature.station.push_back(accumulatedLength + 0.5 * stationLength);
            accumulatedLength += stationLength;
        }
        if (accumulatedLength > 1e-8) {
            for (double& station : signature.station) station /= accumulatedLength;
        }
        return signature;
    }

    static std::filesystem::path dataPath(const std::string& file) {
        const std::filesystem::path requested(file);
        if (requested.is_absolute() || std::filesystem::exists(requested)) return requested;
        const std::filesystem::path local = std::filesystem::path("data") / file;
        if (std::filesystem::exists(local)) return local;
        return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / "data" / file;
    }

    void reload() {
        clearRibbonBlockVisualisation();
        clearStackVisualisation();
        if (m_mesh) scene().removeObject(m_mesh);
        m_mesh = std::make_shared<ComputeMesh>("stereotomy");
        try {
            m_mesh->readFromObj(dataPath("stereotomy.obj").string());
            m_mesh->weld(1e-5f);
            m_mesh->updateHalfEdgeData();
            m_original = std::make_shared<MeshObject>(m_mesh->duplicate());
            m_original->setShowFaces(false);
            m_original->setShowVertices(false);

            m_solver = ProjectionSolver{};
            m_solver.settings.maxIterations = 500;
            m_solver.settings.strength = 1.0f;
            m_solver.settings.tolerance = 1e-5f;
            // This is an iterative damping term in ProjectionSolver, keeping
            // each global solve close to its preceding configuration.
            m_solver.settings.shapePreservationWeight = 0.1f;
            m_solver.settings.fixBoundaryVertices = false;
            m_solver.addConstraint<PlanarFaceConstraint>();

            std::string diagnostic;
            m_valid = extractFaceBlocks(*m_mesh, m_facesPerStrip, m_faceBlocks, &diagnostic);
            if (!m_valid) {
                m_status = "Stereotomy input invalid: " + diagnostic;
                return;
            }
            m_mesh->setColor(Color(0.15f, 0.62f, 0.90f, 1.0f));
            m_mesh->setUseFaceColors(true);
            m_mesh->setShowEdges(true);
            m_mesh->setShowFaces(true);
            // This mesh is the solver input. The visible ribbon is rebuilt as
            // closed, individually coloured solid strip blocks below.
            m_mesh->setVisible(false);
            scene().addObject(m_mesh);
            std::vector<Vec3> rawPositions;
            rawPositions.reserve(m_mesh->getMeshData()->vertices.size());
            for (const MeshVertex& vertex : m_mesh->getMeshData()->vertices) rawPositions.push_back(vertex.position);
            const auto [rawMeanPlanarity, rawMaxPlanarity] = quadPlanarityDistances(*m_mesh->getMeshData(), rawPositions);
            std::cout << "[DevelopableRibbon] input quad plane distance mean/max=" << std::fixed
                      << std::setprecision(8) << rawMeanPlanarity << '/' << rawMaxPlanarity << '\n';
            // Planarisation remains an explicit P action.  Some inputs are
            // already sufficiently planar, and an unnecessary global solve
            // can change the surface that the user intends to stack.
            refreshSignatures();
            m_status = diagnostic + "  " + matchSummary() + "  " + stackSummary();
        } catch (const std::exception& error) {
            m_valid = false;
            m_status = std::string("Could not load stereotomy.obj: ") + error.what();
        }
    }

    void planarise() {
        if (!m_valid) return;
        const int iterations = m_solver.solve(*m_mesh);
        if (m_mesh) m_mesh->getMeshData()->calculateNormals();
        refreshSignatures();
        std::cout << "[DevelopableRibbon] planar solve iterations=" << iterations
                  << "; post-solve quad plane distance mean/max=" << std::fixed << std::setprecision(8)
                  << m_meanPlanarityDistance << '/' << m_maxPlanarityDistance << '\n';
        std::ostringstream report;
        report << "ProjectionSolver planar-face solve: " << iterations << " iterations. "
               << matchSummary() << "  " << stackSummary();
        m_status = report.str();
    }

    void copyRibbonToMesh() {
        MeshData* data = m_mesh && m_mesh->getMeshData() ? m_mesh->getMeshData().get() : nullptr;
        if (!data || data->vertices.size() != m_ribbon.vertices.size()) return;
        for (size_t i = 0; i < data->vertices.size(); ++i) data->vertices[i].position = m_ribbon.vertices[i];
        data->calculateNormals();
        data->triangulationDirty = true;
    }

    void copyMeshToRibbon() {
        const std::shared_ptr<MeshData> data = m_mesh ? m_mesh->getMeshData() : nullptr;
        if (!data || data->vertices.size() != m_ribbon.vertices.size()) return;
        for (size_t i = 0; i < data->vertices.size(); ++i) m_ribbon.vertices[i] = data->vertices[i].position;
    }

    void refreshSignatures() {
        if (!m_valid) return;
        m_facesPerStrip = std::clamp(m_facesPerStrip, 2, kMaxFacesPerStrip);
        m_facesPerStripSlider = static_cast<float>(m_facesPerStrip);
        std::string diagnostic;
        m_valid = extractFaceBlocks(*m_mesh, m_facesPerStrip, m_faceBlocks, &diagnostic);
        if (!m_valid) {
            m_status = "Stereotomy traversal failed: " + diagnostic;
            return;
        }
        const std::shared_ptr<MeshData> data = m_mesh->getMeshData();
        data->calculateNormals();
        m_signatures.clear();
        m_bottomSignatures.clear();
        m_stackBlockIndices.clear();
        std::vector<Vec3> positions;
        positions.reserve(data->vertices.size());
        for (const MeshVertex& vertex : data->vertices) positions.push_back(vertex.position);
        const auto [meanPlanarityDistance, maxPlanarityDistance] = quadPlanarityDistances(*data, positions);
        m_meanPlanarityDistance = meanPlanarityDistance;
        m_maxPlanarityDistance = maxPlanarityDistance;
        m_blockBottomPositions.clear();
        m_blockBottomPositions.reserve(m_faceBlocks.size());
        for (const FaceBlock& block : m_faceBlocks) {
            m_blockBottomPositions.push_back(offsetBlockAlongVertexNormals(*data, block, positions, m_stripThickness));
        }
        // Every block participates at its native face count.  The stacker
        // evaluates signatures on a common normalized-arclength grid, so no
        // geometry or ruling-line resampling is required here.
        for (int blockIndex = 0; blockIndex < static_cast<int>(m_faceBlocks.size()); ++blockIndex) {
            const FaceBlock& block = m_faceBlocks[blockIndex];
            m_signatures.push_back(buildBlockSignature(*data, block, blockIndex, positions));
            m_bottomSignatures.push_back(buildBlockSignature(*data, block, blockIndex, m_blockBottomPositions[blockIndex]));
            m_stackBlockIndices.push_back(blockIndex);
        }
        m_matches = findSimilarRibbonStrips(m_signatures, 3);
        m_stackResult = findBestRibbonStack(m_signatures, m_bottomSignatures);
        m_toolpathBreakAfter.assign(m_stackResult.order.empty() ? 0 : m_stackResult.order.size() - 1, false);
        applyStripColours();
        rebuildRibbonBlockVisualisation();
        m_lastStackThickness = m_stripThickness;
        if (m_showStack) rebuildStackVisualisation();
        printStackReport(diagnostic);
    }

    static Color stripColour(int index, int count) {
        static const Color hotPink(1.0f, 0.08f, 0.58f, 1.0f);
        static const Color blue(0.10f, 0.38f, 1.0f, 1.0f);
        (void)count;
        return index % 2 == 0 ? hotPink : blue;
    }

    void applyStripColours() {
        const std::shared_ptr<MeshData> data = m_mesh ? m_mesh->getMeshData() : nullptr;
        if (!data) return;
        for (MeshFace& face : data->faces) face.color = Color(0.82f, 0.82f, 0.82f, 1.0f);
        for (int strip = 0; strip < static_cast<int>(m_faceBlocks.size()); ++strip) {
            const Color colour = stripColour(strip, static_cast<int>(m_faceBlocks.size()));
            for (const int sourceFace : m_faceBlocks[strip].sourceFaces) {
                if (sourceFace >= 0 && sourceFace < static_cast<int>(data->faces.size())) {
                    data->faces[sourceFace].color = colour;
                }
            }
        }
    }

    std::string matchSummary() const {
        if (m_faceBlocks.empty()) return "No extracted face blocks.";
        std::ostringstream summary;
        summary << m_faceBlocks.size() << " blocks (up to " << m_facesPerStrip << " faces)";
        if (!m_matches.empty()) {
            const RibbonMatch& match = m_matches.front();
            summary << "; closest " << match.stripA << "-" << match.stripB
                    << (match.reversed ? " reversed" : " forward") << ": "
                    << std::fixed << std::setprecision(3) << match.distance;
        }
        return summary.str();
    }

    std::string stackSummary() const {
        if (m_stackResult.order.empty()) return "No stack order.";
        std::ostringstream summary;
        summary << stackRanges().size() << " stacks; worst/total " << std::fixed << std::setprecision(3)
                << m_stackResult.maxInterfaceCost << '/' << m_stackResult.totalCost;
        return summary.str();
    }

    std::vector<std::pair<size_t, size_t>> stackRanges() const {
        std::vector<std::pair<size_t, size_t>> ranges;
        if (m_stackResult.order.empty()) return ranges;
        size_t begin = 0;
        for (size_t layer = 1; layer < m_stackResult.order.size(); ++layer) {
            const double cost = layer - 1 < m_stackResult.interfaceCosts.size() ?
                m_stackResult.interfaceCosts[layer - 1].totalCost : std::numeric_limits<double>::infinity();
            const bool toolpathBreak = layer - 1 < m_toolpathBreakAfter.size() && m_toolpathBreakAfter[layer - 1];
            if (cost > static_cast<double>(m_stackBreakThreshold) || toolpathBreak) {
                ranges.push_back({begin, layer});
                begin = layer;
            }
        }
        ranges.push_back({begin, m_stackResult.order.size()});
        return ranges;
    }

    void printStackPartition() const {
        const auto ranges = stackRanges();
        if (ranges.empty()) return;
        std::cout << "[DevelopableRibbon] stack break threshold=" << std::fixed << std::setprecision(6)
                  << m_stackBreakThreshold << "; toolpath forced breaks="
                  << std::count(m_toolpathBreakAfter.begin(), m_toolpathBreakAfter.end(), true)
                  << "; groups=" << ranges.size() << '\n';
        for (size_t group = 0; group < ranges.size(); ++group) {
            std::cout << "  stack " << group << ':';
            for (size_t layer = ranges[group].first; layer < ranges[group].second; ++layer) {
                std::cout << ' ' << m_stackResult.order[layer];
            }
            std::cout << '\n';
        }
        std::cout.flush();
    }

    // Stack diagnostics belong in the terminal: the on-screen view remains a
    // geometric inspection tool, while this report gives the numerical basis
    // for every selected interface.
    void printStackReport(const std::string& traversalDiagnostic) const {
        int completeBlocks = 0;
        std::vector<int> remainderSizes;
        for (const FaceBlock& block : m_faceBlocks) {
            if (static_cast<int>(block.sourceFaces.size()) == m_facesPerStrip) ++completeBlocks;
            else remainderSizes.push_back(static_cast<int>(block.sourceFaces.size()));
        }
        std::cout << "[DevelopableRibbon] " << traversalDiagnostic
                  << "; thickness=" << std::fixed << std::setprecision(5) << m_stripThickness
                  << "; stack strips=" << m_faceBlocks.size() << " (" << completeBlocks
                  << " x " << m_facesPerStrip << " faces, " << remainderSizes.size() << " short native blocks)"
                  << "; native descriptor samples=" << std::max(0, m_facesPerStrip - 1)
                  << "; quad plane distance mean/max=" << std::setprecision(8)
                  << m_meanPlanarityDistance << '/' << m_maxPlanarityDistance << '\n';
        if (!remainderSizes.empty()) {
            std::cout << "[DevelopableRibbon] short native block face counts:";
            for (const int count : remainderSizes) std::cout << ' ' << count;
            std::cout << '\n';
        }
        if (m_stackResult.order.empty()) {
            std::cout << "[DevelopableRibbon] No valid stack order.\n";
            return;
        }
        std::cout << "[DevelopableRibbon] bottleneck-aware stack: worst="
                  << std::setprecision(8) << m_stackResult.maxInterfaceCost
                  << "; total=" << m_stackResult.totalCost << '\n';
        for (size_t layer = 0; layer < m_stackResult.order.size(); ++layer) {
            const bool reversed = layer < m_stackResult.reversedInOrder.size() && m_stackResult.reversedInOrder[layer];
            std::cout << "  layer " << layer << ": strip " << m_stackResult.order[layer]
                      << (reversed ? " reverse" : " forward");
            if (layer > 0 && layer - 1 < m_stackResult.interfaceCosts.size()) {
                const RibbonPairCompatibility& cost = m_stackResult.interfaceCosts[layer - 1];
                std::cout << " | top(" << m_stackResult.order[layer - 1] << ") -> bottom("
                          << m_stackResult.order[layer] << "): local=" << cost.localCost
                          << ", accumulated=" << cost.accumulatedCost
                          << ", width=" << cost.widthCost
                          << ", total=" << cost.totalCost;
            }
            std::cout << '\n';
        }
        std::cout.flush();
        printStackPartition();
    }

    struct StackStripGeometry {
        std::vector<Vec3> positions;
        std::vector<Vec3> bottomPositions;
        std::vector<std::vector<int>> faces;
        std::vector<std::pair<Vec3, Vec3>> rulings;
        std::vector<std::pair<Vec3, Vec3>> bottomRulings;
        Vec3 labelPosition;
    };

    struct StackVisualLayer {
        int stripIndex = -1;
        bool reversed = false;
        int orderIndex = -1;
        int stackIndex = -1;
        int layerInStack = -1;
        Vec3 labelPosition;
        std::vector<std::pair<Vec3, Vec3>> rulings;
        std::vector<std::pair<Vec3, Vec3>> bottomRulings;
        std::vector<Vec3> topPositions;
        std::vector<Vec3> bottomPositions;
        std::vector<std::vector<int>> faces;
        std::shared_ptr<MeshObject> mesh;
    };

    struct StackBounds {
        Vec3 minimum;
        Vec3 maximum;
    };

    // A zero-radius, two-sided wire line coincident with one displayed
    // ruling.  The target layer is the closest other solid struck by it.
    struct RulingRayHit {
        int sourceLayer = -1;
        int targetLayer = -1;
        int rulingIndex = -1;
        bool bottomSkin = false;
        Vec3 origin;
        Vec3 direction;
        Vec3 point;
    };

    struct RulingRayLine {
        int sourceLayer = -1;
        int rulingIndex = -1;
        bool bottomSkin = false;
        Vec3 origin;
        Vec3 direction;
    };

    StackStripGeometry buildStackStripGeometry(int blockIndex, bool reversed) const {
        StackStripGeometry result;
        const std::shared_ptr<MeshData> data = m_mesh ? m_mesh->getMeshData() : nullptr;
        if (!data || blockIndex < 0 || blockIndex >= static_cast<int>(m_faceBlocks.size())) return result;
        const FaceBlock& block = m_faceBlocks[blockIndex];
        std::vector<Vec3> positions;
        positions.reserve(data->vertices.size());
        for (const MeshVertex& vertex : data->vertices) positions.push_back(vertex.position);
        if (block.sourceFaces.empty()) return result;
        const Vec3 firstCentre = faceCentre(*data, block.sourceFaces.front(), positions);
        const Vec3 lastCentre = faceCentre(*data, block.sourceFaces.back(), positions);
        Vec3 origin;
        Vec3 averageFaceNormal;
        for (int faceIndex : block.sourceFaces) {
            origin += faceCentre(*data, faceIndex, positions);
            averageFaceNormal += faceNormal(*data, faceIndex, positions);
        }
        origin /= static_cast<float>(block.sourceFaces.size());
        Vec3 longitudinal = (lastCentre - firstCentre).normalized();
        if (longitudinal.lengthSquared() <= 1e-8f) longitudinal = Vec3(1.0f, 0.0f, 0.0f);
        Vec3 normal = averageFaceNormal.normalized();
        if (normal.lengthSquared() <= 1e-8f) normal = Vec3(0.0f, 0.0f, 1.0f);
        Vec3 ruling = normal.cross(longitudinal).normalized();
        if (ruling.lengthSquared() <= 1e-8f) ruling = Vec3(0.0f, 1.0f, 0.0f);

        auto toLocal = [&](const Vec3& point) {
            const Vec3 delta = point - origin;
            Vec3 local(delta.dot(longitudinal), delta.dot(ruling), delta.dot(normal));
            // A reverse descriptor is an in-plane 180 degree physical turn:
            // it swaps traversal ends but preserves which side is top.
            if (reversed) {
                local.x = -local.x;
                local.y = -local.y;
            }
            return local;
        };
        std::unordered_map<int, int> vertexMap;
        for (int faceIndex : block.sourceFaces) {
            std::vector<int> face;
            for (int vertex : data->faces[faceIndex].vertices) {
                auto [entry, inserted] = vertexMap.emplace(vertex, static_cast<int>(result.positions.size()));
                if (inserted) {
                    result.positions.push_back(toLocal(positions[vertex]));
                    if (blockIndex < static_cast<int>(m_blockBottomPositions.size())) {
                        result.bottomPositions.push_back(toLocal(m_blockBottomPositions[blockIndex][vertex]));
                    }
                }
                face.push_back(entry->second);
            }
            result.faces.push_back(std::move(face));
        }
        for (const Vec3& position : result.positions) {
            result.labelPosition += position;
        }
        if (!result.positions.empty()) result.labelPosition /= static_cast<float>(result.positions.size());
        for (const auto& [start, end] : blockRulingEdges(*data, block)) {
            result.rulings.push_back({toLocal(positions[start]), toLocal(positions[end])});
            if (blockIndex < static_cast<int>(m_blockBottomPositions.size())) {
                const std::vector<Vec3>& bottom = m_blockBottomPositions[blockIndex];
                result.bottomRulings.push_back({toLocal(bottom[start]), toLocal(bottom[end])});
            }
        }
        return result;
    }

    struct ProjectedTriangle {
        Vec3 a;
        Vec3 b;
        Vec3 c;
    };

    static double cross2D(const Vec3& a, const Vec3& b, const Vec3& c) {
        return static_cast<double>(b.x - a.x) * static_cast<double>(c.y - a.y) -
               static_cast<double>(b.y - a.y) * static_cast<double>(c.x - a.x);
    }

    static bool containsProjectedPoint(const ProjectedTriangle& triangle, const Vec3& point) {
        constexpr double epsilon = 1e-7;
        const double ab = cross2D(triangle.a, triangle.b, point);
        const double bc = cross2D(triangle.b, triangle.c, point);
        const double ca = cross2D(triangle.c, triangle.a, point);
        return (ab >= -epsilon && bc >= -epsilon && ca >= -epsilon) ||
               (ab <= epsilon && bc <= epsilon && ca <= epsilon);
    }

    static bool segmentIntersection2D(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d, Vec3& point) {
        constexpr double epsilon = 1e-8;
        const double rx = static_cast<double>(b.x - a.x);
        const double ry = static_cast<double>(b.y - a.y);
        const double sx = static_cast<double>(d.x - c.x);
        const double sy = static_cast<double>(d.y - c.y);
        const double denominator = rx * sy - ry * sx;
        if (std::abs(denominator) <= epsilon) return false;
        const double qpx = static_cast<double>(c.x - a.x);
        const double qpy = static_cast<double>(c.y - a.y);
        const double t = (qpx * sy - qpy * sx) / denominator;
        const double u = (qpx * ry - qpy * rx) / denominator;
        if (t < -epsilon || t > 1.0 + epsilon || u < -epsilon || u > 1.0 + epsilon) return false;
        point = a + (b - a) * static_cast<float>(std::clamp(t, 0.0, 1.0));
        return true;
    }

    static float projectedHeight(const ProjectedTriangle& triangle, const Vec3& point) {
        const double area = cross2D(triangle.a, triangle.b, triangle.c);
        if (std::abs(area) <= 1e-10) return triangle.a.z;
        const double wa = cross2D(triangle.b, triangle.c, point) / area;
        const double wb = cross2D(triangle.c, triangle.a, point) / area;
        const double wc = 1.0 - wa - wb;
        return static_cast<float>(wa * triangle.a.z + wb * triangle.b.z + wc * triangle.c.z);
    }

    static std::vector<ProjectedTriangle> projectedTriangles(const std::vector<Vec3>& positions,
                                                              const std::vector<std::vector<int>>& faces) {
        std::vector<ProjectedTriangle> result;
        for (const std::vector<int>& face : faces) {
            if (face.size() < 3) continue;
            for (size_t vertex = 1; vertex + 1 < face.size(); ++vertex) {
                const int a = face[0];
                const int b = face[vertex];
                const int c = face[vertex + 1];
                if (a < 0 || b < 0 || c < 0 || a >= static_cast<int>(positions.size()) ||
                    b >= static_cast<int>(positions.size()) || c >= static_cast<int>(positions.size())) continue;
                const ProjectedTriangle triangle{positions[a], positions[b], positions[c]};
                if (std::abs(cross2D(triangle.a, triangle.b, triangle.c)) <= 1e-10) continue;
                result.push_back(triangle);
            }
        }
        return result;
    }

    static float verticalSeparationOffset(const StackStripGeometry& lower, float lowerZ,
                                          const StackStripGeometry& upper) {
        const std::vector<ProjectedTriangle> lowerTriangles = projectedTriangles(lower.positions, lower.faces);
        const std::vector<ProjectedTriangle> upperTriangles = projectedTriangles(upper.bottomPositions, upper.faces);
        constexpr float clearance = 1e-4f;
        float offset = lowerZ;
        for (const ProjectedTriangle& lowerTriangle : lowerTriangles) {
            for (const ProjectedTriangle& upperTriangle : upperTriangles) {
                std::vector<Vec3> contacts;
                const std::array<Vec3, 3> lowerVertices{lowerTriangle.a, lowerTriangle.b, lowerTriangle.c};
                const std::array<Vec3, 3> upperVertices{upperTriangle.a, upperTriangle.b, upperTriangle.c};
                for (const Vec3& point : lowerVertices) if (containsProjectedPoint(upperTriangle, point)) contacts.push_back(point);
                for (const Vec3& point : upperVertices) if (containsProjectedPoint(lowerTriangle, point)) contacts.push_back(point);
                for (size_t lowerEdge = 0; lowerEdge < 3; ++lowerEdge) {
                    const Vec3& la = lowerVertices[lowerEdge];
                    const Vec3& lb = lowerVertices[(lowerEdge + 1) % 3];
                    for (size_t upperEdge = 0; upperEdge < 3; ++upperEdge) {
                        Vec3 point;
                        if (segmentIntersection2D(la, lb, upperVertices[upperEdge], upperVertices[(upperEdge + 1) % 3], point)) {
                            contacts.push_back(point);
                        }
                    }
                }
                for (const Vec3& point : contacts) {
                    offset = std::max(offset, lowerZ + projectedHeight(lowerTriangle, point) -
                                               projectedHeight(upperTriangle, point) + clearance);
                }
            }
        }
        return offset;
    }

    std::shared_ptr<MeshObject> createSolidBlock(const std::string& name,
                                                 const std::vector<Vec3>& topPositions,
                                                 const std::vector<Vec3>& bottomPositions,
                                                 const std::vector<std::vector<int>>& topFaces,
                                                 const Color& colour) const {
        if (topPositions.empty() || topPositions.size() != bottomPositions.size() || topFaces.empty()) return nullptr;

        MeshObject bottom(name + "_bottom");
        bottom.createFromVerticesAndFaces(bottomPositions, topFaces);
        std::vector<Vec3> offsets;
        offsets.reserve(topPositions.size());
        for (size_t i = 0; i < topPositions.size(); ++i) offsets.push_back(topPositions[i] - bottomPositions[i]);
        auto mesh = std::make_shared<MeshObject>(bottom.extrudeMesh(0.0f, MeshExtrudeMode::SmoothSolid, offsets));
        mesh->setUseFaceColors(true);
        mesh->setShowEdges(true);
        mesh->setEdgeWidth(2.0f);
        const std::shared_ptr<MeshData> data = mesh->getMeshData();
        for (MeshFace& face : data->faces) face.color = colour;
        for (MeshEdge& edge : data->edges) edge.color = Color(0.0f, 0.0f, 0.0f, 1.0f);
        return mesh;
    }

    void clearRibbonBlockVisualisation() {
        for (const std::shared_ptr<MeshObject>& block : m_ribbonBlocks) {
            if (block) scene().removeObject(block);
        }
        m_ribbonBlocks.clear();
    }

    void rebuildRibbonBlockVisualisation() {
        clearRibbonBlockVisualisation();
        const std::shared_ptr<MeshData> data = m_mesh ? m_mesh->getMeshData() : nullptr;
        if (!m_valid || !data || m_blockBottomPositions.size() != m_faceBlocks.size()) return;
        for (int strip = 0; strip < static_cast<int>(m_faceBlocks.size()); ++strip) {
            const FaceBlock& sourceBlock = m_faceBlocks[strip];
            std::unordered_map<int, int> vertexMap;
            std::vector<Vec3> topPositions;
            std::vector<Vec3> bottomPositions;
            std::vector<std::vector<int>> faces;
            for (int faceIndex : sourceBlock.sourceFaces) {
                if (faceIndex < 0 || faceIndex >= static_cast<int>(data->faces.size())) continue;
                std::vector<int> face;
                for (int vertex : data->faces[faceIndex].vertices) {
                    if (vertex < 0 || vertex >= static_cast<int>(data->vertices.size())) continue;
                    auto [entry, inserted] = vertexMap.emplace(vertex, static_cast<int>(topPositions.size()));
                    if (inserted) {
                        topPositions.push_back(data->vertices[vertex].position);
                        bottomPositions.push_back(m_blockBottomPositions[strip][vertex]);
                    }
                    face.push_back(entry->second);
                }
                faces.push_back(std::move(face));
            }
            auto solid = createSolidBlock("ribbon_block_" + std::to_string(strip), topPositions, bottomPositions,
                                          faces, stripColour(strip, static_cast<int>(m_faceBlocks.size())));
            if (!solid) continue;
            scene().addObject(solid);
            m_ribbonBlocks.push_back(std::move(solid));
        }
    }

    void clearStackVisualisation() {
        for (const StackVisualLayer& layer : m_stackLayers) {
            if (layer.mesh) scene().removeObject(layer.mesh);
        }
        m_stackLayers.clear();
        m_stackGroupBounds.clear();
        m_rulingRayHits.clear();
        m_rulingRayLines.clear();
        m_stackBoundsValid = false;
    }

    void rebuildStackVisualisation() {
        clearStackVisualisation();
        if (!m_showStack || m_stackResult.order.empty()) return;

        std::vector<StackStripGeometry> geometries;
        geometries.reserve(m_stackResult.order.size());
        for (size_t layer = 0; layer < m_stackResult.order.size(); ++layer) {
            const int stripIndex = m_stackResult.order[layer];
            if (stripIndex < 0 || stripIndex >= static_cast<int>(m_stackBlockIndices.size())) return;
            const bool reversed = layer < m_stackResult.reversedInOrder.size() && m_stackResult.reversedInOrder[layer];
            geometries.push_back(buildStackStripGeometry(m_stackBlockIndices[stripIndex], reversed));
        }

        // Place layers only within their compatible stack group.  A threshold
        // break intentionally creates a separate, side-by-side stack rather
        // than pretending that a high-cost interface nests.  Inter-layer
        // placement clears every overlapping top/bottom triangle pair;
        // bounds are used below only to centre and draw the finished group.
        const std::shared_ptr<MeshData> originalData = m_mesh ? m_mesh->getMeshData() : nullptr;
        Vec3 originalMin, originalMax;
        if (originalData && !originalData->vertices.empty()) originalData->updateBounds(originalMin, originalMax);
        const float displaySeparation = std::max(0.05f, 0.1f * (originalMax.x - originalMin.x));
        m_stackBoundsMin = Vec3(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
        m_stackBoundsMax = Vec3(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
        auto includeInBounds = [&](const Vec3& point) {
            m_stackBoundsMin.x = std::min(m_stackBoundsMin.x, point.x);
            m_stackBoundsMin.y = std::min(m_stackBoundsMin.y, point.y);
            m_stackBoundsMin.z = std::min(m_stackBoundsMin.z, point.z);
            m_stackBoundsMax.x = std::max(m_stackBoundsMax.x, point.x);
            m_stackBoundsMax.y = std::max(m_stackBoundsMax.y, point.y);
            m_stackBoundsMax.z = std::max(m_stackBoundsMax.z, point.z);
        };

        float nextStackX = originalMax.x + displaySeparation;
        const auto ranges = stackRanges();
        for (int stackIndex = 0; stackIndex < static_cast<int>(ranges.size()); ++stackIndex) {
            const auto [begin, end] = ranges[stackIndex];
            std::vector<float> layerZ(end - begin, 0.0f);
            for (size_t layer = begin + 1; layer < end; ++layer) {
                float offset = layerZ[layer - begin - 1];
                // Clear the incoming layer against every layer already in
                // this stack, not merely its immediate predecessor.
                for (size_t lowerLayer = begin; lowerLayer < layer; ++lowerLayer) {
                    offset = std::max(offset, verticalSeparationOffset(geometries[lowerLayer],
                                                                         layerZ[lowerLayer - begin],
                                                                         geometries[layer]));
                }
                layerZ[layer - begin] = offset;
            }
            float stackMinZ = std::numeric_limits<float>::infinity();
            float stackMaxZ = -std::numeric_limits<float>::infinity();
            for (size_t layer = begin; layer < end; ++layer) {
                const float z = layerZ[layer - begin];
                for (const Vec3& position : geometries[layer].positions) {
                    stackMinZ = std::min(stackMinZ, position.z + z);
                    stackMaxZ = std::max(stackMaxZ, position.z + z);
                }
                for (const Vec3& position : geometries[layer].bottomPositions) {
                    stackMinZ = std::min(stackMinZ, position.z + z);
                    stackMaxZ = std::max(stackMaxZ, position.z + z);
                }
            }
            const float centreZ = 0.5f * (stackMinZ + stackMaxZ);
            for (float& z : layerZ) z -= centreZ;

            float minX = std::numeric_limits<float>::infinity();
            float maxX = -std::numeric_limits<float>::infinity();
            float minY = std::numeric_limits<float>::infinity();
            float maxY = -std::numeric_limits<float>::infinity();
            for (size_t layer = begin; layer < end; ++layer) {
                for (const Vec3& position : geometries[layer].positions) {
                    minX = std::min(minX, position.x);
                    maxX = std::max(maxX, position.x);
                    minY = std::min(minY, position.y);
                    maxY = std::max(maxY, position.y);
                }
            }
            const Vec3 stackOffset(nextStackX - minX,
                                   0.5f * (originalMin.y + originalMax.y - minY - maxY), 0.0f);
            nextStackX += maxX - minX + displaySeparation;
            StackBounds groupBounds{
                Vec3(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()),
                Vec3(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity())};
            auto includeInGroupBounds = [&](const Vec3& point) {
                groupBounds.minimum.x = std::min(groupBounds.minimum.x, point.x);
                groupBounds.minimum.y = std::min(groupBounds.minimum.y, point.y);
                groupBounds.minimum.z = std::min(groupBounds.minimum.z, point.z);
                groupBounds.maximum.x = std::max(groupBounds.maximum.x, point.x);
                groupBounds.maximum.y = std::max(groupBounds.maximum.y, point.y);
                groupBounds.maximum.z = std::max(groupBounds.maximum.z, point.z);
            };

            for (size_t layer = begin; layer < end; ++layer) {
                StackStripGeometry& geometry = geometries[layer];
                const int stripIndex = m_stackResult.order[layer];
                const bool reversed = layer < m_stackResult.reversedInOrder.size() && m_stackResult.reversedInOrder[layer];
                const Vec3 offset = stackOffset + Vec3(0.0f, 0.0f, layerZ[layer - begin]);
                for (Vec3& position : geometry.positions) {
                    position += offset;
                    includeInBounds(position);
                    includeInGroupBounds(position);
                }
                for (Vec3& position : geometry.bottomPositions) {
                    position += offset;
                    includeInBounds(position);
                    includeInGroupBounds(position);
                }
                for (auto& ruling : geometry.rulings) {
                    ruling.first += offset;
                    ruling.second += offset;
                }
                for (auto& ruling : geometry.bottomRulings) {
                    ruling.first += offset;
                    ruling.second += offset;
                }
                geometry.labelPosition += offset;
                const int blockIndex = m_stackBlockIndices[stripIndex];
                auto mesh = createSolidBlock("ribbon_stack_" + std::to_string(stackIndex) + "_layer_" +
                                             std::to_string(layer - begin), geometry.positions, geometry.bottomPositions,
                                             geometry.faces, stripColour(blockIndex, static_cast<int>(m_faceBlocks.size())));
                if (!mesh) continue;
                scene().addObject(mesh);
                m_stackLayers.push_back({stripIndex, reversed, static_cast<int>(layer), stackIndex, static_cast<int>(layer - begin),
                                         geometry.labelPosition, std::move(geometry.rulings),
                                         std::move(geometry.bottomRulings), std::move(geometry.positions),
                                         std::move(geometry.bottomPositions), std::move(geometry.faces), std::move(mesh)});
            }
            m_stackGroupBounds.push_back(groupBounds);
        }
        m_stackBoundsValid = !m_stackLayers.empty();
        m_lastStackThickness = m_stripThickness;
        if (m_showRulingRayCheck) rebuildRulingRayCheck();
    }

    static bool intersectRayTriangle(const Vec3& origin, const Vec3& direction,
                                     const Vec3& a, const Vec3& b, const Vec3& c, float& distance) {
        constexpr float epsilon = 1e-6f;
        const Vec3 edgeAB = b - a;
        const Vec3 edgeAC = c - a;
        const Vec3 perpendicular = direction.cross(edgeAC);
        const float determinant = edgeAB.dot(perpendicular);
        if (std::abs(determinant) <= epsilon) return false;
        const float inverseDeterminant = 1.0f / determinant;
        const Vec3 toOrigin = origin - a;
        const float u = toOrigin.dot(perpendicular) * inverseDeterminant;
        if (u < -epsilon || u > 1.0f + epsilon) return false;
        const Vec3 q = toOrigin.cross(edgeAB);
        const float v = direction.dot(q) * inverseDeterminant;
        if (v < -epsilon || u + v > 1.0f + epsilon) return false;
        const float candidate = edgeAC.dot(q) * inverseDeterminant;
        if (candidate <= epsilon) return false;
        distance = candidate;
        return true;
    }

    static bool intersectRayRibbonSkin(const std::vector<Vec3>& positions, const std::vector<std::vector<int>>& faces,
                                       const Vec3& origin, const Vec3& direction, float& distance) {
        bool hit = false;
        distance = std::numeric_limits<float>::infinity();
        for (const std::vector<int>& vertices : faces) {
            if (vertices.size() < 3) continue;
            for (size_t vertex = 1; vertex + 1 < vertices.size(); ++vertex) {
                const int ia = vertices[0];
                const int ib = vertices[vertex];
                const int ic = vertices[vertex + 1];
                if (ia < 0 || ib < 0 || ic < 0 || ia >= static_cast<int>(positions.size()) ||
                    ib >= static_cast<int>(positions.size()) || ic >= static_cast<int>(positions.size())) continue;
                float candidate = 0.0f;
                if (intersectRayTriangle(origin, direction, positions[ia], positions[ib], positions[ic], candidate) &&
                    candidate < distance) {
                    distance = candidate;
                    hit = true;
                }
            }
        }
        return hit;
    }

    void rebuildRulingRayCheck() {
        m_rulingRayHits.clear();
        m_rulingRayLines.clear();
        if (!m_showRulingRayCheck || m_stackLayers.empty()) return;
        for (int sourceIndex = 0; sourceIndex < static_cast<int>(m_stackLayers.size()); ++sourceIndex) {
            const StackVisualLayer& source = m_stackLayers[sourceIndex];
            const std::array<std::pair<const std::vector<std::pair<Vec3, Vec3>>*, bool>, 2> rulingSets{{
                {&source.rulings, false},
                {&source.bottomRulings, true},
            }};
            for (const auto& [rulings, bottomSkin] : rulingSets) {
                for (int rulingIndex = 0; rulingIndex < static_cast<int>(rulings->size()); ++rulingIndex) {
                    const auto& ruling = (*rulings)[rulingIndex];
                    const Vec3 direction = (ruling.second - ruling.first).normalized();
                    if (direction.lengthSquared() <= 1e-8f) continue;
                    const Vec3 origin = (ruling.first + ruling.second) * 0.5f;
                    m_rulingRayLines.push_back({sourceIndex, rulingIndex, bottomSkin, origin, direction});
                    float closestDistance = std::numeric_limits<float>::infinity();
                    int closestTarget = -1;
                    Vec3 closestPoint;
                    for (int targetIndex = 0; targetIndex < static_cast<int>(m_stackLayers.size()); ++targetIndex) {
                        const StackVisualLayer& target = m_stackLayers[targetIndex];
                        if (targetIndex == sourceIndex || target.stackIndex != source.stackIndex) continue;
                        for (const float sign : {1.0f, -1.0f}) {
                            const Vec3 rayDirection = direction * sign;
                            float distance = 0.0f;
                            float topDistance = 0.0f;
                            float bottomDistance = 0.0f;
                            const bool hitsTop = intersectRayRibbonSkin(target.topPositions, target.faces,
                                                                         origin, rayDirection, topDistance);
                            const bool hitsBottom = intersectRayRibbonSkin(target.bottomPositions, target.faces,
                                                                            origin, rayDirection, bottomDistance);
                            if (!hitsTop && !hitsBottom) continue;
                            distance = std::min(hitsTop ? topDistance : std::numeric_limits<float>::infinity(),
                                                hitsBottom ? bottomDistance : std::numeric_limits<float>::infinity());
                            if (distance < closestDistance) {
                                closestDistance = distance;
                                closestTarget = targetIndex;
                                closestPoint = origin + rayDirection * distance;
                            }
                        }
                    }
                    if (closestTarget >= 0) {
                        m_rulingRayHits.push_back({sourceIndex, closestTarget, rulingIndex, bottomSkin,
                                                   origin, direction, closestPoint});
                    }
                }
            }
        }
        std::cout << "[DevelopableRibbon] ruling-ray toolpath check: " << m_rulingRayLines.size()
                  << " top/bottom ruling lines; " << m_rulingRayHits.size()
                  << " top/bottom-skin clashes within their own stacks (zero-radius infinite wire lines)\n";
        constexpr size_t reportLimit = 24;
        for (size_t index = 0; index < std::min(reportLimit, m_rulingRayHits.size()); ++index) {
            const RulingRayHit& hit = m_rulingRayHits[index];
            const StackVisualLayer& source = m_stackLayers[hit.sourceLayer];
            const StackVisualLayer& target = m_stackLayers[hit.targetLayer];
            std::cout << "  K" << source.stackIndex << " L" << source.layerInStack << " / S" << source.stripIndex
                      << (hit.bottomSkin ? " bottom ruling " : " top ruling ") << hit.rulingIndex << " -> K" << target.stackIndex << " L"
                      << target.layerInStack << " / S" << target.stripIndex << '\n';
        }
        if (m_rulingRayHits.size() > reportLimit) {
            std::cout << "  ... " << (m_rulingRayHits.size() - reportLimit) << " additional clashes\n";
        }
        std::cout.flush();
    }

    void toggleRulingRayCheck() {
        m_showRulingRayCheck = !m_showRulingRayCheck;
        if (m_showRulingRayCheck && m_showStack) rebuildRulingRayCheck();
        if (!m_showRulingRayCheck) {
            m_rulingRayHits.clear();
            m_rulingRayLines.clear();
        }
        m_status = m_showRulingRayCheck ?
            "Ruling-ray toolpath check enabled (gray = clear; red = clash)." :
            "Ruling-ray toolpath check disabled.";
    }

    void rebuildToolpathSafeStack() {
        if (!m_showRulingRayCheck) {
            m_status = "Enable the ruling-ray check first with I.";
            return;
        }
        if (!m_showStack) {
            m_status = "Enable stack view first with S, then press K.";
            return;
        }
        const size_t layerCount = m_stackResult.order.size();
        if (layerCount < 2) return;

        // Re-evaluate from the descriptor-threshold partition, then add the
        // smallest set of layer cuts that separates every clashing pair.  A
        // clash gives an interval [a,b]; choosing a cut inside each interval
        // is the standard minimum interval-stabbing problem, solved greedily
        // by selecting its rightmost available boundary.
        m_toolpathBreakAfter.assign(layerCount - 1, false);
        rebuildStackVisualisation();
        for (size_t pass = 0; pass < layerCount && !m_rulingRayHits.empty(); ++pass) {
            std::vector<std::pair<int, int>> intervals;
            for (const RulingRayHit& hit : m_rulingRayHits) {
                if (hit.sourceLayer < 0 || hit.targetLayer < 0 ||
                    hit.sourceLayer >= static_cast<int>(m_stackLayers.size()) ||
                    hit.targetLayer >= static_cast<int>(m_stackLayers.size())) continue;
                const int a = m_stackLayers[hit.sourceLayer].orderIndex;
                const int b = m_stackLayers[hit.targetLayer].orderIndex;
                if (a < 0 || b < 0 || a == b) continue;
                intervals.push_back({std::min(a, b), std::max(a, b)});
            }
            std::sort(intervals.begin(), intervals.end(), [](const auto& a, const auto& b) {
                return a.second == b.second ? a.first < b.first : a.second < b.second;
            });
            bool addedBreak = false;
            for (const auto& [begin, end] : intervals) {
                bool separated = false;
                for (int boundary = begin; boundary < end; ++boundary) {
                    if (boundary >= 0 && boundary < static_cast<int>(m_toolpathBreakAfter.size()) &&
                        m_toolpathBreakAfter[boundary]) {
                        separated = true;
                        break;
                    }
                }
                if (!separated && end > begin) {
                    m_toolpathBreakAfter[end - 1] = true;
                    addedBreak = true;
                }
            }
            if (!addedBreak) break;
            rebuildStackVisualisation();
        }
        const int forcedBreaks = static_cast<int>(std::count(m_toolpathBreakAfter.begin(),
                                                              m_toolpathBreakAfter.end(), true));
        m_status = "Toolpath-aware stack: " + std::to_string(forcedBreaks) + " forced breaks; " +
                   std::to_string(m_rulingRayHits.size()) + " remaining clashes.";
        std::cout << "[DevelopableRibbon] toolpath-aware re-stack: forced breaks=" << forcedBreaks
                  << "; remaining clashes=" << m_rulingRayHits.size() << '\n';
        printStackPartition();
    }

    void toggleStackVisualisation() {
        if (!m_valid) return;
        m_showStack = !m_showStack;
        if (m_mesh) m_mesh->setVisible(false);
        if (m_showStack) rebuildStackVisualisation();
        else clearStackVisualisation();
        m_status = (m_showStack ? "Stack visualisation. " : "Ribbon visualisation. ") + matchSummary() + "  " + stackSummary();
    }

    void drawRulings(Renderer& renderer) const {
        const std::shared_ptr<MeshData> data = m_mesh ? m_mesh->getMeshData() : nullptr;
        if (!data) return;
        for (const FaceBlock& block : m_faceBlocks) {
            for (const auto& [start, end] : blockRulingEdges(*data, block)) {
                if (start < 0 || end < 0 || start >= static_cast<int>(data->vertices.size()) ||
                    end >= static_cast<int>(data->vertices.size())) continue;
                renderer.drawLine(data->vertices[start].position, data->vertices[end].position,
                                  Color(0.95f, 0.22f, 0.08f, 1.0f), 1.3f);
            }
        }
    }

    void drawStripIndices(Renderer& renderer) const {
        renderer.setColor(Color(0.08f, 0.08f, 0.12f, 1.0f));
        const std::shared_ptr<MeshData> data = m_mesh ? m_mesh->getMeshData() : nullptr;
        if (!data) return;
        for (int strip = 0; strip < static_cast<int>(m_faceBlocks.size()); ++strip) {
            Vec3 centre;
            Vec3 normal;
            int pointCount = 0;
            for (int faceIndex : m_faceBlocks[strip].sourceFaces) {
                if (faceIndex < 0 || faceIndex >= static_cast<int>(data->faces.size())) continue;
                const std::vector<int>& face = data->faces[faceIndex].vertices;
                for (int vertex : face) {
                    centre += data->vertices[vertex].position;
                    ++pointCount;
                }
                if (face.size() >= 3) {
                    normal += (data->vertices[face[1]].position - data->vertices[face[0]].position)
                                  .cross(data->vertices[face[2]].position - data->vertices[face[0]].position).normalized();
                }
            }
            if (pointCount == 0) continue;
            centre /= static_cast<float>(pointCount);
            if (normal.lengthSquared() > 1e-8f) centre += normal.normalized() * 0.002f;
            renderer.drawText(std::to_string(strip), centre, 1.1f);
        }
    }

    void drawStackAnnotations(Renderer& renderer) const {
        renderer.setColor(Color(0.08f, 0.08f, 0.12f, 1.0f));
        for (const StackVisualLayer& item : m_stackLayers) {
            renderer.drawText("K" + std::to_string(item.stackIndex) + " L" + std::to_string(item.layerInStack) +
                              " / S" + std::to_string(item.stripIndex) +
                              (item.reversed ? " R" : " F"),
                              item.labelPosition, 1.0f);
            for (const auto& ruling : item.rulings) {
                renderer.drawLine(ruling.first, ruling.second, Color(0.88f, 0.12f, 0.08f, 1.0f), 1.0f);
            }
        }
    }

    void drawStackBounds(Renderer& renderer) const {
        if (!m_stackBoundsValid) return;
        constexpr std::array<std::array<int, 2>, 12> edges = {{{0, 1}, {1, 2}, {2, 3}, {3, 0},
                                                                  {4, 5}, {5, 6}, {6, 7}, {7, 4},
                                                                  {0, 4}, {1, 5}, {2, 6}, {3, 7}}};
        auto drawBounds = [&](const StackBounds& bounds, const Color& color, float width) {
            const Vec3& lo = bounds.minimum;
            const Vec3& hi = bounds.maximum;
            const std::array<Vec3, 8> corners = {
                Vec3(lo.x, lo.y, lo.z), Vec3(hi.x, lo.y, lo.z), Vec3(hi.x, hi.y, lo.z), Vec3(lo.x, hi.y, lo.z),
                Vec3(lo.x, lo.y, hi.z), Vec3(hi.x, lo.y, hi.z), Vec3(hi.x, hi.y, hi.z), Vec3(lo.x, hi.y, hi.z)};
            for (const auto& edge : edges) renderer.drawLine(corners[edge[0]], corners[edge[1]], color, width);
        };
        for (const StackBounds& bounds : m_stackGroupBounds) {
            drawBounds(bounds, Color(0.0f, 0.0f, 0.0f, 1.0f), 1.5f);
        }
    }

    void drawRulingRayHits(Renderer& renderer) const {
        if (!m_stackBoundsValid) return;
        for (const RulingRayLine& ray : m_rulingRayLines) {
            if (ray.sourceLayer < 0 || ray.sourceLayer >= static_cast<int>(m_stackLayers.size())) continue;
            const int stackIndex = m_stackLayers[ray.sourceLayer].stackIndex;
            if (stackIndex < 0 || stackIndex >= static_cast<int>(m_stackGroupBounds.size())) continue;
            Vec3 start;
            Vec3 end;
            if (!clipLineToBounds(ray.origin, ray.direction, m_stackGroupBounds[stackIndex], start, end)) continue;
            renderer.drawLine(start, end, Color(0.42f, 0.42f, 0.42f, 1.0f), 0.8f);
        }
        for (const RulingRayHit& hit : m_rulingRayHits) {
            if (hit.sourceLayer < 0 || hit.sourceLayer >= static_cast<int>(m_stackLayers.size())) continue;
            const int stackIndex = m_stackLayers[hit.sourceLayer].stackIndex;
            if (stackIndex < 0 || stackIndex >= static_cast<int>(m_stackGroupBounds.size())) continue;
            Vec3 start;
            Vec3 end;
            if (!clipLineToBounds(hit.origin, hit.direction, m_stackGroupBounds[stackIndex], start, end)) continue;
            renderer.drawLine(start, end, Color(0.92f, 0.02f, 0.04f, 1.0f), 1.8f);
            renderer.drawPoint(hit.point, Color(1.0f, 0.78f, 0.02f, 1.0f), 6.0f);
        }
    }

    static bool clipLineToBounds(const Vec3& origin, const Vec3& direction, const StackBounds& bounds,
                                 Vec3& start, Vec3& end) {
        constexpr float epsilon = 1e-7f;
        float minimumT = -std::numeric_limits<float>::infinity();
        float maximumT = std::numeric_limits<float>::infinity();
        const std::array<float, 3> originComponents{origin.x, origin.y, origin.z};
        const std::array<float, 3> directionComponents{direction.x, direction.y, direction.z};
        const std::array<float, 3> minimumComponents{bounds.minimum.x, bounds.minimum.y, bounds.minimum.z};
        const std::array<float, 3> maximumComponents{bounds.maximum.x, bounds.maximum.y, bounds.maximum.z};
        for (size_t axis = 0; axis < 3; ++axis) {
            const float coordinate = originComponents[axis];
            const float slope = directionComponents[axis];
            if (std::abs(slope) <= epsilon) {
                if (coordinate < minimumComponents[axis] - epsilon || coordinate > maximumComponents[axis] + epsilon) return false;
                continue;
            }
            float enter = (minimumComponents[axis] - coordinate) / slope;
            float exit = (maximumComponents[axis] - coordinate) / slope;
            if (enter > exit) std::swap(enter, exit);
            minimumT = std::max(minimumT, enter);
            maximumT = std::min(maximumT, exit);
            if (minimumT > maximumT) return false;
        }
        start = origin + direction * minimumT;
        end = origin + direction * maximumT;
        return true;
    }

    static std::string formatCost(double value) {
        std::ostringstream text;
        text << std::fixed << std::setprecision(4) << value;
        return text.str();
    }

    static void drawEdges(Renderer& renderer, const MeshData& mesh, const Color& color, float width) {
        std::vector<Vec3> segments;
        for (const MeshEdge& edge : mesh.edges) {
            if (edge.vertexA < 0 || edge.vertexB < 0 || edge.vertexA >= static_cast<int>(mesh.vertices.size()) ||
                edge.vertexB >= static_cast<int>(mesh.vertices.size())) continue;
            segments.push_back(mesh.vertices[edge.vertexA].position);
            segments.push_back(mesh.vertices[edge.vertexB].position);
        }
        if (!segments.empty()) renderer.drawLines(segments.data(), static_cast<int>(segments.size()), color, width);
    }

    std::shared_ptr<ComputeMesh> m_mesh;
    std::shared_ptr<MeshObject> m_original;
    std::unique_ptr<SimpleUI> m_ui;
    ProjectionSolver m_solver;
    QuadRibbon m_ribbon;
    QuadRibbon m_bottomRibbon;
    std::vector<FaceBlock> m_faceBlocks;
    std::vector<std::vector<Vec3>> m_blockBottomPositions;
    std::vector<RibbonSignature> m_signatures;
    std::vector<RibbonSignature> m_bottomSignatures;
    std::vector<int> m_stackBlockIndices;
    std::vector<RibbonMatch> m_matches;
    RibbonStackResult m_stackResult;
    std::vector<std::shared_ptr<MeshObject>> m_ribbonBlocks;
    std::vector<StackVisualLayer> m_stackLayers;
    std::vector<StackBounds> m_stackGroupBounds;
    std::vector<RulingRayHit> m_rulingRayHits;
    std::vector<RulingRayLine> m_rulingRayLines;
    std::vector<bool> m_toolpathBreakAfter;
    std::string m_status{"Loading stereotomy.obj..."};
    int m_facesPerStrip{12};
    float m_facesPerStripSlider{8.0f};
    float m_stripThickness{0.015f};
    float m_lastStackThickness{-1.0f};
    float m_stackBreakThreshold{0.10f};
    float m_lastStackBreakThreshold{-1.0f};
    double m_meanPlanarityDistance{0.0};
    double m_maxPlanarityDistance{0.0};
    bool m_showOriginal{true};
    bool m_showStack{false};
    bool m_showRulingRayCheck{false};
    bool m_valid{false};
    Vec3 m_stackBoundsMin;
    Vec3 m_stackBoundsMax;
    bool m_stackBoundsValid{false};
};

ALICE2_REGISTER_SKETCH_AUTO(DevelopableRibbonSketch)

#endif
