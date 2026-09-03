#define __MAIN__
#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>
#include <computeGeom/ComputeMesh.h>

#include <algorithm>
#include <filesystem>
#include <iomanip>
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
    }

    void cleanup() override {
        clearRibbonBlockVisualisation();
        clearStackVisualisation();
    }

    void draw(Renderer& renderer, Camera&) override {
        renderer.setColor(Color(0.0f, 0.0f, 0.0f, 1.0f));
        renderer.drawString("r reload | p solve planar faces | s toggle stack | o toggle original wire | [ / ] strip faces", 10.0f, 30.0f);
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
            drawStackAnalysis(renderer);
        }
    }

    bool onKeyPress(unsigned char key, int, int) override {
        switch (key) {
            case 'r': case 'R': reload(); return true;
            case 'p': case 'P': planarise(); return true;
            case 's': case 'S': toggleStackVisualisation(); return true;
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

    static RibbonSignature buildBlockSignature(const MeshData& mesh, const FaceBlock& block, int blockIndex,
                                               const std::vector<Vec3>& positions) {
        RibbonSignature signature;
        signature.startFace = blockIndex; // Maps signature indices back to face blocks.
        signature.faceCount = static_cast<int>(block.sourceFaces.size());
        if (block.sourceFaces.size() < 2 || block.walkEdges.size() != block.sourceFaces.size()) return signature;
        std::vector<Vec3> normals;
        normals.reserve(block.sourceFaces.size());
        for (int face : block.sourceFaces) normals.push_back(faceNormal(mesh, face, positions));
        for (int station = 1; station < static_cast<int>(block.sourceFaces.size()); ++station) {
            const auto [edgeStart, edgeEnd] = block.walkEdges[station];
            const Vec3 ruling = (positions[edgeEnd] - positions[edgeStart]).normalized();
            const Vec3& previousNormal = normals[station - 1];
            const Vec3& nextNormal = normals[station];
            const double bend = std::atan2(ruling.dot(previousNormal.cross(nextNormal)),
                                           std::clamp(static_cast<double>(previousNormal.dot(nextNormal)), -1.0, 1.0));
            const Vec3 previousCentre = faceCentre(mesh, block.sourceFaces[station - 1], positions);
            const int nextFace = std::min(station + 1, static_cast<int>(block.sourceFaces.size()) - 1);
            const Vec3 nextCentre = faceCentre(mesh, block.sourceFaces[nextFace], positions);
            const Vec3 tangent = (nextCentre - previousCentre).normalized();
            Vec3 averageNormal = previousNormal + nextNormal;
            if (averageNormal.lengthSquared() <= 1e-8f) averageNormal = previousNormal;
            else averageNormal.normalize();
            signature.bend.push_back(bend);
            signature.rulingAngle.push_back(std::atan2(averageNormal.dot(tangent.cross(ruling)), tangent.dot(ruling)));
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
        m_bottomPositions.clear();
        m_bottomPositions.reserve(data->vertices.size());
        for (const MeshVertex& vertex : data->vertices) m_bottomPositions.push_back(vertex.position - vertex.normal * m_stripThickness);
        m_signatures.clear();
        m_bottomSignatures.clear();
        m_stackBlockIndices.clear();
        std::vector<Vec3> positions;
        positions.reserve(data->vertices.size());
        for (const MeshVertex& vertex : data->vertices) positions.push_back(vertex.position);
        // Short remainders are still extruded and coloured.  Stack comparison
        // retains its original equal-length descriptor assumption, so only
        // complete N-face blocks participate in that ordering.
        for (int blockIndex = 0; blockIndex < static_cast<int>(m_faceBlocks.size()); ++blockIndex) {
            const FaceBlock& block = m_faceBlocks[blockIndex];
            if (static_cast<int>(block.sourceFaces.size()) != m_facesPerStrip) continue;
            m_signatures.push_back(buildBlockSignature(*data, block, blockIndex, positions));
            m_bottomSignatures.push_back(buildBlockSignature(*data, block, blockIndex, m_bottomPositions));
            m_stackBlockIndices.push_back(blockIndex);
        }
        m_matches = findSimilarRibbonStrips(m_signatures, 3);
        m_stackResult = findBestRibbonStack(m_signatures, m_bottomSignatures);
        applyStripColours();
        rebuildRibbonBlockVisualisation();
        m_lastStackThickness = m_stripThickness;
        if (m_showStack) rebuildStackVisualisation();
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
        summary << "stack cost " << std::fixed << std::setprecision(3) << m_stackResult.totalCost;
        return summary.str();
    }

    struct StackStripGeometry {
        std::vector<Vec3> positions;
        std::vector<Vec3> bottomPositions;
        std::vector<std::vector<int>> faces;
        std::vector<std::pair<Vec3, Vec3>> rulings;
        Vec3 labelPosition;
        float minZ = 0.0f;
        float maxZ = 0.0f;
    };

    struct StackVisualLayer {
        int stripIndex = -1;
        Vec3 labelPosition;
        std::vector<std::pair<Vec3, Vec3>> rulings;
        std::shared_ptr<MeshObject> mesh;
    };

    StackStripGeometry buildStackStripGeometry(int blockIndex) const {
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
        Vec3 longitudinal = (lastCentre - firstCentre).normalized();
        Vec3 averageFaceNormal;
        Vec3 origin;
        for (int faceIndex : block.sourceFaces) {
            origin += faceCentre(*data, faceIndex, positions);
            averageFaceNormal += faceNormal(*data, faceIndex, positions);
        }
        origin /= static_cast<float>(block.sourceFaces.size());
        if (longitudinal.lengthSquared() <= 1e-8f) longitudinal = Vec3(1.0f, 0.0f, 0.0f);
        Vec3 normal = averageFaceNormal.normalized();
        if (normal.lengthSquared() <= 1e-8f) normal = Vec3(0.0f, 0.0f, 1.0f);
        Vec3 ruling = normal.cross(longitudinal).normalized();
        if (ruling.lengthSquared() <= 1e-8f) ruling = Vec3(0.0f, 1.0f, 0.0f);

        auto toLocal = [&](const Vec3& point) {
            const Vec3 delta = point - origin;
            return Vec3(delta.dot(longitudinal), delta.dot(ruling), delta.dot(normal));
        };

        std::unordered_map<int, int> vertexMap;
        for (int faceIndex : block.sourceFaces) {
            std::vector<int> face;
            for (int vertex : data->faces[faceIndex].vertices) {
                auto [entry, inserted] = vertexMap.emplace(vertex, static_cast<int>(result.positions.size()));
                if (inserted) {
                    result.positions.push_back(toLocal(positions[vertex]));
                    result.bottomPositions.push_back(toLocal(m_bottomPositions[vertex]));
                }
                face.push_back(entry->second);
            }
            result.faces.push_back(std::move(face));
        }
        result.minZ = result.maxZ = result.positions.empty() ? 0.0f : result.positions.front().z;
        for (const Vec3& position : result.positions) {
            result.labelPosition += position;
            result.minZ = std::min(result.minZ, position.z);
            result.maxZ = std::max(result.maxZ, position.z);
        }
        if (!result.positions.empty()) result.labelPosition /= static_cast<float>(result.positions.size());
        for (const auto& [start, end] : block.walkEdges) {
            result.rulings.push_back({toLocal(positions[start]), toLocal(positions[end])});
        }
        return result;
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
        if (!m_valid || !data || m_bottomPositions.size() != data->vertices.size()) return;
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
                        bottomPositions.push_back(m_bottomPositions[vertex]);
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
        m_stackBoundsValid = false;
    }

    void rebuildStackVisualisation() {
        clearStackVisualisation();
        if (!m_showStack || m_stackResult.order.empty()) return;

        std::vector<StackStripGeometry> geometries;
        geometries.reserve(m_stackResult.order.size());
        for (int stripIndex : m_stackResult.order) {
            if (stripIndex < 0 || stripIndex >= static_cast<int>(m_stackBlockIndices.size())) return;
            geometries.push_back(buildStackStripGeometry(m_stackBlockIndices[stripIndex]));
        }

        // Place every layer at exact bounding-box contact with the preceding
        // layer. This deliberately adds no arbitrary inter-layer clearance.
        auto bottomMinZ = [&](const StackStripGeometry& geometry) {
            float result = std::numeric_limits<float>::infinity();
            for (const Vec3& position : geometry.bottomPositions) result = std::min(result, position.z);
            return result;
        };
        std::vector<float> layerZ(geometries.size(), 0.0f);
        float stackMinZ = bottomMinZ(geometries.front());
        float stackMaxZ = geometries.front().maxZ;
        for (size_t layer = 1; layer < geometries.size(); ++layer) {
            // The lower face of this upper solid contacts the top face of the
            // preceding solid. The descriptor cost above is therefore read as
            // lower(upper) versus top(lower) compatibility.
            layerZ[layer] = stackMaxZ - bottomMinZ(geometries[layer]);
            stackMaxZ = layerZ[layer] + geometries[layer].maxZ;
        }
        const float centreZ = 0.5f * (stackMinZ + stackMaxZ);
        for (float& z : layerZ) z -= centreZ;

        float stackMinX = std::numeric_limits<float>::infinity();
        float stackMaxX = -std::numeric_limits<float>::infinity();
        float stackMinY = std::numeric_limits<float>::infinity();
        float stackMaxY = -std::numeric_limits<float>::infinity();
        for (const StackStripGeometry& geometry : geometries) {
            for (const Vec3& position : geometry.positions) {
                stackMinX = std::min(stackMinX, position.x);
                stackMaxX = std::max(stackMaxX, position.x);
                stackMinY = std::min(stackMinY, position.y);
                stackMaxY = std::max(stackMaxY, position.y);
            }
        }
        const std::shared_ptr<MeshData> originalData = m_mesh ? m_mesh->getMeshData() : nullptr;
        Vec3 originalMin, originalMax;
        if (originalData && !originalData->vertices.empty()) originalData->updateBounds(originalMin, originalMax);
        const float displaySeparation = std::max(0.05f, 0.1f * std::max(originalMax.x - originalMin.x, stackMaxX - stackMinX));
        const Vec3 stackOffset(originalMax.x - stackMinX + displaySeparation,
                               0.5f * (originalMin.y + originalMax.y - stackMinY - stackMaxY),
                               0.0f);
        m_stackBoundsMin = Vec3(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
        m_stackBoundsMax = Vec3(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
        for (int layer = 0; layer < static_cast<int>(geometries.size()); ++layer) {
            StackStripGeometry& geometry = geometries[layer];
            const int stripIndex = m_stackResult.order[layer];
            const Vec3 offset = stackOffset + Vec3(0.0f, 0.0f, layerZ[layer]);
            for (Vec3& position : geometry.positions) {
                position += offset;
                m_stackBoundsMin.x = std::min(m_stackBoundsMin.x, position.x);
                m_stackBoundsMin.y = std::min(m_stackBoundsMin.y, position.y);
                m_stackBoundsMin.z = std::min(m_stackBoundsMin.z, position.z);
                m_stackBoundsMax.x = std::max(m_stackBoundsMax.x, position.x);
                m_stackBoundsMax.y = std::max(m_stackBoundsMax.y, position.y);
                m_stackBoundsMax.z = std::max(m_stackBoundsMax.z, position.z);
            }
            for (Vec3& position : geometry.bottomPositions) position += offset;
            for (auto& ruling : geometry.rulings) {
                ruling.first += offset;
                ruling.second += offset;
            }
            geometry.labelPosition += offset;

            const int blockIndex = m_stackBlockIndices[stripIndex];
            auto mesh = createSolidBlock("ribbon_stack_layer_" + std::to_string(layer), geometry.positions,
                                         geometry.bottomPositions, geometry.faces,
                                         stripColour(blockIndex, static_cast<int>(m_faceBlocks.size())));
            for (const Vec3& position : geometry.bottomPositions) {
                m_stackBoundsMin.x = std::min(m_stackBoundsMin.x, position.x);
                m_stackBoundsMin.y = std::min(m_stackBoundsMin.y, position.y);
                m_stackBoundsMin.z = std::min(m_stackBoundsMin.z, position.z);
                m_stackBoundsMax.x = std::max(m_stackBoundsMax.x, position.x);
                m_stackBoundsMax.y = std::max(m_stackBoundsMax.y, position.y);
                m_stackBoundsMax.z = std::max(m_stackBoundsMax.z, position.z);
            }
            if (!mesh) continue;
            scene().addObject(mesh);
            m_stackLayers.push_back({stripIndex, geometry.labelPosition, std::move(geometry.rulings), std::move(mesh)});
        }
        m_stackBoundsValid = !m_stackLayers.empty();
        m_lastStackThickness = m_stripThickness;
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
            for (const auto& [start, end] : block.walkEdges) {
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
        for (int layer = 0; layer < static_cast<int>(m_stackLayers.size()); ++layer) {
            const StackVisualLayer& item = m_stackLayers[layer];
            renderer.drawText("L" + std::to_string(layer) + " / S" + std::to_string(item.stripIndex),
                              item.labelPosition, 1.0f);
            for (const auto& ruling : item.rulings) {
                renderer.drawLine(ruling.first, ruling.second, Color(0.88f, 0.12f, 0.08f, 1.0f), 1.0f);
            }
        }
    }

    void drawStackBounds(Renderer& renderer) const {
        if (!m_stackBoundsValid) return;
        const Vec3& lo = m_stackBoundsMin;
        const Vec3& hi = m_stackBoundsMax;
        const std::array<Vec3, 8> corners = {
            Vec3(lo.x, lo.y, lo.z), Vec3(hi.x, lo.y, lo.z), Vec3(hi.x, hi.y, lo.z), Vec3(lo.x, hi.y, lo.z),
            Vec3(lo.x, lo.y, hi.z), Vec3(hi.x, lo.y, hi.z), Vec3(hi.x, hi.y, hi.z), Vec3(lo.x, hi.y, hi.z)};
        constexpr std::array<std::array<int, 2>, 12> edges = {{{0, 1}, {1, 2}, {2, 3}, {3, 0},
                                                                  {4, 5}, {5, 6}, {6, 7}, {7, 4},
                                                                  {0, 4}, {1, 5}, {2, 6}, {3, 7}}};
        for (const auto& edge : edges) renderer.drawLine(corners[edge[0]], corners[edge[1]], Color(0.0f, 0.0f, 0.0f, 1.0f), 1.2f);
    }

    void drawStackAnalysis(Renderer& renderer) const {
        renderer.setColor(Color(0.0f, 0.0f, 0.0f, 1.0f));
        renderer.drawString("Stack compatibility: total = " + formatCost(m_stackResult.totalCost) +
                            "; lower normal-offset face of each upper solid touches the top face below.", 10.0f, 140.0f);
        for (int layer = 1; layer < static_cast<int>(m_stackLayers.size()); ++layer) {
            const int below = m_stackLayers[layer - 1].stripIndex;
            const int above = m_stackLayers[layer].stripIndex;
            const RibbonPairCompatibility& cost = m_stackResult.pairCosts[below][above];
            std::ostringstream line;
            line << "L" << layer - 1 << " S" << below << " -> L" << layer << " S" << above
                 << " : total " << formatCost(cost.totalCost)
                 << " (local " << formatCost(cost.localCost) << ", accumulated " << formatCost(cost.accumulatedCost)
                 << (cost.reversed ? ", reverse candidate" : ", forward candidate") << ")";
            renderer.drawString(line.str(), 10.0f, 160.0f + 17.0f * static_cast<float>(layer - 1));
        }
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
    std::vector<Vec3> m_bottomPositions;
    std::vector<RibbonSignature> m_signatures;
    std::vector<RibbonSignature> m_bottomSignatures;
    std::vector<int> m_stackBlockIndices;
    std::vector<RibbonMatch> m_matches;
    RibbonStackResult m_stackResult;
    std::vector<std::shared_ptr<MeshObject>> m_ribbonBlocks;
    std::vector<StackVisualLayer> m_stackLayers;
    std::string m_status{"Loading stereotomy.obj..."};
    int m_facesPerStrip{12};
    float m_facesPerStripSlider{8.0f};
    float m_stripThickness{0.015f};
    float m_lastStackThickness{-1.0f};
    bool m_showOriginal{true};
    bool m_showStack{false};
    bool m_valid{false};
    Vec3 m_stackBoundsMin;
    Vec3 m_stackBoundsMax;
    bool m_stackBoundsValid{false};
};

ALICE2_REGISTER_SKETCH_AUTO(DevelopableRibbonSketch)

#endif
