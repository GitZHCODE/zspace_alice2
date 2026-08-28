#define __MAIN__
#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>
#include <vector>

using namespace alice2;

class DevelopableRibbonSketch : public ISketch {
public:
    std::string getName() const override { return "Developable Ribbon"; }
    std::string getDescription() const override { return "Planarises an open quad ribbon and finds similar developable strips"; }
    std::string getAuthor() const override { return "alice2 User"; }

    void setup() override {
        scene().setBackgroundColor(Color(1.0f, 1.0f, 1.0f));
        scene().setShowGrid(false);
        scene().setShowAxes(true);
        scene().setAxesLength(0.2f);
        reload();
    }

    void update(float) override {}

    void draw(Renderer& renderer, Camera&) override {
        renderer.drawString(m_status, 10.0f, 30.0f);
        renderer.drawString("r reload | p solve planar faces | o toggle original wire | [ / ] strip faces", 10.0f, 50.0f);
        if (m_showOriginal && m_original) drawEdges(renderer, *m_original->getMeshData(), Color(0.62f, 0.62f, 0.62f, 1.0f), 1.0f);
        if (m_valid) {
            drawRulings(renderer);
            drawStripIndices(renderer);
        }
    }

    bool onKeyPress(unsigned char key, int, int) override {
        switch (key) {
            case 'r': case 'R': reload(); return true;
            case 'p': case 'P': planarise(); return true;
            case 'o': case 'O': m_showOriginal = !m_showOriginal; return true;
            case '[': m_facesPerStrip = std::max(2, m_facesPerStrip - 1); refreshSignatures(); return true;
            case ']': ++m_facesPerStrip; refreshSignatures(); return true;
            default: return false;
        }
    }

private:
    static std::filesystem::path dataPath(const std::string& file) {
        const std::filesystem::path requested(file);
        if (requested.is_absolute() || std::filesystem::exists(requested)) return requested;
        const std::filesystem::path local = std::filesystem::path("data") / file;
        if (std::filesystem::exists(local)) return local;
        return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / "data" / file;
    }

    void reload() {
        if (m_mesh) scene().removeObject(m_mesh);
        m_mesh = std::make_shared<MeshObject>("developable_ribbon");
        try {
            m_mesh->readFromObj(dataPath("ribbon.obj").string());
            // Blender's exported face corners can be duplicated. The ribbon
            // ordering is connectivity based, so restore the shared vertices.
            m_mesh->weld(1e-5f);
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
            m_valid = orderRibbon(*m_mesh->getMeshData(), m_ribbon, &diagnostic);
            if (!m_valid) {
                m_status = "Ribbon input invalid: " + diagnostic;
                return;
            }
            copyRibbonToMesh();
            m_mesh->setColor(Color(0.15f, 0.62f, 0.90f, 1.0f));
            m_mesh->setUseFaceColors(true);
            m_mesh->setShowEdges(true);
            m_mesh->setShowFaces(true);
            scene().addObject(m_mesh);
            refreshSignatures();
            m_status = diagnostic + "  " + matchSummary();
        } catch (const std::exception& error) {
            m_valid = false;
            m_status = std::string("Could not load ribbon.obj: ") + error.what();
        }
    }

    void planarise() {
        if (!m_valid) return;
        const int iterations = m_solver.solve(*m_mesh);
        copyMeshToRibbon();
        refreshSignatures();
        std::ostringstream report;
        report << "ProjectionSolver planar-face solve: " << iterations << " iterations; max residual " << std::scientific
               << std::setprecision(2) << maxRibbonPlanarityError(m_ribbon) << ". "
               << matchSummary();
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
        const int faceCount = static_cast<int>(m_ribbon.faces.size());
        m_facesPerStrip = std::min(std::max(2, m_facesPerStrip), faceCount);
        m_signatures = buildRibbonSignatures(m_ribbon, m_facesPerStrip, m_facesPerStrip);
        m_matches = findSimilarRibbonStrips(m_signatures, 3);
        applyStripColours();
    }

    static Color stripColour(int index, int count) {
        const float t = count <= 1 ? 0.5f : static_cast<float>(index) / static_cast<float>(count - 1);
        return Color::lerp(Color(0.96f, 0.38f, 0.70f, 1.0f), Color(0.30f, 0.68f, 0.96f, 1.0f), t);
    }

    void applyStripColours() {
        const std::shared_ptr<MeshData> data = m_mesh ? m_mesh->getMeshData() : nullptr;
        if (!data) return;
        for (MeshFace& face : data->faces) face.color = Color(0.82f, 0.82f, 0.82f, 1.0f);
        for (int strip = 0; strip < static_cast<int>(m_signatures.size()); ++strip) {
            const RibbonSignature& signature = m_signatures[strip];
            const Color colour = stripColour(strip, static_cast<int>(m_signatures.size()));
            for (int ribbonFace = signature.startFace; ribbonFace < signature.startFace + signature.faceCount &&
                 ribbonFace < static_cast<int>(m_ribbon.sourceFaceIndices.size()); ++ribbonFace) {
                const int sourceFace = m_ribbon.sourceFaceIndices[ribbonFace];
                if (sourceFace >= 0 && sourceFace < static_cast<int>(data->faces.size())) {
                    data->faces[sourceFace].color = colour;
                }
            }
        }
    }

    std::string matchSummary() const {
        if (m_signatures.empty()) return "Need at least two faces per strip.";
        std::ostringstream summary;
        summary << m_signatures.size() << " windows (" << m_facesPerStrip << " faces)";
        if (!m_matches.empty()) {
            const RibbonMatch& match = m_matches.front();
            summary << "; closest " << match.stripA << "-" << match.stripB
                    << (match.reversed ? " reversed" : " forward") << ": "
                    << std::fixed << std::setprecision(3) << match.distance;
        }
        return summary.str();
    }

    void drawRulings(Renderer& renderer) const {
        for (size_t i = 0; i < m_ribbon.railP.size(); ++i) {
            const Vec3& p = m_ribbon.vertices[m_ribbon.railP[i]];
            const Vec3& q = m_ribbon.vertices[m_ribbon.railQ[i]];
            renderer.drawLine(p, q, Color(0.95f, 0.22f, 0.08f, 1.0f), 1.3f);
        }
    }

    void drawStripIndices(Renderer& renderer) const {
        renderer.setColor(Color(0.08f, 0.08f, 0.12f, 1.0f));
        for (int strip = 0; strip < static_cast<int>(m_signatures.size()); ++strip) {
            const RibbonSignature& signature = m_signatures[strip];
            Vec3 centre;
            Vec3 normal;
            int pointCount = 0;
            for (int faceIndex = signature.startFace;
                 faceIndex < signature.startFace + signature.faceCount && faceIndex < static_cast<int>(m_ribbon.faces.size());
                 ++faceIndex) {
                const std::array<int, 4>& face = m_ribbon.faces[faceIndex];
                const Vec3& a = m_ribbon.vertices[face[0]];
                const Vec3& b = m_ribbon.vertices[face[1]];
                const Vec3& d = m_ribbon.vertices[face[3]];
                for (int vertex : face) {
                    centre += m_ribbon.vertices[vertex];
                    ++pointCount;
                }
                normal += (b - a).cross(d - a).normalized();
            }
            if (pointCount == 0) continue;
            centre /= static_cast<float>(pointCount);
            if (normal.lengthSquared() > 1e-8f) centre += normal.normalized() * 0.002f;
            renderer.drawText(std::to_string(strip), centre, 1.1f);
        }
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

    std::shared_ptr<MeshObject> m_mesh;
    std::shared_ptr<MeshObject> m_original;
    ProjectionSolver m_solver;
    QuadRibbon m_ribbon;
    std::vector<RibbonSignature> m_signatures;
    std::vector<RibbonMatch> m_matches;
    std::string m_status{"Loading ribbon.obj..."};
    int m_facesPerStrip{12};
    bool m_showOriginal{true};
    bool m_valid{false};
};

ALICE2_REGISTER_SKETCH_AUTO(DevelopableRibbonSketch)

#endif
