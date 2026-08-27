#define __MAIN__
#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>
#include <tna/TnaSolver.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace alice2;

class TnaSketch : public ISketch {
public:
    std::string getName() const override { return "TNA Form Diagram"; }
    std::string getDescription() const override { return "Stage 1: supported planar form-diagram topology"; }
    std::string getAuthor() const override { return "alice2 User"; }

    void setup() override {
        scene().setBackgroundColor(Color(0.96f, 0.96f, 0.96f, 1.0f));
        scene().setShowGrid(false);
        scene().setShowAxes(true);
        scene().setAxesLength(1.2f);
        reload();
    }

    void update(float) override {}

    void draw(Renderer& renderer, Camera&) override {
        if (m_showInput) drawMesh(renderer, m_inputMesh, Color(0.72f, 0.72f, 0.72f, 1.0f), 0.8f);
        drawMesh(renderer, m_formMesh, Color(0.05f, 0.32f, 0.82f, 1.0f), 1.8f);
        drawSupports(renderer);

        renderer.setColor(Color(0.06f, 0.06f, 0.06f, 1.0f));
        renderer.drawString("r reload OBJ | i toggle original mesh", 10.0f, 30.0f);
        renderer.drawString(m_status, 10.0f, 52.0f);
        renderer.drawString("blue: planar form with appended exterior n-gons | black: supports", 10.0f, 74.0f);
    }

    bool onKeyPress(unsigned char key, int, int) override {
        switch (key) {
            case 'r': case 'R': reload(); return true;
            case 'i': case 'I': m_showInput = !m_showInput; return true;
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
        m_inputObject = std::make_shared<MeshObject>("tna_input");
        try {
            m_inputObject->readFromObj(dataPath(m_objPath).string());
            // Extended OBJ exports frequently duplicate face-corner vertices.
            // Welding restores the shared topology required for boundary walks.
            m_inputObject->weld(1e-3f);
            m_inputMesh = m_inputObject->getMeshData();

            // The form diagram works in plan. Keep m_inputMesh untouched as
            // a grey reference and flatten only its duplicate used by TNA.
            MeshData planarInput = *m_inputMesh;
            for (MeshVertex& vertex : planarInput.vertices) vertex.position.z = 0.0f;
            planarInput.calculateNormals();
            planarInput.triangulationDirty = true;

            // Black vertex colours are explicit support tags. An ordinary OBJ
            // has no such tags, so the builder falls back to all boundaries.
            m_result = m_solver.makeFormDiagram(planarInput, blackVertexSupports(*m_inputMesh));
            if (m_result.success) {
                m_formMesh = std::make_shared<MeshData>(m_result.mesh);
            } else {
                m_formMesh.reset();
            }
            m_status = m_result.diagnostic;
        } catch (const std::exception& error) {
            m_inputMesh.reset();
            m_formMesh.reset();
            m_status = std::string("Failed to load ") + m_objPath + ": " + error.what();
        }
    }

    static std::vector<int> blackVertexSupports(const MeshData& mesh) {
        constexpr float blackThreshold = 0.02f;
        std::vector<int> supports;
        for (int vertex = 0; vertex < static_cast<int>(mesh.vertices.size()); ++vertex) {
            const Color& color = mesh.vertices[vertex].color;
            if (color.r <= blackThreshold && color.g <= blackThreshold && color.b <= blackThreshold) {
                supports.push_back(vertex);
            }
        }
        return supports;
    }

    static void drawMesh(Renderer& renderer, const std::shared_ptr<MeshData>& mesh,
                         const Color& color, float width) {
        if (!mesh) return;
        std::vector<Vec3> segments;
        segments.reserve(mesh->edges.size() * 2);
        for (const MeshEdge& edge : mesh->edges) {
            if (edge.vertexA < 0 || edge.vertexB < 0 ||
                edge.vertexA >= static_cast<int>(mesh->vertices.size()) ||
                edge.vertexB >= static_cast<int>(mesh->vertices.size())) continue;
            segments.push_back(mesh->vertices[edge.vertexA].position);
            segments.push_back(mesh->vertices[edge.vertexB].position);
        }
        if (!segments.empty()) renderer.drawLines(segments.data(), static_cast<int>(segments.size()), color, width);
    }

    void drawSupports(Renderer& renderer) const {
        if (!m_formMesh) return;
        for (const int vertex : m_result.supportVertices) {
            if (vertex < 0 || vertex >= static_cast<int>(m_formMesh->vertices.size())) continue;
            renderer.drawPoint(m_formMesh->vertices[vertex].position, Color(0.0f, 0.0f, 0.0f, 1.0f), 8.0f);
        }
    }

    std::string m_objPath{"tna.obj"};
    std::shared_ptr<MeshObject> m_inputObject;
    std::shared_ptr<MeshData> m_inputMesh;
    std::shared_ptr<MeshData> m_formMesh;
    TnaSolver m_solver;
    TnaFormDiagram m_result;
    std::string m_status{"Loading TNA mesh"};
    bool m_showInput{true};
};

ALICE2_REGISTER_SKETCH_AUTO(TnaSketch)

#endif
