#define __MAIN__

#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace alice2;

class Dev2PqSketch : public ISketch {
public:
    std::string getName() const override { return "Dev2PQ Directional Field"; }
    std::string getDescription() const override { return "Directional power-2 field and curl-projected Dev2PQ prototype"; }
    std::string getAuthor() const override { return "alice2 User"; }

    void setup() override {
        scene().setBackgroundColor(Color(0.96f, 0.96f, 0.96f, 1.0f));
        scene().setShowGrid(false);
        scene().setShowAxes(false);
        m_ui = std::make_unique<SimpleUI>(input());
        m_ui->addSlider("Isoline spacing", Vec2{10.0f, 92.0f}, 190.0f, 0.005f, 0.30f, m_spacing);
        m_ui->addSlider("Directional alignment", Vec2{10.0f, 126.0f}, 190.0f, 0.10f, 12.0f, m_alignmentWeight);
        reload();
    }

    void update(float) override {
        if (std::abs(m_spacing - m_lastSpacing) > 1e-4f ||
            std::abs(m_alignmentWeight - m_lastAlignmentWeight) > 1e-4f) {
            runField();
            m_lastSpacing = m_spacing;
            m_lastAlignmentWeight = m_alignmentWeight;
        }
    }

    void draw(Renderer& renderer, Camera&) override {
        drawSource(renderer);
        if (m_drawField) drawField(renderer);
        if (m_drawIsolines) drawLines(renderer, m_result.isolines, Color(0.05f, 0.45f, 0.92f, 1.0f), 1.8f);
        renderer.setColor(Color(0.05f, 0.05f, 0.05f, 1.0f));
        renderer.drawString("r reload | p raw/Directional field | x directions | l isolines", 10.0f, 30.0f);
        renderer.drawString(m_status, 10.0f, 52.0f);
        renderer.drawString(m_showRaw ? "view: raw libigl curvature rulings" : "view: Directional curl-projected rulings", 10.0f, 74.0f);
        if (m_ui) m_ui->draw(renderer);
    }

    bool onKeyPress(unsigned char key, int, int) override {
        switch (key) {
            case 'r': case 'R': reload(); return true;
            case 'p': case 'P': m_showRaw = !m_showRaw; return true;
            case 'x': case 'X': m_drawField = !m_drawField; return true;
            case 'l': case 'L': m_drawIsolines = !m_drawIsolines; return true;
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
    static std::filesystem::path dataPath(const std::string& file) {
        const std::filesystem::path requested(file);
        if (requested.is_absolute() || std::filesystem::exists(requested)) return requested;
        const std::filesystem::path working = std::filesystem::path("data") / file;
        if (std::filesystem::exists(working)) return working;
        return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / "data" / file;
    }

    void reload() {
        m_source = std::make_shared<MeshObject>("dev2pq_input");
        try {
            m_source->readFromObj(dataPath(m_objPath).string());
            m_source->setShowFaces(false);
            runField();
        } catch (const std::exception& error) {
            m_status = std::string("Failed to load ") + m_objPath + ": " + error.what();
        }
    }

    void runField() {
        const auto mesh = m_source ? m_source->getMeshData() : nullptr;
        if (!mesh || mesh->faces.empty()) {
            m_status = "No OBJ mesh loaded";
            return;
        }
        Dev2PqOptions options;
        options.stripSpacing = m_spacing;
        options.alignmentWeight = m_alignmentWeight;
        m_result = m_remesher.remesh(*mesh, options);
        if (m_result.success) {
            m_status = m_result.diagnostic + " | curl " + std::to_string(m_result.maxCurlBefore) + " -> " +
                       std::to_string(m_result.maxCurlAfter) + " | singularities " + std::to_string(m_result.singularityCount);
        } else {
            m_status = "Dev2PQ rejected input: " + m_result.diagnostic;
        }
    }

    void drawSource(Renderer& renderer) const {
        const auto mesh = m_source ? m_source->getMeshData() : nullptr;
        if (!mesh) return;
        std::vector<Vec3> segments;
        for (const MeshEdge& edge : mesh->edges) {
            if (edge.vertexA < 0 || edge.vertexB < 0 || edge.vertexA >= static_cast<int>(mesh->vertices.size()) || edge.vertexB >= static_cast<int>(mesh->vertices.size())) continue;
            segments.push_back(mesh->vertices[edge.vertexA].position);
            segments.push_back(mesh->vertices[edge.vertexB].position);
        }
        if (!segments.empty()) renderer.drawLines(segments.data(), static_cast<int>(segments.size()), Color(0.62f, 0.62f, 0.62f, 1.0f), 0.9f);
    }

    void drawField(Renderer& renderer) const {
        const auto& rulings = m_showRaw ? m_result.rawRulings : m_result.optimizedRulings;
        const int count = std::min(static_cast<int>(m_result.faceCentres.size()), static_cast<int>(rulings.size()));
        for (int face = 0; face < count; ++face) {
            const float confidence = face < static_cast<int>(m_result.confidence.size()) ? m_result.confidence[face] : 0.0f;
            const Color color = Color::lerp(Color(0.88f, 0.20f, 0.08f, 1.0f), Color(0.08f, 0.65f, 0.20f, 1.0f), confidence);
            const Vec3 direction = rulings[face] * (0.035f + 0.060f * confidence);
            renderer.drawLine(m_result.faceCentres[face] - direction, m_result.faceCentres[face] + direction, color, 1.25f);
        }
    }

    static void drawLines(Renderer& renderer, const std::vector<std::vector<Vec3>>& lines, const Color& color, float width) {
        std::vector<Vec3> segments;
        for (const auto& line : lines)
            for (int index = 0; index + 1 < static_cast<int>(line.size()); ++index) {
                segments.push_back(line[index]);
                segments.push_back(line[index + 1]);
            }
        if (!segments.empty()) renderer.drawLines(segments.data(), static_cast<int>(segments.size()), color, width);
    }

    std::string m_objPath{"tna_tri_optimised.obj"};
    std::shared_ptr<MeshObject> m_source;
    Dev2PqRemesher m_remesher;
    Dev2PqResult m_result;
    std::unique_ptr<SimpleUI> m_ui;
    std::string m_status{"Loading Dev2PQ input"};
    float m_spacing{0.08f};
    float m_alignmentWeight{3.0f};
    float m_lastSpacing{-1.0f};
    float m_lastAlignmentWeight{-1.0f};
    bool m_showRaw{false};
    bool m_drawField{true};
    bool m_drawIsolines{true};
};

ALICE2_REGISTER_SKETCH_AUTO(Dev2PqSketch)

#endif
