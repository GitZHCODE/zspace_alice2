#define __MAIN__

#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace alice2;

class Dev2PqSketch : public ISketch {
public:
    std::string getName() const override { return "Dev2PQ Developable Remesh"; }
    std::string getDescription() const override { return "Ruling-aligned planar quad-strip remeshing prototype"; }
    std::string getAuthor() const override { return "alice2 User"; }

    void setup() override {
        scene().setBackgroundColor(Color(0.96f, 0.96f, 0.96f, 1.0f));
        scene().setShowGrid(false);
        scene().setShowAxes(false);
        m_ui = std::make_unique<SimpleUI>(input());
        m_ui->addSlider("Strip spacing", Vec2{10.0f, 92.0f}, 190.0f, 0.02f, 0.30f, m_spacing);
        m_ui->addSlider("Ruling confidence", Vec2{10.0f, 126.0f}, 190.0f, 0.02f, 0.85f, m_confidenceThreshold);
        reload();
    }

    void update(float) override {
        if (std::abs(m_spacing - m_lastSpacing) > 1e-4f ||
            std::abs(m_confidenceThreshold - m_lastConfidence) > 1e-4f) {
            runRemesher();
            m_lastSpacing = m_spacing;
            m_lastConfidence = m_confidenceThreshold;
        }
    }

    void draw(Renderer& renderer, Camera&) override {
        drawSource(renderer);
        if (m_drawRulings) drawRulings(renderer);
        if (m_drawIsolines) drawLines(renderer, m_result.rulingIsolines, Color(0.05f, 0.45f, 0.92f, 1.0f), 1.6f);
        if (m_drawOutput) drawOutput(renderer);
        renderer.setColor(Color(0.05f, 0.05f, 0.05f, 1.0f));
        renderer.drawString("r reload | m curvature estimator | x field | v raw/processed | l isolines | q output | e export OBJ", 10.0f, 30.0f);
        renderer.drawString(m_status, 10.0f, 52.0f);
        if (m_ui) m_ui->draw(renderer);
    }

    bool onKeyPress(unsigned char key, int, int) override {
        switch (key) {
            case 'r': case 'R': reload(); return true;
            case 'm': case 'M':
                m_curvatureEstimator = m_curvatureEstimator == Dev2PqCurvatureEstimator::Libigl
                                           ? Dev2PqCurvatureEstimator::MeshObject
                                           : Dev2PqCurvatureEstimator::Libigl;
                runRemesher();
                return true;
            case 'x': case 'X': m_drawRulings = !m_drawRulings; return true;
            case 'v': case 'V': m_showProcessedRulings = !m_showProcessedRulings; return true;
            case 'l': case 'L': m_drawIsolines = !m_drawIsolines; return true;
            case 'q': case 'Q': m_drawOutput = !m_drawOutput; return true;
            case 'e': case 'E': exportResult(); return true;
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
        const std::filesystem::path requestedPath(file);
        if (requestedPath.is_absolute() || std::filesystem::exists(requestedPath)) return requestedPath;
        const std::filesystem::path workingDirectoryPath = std::filesystem::path("data") / file;
        if (std::filesystem::exists(workingDirectoryPath)) return workingDirectoryPath;
        return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / "data" / file;
    }

    void reload() {
        m_source = std::make_shared<MeshObject>("dev2pq_input");
        try {
            const std::filesystem::path path = dataPath(m_objPath);
            m_source->readFromObj(path.string());
            m_source->setShowFaces(false);
            runRemesher();
        } catch (const std::exception& error) {
            m_status = std::string("Failed to load ") + m_objPath + ": " + error.what();
        }
    }

    void runRemesher() {
        const auto data = m_source ? m_source->getMeshData() : nullptr;
        if (!data || data->faces.empty()) {
            m_status = "No OBJ mesh loaded";
            return;
        }
        Dev2PqOptions options;
        options.stripSpacing = m_spacing;
        options.confidenceThreshold = m_confidenceThreshold;
        options.curvatureEstimator = m_curvatureEstimator;
        m_result = m_remesher.remesh(*data, options);
        const char* estimator = m_curvatureEstimator == Dev2PqCurvatureEstimator::Libigl ? "libigl" : "MeshObject";
        m_status = m_result.success ? std::string(estimator) + ": " + m_result.diagnostic
                                    : "Dev2PQ rejected input: " + m_result.diagnostic;
    }

    void exportResult() {
        if (!m_result.success || !m_result.mesh || m_result.mesh->faces.empty()) {
            m_status = "No Dev2PQ mesh available to export";
            return;
        }
        MeshObject output("dev2pq_remesh");
        output.setMeshData(m_result.mesh);
        output.writeToObj(dataPath("dev2pq_remesh.obj").string());
        m_status = "Exported data/dev2pq_remesh.obj";
    }

    void drawSource(Renderer& renderer) const {
        const auto data = m_source ? m_source->getMeshData() : nullptr;
        if (!data) return;
        std::vector<Vec3> segments;
        for (const MeshEdge& edge : data->edges) {
            if (edge.vertexA < 0 || edge.vertexB < 0 || edge.vertexA >= static_cast<int>(data->vertices.size()) || edge.vertexB >= static_cast<int>(data->vertices.size())) continue;
            segments.push_back(data->vertices[edge.vertexA].position);
            segments.push_back(data->vertices[edge.vertexB].position);
        }
        if (!segments.empty()) renderer.drawLines(segments.data(), static_cast<int>(segments.size()), Color(0.60f, 0.60f, 0.60f, 1.0f), 0.9f);
    }

    void drawRulings(Renderer& renderer) const {
        const auto data = m_source ? m_source->getMeshData() : nullptr;
        if (!data) return;
        const std::vector<Vec3>& directions = m_showProcessedRulings ? m_result.faceRulings : m_result.rawFaceRulings;
        const std::vector<float>& confidences = m_showProcessedRulings ? m_result.faceConfidence : m_result.rawFaceConfidence;
        const int count = std::min(static_cast<int>(data->faces.size()), static_cast<int>(directions.size()));
        for (int faceIndex = 0; faceIndex < count; ++faceIndex) {
            const MeshFace& face = data->faces[faceIndex];
            if (face.vertices.size() < 3) continue;
            Vec3 centre;
            for (int vertex : face.vertices) centre += data->vertices[vertex].position;
            centre /= static_cast<float>(face.vertices.size());
            const float confidence = faceIndex < static_cast<int>(confidences.size()) ? confidences[faceIndex] : 0.0f;
            const Color color = Color::lerp(Color(0.85f, 0.24f, 0.06f, 1.0f), Color(0.08f, 0.65f, 0.20f, 1.0f), confidence);
            const Vec3 direction = directions[faceIndex] * (0.04f + 0.05f * confidence);
            renderer.drawLine(centre - direction, centre + direction, color, 1.3f);
        }
    }

    void drawLines(Renderer& renderer, const std::vector<std::vector<Vec3>>& lines, const Color& color, float width) const {
        std::vector<Vec3> segments;
        for (const auto& line : lines) {
            for (int index = 0; index + 1 < static_cast<int>(line.size()); ++index) {
                segments.push_back(line[index]);
                segments.push_back(line[index + 1]);
            }
        }
        if (!segments.empty()) renderer.drawLines(segments.data(), static_cast<int>(segments.size()), color, width);
    }

    void drawOutput(Renderer& renderer) const {
        if (!m_result.mesh) return;
        std::vector<Vec3> quads;
        std::vector<Vec3> planar;
        for (const MeshFace& face : m_result.mesh->faces) {
            std::vector<Vec3>& segments = face.vertices.size() == 4 ? quads : planar;
            for (int index = 0; index < static_cast<int>(face.vertices.size()); ++index) {
                const int a = face.vertices[index];
                const int b = face.vertices[(index + 1) % face.vertices.size()];
                segments.push_back(m_result.mesh->vertices[a].position);
                segments.push_back(m_result.mesh->vertices[b].position);
            }
        }
        if (!quads.empty()) renderer.drawLines(quads.data(), static_cast<int>(quads.size()), Color(0.04f, 0.34f, 0.12f, 1.0f), 2.2f);
        if (!planar.empty()) renderer.drawLines(planar.data(), static_cast<int>(planar.size()), Color(0.92f, 0.42f, 0.04f, 1.0f), 1.5f);
    }

    // Set this to an OBJ in alice2/data, a working-directory-relative path, or an absolute OBJ path.
    std::string m_objPath{"dev2pq.obj"};
    // std::string m_objPath{"dev2pq_2c.obj"};
    std::shared_ptr<MeshObject> m_source;
    Dev2PqRemesher m_remesher;
    Dev2PqResult m_result;
    std::unique_ptr<SimpleUI> m_ui;
    std::string m_status{"Loading Dev2PQ input"};
    float m_spacing{0.08f};
    float m_confidenceThreshold{0.16f};
    float m_lastSpacing{-1.0f};
    float m_lastConfidence{-1.0f};
    Dev2PqCurvatureEstimator m_curvatureEstimator{Dev2PqCurvatureEstimator::Libigl};
    bool m_drawRulings{true};
    bool m_showProcessedRulings{false};
    bool m_drawIsolines{true};
    bool m_drawOutput{true};
};

ALICE2_REGISTER_SKETCH_AUTO(Dev2PqSketch)

#endif
