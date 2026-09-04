#define __MAIN__
#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>

using namespace alice2;

class RuledSurfaceStackingSketch : public ISketch {
public:
    std::string getName() const override { return "Ruled Surface Stacking"; }
    std::string getDescription() const override { return "Sampled vertical nesting of diagnostic planar ruled strips"; }
    std::string getAuthor() const override { return "alice2 User"; }

    void setup() override {
        scene().setBackgroundColor(Color(0.98f, 0.98f, 0.98f, 1.0f));
        scene().setShowGrid(false);
        scene().setShowAxes(true);
        scene().setAxesLength(1.0f);
        m_ui = std::make_unique<SimpleUI>(input());
        m_ui->setTheme(SimpleUI::UITheme::Dark);
        m_ui->addSlider("Surfaces", Vec2{10.0f, 78.0f}, 210.0f, 2.0f,
                        static_cast<float>(kMaxSurfaceCount), m_surfaceCountSlider);
        m_ui->addSlider("Planes per surface", Vec2{10.0f, 106.0f}, 210.0f, 2.0f,
                        static_cast<float>(kMaxFacesPerSurface), m_facesPerSurfaceSlider);
        m_ui->addSlider("Random variation", Vec2{10.0f, 134.0f}, 210.0f, 0.0f, 1.0f, m_randomnessSlider);
        m_ui->addSlider("Length variation", Vec2{10.0f, 162.0f}, 210.0f, 0.0f, 1.0f, m_lengthRandomnessSlider);
        m_ui->addSlider("Ruling turn", Vec2{10.0f, 190.0f}, 210.0f, 0.0f, 1.0f, m_rulingRotationSlider);
        m_ui->addToggle("Optimise physical flips", UIRect{10.0f, 218.0f, 210.0f, 22.0f}, m_optimiseFlips);
        m_ui->addToggle("Hot-wire collision", UIRect{10.0f, 246.0f, 210.0f, 22.0f}, m_hotWireCollision);
        m_ui->addToggle("Show face normals", UIRect{10.0f, 274.0f, 210.0f, 22.0f}, m_showFaceNormals);
        solveAndBuild();
    }

    void cleanup() override { clearMeshes(); }

    void update(float) override {
        const int surfaceCount = std::clamp(static_cast<int>(std::lround(m_surfaceCountSlider)), 2, kMaxSurfaceCount);
        const int faceCount = std::clamp(static_cast<int>(std::lround(m_facesPerSurfaceSlider)), 2, kMaxFacesPerSurface);
        const double randomness = std::clamp(static_cast<double>(m_randomnessSlider), 0.0, 1.0);
        const double lengthRandomness = std::clamp(static_cast<double>(m_lengthRandomnessSlider), 0.0, 1.0);
        const double rulingRotation = std::clamp(static_cast<double>(m_rulingRotationSlider), 0.0, 1.0);
        if (surfaceCount != m_generation.surfaceCount || faceCount != m_generation.faceCount ||
            std::abs(randomness - m_generation.randomness) > 1e-6 ||
            std::abs(lengthRandomness - m_generation.lengthRandomness) > 1e-6 ||
            std::abs(rulingRotation - m_generation.rulingRotation) > 1e-6 ||
            m_optimiseFlips != m_lastOptimiseFlips ||
            m_hotWireCollision != m_lastHotWireCollision) {
            m_generation.surfaceCount = surfaceCount;
            m_generation.faceCount = faceCount;
            m_generation.randomness = randomness;
            m_generation.lengthRandomness = lengthRandomness;
            m_generation.rulingRotation = rulingRotation;
            m_lastOptimiseFlips = m_optimiseFlips;
            m_lastHotWireCollision = m_hotWireCollision;
            solveAndBuild();
        }
    }

    void draw(Renderer& renderer, Camera&) override {
        renderer.setColor(Color(0.05f, 0.05f, 0.05f, 1.0f));
        renderer.drawString("r rebuild | n new random family | red: rulings | blue: normals | grey: group bounds", 10.0f, 30.0f);
        renderer.drawString(m_summary, 10.0f, 52.0f);
        if (m_ui) m_ui->draw(renderer);
        drawRulingsAndNormals(renderer);
        drawBounds(renderer);
    }

    bool onKeyPress(unsigned char key, int, int) override {
        if (key == 'r' || key == 'R') {
            solveAndBuild();
            return true;
        }
        if (key == 'n' || key == 'N') {
            ++m_generation.seed;
            solveAndBuild();
            return true;
        }
        return false;
    }

    bool onMousePress(int button, int state, int x, int y) override {
        return m_ui && m_ui->onMousePress(button, state, x, y);
    }

    bool onMouseMove(int x, int y) override {
        return m_ui && m_ui->onMouseMove(x, y);
    }

private:
    static constexpr double kClearance = 0.05;
    static constexpr int kMaxSurfaceCount = 50;
    static constexpr int kMaxFacesPerSurface = 24;

    void clearMeshes() {
        for (const auto& mesh : m_meshes) {
            if (mesh) scene().removeObject(mesh);
        }
        m_meshes.clear();
    }

    void solveAndBuild() {
        clearMeshes();
        m_surfaces = makeProceduralRuledSurfaces(m_generation);
        const RuledSurfaceBounds2D foamFootprint = ruledSurfaceGroupBoundsXY(m_surfaces);
        if (m_optimiseFlips) {
            m_solution = findRuledSurfaceStackWithFlips(
                m_surfaces, foamFootprint, kClearance, m_hotWireCollision);
            for (size_t i = 0; i < m_surfaces.size() && i < m_solution.flippedBySurface.size(); ++i) {
                if (!m_solution.flippedBySurface[i]) continue;
                if (const auto flipped = flipRuledSurfaceForStack(m_surfaces[i])) m_surfaces[i] = *flipped;
            }
        } else {
            m_solution = RuledSurfaceStackSolution{};
        }
        const Eigen::Vector3d stackDirection = Eigen::Vector3d::UnitZ();
        m_valid.assign(m_surfaces.size(), false);
        for (size_t i = 0; i < m_surfaces.size(); ++i) {
            m_valid[i] = isValidForStackDirection(m_surfaces[i], stackDirection);
        }
        m_gapMatrix = m_hotWireCollision
            ? buildExtendedSweepGapMatrix(m_surfaces, foamFootprint, kClearance)
            : buildRuledSurfaceGapMatrix(m_surfaces, kClearance);
        if (!m_optimiseFlips) m_solution = findRuledSurfaceStack(m_surfaces, m_gapMatrix);
        buildMeshes();
        writeDiagnostics();

        std::ostringstream summary;
        summary << std::fixed << std::setprecision(3) << "Order: ";
        for (size_t layer = 0; layer < m_solution.order.size(); ++layer) {
            if (layer) summary << " -> ";
            summary << m_solution.order[layer];
        }
        summary << "    final height: " << m_solution.totalHeight;
        m_summary = summary.str();
    }

    void buildMeshes() {
        static const std::array<Color, 8> colors{{
            Color(0.90f, 0.22f, 0.18f, 0.82f), Color(0.12f, 0.48f, 0.90f, 0.82f),
            Color(0.16f, 0.70f, 0.28f, 0.82f), Color(0.82f, 0.45f, 0.08f, 0.82f),
            Color(0.62f, 0.20f, 0.78f, 0.82f), Color(0.05f, 0.68f, 0.70f, 0.82f),
            Color(0.88f, 0.12f, 0.55f, 0.82f), Color(0.42f, 0.58f, 0.10f, 0.82f)}};
        m_meshes.resize(m_surfaces.size());
        m_stackZBySurface.assign(m_surfaces.size(), 0.0);
        for (size_t layer = 0; layer < m_solution.order.size(); ++layer) {
            m_stackZBySurface[m_solution.order[layer]] = m_solution.stackZ[layer];
        }
        for (size_t surfaceIndex = 0; surfaceIndex < m_surfaces.size(); ++surfaceIndex) {
            std::vector<Vec3> vertices;
            std::vector<std::vector<int>> faces;
            vertices.reserve(m_surfaces[surfaceIndex].faces.size() * 4);
            faces.reserve(m_surfaces[surfaceIndex].faces.size());
            for (const RuledSurfaceFace& face : m_surfaces[surfaceIndex].faces) {
                std::vector<int> quad;
                for (const Eigen::Vector3d& vertex : face.vertices) {
                    quad.push_back(static_cast<int>(vertices.size()));
                    vertices.emplace_back(static_cast<float>(vertex.x()), static_cast<float>(vertex.y()),
                                          static_cast<float>(vertex.z() + m_stackZBySurface[surfaceIndex]));
                }
                faces.push_back(std::move(quad));
            }
            auto mesh = std::make_shared<MeshObject>("ruled_surface_" + std::to_string(surfaceIndex));
            mesh->createFromVerticesAndFaces(vertices, faces, {}, std::vector<Color>(vertices.size(), colors[surfaceIndex % colors.size()]));
            mesh->setUseFaceColors(true);
            mesh->setOpacity(0.82f);
            mesh->setShowEdges(true);
            mesh->setShowFaces(true);
            mesh->setColor(colors[surfaceIndex % colors.size()]);
            scene().addObject(mesh);
            m_meshes[surfaceIndex] = mesh;
        }
    }

    void drawRulingsAndNormals(Renderer& renderer) const {
        for (size_t surfaceIndex = 0; surfaceIndex < m_surfaces.size(); ++surfaceIndex) {
            const float z = static_cast<float>(m_stackZBySurface[surfaceIndex]);
            for (const RuledSurfaceRuling& ruling : m_surfaces[surfaceIndex].rulings) {
                renderer.drawLine(toVec3(ruling.left, z), toVec3(ruling.right, z),
                                  Color(0.88f, 0.10f, 0.08f, 1.0f), 1.4f);
            }
            if (!m_showFaceNormals) continue;
            for (const RuledSurfaceFace& face : m_surfaces[surfaceIndex].faces) {
                Eigen::Vector3d centre = Eigen::Vector3d::Zero();
                for (const Eigen::Vector3d& vertex : face.vertices) centre += vertex;
                centre /= 4.0;
                renderer.drawLine(toVec3(centre, z), toVec3(centre + face.plane.n * 0.30, z),
                                  Color(0.05f, 0.20f, 0.95f, 1.0f), 1.2f);
            }
        }
    }

    void drawBounds(Renderer& renderer) const {
        if (m_surfaces.empty()) return;
        Eigen::Vector3d minimum = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
        Eigen::Vector3d maximum = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());
        for (size_t i = 0; i < m_surfaces.size(); ++i) {
            for (const RuledSurfaceFace& face : m_surfaces[i].faces) {
                for (Eigen::Vector3d vertex : face.vertices) {
                    vertex.z() += m_stackZBySurface[i];
                    minimum = minimum.cwiseMin(vertex);
                    maximum = maximum.cwiseMax(vertex);
                }
            }
        }
        const std::array<Eigen::Vector3d, 8> corners{{
            {minimum.x(), minimum.y(), minimum.z()}, {maximum.x(), minimum.y(), minimum.z()},
            {maximum.x(), maximum.y(), minimum.z()}, {minimum.x(), maximum.y(), minimum.z()},
            {minimum.x(), minimum.y(), maximum.z()}, {maximum.x(), minimum.y(), maximum.z()},
            {maximum.x(), maximum.y(), maximum.z()}, {minimum.x(), maximum.y(), maximum.z()}}};
        constexpr std::array<std::array<int, 2>, 12> edges{{
            {{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 0}}, {{4, 5}}, {{5, 6}},
            {{6, 7}}, {{7, 4}}, {{0, 4}}, {{1, 5}}, {{2, 6}}, {{3, 7}}}};
        for (const auto& edge : edges) {
            renderer.drawLine(toVec3(corners[edge[0]]), toVec3(corners[edge[1]]),
                              Color(0.36f, 0.36f, 0.36f, 1.0f), 1.0f);
        }
    }

    static Vec3 toVec3(const Eigen::Vector3d& point, float zOffset = 0.0f) {
        return {static_cast<float>(point.x()), static_cast<float>(point.y()),
                static_cast<float>(point.z()) + zOffset};
    }

    void writeDiagnostics() const {
        std::cout << "Procedural ruled surface stack: count=" << m_generation.surfaceCount
                  << " faces=" << m_generation.faceCount << " randomness=" << m_generation.randomness
                  << " length randomness=" << m_generation.lengthRandomness << " seed=" << m_generation.seed
                  << " ruling turn=" << m_generation.rulingRotation
                  << " gaps=" << (m_hotWireCollision ? "exact extended-ruling sweep" : "sampled finite strips")
                  << " flips=" << (m_optimiseFlips ? "optimised" : "off")
                  << " order=" << (m_generation.surfaceCount <= 8 ? "brute force" : "multi-start greedy")
                  << " (clearance " << kClearance << ")\n";
        for (size_t i = 0; i < m_surfaces.size(); ++i) {
            double minDot = std::numeric_limits<double>::infinity();
            double minZ = std::numeric_limits<double>::infinity();
            double maxZ = -std::numeric_limits<double>::infinity();
            for (const RuledSurfaceFace& face : m_surfaces[i].faces) {
                minDot = std::min(minDot, face.plane.n.z());
                for (const Eigen::Vector3d& vertex : face.vertices) {
                    minZ = std::min(minZ, vertex.z());
                    maxZ = std::max(maxZ, vertex.z());
                }
            }
            std::cout << "Surface " << i << ": faces=" << m_surfaces[i].faces.size()
                      << " valid=" << (m_valid[i] ? "yes" : "no") << " min normal dot Z=" << minDot
                      << " z range=[" << minZ << ", " << maxZ << "]\n";
        }
        std::cout << "Gap matrix (row below, column above):\n" << m_gapMatrix << "\nBest order: ";
        for (size_t layer = 0; layer < m_solution.order.size(); ++layer) {
            if (layer) std::cout << " -> ";
            std::cout << m_solution.order[layer] << " (z=" << m_solution.stackZ[layer] << ')';
        }
        if (m_solution.flippedBySurface.size() == m_surfaces.size()) {
            std::cout << "\nFlipped surfaces:";
            bool anyFlipped = false;
            for (size_t i = 0; i < m_solution.flippedBySurface.size(); ++i) {
                if (m_solution.flippedBySurface[i]) {
                    std::cout << ' ' << i;
                    anyFlipped = true;
                }
            }
            if (!anyFlipped) std::cout << " none";
        }
        std::cout << "\nFinal stack height: " << m_solution.totalHeight << "\n";
    }

    std::vector<RuledSurface> m_surfaces;
    std::vector<std::shared_ptr<MeshObject>> m_meshes;
    std::vector<bool> m_valid;
    std::vector<double> m_stackZBySurface;
    std::unique_ptr<SimpleUI> m_ui;
    Eigen::MatrixXd m_gapMatrix;
    RuledSurfaceStackSolution m_solution;
    RuledSurfaceProceduralSettings m_generation;
    float m_surfaceCountSlider{6.0f};
    float m_facesPerSurfaceSlider{6.0f};
    float m_randomnessSlider{0.50f};
    float m_lengthRandomnessSlider{0.0f};
    float m_rulingRotationSlider{0.0f};
    bool m_optimiseFlips{true};
    bool m_lastOptimiseFlips{true};
    bool m_hotWireCollision{true};
    bool m_lastHotWireCollision{true};
    bool m_showFaceNormals{true};
    std::string m_summary;
};

ALICE2_REGISTER_SKETCH_AUTO(RuledSurfaceStackingSketch)

#endif
