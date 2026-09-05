#define __MAIN__
#ifdef __MAIN__

#include <alice2.h>
#include <computeGeom/ComputeMesh.h>
#include <geometry/Stereotomy.h>
#include <sketches/SketchRegistry.h>
#include <stack/RuledSurfaceStackSolver.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_map>

using namespace alice2;
using namespace alice2::stack;

namespace {

Vec3 alice(const Eigen::Vector3d& point) {
    return {static_cast<float>(point.x()), static_cast<float>(point.y()), static_cast<float>(point.z())};
}

std::filesystem::path dataPath(const std::string& file) {
    const std::filesystem::path requested(file);
    if (requested.is_absolute() || std::filesystem::exists(requested)) return requested;
    const std::filesystem::path local = std::filesystem::path("data") / file;
    if (std::filesystem::exists(local)) return local;
    return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / "data" / file;
}

Color blockColour(int index) {
    static const Color hotPink(1.0f, 0.08f, 0.58f, 1.0f);
    static const Color blue(0.10f, 0.38f, 1.0f, 1.0f);
    return index % 2 == 0 ? hotPink : blue;
}

Vec3 faceCentre(const std::vector<Vec3>& positions, const std::vector<int>& face) {
    Vec3 centre;
    for (const int vertex : face) centre += positions[vertex];
    return face.empty() ? centre : centre / static_cast<float>(face.size());
}

struct StackableBlock {
    int sourceBlock = -1;
    RuledSurface surface;
};

// Stacking compares the blocks in a common, material-local frame: X follows
// the face walk, Y follows a ruling, and Z is the block's averaged normal.
std::optional<StackableBlock> makeStackableBlock(const geometry::StereotomySolid& solid) {
    if (solid.faces.size() < 2 || solid.rulingEdges.size() < 2) return std::nullopt;
    Vec3 origin, normal;
    for (const Vec3& point : solid.topVertices) origin += point;
    if (solid.topVertices.empty()) return std::nullopt;
    origin /= static_cast<float>(solid.topVertices.size());
    for (const auto& face : solid.faces) {
        if (face.size() < 3) continue;
        normal += (solid.topVertices[face[1]] - solid.topVertices[face[0]])
                      .cross(solid.topVertices[face[2]] - solid.topVertices[face[0]]).normalized();
    }
    normal.normalize();
    Vec3 longitudinal = faceCentre(solid.topVertices, solid.faces.back()) -
                         faceCentre(solid.topVertices, solid.faces.front());
    longitudinal.normalize();
    if (normal.lengthSquared() <= 1e-8f || longitudinal.lengthSquared() <= 1e-8f) return std::nullopt;
    Vec3 ruling = normal.cross(longitudinal).normalized();
    if (ruling.lengthSquared() <= 1e-8f) return std::nullopt;

    const auto toLocal = [&](const Vec3& point) {
        const Vec3 delta = point - origin;
        return Eigen::Vector3d(delta.dot(longitudinal), delta.dot(ruling), delta.dot(normal));
    };
    std::vector<RuledSurfaceRuling> rulings;
    rulings.reserve(solid.rulingEdges.size());
    for (const auto& [a, b] : solid.rulingEdges) {
        if (a < 0 || b < 0 || a >= static_cast<int>(solid.topVertices.size()) || b >= static_cast<int>(solid.topVertices.size())) continue;
        rulings.push_back({toLocal(solid.topVertices[a]), toLocal(solid.topVertices[b])});
    }
    // The terminal boundary returned by the quad walk can have the opposite
    // endpoint order.  Align every ruling to its predecessor before forming
    // quads, otherwise only the final face becomes a twisted, flipped-winding
    // quad even though the source solid is correctly oriented.
    for (size_t i = 1; i < rulings.size(); ++i) {
        const double aligned = (rulings[i - 1].a - rulings[i].a).squaredNorm() +
                               (rulings[i - 1].b - rulings[i].b).squaredNorm();
        const double reversed = (rulings[i - 1].a - rulings[i].b).squaredNorm() +
                                (rulings[i - 1].b - rulings[i].a).squaredNorm();
        if (reversed < aligned) std::swap(rulings[i].a, rulings[i].b);
    }
    RuledSurface surface = makeRuledSurface(rulings);
    if (!isValidForStackDirection(surface)) {
        for (RuledSurfaceRuling& rulingLine : rulings) std::swap(rulingLine.a, rulingLine.b);
        surface = makeRuledSurface(rulings);
    }
    if (!isValidForStackDirection(surface)) return std::nullopt;
    return StackableBlock{solid.blockIndex, std::move(surface)};
}

geometry::StereotomySolid solidForSurface(const RuledSurface& surface, float thickness) {
    geometry::StereotomySolid solid;
    solid.topVertices.reserve(surface.rulings.size() * 2);
    solid.bottomVertices.reserve(surface.rulings.size() * 2);
    for (const RuledSurfaceRuling& ruling : surface.rulings) {
        solid.topVertices.push_back(alice(ruling.a));
        solid.topVertices.push_back(alice(ruling.b));
        solid.bottomVertices.push_back(alice(ruling.a - Eigen::Vector3d(0.0, 0.0, thickness)));
        solid.bottomVertices.push_back(alice(ruling.b - Eigen::Vector3d(0.0, 0.0, thickness)));
    }
    for (size_t i = 0; i + 1 < surface.rulings.size(); ++i) {
        solid.faces.push_back({static_cast<int>(2 * i), static_cast<int>(2 * i + 2),
                               static_cast<int>(2 * i + 3), static_cast<int>(2 * i + 1)});
    }
    return solid;
}

std::string fixed(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << value;
    return out.str();
}

} // namespace

class StereotomyStackingSketch final : public ISketch {
public:
    std::string getName() const override { return "Stereotomy Solid Stack"; }
    std::string getDescription() const override { return "Centre-graph stereotomy solids placed by the direct-Z ruled-surface solver"; }
    std::string getAuthor() const override { return "alice2 User"; }

    void setup() override {
        scene().setBackgroundColor(Color(1.0f, 1.0f, 1.0f));
        scene().setShowGrid(false);
        scene().setShowAxes(true); scene().setAxesLength(0.2f);
        m_ui = std::make_unique<SimpleUI>(input());
        m_ui->setTheme(SimpleUI::UITheme::Dark);
        m_ui->addSlider("Faces per solid", {14, 94}, 240.0f, 2.0f, 36.0f, m_facesPerSolid);
        m_ui->addSlider("Solid thickness", {14, 124}, 240.0f, 0.005f, 0.15f, m_thickness);
        m_ui->addToggle("Optimise physical flips", {14, 154, 220, 22}, m_optimiseFlips);
        m_ui->addToggle("Hot-wire collision", {14, 182, 190, 22}, m_hotWireCollision);
        loadSource();
    }

    void update(float) override {
        const int faces = std::clamp(static_cast<int>(std::lround(m_facesPerSolid)), 2, 36);
        if (faces != m_lastFaces || std::abs(m_thickness - m_lastThickness) > 1e-6f) rebuildStereotomy();
        if (m_optimiseFlips != m_lastOptimiseFlips || m_hotWireCollision != m_lastHotWireCollision) {
            invalidateStack();
            m_lastOptimiseFlips = m_optimiseFlips;
            m_lastHotWireCollision = m_hotWireCollision;
            m_status = "Stack settings changed. Press P to compute the stack.";
        }
    }

    void cleanup() override { clearSourceSolids(); clearStackSolids(); }

    void draw(Renderer& renderer, Camera&) override {
        renderer.setColor(Color(0.0f, 0.0f, 0.0f, 1.0f));
        renderer.drawString("Stereotomy Solid Stack — centre-graph blocks + direct-Z solver", 14, 24);
        renderer.drawString("p compute stack   s toggle stack   r reload source", 14, 46);
        renderer.drawString(m_status, 14, 68);
        if (m_ui) m_ui->draw(renderer);
        if (m_showStack && m_stackBoundsValid) drawStackBounds(renderer);
    }

    bool onKeyPress(unsigned char key, int, int) override {
        if (key == 'r' || key == 'R') { loadSource(); return true; }
        if (key == 'p' || key == 'P') { computeStack(); return true; }
        if (key == 's' || key == 'S') {
            m_showStack = !m_showStack;
            updateVisibility();
            m_status = m_showStack && !m_stackReady ?
                "Stack view selected; press P to compute it." :
                (m_showStack ? "Showing source solids and computed stack." : "Showing source solids; stack hidden.");
            return true;
        }
        return false;
    }
    bool onMousePress(int button, int state, int x, int y) override {
        return m_ui && m_ui->onMousePress(button, state, x, y);
    }
    bool onMouseMove(int x, int y) override { return m_ui && m_ui->onMouseMove(x, y); }

private:
    void loadSource() {
        clearSourceSolids();
        clearStackSolids();
        if (m_source) scene().removeObject(m_source);
        try {
            m_source = std::make_shared<ComputeMesh>("stereotomy_source");
            m_source->readFromObj(dataPath("stereotomy.obj").string());
            m_source->weld(1e-5f);
            m_source->updateHalfEdgeData();
            m_source->setVisible(false);
            scene().addObject(m_source);
            rebuildStereotomy();
        } catch (const std::exception& error) {
            m_status = std::string("Could not load stereotomy.obj: ") + error.what();
        }
    }

    void rebuildStereotomy() {
        clearSourceSolids();
        invalidateStack();
        if (!m_source) return;
        const int faces = std::clamp(static_cast<int>(std::lround(m_facesPerSolid)), 2, 36);
        std::string diagnostic;
        if (!m_stereotomy.rebuild(*m_source, faces, m_thickness, &diagnostic)) {
            m_status = "Stereotomy traversal failed: " + diagnostic;
            return;
        }
        m_stackable.clear();
        m_rejectedBlocks = 0;
        for (const auto& solid : m_stereotomy.solids()) {
            auto block = makeStackableBlock(solid);
            if (block) m_stackable.push_back(std::move(*block)); else ++m_rejectedBlocks;
            auto mesh = geometry::Stereotomy::makeSolidMesh("stereotomy_source_" + std::to_string(solid.blockIndex),
                                                             solid, blockColour(solid.blockIndex));
            if (!mesh) continue;
            scene().addObject(mesh);
            m_sourceSolids.push_back(std::move(mesh));
        }
        m_lastFaces = faces; m_lastThickness = m_thickness;
        if (m_stackable.empty()) {
            m_status = diagnostic + " No block has two valid +Z ruled faces.";
            return;
        }
        m_status = diagnostic + "  " + std::to_string(m_sourceSolids.size()) +
            " separated solids. Press P to compute the stack.";
        updateVisibility();
    }

    void computeStack() {
        clearStackSolids();
        m_stackReady = false;
        if (m_stackable.empty()) {
            m_status = "No stackable stereotomy blocks. Reload the source or change the face count.";
            return;
        }
        try {
            std::vector<RuledSurface> surfaces;
            surfaces.reserve(m_stackable.size());
            for (const auto& block : m_stackable) surfaces.push_back(block.surface);
            const RuledSurfaceBounds2D bounds = ruledSurfaceGroupBoundsXY(surfaces);
            const double scale = std::max({bounds.max.x() - bounds.min.x(), bounds.max.y() - bounds.min.y(), 1.0});
            m_geometrySettings.geomEpsilon = 1e-10 * scale;
            m_geometrySettings.mergeEpsilon = 1e-9 * scale;
            // The interval solver works on the top ruled skins.  Each rendered
            // solid extends exactly m_thickness below that skin, so reserving a
            // full thickness here makes every overlapping upper bottom clear
            // the lower top, with a small visible manufacturing gap.
            m_geometrySettings.clearance = static_cast<double>(m_thickness) + 1e-4;
            RuledSurfaceBounds2D foam = bounds;
            foam.min.array() -= 0.05 * scale; foam.max.array() += 0.05 * scale;
            m_variants = makeStackSurfaceVariants(surfaces, foam, m_geometrySettings.geomEpsilon);
            m_geometryStats = {};
            m_pairs = buildPairConstraintData(m_variants, m_geometrySettings, &m_geometryStats);
            m_solverSettings.numericalEpsilon = m_geometrySettings.geomEpsilon;
            m_solverSettings.optimiseFlips = m_optimiseFlips;
            m_solverSettings.useHotWireCollision = m_hotWireCollision;
            // The stereotomy source can yield many blocks. Keep P responsive:
            // retain the feasible greedy incumbent if the exact search is cut off.
            m_solverSettings.enableStrongBranching = false;
            m_solverSettings.maxSearchNodes = m_optimiseFlips ? 128 : 512;
            m_solverSettings.flipPairCandidateCount = 4;
            m_solverStats = {};
            m_solution = solveRuledSurfaceStackFast(m_variants, m_pairs, m_solverSettings, &m_solverStats);
            if (!m_solution.feasible) {
                m_status = "Solver found no feasible stack.";
                return;
            }
            std::vector<geometry::StereotomySolid> stackedSolids;
            stackedSolids.reserve(m_variants.size());
            Vec3 stackMin(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::infinity());
            Vec3 stackMax(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity());
            const auto includeInStackBounds = [&](const Vec3& point) {
                stackMin.x = std::min(stackMin.x, point.x); stackMin.y = std::min(stackMin.y, point.y); stackMin.z = std::min(stackMin.z, point.z);
                stackMax.x = std::max(stackMax.x, point.x); stackMax.y = std::max(stackMax.y, point.y); stackMax.z = std::max(stackMax.z, point.z);
            };
            for (size_t i = 0; i < m_variants.size(); ++i) {
                const bool flipped = i < m_solution.flippedBySurface.size() && m_solution.flippedBySurface[i] &&
                                     m_variants[i].flipped.has_value();
                const RuledSurface& surface = flipped ? m_variants[i].flipped->geometry : m_variants[i].normal.geometry;
                geometry::StereotomySolid solid = solidForSurface(surface, m_thickness);
                for (Vec3& point : solid.topVertices) point.z += static_cast<float>(m_solution.zBySurface[i]);
                for (Vec3& point : solid.bottomVertices) point.z += static_cast<float>(m_solution.zBySurface[i]);
                for (const Vec3& point : solid.topVertices) includeInStackBounds(point);
                for (const Vec3& point : solid.bottomVertices) includeInStackBounds(point);
                stackedSolids.push_back(std::move(solid));
            }
            Vec3 sourceMin, sourceMax;
            const auto sourceData = m_source ? m_source->getMeshData() : nullptr;
            if (sourceData && !sourceData->vertices.empty()) sourceData->updateBounds(sourceMin, sourceMax);
            const float sourceWidth = std::max(1.0f, sourceMax.x - sourceMin.x);
            const Vec3 stackOffset(sourceMax.x + 0.12f * sourceWidth - stackMin.x,
                                   0.5f * (sourceMin.y + sourceMax.y - stackMin.y - stackMax.y),
                                   0.5f * (sourceMin.z + sourceMax.z - stackMin.z - stackMax.z));
            m_stackBoundsMin = stackMin + stackOffset;
            m_stackBoundsMax = stackMax + stackOffset;
            m_stackBoundsValid = true;
            for (size_t i = 0; i < stackedSolids.size(); ++i) {
                geometry::StereotomySolid& solid = stackedSolids[i];
                for (Vec3& point : solid.topVertices) point += stackOffset;
                for (Vec3& point : solid.bottomVertices) point += stackOffset;
                auto mesh = geometry::Stereotomy::makeSolidMesh("stereotomy_stack_" + std::to_string(i), solid,
                                                                 blockColour(m_stackable[i].sourceBlock));
                if (!mesh) continue;
                scene().addObject(mesh);
                m_stackSolids.push_back(std::move(mesh));
            }
            m_stackReady = true;
            updateVisibility();
            m_status = "Computed " + std::to_string(m_stackSolids.size()) + " stacked solids" +
                (m_rejectedBlocks ? " (" + std::to_string(m_rejectedBlocks) + " rejected)" : "") +
                "; height " + fixed(m_solution.totalHeight) + "; nodes " + std::to_string(m_solverStats.nodesVisited) +
                (m_solverStats.searchLimitReached ? " (bounded search)" : " (exact search)");
        } catch (const std::exception& error) {
            m_status = std::string("Stack solver error: ") + error.what();
        }
    }

    void clearSourceSolids() {
        for (const auto& solid : m_sourceSolids) if (solid) scene().removeObject(solid);
        m_sourceSolids.clear();
    }

    void clearStackSolids() {
        for (const auto& solid : m_stackSolids) if (solid) scene().removeObject(solid);
        m_stackSolids.clear();
        m_stackBoundsValid = false;
    }

    void invalidateStack() {
        clearStackSolids();
        m_stackReady = false;
        updateVisibility();
    }

    void updateVisibility() {
        const bool showStack = m_showStack && m_stackReady;
        for (const auto& solid : m_sourceSolids) if (solid) solid->setVisible(true);
        for (const auto& solid : m_stackSolids) if (solid) solid->setVisible(showStack);
    }

    void drawStackBounds(Renderer& renderer) const {
        constexpr std::array<std::array<int, 2>, 12> edges = {{{0, 1}, {1, 2}, {2, 3}, {3, 0},
                                                                  {4, 5}, {5, 6}, {6, 7}, {7, 4},
                                                                  {0, 4}, {1, 5}, {2, 6}, {3, 7}}};
        const Vec3& lo = m_stackBoundsMin;
        const Vec3& hi = m_stackBoundsMax;
        const std::array<Vec3, 8> corners = {
            Vec3(lo.x, lo.y, lo.z), Vec3(hi.x, lo.y, lo.z), Vec3(hi.x, hi.y, lo.z), Vec3(lo.x, hi.y, lo.z),
            Vec3(lo.x, lo.y, hi.z), Vec3(hi.x, lo.y, hi.z), Vec3(hi.x, hi.y, hi.z), Vec3(lo.x, hi.y, hi.z)};
        for (const auto& edge : edges) renderer.drawLine(corners[edge[0]], corners[edge[1]], Color(0.0f, 0.0f, 0.0f, 1.0f), 1.5f);
    }

    std::shared_ptr<ComputeMesh> m_source;
    std::unique_ptr<SimpleUI> m_ui;
    geometry::Stereotomy m_stereotomy;
    std::vector<std::shared_ptr<MeshObject>> m_sourceSolids;
    std::vector<std::shared_ptr<MeshObject>> m_stackSolids;
    std::vector<StackableBlock> m_stackable;
    std::vector<SurfaceOrientationVariants> m_variants;
    std::vector<PairConstraintData> m_pairs;
    StackGeometrySettings m_geometrySettings;
    StackSolveSettings m_solverSettings;
    StackGeometryStats m_geometryStats;
    StackSolveStats m_solverStats;
    RuledSurfaceStackSolution m_solution;
    float m_facesPerSolid = 12.0f;
    float m_thickness = 0.03f;
    int m_lastFaces = -1;
    float m_lastThickness = -1.0f;
    bool m_optimiseFlips = false, m_hotWireCollision = true;
    bool m_lastOptimiseFlips = false, m_lastHotWireCollision = true;
    int m_rejectedBlocks = 0;
    bool m_showStack = false;
    bool m_stackReady = false;
    Vec3 m_stackBoundsMin;
    Vec3 m_stackBoundsMax;
    bool m_stackBoundsValid = false;
    std::string m_status{"Loading stereotomy.obj..."};
};

ALICE2_REGISTER_SKETCH_AUTO(StereotomyStackingSketch)

#endif // __MAIN__
