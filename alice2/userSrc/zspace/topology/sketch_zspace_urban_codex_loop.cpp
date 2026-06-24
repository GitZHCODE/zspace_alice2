#define __MAIN__
#ifdef __MAIN__

#include <zspace/interface.h>
#include <zspace/io.h>

#include <alice2.h>
#include <sketches/SketchRegistry.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace alice2;

class zSpaceUrbanCodexLoopSketch : public ISketch {
public:
    std::string getName() const override { return "zSpace Urban Codex Loop"; }
    std::string getDescription() const override { return "Clean minimal urban massing sketch for Codex/VLM critique iterations."; }
    std::string getAuthor() const override { return "Codex + alice2 + zspace_core"; }

    void setup() override
    {
        scene().setBackgroundColor(Color(1.0f, 1.0f, 1.0f, 1.0f));
        scene().setShowGrid(false);
        scene().setShowAxes(false);

        m_ui = std::make_unique<SimpleUI>(input());
        m_ui->setTheme(SimpleUI::UITheme::Dark);
        m_ui->addSlider("p", Vec2{14.0f, 82.0f}, 240.0f, 0.05f, 0.50f, m_p);
        m_ui->addToggle("Field Mesh", UIRect{14.0f, 112.0f, 130.0f, 24.0f}, m_drawStreetFieldMesh);

        loadMesh();
        if (!m_loaded) return;

        std::cout << "[URBAN CODEX LOOP] Clean neutral baseline loaded." << std::endl;
        std::cout << "[URBAN CODEX LOOP] Faces: " << zSpace::zFnMesh(m_mesh).numPolygons() << std::endl;
    }

    void update(float) override
    {
        if (!m_loaded || m_screenshotTaken) return;

        if (std::abs(m_p - m_lastBuiltP) > 0.001f) {
            zSpace::zFnMesh fn(m_mesh);
            buildStreetEdges(fn);
            buildPlotRecords(fn);
            buildStreetSdfField();
            buildTypeACenterlineGraphs();
        }

        m_frameCount++;
        if (m_frameCount == 10) {
            setPlanCamera();
        }
        else if (m_autoCapture && m_frameCount > 30) {
            Application::getInstance()->takeScreenshot();
            m_screenshotTaken = true;
            std::cout << "[URBAN CODEX LOOP] Screenshot captured. Exiting." << std::endl;
            exit(0);
        }
    }

    void draw(Renderer& renderer, Camera&) override
    {
        if (!m_loaded) return;

        zSpace::zFnMesh fn(m_mesh);
        drawNeutralBaseMesh(renderer, fn);
        drawStreetSdfGeometry(renderer);
        drawOpenSpaceSdf(renderer, fn);
        drawTypeACenterlineGraphs(renderer);

        if (m_ui) {
            m_ui->draw(renderer);
        }
    }

    bool onKeyPress(unsigned char key, int, int) override
    {
        if (key == 's' || key == 'S') {
            Application::getInstance()->takeScreenshot();
            m_screenshotTaken = true;
            std::cout << "[URBAN CODEX LOOP] Manual screenshot captured. Exiting." << std::endl;
            exit(0);
        }
        return false;
    }

    bool onMousePress(int button, int state, int x, int y) override
    {
        if (m_ui && m_ui->onMousePress(button, state, x, y)) return true;
        return false;
    }

    bool onMouseMove(int x, int y) override
    {
        if (m_ui && m_ui->onMouseMove(x, y)) return true;
        return false;
    }

private:
    zSpace::zObjectMesh m_mesh;
    std::string m_meshPath = "data/input_grid_01.obj";
    std::unique_ptr<SimpleUI> m_ui;

    bool m_loaded = false;
    bool m_screenshotTaken = false;
    bool m_autoCapture = false;
    bool m_drawStreetFieldMesh = false;
    int m_frameCount = 0;

    zSpace::zPoint m_boundsMin;
    zSpace::zPoint m_boundsMax;
    zSpace::zPoint m_meshCenter;
    float m_maxDistance = 1.0f;

    // First-pass parameters. No gradient, greenery, open-space hierarchy, or SDF field yet.
    // Codex will add those methods later only after VLM critique calls for them.
    float m_massingZ = 0.004f;
    float m_baseZ = -0.004f;
    float m_massingCoverageStep = 2.0f;
    float m_parcelCoverage = 0.54f;
    float m_minBuildingLength = 0.11f;
    float m_minBuildingDepth = 0.060f;
    float m_maxBuildingAspect = 2.6f;
    float m_edgeClearanceFactor = 0.82f;
    float m_typeAMinWidthMeters = 1.2f;
    float m_typeAMaxWidthMeters = 2.0f;
    float m_typeARoadSetbackMeters = 0.5f;
    float m_typeALocalSetbackMeters = 0.2f;
    float m_openSpaceZ = 0.001f;
    float m_p = 0.30f;
    float m_lastBuiltP = -1.0f;
    float m_siteLongDimensionMeters = 500.0f;
    float m_modelUnitsPerMeter = 1.0f;
    float m_civicSpineWidth = 0.055f;
    float m_civicPlazaRadius = 0.135f;
    float m_neighborhoodPlazaRadius = 0.105f;
    int m_streetFieldResolution = 320;
    Vec3 m_civicSpineA;
    Vec3 m_civicSpineB;
    Vec3 m_neighborhoodPlazaA;
    Vec3 m_neighborhoodPlazaB;

    enum class StreetClass {
        Primary,
        Secondary,
        Tertiary
    };

    enum class PlotBoundaryType {
        PrimaryRoad,
        SecondaryRoad,
        TertiaryRoad,
        PlotSplitLine
    };

    struct StreetEdge {
        Vec3 a;
        Vec3 b;
        StreetClass streetClass;
        float offsetWidth;
        Color color;
    };

    struct PlotBoundaryEdge {
        Vec3 a;
        Vec3 b;
        PlotBoundaryType boundaryType;
        int streetEdgeIndex;
    };

    class plot {
    public:
        struct CenterlineGraphVertex {
            Vec3 position;
            int sourceVertexIndex;
        };

        struct CenterlineGraphEdge {
            int startVertexIndex;
            int endVertexIndex;
            PlotBoundaryType boundaryType;
            float offsetDistance;
        };

        int id;
        int faceIndex;
        Vec3 center;
        std::vector<Vec3> vertices;
        std::vector<PlotBoundaryEdge> boundaryEdges;
        std::vector<CenterlineGraphVertex> centerlineGraphVertices;
        std::vector<CenterlineGraphEdge> centerlineGraphEdges;

        void buildCenterlineGraph(
            float roadSetback,
            float localSetback,
            float buildingWidth,
            float primaryRoadHalfWidth,
            float secondaryRoadHalfWidth,
            float tertiaryRoadHalfWidth
        )
        {
            centerlineGraphVertices.clear();
            centerlineGraphEdges.clear();
            if (vertices.size() < 3 || boundaryEdges.size() != vertices.size()) return;

            struct OffsetLine {
                Vec3 a;
                Vec3 b;
                float offsetDistance;
            };

            std::vector<OffsetLine> offsetLines;
            offsetLines.reserve(boundaryEdges.size());

            for (const auto& edge : boundaryEdges) {
                Vec3 edgeVector = edge.b - edge.a;
                Vec3 tangent = normalized2d(edgeVector);
                Vec3 normalA(-tangent.y, tangent.x, 0.0f);
                Vec3 midpoint = (edge.a + edge.b) * 0.5f;
                Vec3 inwardNormal = dot2d(center - midpoint, normalA) >= 0.0f ? normalA : normalA * -1.0f;

                float setback = isPrimaryOrSecondary(edge.boundaryType) ? roadSetback : localSetback;
                float roadHalfWidth = roadHalfWidthForBoundary(
                    edge.boundaryType,
                    primaryRoadHalfWidth,
                    secondaryRoadHalfWidth,
                    tertiaryRoadHalfWidth
                );
                float offsetDistance = setback + buildingWidth * 0.5f + roadHalfWidth;
                Vec3 offset = inwardNormal * offsetDistance;
                offsetLines.push_back({ edge.a + offset, edge.b + offset, offsetDistance });
            }

            centerlineGraphVertices.reserve(vertices.size());
            for (size_t i = 0; i < vertices.size(); ++i) {
                size_t previous = (i + offsetLines.size() - 1) % offsetLines.size();
                Vec3 position;
                if (!intersectLines2d(offsetLines[previous].a, offsetLines[previous].b, offsetLines[i].a, offsetLines[i].b, position)) {
                    position = (offsetLines[previous].b + offsetLines[i].a) * 0.5f;
                }
                centerlineGraphVertices.push_back({ position, static_cast<int>(i) });
            }

            centerlineGraphEdges.reserve(boundaryEdges.size());
            for (size_t i = 0; i < boundaryEdges.size(); ++i) {
                centerlineGraphEdges.push_back({
                    static_cast<int>(i),
                    static_cast<int>((i + 1) % boundaryEdges.size()),
                    boundaryEdges[i].boundaryType,
                    offsetLines[i].offsetDistance
                });
            }
        }

    private:
        static float dot2d(const Vec3& a, const Vec3& b)
        {
            return a.x * b.x + a.y * b.y;
        }

        static float cross2d(const Vec3& a, const Vec3& b)
        {
            return a.x * b.y - a.y * b.x;
        }

        static Vec3 normalized2d(const Vec3& v)
        {
            float len = std::sqrt(v.x * v.x + v.y * v.y);
            if (len < 1e-6f) return Vec3(1.0f, 0.0f, 0.0f);
            return Vec3(v.x / len, v.y / len, 0.0f);
        }

        static bool isPrimaryOrSecondary(PlotBoundaryType boundaryType)
        {
            return boundaryType == PlotBoundaryType::PrimaryRoad || boundaryType == PlotBoundaryType::SecondaryRoad;
        }

        static float roadHalfWidthForBoundary(
            PlotBoundaryType boundaryType,
            float primaryRoadHalfWidth,
            float secondaryRoadHalfWidth,
            float tertiaryRoadHalfWidth
        )
        {
            switch (boundaryType) {
                case PlotBoundaryType::PrimaryRoad: return primaryRoadHalfWidth;
                case PlotBoundaryType::SecondaryRoad: return secondaryRoadHalfWidth;
                case PlotBoundaryType::TertiaryRoad: return tertiaryRoadHalfWidth;
                case PlotBoundaryType::PlotSplitLine: return 0.0f;
            }
            return 0.0f;
        }

        static bool intersectLines2d(const Vec3& a0, const Vec3& a1, const Vec3& b0, const Vec3& b1, Vec3& result)
        {
            Vec3 da = a1 - a0;
            Vec3 db = b1 - b0;
            float denominator = cross2d(da, db);
            if (std::abs(denominator) < 1e-6f) return false;

            float t = cross2d(b0 - a0, db) / denominator;
            result = a0 + da * t;
            result.z = 0.0f;
            return true;
        }
    };

    std::vector<StreetEdge> m_streetEdges;
    std::vector<plot> m_plots;
    zSpace::zObjectMeshScalarField m_streetSdfField;
    zSpace::zObjectGraph m_streetIsoContour;

    static Vec3 toVec3(const zSpace::zVector& p)
    {
        return Vec3(static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z));
    }

    Vec3 withZ(const Vec3& p, float z) const
    {
        return Vec3(p.x, p.y, z);
    }

    float dot2d(const Vec3& a, const Vec3& b) const
    {
        return a.x * b.x + a.y * b.y;
    }

    Vec3 normalized2d(const Vec3& v) const
    {
        float len = std::sqrt(v.x * v.x + v.y * v.y);
        if (len < 1e-6f) return Vec3(1.0f, 0.0f, 0.0f);
        return Vec3(v.x / len, v.y / len, 0.0f);
    }

    float saturate(float value) const
    {
        return std::clamp(value, 0.0f, 1.0f);
    }

    float smoothstep(float edge0, float edge1, float x) const
    {
        float t = saturate((x - edge0) / (edge1 - edge0));
        return t * t * (3.0f - 2.0f * t);
    }

    float metersToModelUnits(float meters) const
    {
        return meters * m_modelUnitsPerMeter;
    }

    float distanceToSegment2d(const Vec3& p, const Vec3& a, const Vec3& b) const
    {
        Vec3 ab = b - a;
        float lenSq = dot2d(ab, ab);
        if (lenSq < 1e-8f) return (p - a).length();

        float t = dot2d(p - a, ab) / lenSq;
        t = std::clamp(t, 0.0f, 1.0f);
        Vec3 closest = a + ab * t;
        return (p - closest).length();
    }

    float civicOpenSpaceSdf(const Vec3& p) const
    {
        float spine = distanceToSegment2d(p, m_civicSpineA, m_civicSpineB) - m_civicSpineWidth;
        float plaza = (p - toVec3(m_meshCenter)).length() - m_civicPlazaRadius;
        float plazaB = (p - m_neighborhoodPlazaA).length() - m_neighborhoodPlazaRadius;
        float plazaC = (p - m_neighborhoodPlazaB).length() - m_neighborhoodPlazaRadius;
        return std::min(std::min(spine, plaza), std::min(plazaB, plazaC));
    }

    bool isCivicOpenSpace(const Vec3& p) const
    {
        return civicOpenSpaceSdf(p) < 0.0f;
    }

    float streetOffsetSdf(const Vec3& p) const
    {
        float d = 1e9f;
        for (const auto& edge : m_streetEdges) {
            d = std::min(d, distanceToSegment2d(p, edge.a, edge.b) - edge.offsetWidth);
        }
        return d;
    }

    const StreetEdge* nearestStreetEdge(const Vec3& p) const
    {
        const StreetEdge* nearest = nullptr;
        float bestDistance = 1e9f;
        for (const auto& edge : m_streetEdges) {
            float distance = distanceToSegment2d(p, edge.a, edge.b) - edge.offsetWidth;
            if (distance < bestDistance) {
                bestDistance = distance;
                nearest = &edge;
            }
        }
        return nearest;
    }

    bool isStreetSpace(const Vec3& p) const
    {
        return streetOffsetSdf(p) < 0.0f;
    }

    Color lerpColor(const Color& a, const Color& b, float t) const
    {
        t = saturate(t);
        return Color(
            a.r + (b.r - a.r) * t,
            a.g + (b.g - a.g) * t,
            a.b + (b.b - a.b) * t,
            a.a + (b.a - a.a) * t
        );
    }

    Color densityBaseColor(const Vec3& p) const
    {
        float density = densityValue(p);
        const Color low(0.90f, 0.90f, 0.84f, 1.0f);
        const Color mid(0.95f, 0.84f, 0.56f, 1.0f);
        const Color high(0.82f, 0.40f, 0.30f, 1.0f);

        if (density < 0.5f) return lerpColor(low, mid, density * 2.0f);
        return lerpColor(mid, high, (density - 0.5f) * 2.0f);
    }

    float densityValue(const Vec3& p) const
    {
        float centerDistance = (p - toVec3(m_meshCenter)).length() / m_maxDistance;
        float core = 1.0f - smoothstep(0.12f, 0.58f, centerDistance);
        float primaryProximity = 0.0f;
        for (const auto& edge : m_streetEdges) {
            if (edge.streetClass != StreetClass::Primary) continue;
            primaryProximity = std::max(primaryProximity, 1.0f - smoothstep(0.04f, 0.34f, distanceToSegment2d(p, edge.a, edge.b)));
        }
        float spineProximity = std::max(primaryProximity, 1.0f - smoothstep(0.04f, 0.34f, distanceToSegment2d(p, m_civicSpineA, m_civicSpineB)));
        return saturate(core * 0.72f + spineProximity * 0.28f);
    }

    void loadMesh()
    {
        auto result = zSpace::zIO::readMesh(m_meshPath, m_mesh);
        if (!result) {
            m_loaded = false;
            std::cout << "[ERROR] Could not read mesh: " << result.message() << std::endl;
            return;
        }

        zSpace::zFnMesh fn(m_mesh);
        if (fn.numVertices() <= 0 || fn.numEdges() <= 0 || fn.numPolygons() <= 0) {
            m_loaded = false;
            std::cout << "[ERROR] Loaded mesh has empty topology." << std::endl;
            return;
        }

        fn.getBounds(m_boundsMin, m_boundsMax);
        m_meshCenter = (m_boundsMin + m_boundsMax) * 0.5;
        Vec3 bMin = toVec3(m_boundsMin);
        Vec3 bMax = toVec3(m_boundsMax);
        Vec3 span = bMax - bMin;
        float siteLongDimensionModelUnits = std::max(span.x, span.y);
        if (siteLongDimensionModelUnits > 1e-6f) {
            m_modelUnitsPerMeter = siteLongDimensionModelUnits / m_siteLongDimensionMeters;
        }
        m_civicSpineA = Vec3(bMin.x + span.x * 0.18f, bMin.y + span.y * 0.45f, 0.0f);
        m_civicSpineB = Vec3(bMax.x - span.x * 0.18f, bMin.y + span.y * 0.58f, 0.0f);
        m_neighborhoodPlazaA = Vec3(bMin.x + span.x * 0.32f, bMin.y + span.y * 0.47f, 0.0f);
        m_neighborhoodPlazaB = Vec3(bMin.x + span.x * 0.74f, bMin.y + span.y * 0.61f, 0.0f);

        zSpace::zPointArray vertices;
        fn.getVertexPositions(vertices);
        m_maxDistance = 0.0f;
        for (const auto& v : vertices) {
            m_maxDistance = std::max(m_maxDistance, (toVec3(v) - toVec3(m_meshCenter)).length());
        }
        if (m_maxDistance < 1e-5f) m_maxDistance = 1.0f;
        buildStreetEdges(fn);
        buildPlotRecords(fn);
        buildStreetSdfField();
        buildTypeACenterlineGraphs();

        m_loaded = true;
    }

    void setPlanCamera()
    {
        Vec3 bMin = toVec3(m_boundsMin);
        Vec3 bMax = toVec3(m_boundsMax);
        Vec3 center = (bMin + bMax) * 0.5f;
        Vec3 size = bMax - bMin;
        float maxSize = std::max(size.x, size.y);
        float aspect = camera().getAspectRatio();
        float halfSize = maxSize * 0.58f;

        camera().setOrthographic(-halfSize * aspect, halfSize * aspect, -halfSize, halfSize, 0.1f, 1000.0f);

        CameraState planState;
        planState.mode = CameraMode::Orbit;
        planState.orbitCenter = center;
        planState.orbitDistance = maxSize * 2.0f;
        planState.position = Vec3(center.x, center.y, center.z + planState.orbitDistance);
        planState.rotation = alice2::Quaternion::fromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -90.0f * ALICE2_DEG_TO_RAD);
        Application::getInstance()->getCameraController().setCameraState(planState);
    }

    void buildStreetEdges(zSpace::zFnMesh& fn)
    {
        m_streetEdges.clear();

        std::vector<std::pair<Vec3, Vec3>> uniqueEdges;
        for (int i = 0; i < fn.numPolygons(); ++i) {
            zSpace::zItMeshFace face(m_mesh, i);
            if (!face.isActive()) continue;

            std::vector<zSpace::zVector> positions;
            face.getVertexPositions(positions);
            if (positions.size() < 2) continue;

            for (size_t j = 0; j < positions.size(); ++j) {
                Vec3 a = toVec3(positions[j]);
                Vec3 b = toVec3(positions[(j + 1) % positions.size()]);
                if (!edgeAlreadyAdded(uniqueEdges, a, b)) {
                    uniqueEdges.emplace_back(a, b);
                }
            }
        }

        if (uniqueEdges.empty()) return;

        float longest = 0.0f;
        for (const auto& edge : uniqueEdges) {
            longest = std::max(longest, (edge.second - edge.first).length());
        }

        std::vector<std::pair<Vec3, Vec3>> primaryEdges;
        for (const auto& edge : uniqueEdges) {
            float length = (edge.second - edge.first).length();
            if (isPrimaryStreetEdge(edge.first, edge.second, length, longest)) {
                primaryEdges.push_back(edge);
            }
        }

        std::vector<std::pair<Vec3, Vec3>> secondaryEdges;
        for (const auto& edge : uniqueEdges) {
            float length = (edge.second - edge.first).length();
            if (!isEdgeInList(primaryEdges, edge.first, edge.second) &&
                isSecondaryStreetEdge(edge.first, edge.second, length, longest, primaryEdges)) {
                secondaryEdges.push_back(edge);
            }
        }

        for (const auto& edge : uniqueEdges) {
            Vec3 a = edge.first;
            Vec3 b = edge.second;
            float length = (b - a).length();
            StreetClass streetClass = StreetClass::Tertiary;
            if (!tryClassifyStreetEdge(a, b, length, longest, primaryEdges, secondaryEdges, streetClass)) {
                continue;
            }

            m_streetEdges.push_back({
                a,
                b,
                streetClass,
                streetOffsetWidth(streetClass),
                streetColor(streetClass)
            });
        }

        m_lastBuiltP = m_p;
        std::cout << "[URBAN CODEX LOOP] Street edges: " << m_streetEdges.size() << " | p=" << m_p << std::endl;
    }

    void buildPlotRecords(zSpace::zFnMesh& fn)
    {
        m_plots.clear();

        for (int i = 0; i < fn.numPolygons(); ++i) {
            zSpace::zItMeshFace face(m_mesh, i);
            if (!face.isActive()) continue;

            std::vector<zSpace::zVector> positions;
            face.getVertexPositions(positions);
            if (positions.size() < 3) continue;

            plot plotData;
            plotData.id = static_cast<int>(m_plots.size());
            plotData.faceIndex = i;
            plotData.center = toVec3(face.getCenter());
            plotData.vertices.reserve(positions.size());
            plotData.boundaryEdges.reserve(positions.size());

            for (const auto& position : positions) {
                plotData.vertices.push_back(toVec3(position));
            }

            for (size_t j = 0; j < positions.size(); ++j) {
                Vec3 a = toVec3(positions[j]);
                Vec3 b = toVec3(positions[(j + 1) % positions.size()]);
                int streetIndex = findStreetEdgeIndex(a, b);
                plotData.boundaryEdges.push_back({
                    a,
                    b,
                    boundaryTypeForStreetEdge(streetIndex),
                    streetIndex
                });
            }

            m_plots.push_back(plotData);
        }

        std::cout << "[URBAN CODEX LOOP] Plot records: " << m_plots.size() << std::endl;
        logPlotBoundarySummary();
    }

    void logPlotBoundarySummary() const
    {
        int primary = 0;
        int secondary = 0;
        int tertiary = 0;
        int split = 0;

        for (const auto& plot : m_plots) {
            for (const auto& edge : plot.boundaryEdges) {
                switch (edge.boundaryType) {
                    case PlotBoundaryType::PrimaryRoad: primary++; break;
                    case PlotBoundaryType::SecondaryRoad: secondary++; break;
                    case PlotBoundaryType::TertiaryRoad: tertiary++; break;
                    case PlotBoundaryType::PlotSplitLine: split++; break;
                }
            }
        }

        std::cout << "[URBAN CODEX LOOP] Plot boundary edges | primary: " << primary
                  << " secondary: " << secondary
                  << " tertiary: " << tertiary
                  << " split: " << split << std::endl;
    }

    void buildTypeACenterlineGraphs()
    {
        const float width = metersToModelUnits(m_typeAMaxWidthMeters);
        const float minWidth = metersToModelUnits(m_typeAMinWidthMeters);
        if (width < minWidth || width <= 1e-6f) return;

        int graphEdges = 0;
        for (auto& plotData : m_plots) {
            plotData.buildCenterlineGraph(
                metersToModelUnits(m_typeARoadSetbackMeters),
                metersToModelUnits(m_typeALocalSetbackMeters),
                width,
                primaryStreetWidth() * 0.5f,
                secondaryStreetWidth() * 0.5f,
                tertiaryStreetWidth() * 0.5f
            );
            graphEdges += static_cast<int>(plotData.centerlineGraphEdges.size());
        }

        std::cout << "[URBAN CODEX LOOP] Type A plot centerline graph edges: " << graphEdges
                  << " | width " << m_typeAMaxWidthMeters << "m"
                  << " | road setback " << m_typeARoadSetbackMeters << "m"
                  << " | local setback " << m_typeALocalSetbackMeters << "m" << std::endl;
    }

    void buildStreetSdfField()
    {
        Vec3 bMin = toVec3(m_boundsMin);
        Vec3 bMax = toVec3(m_boundsMax);
        Vec3 span = bMax - bMin;
        float pad = std::max(span.x, span.y) * 0.05f;
        zSpace::zPoint fieldMin(bMin.x - pad, bMin.y - pad, 0.0f);
        zSpace::zPoint fieldMax(bMax.x + pad, bMax.y + pad, 0.0f);

        zSpace::zFnMeshScalarField fn(m_streetSdfField);
        fn.create(fieldMin, fieldMax, m_streetFieldResolution, m_streetFieldResolution, 1, true, false);

        zSpace::zPointArray positions;
        fn.getPositions(positions);

        zSpace::zScalarArray values;
        values.reserve(positions.size());
        for (const auto& p : positions) {
            values.push_back(streetOffsetSdf(toVec3(p)));
        }

        fn.setFieldValues(values, zSpace::zFieldColorType::zFieldSDF, primaryStreetWidth());
        fn.updateColors(zSpace::zFieldColorType::zFieldSDF, primaryStreetWidth());
        fn.getIsocontour(m_streetIsoContour, 0.0f);
        liftStreetIsoGeometry();
    }

    void liftStreetIsoGeometry()
    {
        zSpace::zFnGraph contourFn(m_streetIsoContour);
        zSpace::zPointArray contourPositions;
        contourFn.getVertexPositions(contourPositions);
        for (auto& p : contourPositions) {
            p.z = m_openSpaceZ + 0.004f;
        }
        if (!contourPositions.empty()) {
            contourFn.setVertexPositions(contourPositions);
        }

    }

    bool isPrimaryStreetEdge(const Vec3& a, const Vec3& b, float length, float longest) const
    {
        Vec3 bMin = toVec3(m_boundsMin);
        Vec3 bMax = toVec3(m_boundsMax);
        Vec3 span = bMax - bMin;
        Vec3 mid = (a + b) * 0.5f;
        Vec3 dir = normalized2d(b - a);

        float maxSpan = std::max(span.x, span.y);
        float primaryBand = maxSpan * 0.051f;
        float verticalScore = std::abs(dir.y);
        float normalizedLength = (longest > 1e-6f) ? length / longest : 0.0f;

        bool nearLeftBoundary = std::abs(mid.x - bMin.x) < primaryBand;
        bool nearRightBoundary = std::abs(mid.x - bMax.x) < primaryBand;
        bool majorVertical = verticalScore > 0.35f && normalizedLength > 0.18f;
        return (nearLeftBoundary || nearRightBoundary) && majorVertical;
    }

    bool isSecondaryStreetEdge(
        const Vec3& a,
        const Vec3& b,
        float length,
        float longest,
        const std::vector<std::pair<Vec3, Vec3>>& primaryEdges
    ) const
    {
        (void)primaryEdges;
        Vec3 bMin = toVec3(m_boundsMin);
        Vec3 bMax = toVec3(m_boundsMax);
        Vec3 span = bMax - bMin;
        Vec3 dir = normalized2d(b - a);
        Vec3 mid = (a + b) * 0.5f;
        float horizontalScore = std::abs(dir.x);
        float verticalScore = std::abs(dir.y);
        float normalizedLength = (longest > 1e-6f) ? length / longest : 0.0f;
        float maxSpan = std::max(span.x, span.y);
        float boundaryBand = maxSpan * 0.055f;
        float nx = span.x > 1e-6f ? (mid.x - bMin.x) / span.x : 0.0f;

        bool nearTopBoundary = std::abs(mid.y - bMax.y) < boundaryBand;
        bool nearBottomBoundary = std::abs(mid.y - bMin.y) < boundaryBand;
        bool perimeterRoute = (nearTopBoundary || nearBottomBoundary) && horizontalScore > 0.38f && normalizedLength > 0.14f;
        bool rightSideCollector = nx > 0.48f && verticalScore > 0.42f && normalizedLength > 0.16f;
        bool longBlueRoute = horizontalScore > 0.62f && normalizedLength > 0.32f;

        return perimeterRoute || rightSideCollector || longBlueRoute;
    }

    bool isTertiaryStreetEdge(
        const Vec3& a,
        const Vec3& b,
        float length,
        float longest,
        const std::vector<std::pair<Vec3, Vec3>>& secondaryEdges
    ) const
    {
        (void)secondaryEdges;
        Vec3 bMin = toVec3(m_boundsMin);
        Vec3 bMax = toVec3(m_boundsMax);
        Vec3 span = bMax - bMin;
        Vec3 dir = normalized2d(b - a);
        Vec3 mid = (a + b) * 0.5f;
        float horizontalScore = std::abs(dir.x);
        float verticalScore = std::abs(dir.y);
        float normalizedLength = (longest > 1e-6f) ? length / longest : 0.0f;

        float nx = span.x > 1e-6f ? (mid.x - bMin.x) / span.x : 0.0f;
        float maxSpan = std::max(span.x, span.y);
        float boundaryBand = maxSpan * 0.050f;
        bool nearLeftOrRightBoundary = std::abs(mid.x - bMin.x) < boundaryBand || std::abs(mid.x - bMax.x) < boundaryBand;
        bool interiorConnector = !nearLeftOrRightBoundary && nx > 0.05f && nx < 0.96f;
        bool routeLike = horizontalScore > 0.34f || verticalScore > 0.42f;

        return interiorConnector && routeLike && normalizedLength > 0.12f;
    }

    bool tryClassifyStreetEdge(
        const Vec3& a,
        const Vec3& b,
        float length,
        float longest,
        const std::vector<std::pair<Vec3, Vec3>>& primaryEdges,
        const std::vector<std::pair<Vec3, Vec3>>& secondaryEdges,
        StreetClass& streetClass
    ) const
    {
        if (isPrimaryStreetEdge(a, b, length, longest)) {
            streetClass = StreetClass::Primary;
            return true;
        }
        if (isEdgeInList(secondaryEdges, a, b)) {
            streetClass = StreetClass::Secondary;
            return true;
        }
        if (isTertiaryStreetEdge(a, b, length, longest, secondaryEdges)) {
            streetClass = StreetClass::Tertiary;
            return true;
        }

        return false;
    }

    float nearestPrimaryStreetDistance(const Vec3& p, const std::vector<std::pair<Vec3, Vec3>>& primaryEdges) const
    {
        return nearestEdgeDistance(p, primaryEdges);
    }

    float nearestEdgeDistance(const Vec3& p, const std::vector<std::pair<Vec3, Vec3>>& edges) const
    {
        float nearest = 1e9f;
        for (const auto& edge : edges) {
            nearest = std::min(nearest, distanceToSegment2d(p, edge.first, edge.second));
        }
        return nearest;
    }

    bool isEdgeInList(const std::vector<std::pair<Vec3, Vec3>>& edges, const Vec3& a, const Vec3& b) const
    {
        const float eps = 1e-4f;
        for (const auto& edge : edges) {
            bool same = (edge.first - a).length() < eps && (edge.second - b).length() < eps;
            bool reverse = (edge.first - b).length() < eps && (edge.second - a).length() < eps;
            if (same || reverse) return true;
        }
        return false;
    }

    bool edgeAlreadyAdded(const std::vector<std::pair<Vec3, Vec3>>& edges, const Vec3& a, const Vec3& b) const
    {
        const float eps = 1e-4f;
        for (const auto& edge : edges) {
            bool same = (edge.first - a).length() < eps && (edge.second - b).length() < eps;
            bool reverse = (edge.first - b).length() < eps && (edge.second - a).length() < eps;
            if (same || reverse) return true;
        }
        return false;
    }

    int findStreetEdgeIndex(const Vec3& a, const Vec3& b) const
    {
        const float eps = 1e-4f;
        for (int i = 0; i < static_cast<int>(m_streetEdges.size()); ++i) {
            const auto& edge = m_streetEdges[i];
            bool same = (edge.a - a).length() < eps && (edge.b - b).length() < eps;
            bool reverse = (edge.a - b).length() < eps && (edge.b - a).length() < eps;
            if (same || reverse) return i;
        }
        return -1;
    }

    PlotBoundaryType boundaryTypeForStreetEdge(int streetEdgeIndex) const
    {
        if (streetEdgeIndex < 0 || streetEdgeIndex >= static_cast<int>(m_streetEdges.size())) {
            return PlotBoundaryType::PlotSplitLine;
        }

        switch (m_streetEdges[streetEdgeIndex].streetClass) {
            case StreetClass::Primary: return PlotBoundaryType::PrimaryRoad;
            case StreetClass::Secondary: return PlotBoundaryType::SecondaryRoad;
            case StreetClass::Tertiary: return PlotBoundaryType::TertiaryRoad;
        }

        return PlotBoundaryType::PlotSplitLine;
    }

    float streetOffsetWidth(StreetClass streetClass) const
    {
        switch (streetClass) {
            case StreetClass::Primary: return primaryStreetWidth() * 0.5f;
            case StreetClass::Secondary: return secondaryStreetWidth() * 0.5f;
            case StreetClass::Tertiary: return tertiaryStreetWidth() * 0.5f;
        }
        return tertiaryStreetWidth() * 0.5f;
    }

    float primaryStreetWidth() const
    {
        return std::max(0.01f, m_p);
    }

    float secondaryStreetWidth() const
    {
        return primaryStreetWidth() * (2.0f / 3.0f);
    }

    float tertiaryStreetWidth() const
    {
        return primaryStreetWidth() * (1.0f / 3.0f);
    }

    Color streetColor(StreetClass streetClass) const
    {
        switch (streetClass) {
            case StreetClass::Primary: return Color(1.0f, 0.0f, 0.0f, 1.0f);
            case StreetClass::Secondary: return Color(0.0f, 0.18f, 0.86f, 1.0f);
            case StreetClass::Tertiary: return Color(0.0f, 0.82f, 0.0f, 1.0f);
        }
        return Color(0.0f, 0.82f, 0.0f, 1.0f);
    }

    Color streetOffsetColor(StreetClass streetClass) const
    {
        (void)streetClass;
        return Color(0.5f, 0.5f, 0.5f, 1.0f);
    }

    void drawNeutralBaseMesh(Renderer& renderer, zSpace::zFnMesh& fn)
    {
        for (int i = 0; i < fn.numPolygons(); ++i) {
            zSpace::zItMeshFace face(m_mesh, i);
            if (!face.isActive()) continue;

            std::vector<zSpace::zVector> positions;
            face.getVertexPositions(positions);
            if (positions.size() < 3) continue;

            Vec3 center = withZ(toVec3(face.getCenter()), m_baseZ);
            Color faceColor = densityBaseColor(toVec3(face.getCenter()));

            for (size_t j = 0; j < positions.size(); ++j) {
                Vec3 p1 = withZ(toVec3(positions[j]), m_baseZ);
                Vec3 p2 = withZ(toVec3(positions[(j + 1) % positions.size()]), m_baseZ);
                renderer.drawTriangle(center, p1, p2, faceColor);
            }

            for (size_t j = 0; j < positions.size(); ++j) {
                Vec3 p1 = withZ(toVec3(positions[j]), m_baseZ + 0.001f);
                Vec3 p2 = withZ(toVec3(positions[(j + 1) % positions.size()]), m_baseZ + 0.001f);
                renderer.drawLine(p1, p2, Color(0.78f, 0.78f, 0.74f, 1.0f), 1.0f);
            }
        }
    }

    void drawTypeACenterlineGraphs(Renderer& renderer)
    {
        const Color graphColor(1.0f, 0.0f, 1.0f, 1.0f);

        for (const auto& plotData : m_plots) {
            for (const auto& edge : plotData.centerlineGraphEdges) {
                if (edge.startVertexIndex < 0 || edge.endVertexIndex < 0) continue;
                if (edge.startVertexIndex >= static_cast<int>(plotData.centerlineGraphVertices.size())) continue;
                if (edge.endVertexIndex >= static_cast<int>(plotData.centerlineGraphVertices.size())) continue;

                Vec3 a = withZ(plotData.centerlineGraphVertices[edge.startVertexIndex].position, m_massingZ + 0.002f);
                Vec3 b = withZ(plotData.centerlineGraphVertices[edge.endVertexIndex].position, m_massingZ + 0.002f);
                renderer.drawLine(a, b, graphColor, 2.0f);
            }

            for (const auto& vertex : plotData.centerlineGraphVertices) {
                renderer.drawPoint(withZ(vertex.position, m_massingZ + 0.003f), graphColor, 5.0f);
            }
        }
    }

    void drawSimpleMassing(Renderer& renderer, zSpace::zFnMesh& fn)
    {
        for (int i = 0; i < fn.numPolygons(); ++i) {
            zSpace::zItMeshFace face(m_mesh, i);
            if (!face.isActive()) continue;

            Vec3 center = toVec3(face.getCenter());
            if (isCivicOpenSpace(center)) continue;
            if (isStreetSpace(center)) continue;

            float density = densityValue(center);
            if (density < 0.42f && std::fmod(static_cast<float>(i), m_massingCoverageStep) > 0.01f) continue;
            if (density >= 0.42f && density < 0.68f && std::fmod(static_cast<float>(i), 2.0f) > 0.01f) continue;

            std::vector<zSpace::zVector> positions;
            face.getVertexPositions(positions);
            if (positions.size() < 3) continue;

            float coverage = m_parcelCoverage + density * 0.22f;
            std::vector<Vec3> footprint = makeConstrainedBuildingFootprint(center, positions, coverage);
            if (footprint.empty()) continue;

            Vec3 c = withZ(center, m_massingZ);
            for (size_t j = 0; j < footprint.size(); ++j) {
                renderer.drawTriangle(c, footprint[j], footprint[(j + 1) % footprint.size()], Color(0.0f, 0.0f, 0.0f, 1.0f));
            }
        }
    }

    void drawOpenSpaceSdf(Renderer& renderer, zSpace::zFnMesh& fn)
    {
        for (int i = 0; i < fn.numPolygons(); ++i) {
            zSpace::zItMeshFace face(m_mesh, i);
            if (!face.isActive()) continue;

            Vec3 center = toVec3(face.getCenter());
            if (!isCivicOpenSpace(center)) continue;

            std::vector<zSpace::zVector> positions;
            face.getVertexPositions(positions);
            if (positions.size() < 3) continue;

            Vec3 c = withZ(center, m_openSpaceZ);
            Color openSpaceColor(0.74f, 0.84f, 0.67f, 1.0f);
            for (size_t j = 0; j < positions.size(); ++j) {
                Vec3 p1 = withZ(toVec3(positions[j]), m_openSpaceZ);
                Vec3 p2 = withZ(toVec3(positions[(j + 1) % positions.size()]), m_openSpaceZ);
                renderer.drawTriangle(c, p1, p2, openSpaceColor);
            }
        }
    }

    void drawStreetSdfGeometry(Renderer& renderer)
    {
        (void)renderer;
        if (m_drawStreetFieldMesh) {
            zDisplayMeshSetting fieldDisplay;
            fieldDisplay.showFaces = true;
            fieldDisplay.showEdges = false;
            fieldDisplay.showVertices = false;
            fieldDisplay.useVertexColors = true;
            scene().draw(m_streetSdfField, fieldDisplay);
        }

        zDisplayGraphSetting contourDisplay;
        contourDisplay.showEdges = true;
        contourDisplay.showVertices = false;
        contourDisplay.edgeColor = Color(0.12f, 0.12f, 0.12f, 1.0f);
        contourDisplay.edgeWidth = 2.0f;
        scene().draw(m_streetIsoContour, contourDisplay);

    }

    std::vector<Vec3> makeConstrainedBuildingFootprint(const Vec3& center, const std::vector<zSpace::zVector>& positions, float parcelCoverage) const
    {
        Vec3 axisX(1.0f, 0.0f, 0.0f);
        float longestEdge = 0.0f;

        for (size_t i = 0; i < positions.size(); ++i) {
            Vec3 a = toVec3(positions[i]);
            Vec3 b = toVec3(positions[(i + 1) % positions.size()]);
            Vec3 edge = b - a;
            float length = std::sqrt(edge.x * edge.x + edge.y * edge.y);
            if (length > longestEdge) {
                longestEdge = length;
                axisX = normalized2d(edge);
            }
        }

        Vec3 axisY(-axisX.y, axisX.x, 0.0f);
        float minX = 1e9f;
        float maxX = -1e9f;
        float minY = 1e9f;
        float maxY = -1e9f;

        for (const auto& p : positions) {
            Vec3 rel = toVec3(p) - center;
            float x = dot2d(rel, axisX);
            float y = dot2d(rel, axisY);
            minX = std::min(minX, x);
            maxX = std::max(maxX, x);
            minY = std::min(minY, y);
            maxY = std::max(maxY, y);
        }

        float availableLength = (maxX - minX) * m_edgeClearanceFactor;
        float availableDepth = (maxY - minY) * m_edgeClearanceFactor;
        if (availableLength < m_minBuildingLength || availableDepth < m_minBuildingDepth) {
            return {};
        }

        float length = std::max(m_minBuildingLength, availableLength * parcelCoverage);
        float depth = std::max(m_minBuildingDepth, availableDepth * parcelCoverage);

        length = std::min(length, availableLength);
        depth = std::min(depth, availableDepth);

        if (length / depth > m_maxBuildingAspect) {
            length = std::min(availableLength, depth * m_maxBuildingAspect);
        }
        if (depth / length > m_maxBuildingAspect) {
            depth = std::min(availableDepth, length * m_maxBuildingAspect);
        }

        if (length < m_minBuildingLength || depth < m_minBuildingDepth) {
            return {};
        }

        float halfLength = length * 0.5f;
        float halfDepth = depth * 0.5f;
        return {
            withZ(center - axisX * halfLength - axisY * halfDepth, m_massingZ),
            withZ(center + axisX * halfLength - axisY * halfDepth, m_massingZ),
            withZ(center + axisX * halfLength + axisY * halfDepth, m_massingZ),
            withZ(center - axisX * halfLength + axisY * halfDepth, m_massingZ)
        };
    }
};

ALICE2_REGISTER_SKETCH_AUTO(zSpaceUrbanCodexLoopSketch)

#endif
