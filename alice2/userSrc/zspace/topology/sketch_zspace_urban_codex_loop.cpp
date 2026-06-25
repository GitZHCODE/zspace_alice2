#define __MAIN__
#ifdef __MAIN__

#include <zspace/interface.h>
#include <zspace/io.h>

#include <alice2.h>
#include <sketches/SketchRegistry.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
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
        m_ui->addSlider("p", Vec2{14.0f, 82.0f}, 240.0f, 0.0f, 100.0f, m_p);
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
            buildTypeBCenterlineGraphs();
            buildTypeCCenterlineGraphs();
            buildEffectiveTypologyGraphs();
            buildTypeBSdfField();
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
        drawTypeBSdfContour(renderer);
        drawEffectiveTypologyGraphs(renderer);

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
    Vec3 m_plotCenterMin;
    Vec3 m_plotCenterMax;
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
    float m_typeAMinWidthMeters = 15.0f;
    float m_typeAMaxWidthMeters = 25.0f;
    float m_typeARoadSetbackMeters = 5.0f;
    float m_typeALocalSetbackMeters = 2.0f;
    float m_openSpaceZ = 0.001f;
    float m_p = 12.0f;
    float m_lastBuiltP = -1.0f;
    float m_siteLongDimensionMeters = 500.0f;
    float m_modelUnitsPerMeter = 1.0f;
    float m_globalParameterScale = 1.0f;
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

    enum class BuildingType {
        TypeA,
        TypeB,
        TypeC
    };

    struct ShapeParams {
        float typeAWeight = 1.0f;
        float typeBWeight = 0.0f;
        float typeCWeight = 0.0f;
        float buildingWidthMeters = 20.0f;
        float typeAEdgeLengthFraction = 0.5f;
        float typeBXFraction = 0.5f;
        float typeBInternalEdgeFraction = 0.25f;
        float typeCEdgeFraction = 0.75f;
        float typeBOrientationIndex = 0.0f;
        float typeCOrientationIndex = 0.0f;
    };

    struct TypologyAnchor {
        Vec3 position;
        ShapeParams params;
        float strength = 1.0f;
        float radius = 1.0f;
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

    struct TypeASdfBox {
        Vec3 center;
        Vec3 axisX;
        Vec3 axisY;
        float halfX;
        float halfY;
    };

    struct TypeASetbackPlane {
        Vec3 point;
        Vec3 inwardNormal;
    };

    struct TypeAPlotSdf {
        std::vector<TypeASdfBox> sdfA;
        std::vector<TypeASdfBox> sdfB;
        std::vector<TypeASetbackPlane> setbackPlanes;
    };

    struct TypeBPlotSdf {
        std::vector<std::pair<Vec3, Vec3>> graphSegments;
        std::vector<Vec3> graphJointPoints;
        float graphHalfWidth = 0.0f;
        std::vector<TypeASetbackPlane> setbackPlanes;
    };

    class plot {
    public:
        struct CenterlineGraphEdge {
            int startVertexIndex;
            int endVertexIndex;
            PlotBoundaryType boundaryType;
            float offsetDistance;
        };

        struct TypeBGraphSegment {
            Vec3 start;
            Vec3 end;
        };

        struct WeightedGraphSegment {
            TypeBGraphSegment segment;
            float weight = 0.0f;
        };

        int id;
        int faceIndex;
        BuildingType buildingType = BuildingType::TypeA;
        Vec3 center;
        float typeABlendWeight = 1.0f;
        float typeBBlendWeight = 0.0f;
        float typeCBlendWeight = 0.0f;
        float typeABuildingWidthMeters = 25.0f;
        float typeAEdgeLengthFraction = 1.0f;
        float typeBXFraction = 0.5f;
        float typeBYFraction = 0.5f;
        float typeBInternalEdgeFraction = 0.5f;
        float typeCEdgeFraction = 0.75f;
        int typeBOrientationIndex = 0;
        int typeCOrientationIndex = 0;
        std::vector<Vec3> vertices;
        std::vector<PlotBoundaryEdge> boundaryEdges;
        std::vector<CenterlineGraphEdge> centerlineGraphEdges;
        std::vector<TypeBGraphSegment> typeBGraphSegments;
        std::vector<TypeBGraphSegment> typeCGraphSegments;
        std::vector<TypeBGraphSegment> effectiveGraphSegments;
        zSpace::zObjectGraph centerlineGraph;
        zSpace::zObjectGraph typeBCenterlineGraph;
        zSpace::zObjectGraph typeCCenterlineGraph;
        zSpace::zObjectGraph effectiveCenterlineGraph;

        void buildCenterlineGraph(
            float roadSetback,
            float localSetback,
            float buildingWidth,
            float primaryRoadHalfWidth,
            float secondaryRoadHalfWidth,
            float tertiaryRoadHalfWidth,
            float graphZ
        )
        {
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

            zSpace::zPointArray graphPositions;
            graphPositions.reserve(vertices.size());
            for (size_t i = 0; i < vertices.size(); ++i) {
                size_t previous = (i + offsetLines.size() - 1) % offsetLines.size();
                Vec3 position;
                if (!intersectLines2d(offsetLines[previous].a, offsetLines[previous].b, offsetLines[i].a, offsetLines[i].b, position)) {
                    position = (offsetLines[previous].b + offsetLines[i].a) * 0.5f;
                }
                position.z = graphZ;
                graphPositions.push_back(zSpace::zPoint(position.x, position.y, position.z));
            }

            centerlineGraphEdges.reserve(boundaryEdges.size());
            zSpace::zIntArray graphEdgeConnects;
            graphEdgeConnects.reserve(boundaryEdges.size() * 2);
            for (size_t i = 0; i < boundaryEdges.size(); ++i) {
                int start = static_cast<int>(i);
                int end = static_cast<int>((i + 1) % boundaryEdges.size());
                centerlineGraphEdges.push_back({
                    start,
                    end,
                    boundaryEdges[i].boundaryType,
                    offsetLines[i].offsetDistance
                });
                graphEdgeConnects.push_back(start);
                graphEdgeConnects.push_back(end);
            }

            zSpace::zFnGraph graphFn(centerlineGraph);
            graphFn.create(graphPositions, graphEdgeConnects);
        }

        void buildTypeBSGraph(float xFraction, float internalEdgeFraction, int orientationIndex, float graphZ)
        {
            xFraction = std::clamp(xFraction, 0.0f, 1.0f);
            internalEdgeFraction = std::clamp(internalEdgeFraction, 0.0f, 0.5f);
            typeBXFraction = xFraction;
            typeBYFraction = 1.0f - typeBXFraction;
            typeBInternalEdgeFraction = internalEdgeFraction;
            typeBOrientationIndex = orientationIndex;
            typeBGraphSegments.clear();

            zSpace::zFnGraph sourceFn(centerlineGraph);
            zSpace::zPointArray sourcePositions;
            sourceFn.getVertexPositions(sourcePositions);
            if (sourcePositions.size() < 4) return;

            std::vector<Vec3> p;
            p.reserve(sourcePositions.size());
            for (const auto& sourcePosition : sourcePositions) {
                p.push_back(Vec3(sourcePosition.x, sourcePosition.y, graphZ));
            }

            const int n = static_cast<int>(p.size());
            const int base = wrappedIndex(orientationIndex, n);
            const int a = base;
            const int afterA = (a + 1) % n;
            const int b = (a + n / 2) % n;
            const int afterB = (b + 1) % n;

            Vec3 midA = lerp(p[a], p[afterA], typeBXFraction);
            Vec3 midB = lerp(p[b], p[afterB], typeBXFraction);

            Vec3 partialA = lerp(midA, midB, typeBInternalEdgeFraction);
            Vec3 partialB = lerp(midB, midA, typeBInternalEdgeFraction);

            zSpace::zPointArray graphPositions;
            graphPositions.push_back(zSpace::zPoint(p[a].x, p[a].y, graphZ));
            graphPositions.push_back(zSpace::zPoint(midA.x, midA.y, graphZ));
            graphPositions.push_back(zSpace::zPoint(partialA.x, partialA.y, graphZ));
            graphPositions.push_back(zSpace::zPoint(partialB.x, partialB.y, graphZ));
            graphPositions.push_back(zSpace::zPoint(midB.x, midB.y, graphZ));
            graphPositions.push_back(zSpace::zPoint(p[b].x, p[b].y, graphZ));

            zSpace::zIntArray graphEdgeConnects;
            graphEdgeConnects.push_back(0);
            graphEdgeConnects.push_back(1);
            graphEdgeConnects.push_back(1);
            graphEdgeConnects.push_back(2);
            graphEdgeConnects.push_back(5);
            graphEdgeConnects.push_back(4);
            graphEdgeConnects.push_back(4);
            graphEdgeConnects.push_back(3);

            typeBGraphSegments.push_back({ p[a], midA });
            if ((partialA - midA).length() > 1e-6f) {
                typeBGraphSegments.push_back({ midA, partialA });
            }
            typeBGraphSegments.push_back({ p[b], midB });
            if ((partialB - midB).length() > 1e-6f) {
                typeBGraphSegments.push_back({ midB, partialB });
            }

            zSpace::zFnGraph graphFn(typeBCenterlineGraph);
            graphFn.create(graphPositions, graphEdgeConnects);
        }

        void buildTypeCParallelGraph(float edgeFraction, int orientationIndex, float graphZ)
        {
            edgeFraction = std::clamp(edgeFraction, 0.5f, 1.0f);
            typeCEdgeFraction = edgeFraction;
            typeCOrientationIndex = orientationIndex;
            typeCGraphSegments.clear();

            zSpace::zFnGraph sourceFn(centerlineGraph);
            zSpace::zPointArray sourcePositions;
            sourceFn.getVertexPositions(sourcePositions);
            if (sourcePositions.size() < 4) return;

            std::vector<Vec3> p;
            p.reserve(sourcePositions.size());
            for (const auto& sourcePosition : sourcePositions) {
                p.push_back(Vec3(sourcePosition.x, sourcePosition.y, graphZ));
            }

            const int n = static_cast<int>(p.size());
            const int a0 = wrappedIndex(orientationIndex, n);
            const int a1 = (a0 + 1) % n;
            const int b0 = (a0 + n / 2) % n;
            const int b1 = (b0 + 1) % n;

            Vec3 aEnd = lerp(p[a0], p[a1], edgeFraction);
            Vec3 bEnd = lerp(p[b0], p[b1], edgeFraction);

            zSpace::zPointArray graphPositions;
            graphPositions.push_back(zSpace::zPoint(p[a0].x, p[a0].y, graphZ));
            graphPositions.push_back(zSpace::zPoint(aEnd.x, aEnd.y, graphZ));
            graphPositions.push_back(zSpace::zPoint(p[b0].x, p[b0].y, graphZ));
            graphPositions.push_back(zSpace::zPoint(bEnd.x, bEnd.y, graphZ));

            zSpace::zIntArray graphEdgeConnects;
            graphEdgeConnects.push_back(0);
            graphEdgeConnects.push_back(1);
            graphEdgeConnects.push_back(2);
            graphEdgeConnects.push_back(3);

            typeCGraphSegments.push_back({ p[a0], aEnd });
            typeCGraphSegments.push_back({ p[b0], bEnd });

            zSpace::zFnGraph graphFn(typeCCenterlineGraph);
            graphFn.create(graphPositions, graphEdgeConnects);
        }

        void buildEffectiveGraph(float graphZ)
        {
            effectiveGraphSegments.clear();

            zSpace::zFnGraph sourceFn(centerlineGraph);
            zSpace::zPointArray sourcePositions;
            sourceFn.getVertexPositions(sourcePositions);
            if (sourcePositions.size() < 4) return;

            std::vector<Vec3> p;
            p.reserve(sourcePositions.size());
            for (const auto& sourcePosition : sourcePositions) {
                p.push_back(Vec3(sourcePosition.x, sourcePosition.y, graphZ));
            }

            float totalWeight = typeABlendWeight + typeBBlendWeight + typeCBlendWeight;
            if (totalWeight <= 0.001f) return;

            float aWeight = typeABlendWeight / totalWeight;
            float bWeight = typeBBlendWeight / totalWeight;
            float cWeight = typeCBlendWeight / totalWeight;

            std::vector<WeightedGraphSegment> overlaySegments;
            addWeightedOverlaySegments(overlaySegments, makeTypeAOverlaySegments(p, typeAEdgeLengthFraction), aWeight);
            addWeightedOverlaySegments(
                overlaySegments,
                makeTypeBOverlaySegments(p, typeBXFraction, typeBInternalEdgeFraction, typeBOrientationIndex),
                bWeight
            );
            addWeightedOverlaySegments(
                overlaySegments,
                makeTypeCOverlaySegments(p, typeCEdgeFraction, typeCOrientationIndex),
                cWeight
            );

            effectiveGraphSegments = overlayToEffectiveSegments(overlaySegments, 0.08f);
            createGraphFromSegments(effectiveGraphSegments, effectiveCenterlineGraph, graphZ);
        }

    private:
        static void addWeightedOverlaySegments(
            std::vector<WeightedGraphSegment>& target,
            const std::vector<TypeBGraphSegment>& source,
            float weight
        )
        {
            if (weight <= 0.001f) return;
            for (const auto& segment : source) {
                if ((segment.end - segment.start).length() <= 1e-6f) continue;
                target.push_back({ segment, weight });
            }
        }

        static std::vector<TypeBGraphSegment> overlayToEffectiveSegments(
            const std::vector<WeightedGraphSegment>& overlaySegments,
            float weightThreshold
        )
        {
            std::vector<TypeBGraphSegment> effectiveSegments;
            effectiveSegments.reserve(overlaySegments.size());

            for (const auto& overlay : overlaySegments) {
                if (overlay.weight < weightThreshold) continue;
                effectiveSegments.push_back(overlay.segment);
            }

            return effectiveSegments;
        }

        static std::vector<TypeBGraphSegment> makeTypeAOverlaySegments(const std::vector<Vec3>& p, float edgeLengthFraction)
        {
            std::vector<TypeBGraphSegment> segments;
            const int n = static_cast<int>(p.size());
            if (n < 4) return segments;

            edgeLengthFraction = std::clamp(edgeLengthFraction, 0.25f, 1.0f);
            if (edgeLengthFraction >= 0.999f) {
                segments.reserve(n);
                for (int i = 0; i < n; ++i) {
                    segments.push_back({ p[i], p[(i + 1) % n] });
                }
                return segments;
            }

            int corners[2] = { 0, n / 2 };
            segments.reserve(4);
            for (int corner : corners) {
                int prev = wrappedIndex(corner - 1, n);
                int next = wrappedIndex(corner + 1, n);
                segments.push_back({ p[corner], lerp(p[corner], p[prev], edgeLengthFraction) });
                segments.push_back({ p[corner], lerp(p[corner], p[next], edgeLengthFraction) });
            }

            return segments;
        }

        static std::vector<TypeBGraphSegment> makeTypeBOverlaySegments(
            const std::vector<Vec3>& p,
            float xFraction,
            float internalEdgeFraction,
            int orientationIndex
        )
        {
            std::vector<TypeBGraphSegment> segments;
            const int n = static_cast<int>(p.size());
            if (n < 4) return segments;

            xFraction = std::clamp(xFraction, 0.25f, 0.75f);
            internalEdgeFraction = std::clamp(internalEdgeFraction, 0.0f, 0.5f);

            int a = wrappedIndex(orientationIndex, n);
            int afterA = (a + 1) % n;
            int b = (a + n / 2) % n;
            int afterB = (b + 1) % n;

            Vec3 midA = lerp(p[a], p[afterA], xFraction);
            Vec3 midB = lerp(p[b], p[afterB], xFraction);
            Vec3 partialA = lerp(midA, midB, internalEdgeFraction);
            Vec3 partialB = lerp(midB, midA, internalEdgeFraction);

            segments.push_back({ p[a], midA });
            if ((partialA - midA).length() > 1e-6f) {
                segments.push_back({ midA, partialA });
            }
            segments.push_back({ p[b], midB });
            if ((partialB - midB).length() > 1e-6f) {
                segments.push_back({ midB, partialB });
            }

            return segments;
        }

        static std::vector<TypeBGraphSegment> makeTypeCOverlaySegments(
            const std::vector<Vec3>& p,
            float edgeFraction,
            int orientationIndex
        )
        {
            std::vector<TypeBGraphSegment> segments;
            const int n = static_cast<int>(p.size());
            if (n < 4) return segments;

            edgeFraction = std::clamp(edgeFraction, 0.5f, 1.0f);
            int a0 = wrappedIndex(orientationIndex, n);
            int a1 = (a0 + 1) % n;
            int b0 = (a0 + n / 2) % n;
            int b1 = (b0 + 1) % n;

            segments.push_back({ p[a0], lerp(p[a0], p[a1], edgeFraction) });
            segments.push_back({ p[b0], lerp(p[b0], p[b1], edgeFraction) });

            return segments;
        }

        static Vec3 lerp(const Vec3& a, const Vec3& b, float t)
        {
            return a + (b - a) * t;
        }

        static int wrappedIndex(int index, int size)
        {
            if (size <= 0) return 0;
            int result = index % size;
            return result < 0 ? result + size : result;
        }

        static void createGraphFromSegments(
            const std::vector<TypeBGraphSegment>& segments,
            zSpace::zObjectGraph& graph,
            float graphZ
        )
        {
            zSpace::zPointArray graphPositions;
            zSpace::zIntArray graphEdgeConnects;
            graphPositions.reserve(segments.size() * 2);
            graphEdgeConnects.reserve(segments.size() * 2);

            for (const auto& segment : segments) {
                int startIndex = findOrAddGraphPosition(graphPositions, segment.start, graphZ);
                int endIndex = findOrAddGraphPosition(graphPositions, segment.end, graphZ);
                if (startIndex == endIndex) continue;
                graphEdgeConnects.push_back(startIndex);
                graphEdgeConnects.push_back(endIndex);
            }

            zSpace::zFnGraph graphFn(graph);
            graphFn.create(graphPositions, graphEdgeConnects);
        }

        static int findOrAddGraphPosition(zSpace::zPointArray& graphPositions, const Vec3& position, float graphZ)
        {
            for (int i = 0; i < static_cast<int>(graphPositions.size()); ++i) {
                Vec3 existing(graphPositions[i].x, graphPositions[i].y, graphZ);
                if ((existing - Vec3(position.x, position.y, graphZ)).length() < 1e-6f) {
                    return i;
                }
            }

            graphPositions.push_back(zSpace::zPoint(position.x, position.y, graphZ));
            return static_cast<int>(graphPositions.size() - 1);
        }

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
    std::vector<TypologyAnchor> m_typologyAnchors;
    zSpace::zObjectMeshScalarField m_streetSdfField;
    zSpace::zObjectGraph m_streetIsoContour;
    zSpace::zObjectMeshScalarField m_typeASdfField;
    zSpace::zObjectGraph m_typeAIsoContour;
    zSpace::zObjectMeshScalarField m_typeBSdfField;
    zSpace::zObjectGraph m_typeBIsoContour;
    std::vector<TypeAPlotSdf> m_typeASdfPlots;
    std::vector<TypeBPlotSdf> m_typeBSdfPlots;

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
        return meters * m_modelUnitsPerMeter * m_globalParameterScale;
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
        initializeTypologyAnchors(bMin, bMax);

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
        buildTypeBCenterlineGraphs();
        buildTypeCCenterlineGraphs();
        buildEffectiveTypologyGraphs();
        buildTypeBSdfField();

        m_loaded = true;
    }

    void initializeTypologyAnchors(const Vec3& bMin, const Vec3& bMax)
    {
        m_typologyAnchors.clear();

        Vec3 span = bMax - bMin;
        float radius = std::max(span.x, span.y);
        Vec3 bottomLeft(bMin.x, bMin.y, 0.0f);
        Vec3 bottomRight(bMax.x, bMin.y, 0.0f);
        Vec3 topLeft(bMin.x, bMax.y, 0.0f);
        Vec3 topRight(bMax.x, bMax.y, 0.0f);

        ShapeParams bHalf;
        bHalf.typeAWeight = 0.0f;
        bHalf.typeBWeight = 1.0f;
        bHalf.typeCWeight = 0.0f;
        bHalf.buildingWidthMeters = 20.0f;
        bHalf.typeBXFraction = 0.5f;
        bHalf.typeBInternalEdgeFraction = 0.25f;
        bHalf.typeBOrientationIndex = 1.0f;

        ShapeParams aFull;
        aFull.typeAWeight = 1.0f;
        aFull.typeBWeight = 0.0f;
        aFull.typeCWeight = 0.0f;
        aFull.buildingWidthMeters = 22.0f;
        aFull.typeAEdgeLengthFraction = 1.0f;

        ShapeParams cParallel;
        cParallel.typeAWeight = 0.0f;
        cParallel.typeBWeight = 0.0f;
        cParallel.typeCWeight = 1.0f;
        cParallel.buildingWidthMeters = 18.0f;
        cParallel.typeCEdgeFraction = 1.0f;
        cParallel.typeCOrientationIndex = 1.0f;

        ShapeParams aLong;
        aLong.typeAWeight = 1.0f;
        aLong.typeBWeight = 0.0f;
        aLong.typeCWeight = 0.0f;
        aLong.buildingWidthMeters = 20.0f;
        aLong.typeAEdgeLengthFraction = 0.70f;

        m_typologyAnchors.push_back({ bottomLeft, bHalf, 1.0f, radius });
        m_typologyAnchors.push_back({ bottomRight, aFull, 1.0f, radius });
        m_typologyAnchors.push_back({ topLeft, cParallel, 1.0f, radius });
        m_typologyAnchors.push_back({ topRight, aLong, 1.0f, radius });

        std::cout << "[URBAN CODEX LOOP] Typology anchors: " << m_typologyAnchors.size()
                  << " | arbitrary anchor field enabled" << std::endl;
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
        m_plotCenterMin = Vec3(1e9f, 1e9f, 0.0f);
        m_plotCenterMax = Vec3(-1e9f, -1e9f, 0.0f);

        for (int i = 0; i < fn.numPolygons(); ++i) {
            zSpace::zItMeshFace face(m_mesh, i);
            if (!face.isActive()) continue;

            std::vector<zSpace::zVector> positions;
            face.getVertexPositions(positions);
            if (positions.size() < 3) continue;
            std::vector<Vec3> orderedVertices = orderedPlotVertices(positions);
            if (orderedVertices.size() < 3) continue;

            plot plotData;
            plotData.id = static_cast<int>(m_plots.size());
            plotData.faceIndex = i;
            plotData.center = toVec3(face.getCenter());
            m_plotCenterMin.x = std::min(m_plotCenterMin.x, plotData.center.x);
            m_plotCenterMin.y = std::min(m_plotCenterMin.y, plotData.center.y);
            m_plotCenterMax.x = std::max(m_plotCenterMax.x, plotData.center.x);
            m_plotCenterMax.y = std::max(m_plotCenterMax.y, plotData.center.y);
            plotData.vertices.reserve(positions.size());
            plotData.boundaryEdges.reserve(positions.size());

            for (const auto& position : orderedVertices) {
                plotData.vertices.push_back(position);
            }

            for (size_t j = 0; j < orderedVertices.size(); ++j) {
                Vec3 a = orderedVertices[j];
                Vec3 b = orderedVertices[(j + 1) % orderedVertices.size()];
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

        if (m_plots.empty()) {
            m_plotCenterMin = Vec3(0.0f, 0.0f, 0.0f);
            m_plotCenterMax = Vec3(1.0f, 1.0f, 0.0f);
        }
        else {
            for (auto& plotData : m_plots) {
                applyTypologyGene(plotData);
            }
        }

        std::cout << "[URBAN CODEX LOOP] Plot records: " << m_plots.size() << std::endl;
        logPlotBoundarySummary();
        logBuildingTypeSummary();
    }

    std::vector<Vec3> orderedPlotVertices(const std::vector<zSpace::zVector>& positions) const
    {
        std::vector<Vec3> ordered;
        ordered.reserve(positions.size());
        Vec3 center(0.0f, 0.0f, 0.0f);
        for (const auto& position : positions) {
            Vec3 p = toVec3(position);
            ordered.push_back(p);
            center += p;
        }

        if (ordered.empty()) return ordered;
        center = center * (1.0f / static_cast<float>(ordered.size()));

        std::sort(ordered.begin(), ordered.end(), [center](const Vec3& a, const Vec3& b) {
            float angleA = std::atan2(a.y - center.y, a.x - center.x);
            float angleB = std::atan2(b.y - center.y, b.x - center.x);
            return angleA < angleB;
        });

        if (signedArea2d(ordered) < 0.0f) {
            std::reverse(ordered.begin(), ordered.end());
        }

        int startIndex = 0;
        float bestScore = std::numeric_limits<float>::max();
        for (int i = 0; i < static_cast<int>(ordered.size()); ++i) {
            float score = ordered[i].x + ordered[i].y;
            if (score < bestScore) {
                bestScore = score;
                startIndex = i;
            }
        }

        std::rotate(ordered.begin(), ordered.begin() + startIndex, ordered.end());
        return ordered;
    }

    static float signedArea2d(const std::vector<Vec3>& polygon)
    {
        float area = 0.0f;
        for (size_t i = 0; i < polygon.size(); ++i) {
            const Vec3& a = polygon[i];
            const Vec3& b = polygon[(i + 1) % polygon.size()];
            area += a.x * b.y - b.x * a.y;
        }
        return area * 0.5f;
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

    void logBuildingTypeSummary() const
    {
        int typeA = 0;
        int typeB = 0;
        int typeC = 0;
        for (const auto& plotData : m_plots) {
            if (plotData.buildingType == BuildingType::TypeA) {
                typeA++;
            }
            else if (plotData.buildingType == BuildingType::TypeB) {
                typeB++;
            }
            else if (plotData.buildingType == BuildingType::TypeC) {
                typeC++;
            }
        }

        std::cout << "[URBAN CODEX LOOP] Building type assignment | Type A: " << typeA
                  << " Type B: " << typeB
                  << " Type C: " << typeC << std::endl;
    }

    ShapeParams computeTypologyGene(const Vec3& position) const
    {
        if (m_typologyAnchors.empty()) {
            return fallbackShapeParams(0);
        }

        if (m_typologyAnchors.size() == 4) {
            float u = saturate((position.x - m_plotCenterMin.x) / std::max(m_plotCenterMax.x - m_plotCenterMin.x, 1e-6f));
            float v = saturate((position.y - m_plotCenterMin.y) / std::max(m_plotCenterMax.y - m_plotCenterMin.y, 1e-6f));

            std::vector<float> weights = {
                (1.0f - u) * (1.0f - v),
                u * (1.0f - v),
                (1.0f - u) * v,
                u * v
            };
            return blendedShapeParams(weights);
        }

        std::vector<float> weights;
        weights.reserve(m_typologyAnchors.size());
        for (const auto& anchor : m_typologyAnchors) {
            float radius = std::max(anchor.radius, 1e-6f);
            float d = (position - anchor.position).length() / radius;
            weights.push_back(anchor.strength / (d * d + 0.015f));
        }

        return blendedShapeParams(weights);
    }

    ShapeParams blendedShapeParams(const std::vector<float>& weights) const
    {
        ShapeParams result;
        result.typeAWeight = 0.0f;
        result.typeBWeight = 0.0f;
        result.typeCWeight = 0.0f;
        result.buildingWidthMeters = 0.0f;
        result.typeAEdgeLengthFraction = 0.0f;
        result.typeBXFraction = 0.0f;
        result.typeBInternalEdgeFraction = 0.0f;
        result.typeCEdgeFraction = 0.0f;
        result.typeBOrientationIndex = 0.0f;
        result.typeCOrientationIndex = 0.0f;
        float totalWeight = 0.0f;

        for (size_t i = 0; i < m_typologyAnchors.size() && i < weights.size(); ++i) {
            float w = std::max(weights[i], 0.0f);
            const auto& anchor = m_typologyAnchors[i];
            totalWeight += w;

            result.typeAWeight += anchor.params.typeAWeight * w;
            result.typeBWeight += anchor.params.typeBWeight * w;
            result.typeCWeight += anchor.params.typeCWeight * w;
            result.buildingWidthMeters += anchor.params.buildingWidthMeters * w;
            result.typeAEdgeLengthFraction += anchor.params.typeAEdgeLengthFraction * w;
            result.typeBXFraction += anchor.params.typeBXFraction * w;
            result.typeBInternalEdgeFraction += anchor.params.typeBInternalEdgeFraction * w;
            result.typeCEdgeFraction += anchor.params.typeCEdgeFraction * w;
            result.typeBOrientationIndex += anchor.params.typeBOrientationIndex * w;
            result.typeCOrientationIndex += anchor.params.typeCOrientationIndex * w;
        }

        if (totalWeight <= 1e-6f) {
            return fallbackShapeParams(0);
        }

        result.typeAWeight = std::clamp(result.typeAWeight / totalWeight, 0.0f, 1.0f);
        result.typeBWeight = std::clamp(result.typeBWeight / totalWeight, 0.0f, 1.0f);
        result.typeCWeight = std::clamp(result.typeCWeight / totalWeight, 0.0f, 1.0f);
        result.buildingWidthMeters = std::clamp(result.buildingWidthMeters / totalWeight, m_typeAMinWidthMeters, m_typeAMaxWidthMeters);
        result.typeAEdgeLengthFraction = sanitizeTypeAEdgeLengthFraction(result.typeAEdgeLengthFraction / totalWeight);
        result.typeBXFraction = std::clamp(result.typeBXFraction / totalWeight, 0.25f, 0.75f);
        result.typeBInternalEdgeFraction = std::clamp(result.typeBInternalEdgeFraction / totalWeight, 0.0f, 0.5f);
        result.typeCEdgeFraction = std::clamp(result.typeCEdgeFraction / totalWeight, 0.5f, 1.0f);
        result.typeBOrientationIndex = std::clamp(result.typeBOrientationIndex / totalWeight, 0.0f, 1.0f);
        result.typeCOrientationIndex = std::clamp(result.typeCOrientationIndex / totalWeight, 0.0f, 1.0f);
        return result;
    }

    ShapeParams fallbackShapeParams(int plotId) const
    {
        ShapeParams params;
        bool isTypeB = randomBuildingType(plotId) == BuildingType::TypeB;
        params.typeAWeight = isTypeB ? 0.0f : 1.0f;
        params.typeBWeight = isTypeB ? 1.0f : 0.0f;
        params.typeCWeight = 0.0f;
        params.buildingWidthMeters = randomTypeABuildingWidthMeters(plotId);
        params.typeAEdgeLengthFraction = randomTypeAEdgeLengthFraction(plotId);
        params.typeBXFraction = randomTypeBXFraction(plotId);
        params.typeBInternalEdgeFraction = randomTypeBInternalEdgeFraction(plotId);
        params.typeCEdgeFraction = randomTypeCEdgeFraction(plotId);
        params.typeBOrientationIndex = 0.0f;
        params.typeCOrientationIndex = 0.0f;
        return params;
    }

    void applyTypologyGene(plot& plotData) const
    {
        ShapeParams gene = computeTypologyGene(plotData.center);
        float totalWeight = gene.typeAWeight + gene.typeBWeight + gene.typeCWeight;
        if (totalWeight <= 1e-6f) totalWeight = 1.0f;
        float typeAWeight = std::clamp(gene.typeAWeight / totalWeight, 0.0f, 1.0f);
        float typeBWeight = std::clamp(gene.typeBWeight / totalWeight, 0.0f, 1.0f);
        float typeCWeight = std::clamp(gene.typeCWeight / totalWeight, 0.0f, 1.0f);
        plotData.typeABlendWeight = typeAWeight;
        plotData.typeBBlendWeight = typeBWeight;
        plotData.typeCBlendWeight = typeCWeight;
        plotData.buildingType = BuildingType::TypeA;
        if (typeBWeight >= typeAWeight && typeBWeight >= typeCWeight) {
            plotData.buildingType = BuildingType::TypeB;
        }
        else if (typeCWeight >= typeAWeight && typeCWeight >= typeBWeight) {
            plotData.buildingType = BuildingType::TypeC;
        }
        plotData.typeABuildingWidthMeters = gene.buildingWidthMeters;
        plotData.typeAEdgeLengthFraction = gene.typeAEdgeLengthFraction;
        plotData.typeBXFraction = gene.typeBXFraction;
        plotData.typeBYFraction = 1.0f - plotData.typeBXFraction;
        plotData.typeBInternalEdgeFraction = gene.typeBInternalEdgeFraction;
        plotData.typeCEdgeFraction = gene.typeCEdgeFraction;
        plotData.typeBOrientationIndex = static_cast<int>(std::round(gene.typeBOrientationIndex));
        plotData.typeCOrientationIndex = static_cast<int>(std::round(gene.typeCOrientationIndex));
    }

    float deterministicUnitRandom(int id, int salt) const
    {
        float value = std::sin(static_cast<float>((id + 1) * 73 + salt * 193) * 12.9898f) * 43758.5453f;
        return value - std::floor(value);
    }

    BuildingType randomBuildingType(int plotId) const
    {
        return deterministicUnitRandom(plotId, 4) < 0.5f ? BuildingType::TypeA : BuildingType::TypeB;
    }

    float randomTypeABuildingWidthMeters(int plotId) const
    {
        float t = deterministicUnitRandom(plotId, 1);
        return m_typeAMinWidthMeters + (m_typeAMaxWidthMeters - m_typeAMinWidthMeters) * t;
    }

    float randomTypeAEdgeLengthFraction(int plotId) const
    {
        static const float edgeBins[] = { 0.25f, 0.40f, 0.55f, 0.70f, 1.0f };
        int bin = std::abs(plotId) % 5;
        return sanitizeTypeAEdgeLengthFraction(edgeBins[bin]);
    }

    float sanitizeTypeAEdgeLengthFraction(float value) const
    {
        return std::clamp(value, 0.25f, 1.0f);
    }

    float randomTypeBXFraction(int plotId) const
    {
        return 0.25f + deterministicUnitRandom(plotId, 3) * 0.5f;
    }

    float randomTypeBInternalEdgeFraction(int plotId) const
    {
        return deterministicUnitRandom(plotId, 5) * 0.5f;
    }

    float randomTypeCEdgeFraction(int plotId) const
    {
        return 0.5f + deterministicUnitRandom(plotId, 6) * 0.5f;
    }

    void buildTypeACenterlineGraphs()
    {
        const float minWidth = metersToModelUnits(m_typeAMinWidthMeters);

        int graphEdges = 0;
        for (auto& plotData : m_plots) {
            float width = metersToModelUnits(plotData.typeABuildingWidthMeters);
            if (width < minWidth || width <= 1e-6f) continue;

            plotData.buildCenterlineGraph(
                metersToModelUnits(m_typeARoadSetbackMeters),
                metersToModelUnits(m_typeALocalSetbackMeters),
                width,
                primaryStreetWidth() * 0.5f,
                secondaryStreetWidth() * 0.5f,
                tertiaryStreetWidth() * 0.5f,
                m_massingZ + 0.003f
            );
            graphEdges += static_cast<int>(plotData.centerlineGraphEdges.size());
        }

        std::cout << "[URBAN CODEX LOOP] Type A plot centerline graph edges: " << graphEdges
                  << " | width range " << m_typeAMinWidthMeters << "-" << m_typeAMaxWidthMeters << "m"
                  << " | road setback " << m_typeARoadSetbackMeters << "m"
                  << " | local setback " << m_typeALocalSetbackMeters << "m" << std::endl;
    }

    void buildTypeBCenterlineGraphs()
    {
        int graphCount = 0;
        int graphEdges = 0;
        for (auto& plotData : m_plots) {
            if (plotData.typeBBlendWeight <= 0.001f) continue;

            plotData.typeBYFraction = 1.0f - plotData.typeBXFraction;
            plotData.buildTypeBSGraph(
                plotData.typeBXFraction,
                plotData.typeBInternalEdgeFraction,
                plotData.typeBOrientationIndex,
                m_massingZ + 0.009f
            );
            graphCount++;
            graphEdges += static_cast<int>(plotData.typeBGraphSegments.size());
        }

        std::cout << "[URBAN CODEX LOOP] Type B S graphs: " << graphCount
                  << " | graph edges: " << graphEdges
                  << " | X random range 0.25-0.75"
                  << " | Y = 1 - X"
                  << " | internal edge random range 0.0-0.5" << std::endl;
    }

    void buildTypeCCenterlineGraphs()
    {
        int graphCount = 0;
        int graphEdges = 0;
        for (auto& plotData : m_plots) {
            if (plotData.typeCBlendWeight <= 0.001f) continue;

            plotData.buildTypeCParallelGraph(
                plotData.typeCEdgeFraction,
                plotData.typeCOrientationIndex,
                m_massingZ + 0.009f
            );
            graphCount++;
            graphEdges += static_cast<int>(plotData.typeCGraphSegments.size());
        }

        std::cout << "[URBAN CODEX LOOP] Type C parallel graphs: " << graphCount
                  << " | graph edges: " << graphEdges
                  << " | edge random range 0.5-1.0" << std::endl;
    }

    void buildEffectiveTypologyGraphs()
    {
        int graphCount = 0;
        int graphEdges = 0;
        for (auto& plotData : m_plots) {
            plotData.buildEffectiveGraph(m_massingZ + 0.011f);
            if (plotData.effectiveGraphSegments.empty()) continue;

            graphCount++;
            graphEdges += static_cast<int>(plotData.effectiveGraphSegments.size());
        }

        std::cout << "[URBAN CODEX LOOP] Effective typology transport graphs: " << graphCount
                  << " | graph edges: " << graphEdges
                  << " | shared parametric topology" << std::endl;
    }

    void buildTypeASdfField()
    {
        buildTypeASdfPrimitives();

        Vec3 bMin = toVec3(m_boundsMin);
        Vec3 bMax = toVec3(m_boundsMax);
        Vec3 span = bMax - bMin;
        float pad = std::max(span.x, span.y) * 0.05f;
        zSpace::zPoint fieldMin(bMin.x - pad, bMin.y - pad, 0.0f);
        zSpace::zPoint fieldMax(bMax.x + pad, bMax.y + pad, 0.0f);

        zSpace::zFnMeshScalarField fn(m_typeASdfField);
        fn.create(fieldMin, fieldMax, m_streetFieldResolution, m_streetFieldResolution, 1, true, false);

        zSpace::zPointArray positions;
        fn.getPositions(positions);

        zSpace::zScalarArray values;
        values.reserve(positions.size());
        for (const auto& p : positions) {
            values.push_back(typeASdf(toVec3(p)));
        }

        fn.setFieldValues(values, zSpace::zFieldColorType::zFieldSDF, metersToModelUnits(m_typeAMaxWidthMeters));
        fn.updateColors(zSpace::zFieldColorType::zFieldSDF, metersToModelUnits(m_typeAMaxWidthMeters));
        fn.getIsocontour(m_typeAIsoContour, 0.0f);
        liftTypeAIsoGeometry();
    }

    void buildTypeBSdfField()
    {
        Vec3 bMin = toVec3(m_boundsMin);
        Vec3 bMax = toVec3(m_boundsMax);
        Vec3 span = bMax - bMin;
        float pad = std::max(span.x, span.y) * 0.05f;
        zSpace::zPoint fieldMin(bMin.x - pad, bMin.y - pad, 0.0f);
        zSpace::zPoint fieldMax(bMax.x + pad, bMax.y + pad, 0.0f);

        zSpace::zFnMeshScalarField fn(m_typeBSdfField);
        fn.create(fieldMin, fieldMax, m_streetFieldResolution, m_streetFieldResolution, 1, true, false);

        zSpace::zPointArray positions;
        fn.getPositions(positions);

        zSpace::zScalarArray values(positions.size(), 1e9f);
        for (const auto& plotData : m_plots) {
            if (plotData.effectiveGraphSegments.empty()) continue;

            const float buildingWidth = metersToModelUnits(plotData.typeABuildingWidthMeters);
            const float edgeHalfDepth = buildingWidth * 0.5f;
            if (edgeHalfDepth <= 1e-6f) continue;

            zSpace::zScalarArray edgeValues;
            fn.getScalarsAsEdgeDistance(edgeValues, const_cast<zSpace::zObjectGraph&>(plotData.effectiveCenterlineGraph), edgeHalfDepth, false);
            if (edgeValues.size() != positions.size()) continue;

            TypeBPlotSdf plotSdf;
            addTypeBSetbackClipPlanes(plotData, plotSdf);

            zSpace::zFnGraph graphFn(const_cast<zSpace::zObjectGraph&>(plotData.effectiveCenterlineGraph));
            zSpace::zPointArray graphPositions;
            graphFn.getVertexPositions(graphPositions);

            for (size_t i = 0; i < positions.size(); ++i) {
                Vec3 sample = toVec3(positions[i]);
                float vertexSdf = graphVertexSquareSdf(sample, graphPositions, edgeHalfDepth);
                float graphSdf = std::min(static_cast<float>(edgeValues[i]), vertexSdf);
                float clipSdf = typeBSetbackClipSdf(sample, plotSdf);
                values[i] = std::min(values[i], std::max(graphSdf, clipSdf));
            }
        }

        fn.setFieldValues(values, zSpace::zFieldColorType::zFieldSDF, metersToModelUnits(m_typeAMaxWidthMeters));
        fn.updateColors(zSpace::zFieldColorType::zFieldSDF, metersToModelUnits(m_typeAMaxWidthMeters));
        fn.getIsocontour(m_typeBIsoContour, 0.0f);
        liftTypeBIsoGeometry();
    }

    void buildTypeASdfPrimitives()
    {
        m_typeASdfPlots.clear();

        for (auto& plotData : m_plots) {
            if (plotData.typeABlendWeight <= 0.001f) continue;

            const float buildingWidth = metersToModelUnits(plotData.typeABuildingWidthMeters);
            const float cornerHalfSize = buildingWidth * 0.6f;
            const float edgeHalfDepth = buildingWidth * 0.5f;
            const float edgeLengthFraction = sanitizeTypeAEdgeLengthFraction(plotData.typeAEdgeLengthFraction);
            const bool fullGraph = edgeLengthFraction >= 0.999f;

            TypeAPlotSdf plotSdf;
            addTypeASetbackClipPlanes(plotData, plotSdf);

            zSpace::zFnGraph graphFn(plotData.centerlineGraph);
            zSpace::zPointArray graphPositions;
            graphFn.getVertexPositions(graphPositions);
            if (graphPositions.empty()) continue;

            std::vector<int> cornerIndices = selectedTypeACorners(graphPositions);
            if (fullGraph) {
                cornerIndices.clear();
                for (int i = 0; i < static_cast<int>(graphPositions.size()); ++i) {
                    cornerIndices.push_back(i);
                }
            }

            for (int cornerIndex : cornerIndices) {
                if (cornerIndex < 0 || cornerIndex >= static_cast<int>(graphPositions.size())) continue;
                plotSdf.sdfA.push_back({
                    toVec3(graphPositions[cornerIndex]),
                    Vec3(1.0f, 0.0f, 0.0f),
                    Vec3(0.0f, 1.0f, 0.0f),
                    cornerHalfSize,
                    cornerHalfSize
                });
            }

            if (fullGraph) {
                for (const auto& edge : plotData.centerlineGraphEdges) {
                    if (edge.startVertexIndex < 0 || edge.endVertexIndex < 0) continue;
                    if (edge.startVertexIndex >= static_cast<int>(graphPositions.size())) continue;
                    if (edge.endVertexIndex >= static_cast<int>(graphPositions.size())) continue;

                    addTypeAEdgeStrip(
                        plotSdf,
                        toVec3(graphPositions[edge.startVertexIndex]),
                        toVec3(graphPositions[edge.endVertexIndex]),
                        1.0f,
                        edgeHalfDepth
                    );
                }
                m_typeASdfPlots.push_back(plotSdf);
                continue;
            }

            for (int cornerIndex : cornerIndices) {
                addTypeAIncidentEdgeStrips(plotData, plotSdf, graphPositions, cornerIndex, edgeLengthFraction, edgeHalfDepth);
            }

            if (!plotSdf.sdfA.empty() || !plotSdf.sdfB.empty()) {
                m_typeASdfPlots.push_back(plotSdf);
            }
        }
    }

    void addTypeASetbackClipPlanes(const plot& plotData, TypeAPlotSdf& plotSdf) const
    {
        for (const auto& edge : plotData.boundaryEdges) {
            Vec3 edgeVector = edge.b - edge.a;
            float edgeLength = std::sqrt(edgeVector.x * edgeVector.x + edgeVector.y * edgeVector.y);
            float clearance = setbackClearanceForTypeABoundary(edge.boundaryType);
            if (edgeLength < 1e-6f || clearance <= 1e-6f) continue;

            Vec3 tangent = normalized2d(edgeVector);
            Vec3 normalA(-tangent.y, tangent.x, 0.0f);
            Vec3 midpoint = (edge.a + edge.b) * 0.5f;
            Vec3 inwardNormal = dot2d(plotData.center - midpoint, normalA) >= 0.0f ? normalA : normalA * -1.0f;
            plotSdf.setbackPlanes.push_back({ edge.a + inwardNormal * clearance, inwardNormal });
        }
    }

    void buildTypeBSdfPrimitives()
    {
        m_typeBSdfPlots.clear();

        for (auto& plotData : m_plots) {
            if (plotData.effectiveGraphSegments.empty()) continue;

            const float buildingWidth = metersToModelUnits(plotData.typeABuildingWidthMeters);
            const float edgeHalfDepth = buildingWidth * 0.5f;
            if (edgeHalfDepth <= 1e-6f) continue;

            addTypeBGraphSdf(plotData, plotData.effectiveGraphSegments, edgeHalfDepth);
        }
    }

    void addTypeBGraphSdf(
        const plot& plotData,
        const std::vector<plot::TypeBGraphSegment>& sourceSegments,
        float graphHalfWidth
    )
    {
        if (sourceSegments.empty() || graphHalfWidth <= 1e-6f) return;

        TypeBPlotSdf plotSdf;
        addTypeBSetbackClipPlanes(plotData, plotSdf);
        plotSdf.graphHalfWidth = graphHalfWidth;
        plotSdf.graphSegments.reserve(sourceSegments.size());
        plotSdf.graphJointPoints.reserve(sourceSegments.size() * 2);
        addGraphSegmentsToSdf(sourceSegments, plotSdf);

        if (!plotSdf.graphSegments.empty()) {
            m_typeBSdfPlots.push_back(plotSdf);
        }
    }

    void addGraphSegmentsToSdf(const std::vector<plot::TypeBGraphSegment>& sourceSegments, TypeBPlotSdf& plotSdf) const
    {
        for (const auto& graphSegment : sourceSegments) {
            plotSdf.graphSegments.push_back({ graphSegment.start, graphSegment.end });
            plotSdf.graphJointPoints.push_back(graphSegment.start);
            plotSdf.graphJointPoints.push_back(graphSegment.end);
        }
    }

    float graphVertexSquareSdf(const Vec3& p, const zSpace::zPointArray& graphPositions, float halfSize) const
    {
        float d = 1e9f;
        Vec3 axisX(1.0f, 0.0f, 0.0f);
        Vec3 axisY(0.0f, 1.0f, 0.0f);
        for (const auto& graphPosition : graphPositions) {
            d = std::min(d, orientedBoxSdf(p, toVec3(graphPosition), axisX, axisY, halfSize, halfSize));
        }
        return d;
    }

    float graphSegmentCost(const plot::TypeBGraphSegment& a, const plot::TypeBGraphSegment& b) const
    {
        Vec3 aMid = (a.start + a.end) * 0.5f;
        Vec3 bMid = (b.start + b.end) * 0.5f;
        Vec3 aVector = a.end - a.start;
        Vec3 bVector = b.end - b.start;
        float aLength = std::max(aVector.length(), 1e-6f);
        float bLength = std::max(bVector.length(), 1e-6f);
        Vec3 aDir = aVector * (1.0f / aLength);
        Vec3 bDir = bVector * (1.0f / bLength);
        float directionCost = 1.0f - std::abs(dot2d(aDir, bDir));
        float meanLength = (aLength + bLength) * 0.5f;
        return (aMid - bMid).length() + std::abs(aLength - bLength) * 0.25f + directionCost * meanLength * 0.15f;
    }

    float averageSegmentLength(const std::vector<plot::TypeBGraphSegment>& segments) const
    {
        if (segments.empty()) return 1.0f;

        float totalLength = 0.0f;
        for (const auto& segment : segments) {
            totalLength += (segment.end - segment.start).length();
        }
        return totalLength / static_cast<float>(segments.size());
    }

    void addTypeBSetbackClipPlanes(const plot& plotData, TypeBPlotSdf& plotSdf) const
    {
        for (const auto& edge : plotData.boundaryEdges) {
            Vec3 edgeVector = edge.b - edge.a;
            float edgeLength = std::sqrt(edgeVector.x * edgeVector.x + edgeVector.y * edgeVector.y);
            float clearance = setbackClearanceForTypeABoundary(edge.boundaryType);
            if (edgeLength < 1e-6f || clearance <= 1e-6f) continue;

            Vec3 tangent = normalized2d(edgeVector);
            Vec3 normalA(-tangent.y, tangent.x, 0.0f);
            Vec3 midpoint = (edge.a + edge.b) * 0.5f;
            Vec3 inwardNormal = dot2d(plotData.center - midpoint, normalA) >= 0.0f ? normalA : normalA * -1.0f;
            plotSdf.setbackPlanes.push_back({ edge.a + inwardNormal * clearance, inwardNormal });
        }
    }

    std::vector<int> selectedTypeACorners(const zSpace::zPointArray& graphPositions) const
    {
        std::vector<int> result;
        if (graphPositions.empty()) return result;

        result.push_back(0);
        if (graphPositions.size() == 1) return result;

        int opposite = static_cast<int>(graphPositions.size() / 2);
        if (opposite == 0) opposite = 1;
        result.push_back(opposite);
        return result;
    }

    void addTypeAIncidentEdgeStrips(
        const plot& plotData,
        TypeAPlotSdf& plotSdf,
        const zSpace::zPointArray& graphPositions,
        int cornerIndex,
        float edgeLengthFraction,
        float edgeHalfDepth
    )
    {
        if (edgeLengthFraction <= 0.0f) return;

        for (const auto& edge : plotData.centerlineGraphEdges) {
            int otherIndex = -1;
            if (edge.startVertexIndex == cornerIndex) {
                otherIndex = edge.endVertexIndex;
            }
            else if (edge.endVertexIndex == cornerIndex) {
                otherIndex = edge.startVertexIndex;
            }
            else {
                continue;
            }

            if (otherIndex < 0 || otherIndex >= static_cast<int>(graphPositions.size())) continue;
            addTypeAEdgeStrip(
                plotSdf,
                toVec3(graphPositions[cornerIndex]),
                toVec3(graphPositions[otherIndex]),
                edgeLengthFraction,
                edgeHalfDepth
            );
        }
    }

    void addTypeAEdgeStrip(TypeAPlotSdf& plotSdf, const Vec3& start, const Vec3& end, float lengthFraction, float edgeHalfDepth)
    {
        Vec3 edgeVector = end - start;
        float edgeLength = std::sqrt(edgeVector.x * edgeVector.x + edgeVector.y * edgeVector.y);
        if (edgeLength < 1e-6f || lengthFraction <= 0.0f) return;

        Vec3 tangent = normalized2d(edgeVector);
        Vec3 normal(-tangent.y, tangent.x, 0.0f);
        float segmentLength = edgeLength * saturate(lengthFraction);
        Vec3 center = start + tangent * (segmentLength * 0.5f);
        plotSdf.sdfB.push_back({ center, tangent, normal, segmentLength * 0.5f, edgeHalfDepth });
    }

    void liftTypeAIsoGeometry()
    {
        zSpace::zFnGraph contourFn(m_typeAIsoContour);
        zSpace::zPointArray contourPositions;
        contourFn.getVertexPositions(contourPositions);
        for (auto& p : contourPositions) {
            p.z = m_massingZ + 0.006f;
        }
        if (!contourPositions.empty()) {
            contourFn.setVertexPositions(contourPositions);
        }
    }

    void liftTypeBIsoGeometry()
    {
        zSpace::zFnGraph contourFn(m_typeBIsoContour);
        zSpace::zPointArray contourPositions;
        contourFn.getVertexPositions(contourPositions);
        for (auto& p : contourPositions) {
            p.z = m_massingZ + 0.007f;
        }
        if (!contourPositions.empty()) {
            contourFn.setVertexPositions(contourPositions);
        }
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
        return metersToModelUnits(std::max(0.0f, m_p));
    }

    float secondaryStreetWidth() const
    {
        return primaryStreetWidth() * (2.0f / 3.0f);
    }

    float tertiaryStreetWidth() const
    {
        return primaryStreetWidth() * (1.0f / 3.0f);
    }

    float setbackForTypeABoundary(PlotBoundaryType boundaryType) const
    {
        switch (boundaryType) {
            case PlotBoundaryType::PrimaryRoad:
            case PlotBoundaryType::SecondaryRoad:
                return metersToModelUnits(m_typeARoadSetbackMeters);
            case PlotBoundaryType::TertiaryRoad:
            case PlotBoundaryType::PlotSplitLine:
                return metersToModelUnits(m_typeALocalSetbackMeters);
        }
        return metersToModelUnits(m_typeALocalSetbackMeters);
    }

    float roadHalfWidthForTypeABoundary(PlotBoundaryType boundaryType) const
    {
        switch (boundaryType) {
            case PlotBoundaryType::PrimaryRoad: return primaryStreetWidth() * 0.5f;
            case PlotBoundaryType::SecondaryRoad: return secondaryStreetWidth() * 0.5f;
            case PlotBoundaryType::TertiaryRoad: return tertiaryStreetWidth() * 0.5f;
            case PlotBoundaryType::PlotSplitLine: return 0.0f;
        }
        return 0.0f;
    }

    float setbackClearanceForTypeABoundary(PlotBoundaryType boundaryType) const
    {
        return setbackForTypeABoundary(boundaryType) + roadHalfWidthForTypeABoundary(boundaryType);
    }

    float typeASdf(const Vec3& p) const
    {
        float d = 1e9f;

        for (const auto& plotSdf : m_typeASdfPlots) {
            float ab = 1e9f;
            for (const auto& box : plotSdf.sdfA) {
                ab = std::min(ab, orientedBoxSdf(p, box.center, box.axisX, box.axisY, box.halfX, box.halfY));
            }
            for (const auto& box : plotSdf.sdfB) {
                ab = std::min(ab, orientedBoxSdf(p, box.center, box.axisX, box.axisY, box.halfX, box.halfY));
            }

            float clip = typeASetbackClipSdf(p, plotSdf);
            d = std::min(d, std::max(ab, clip));
        }

        return d;
    }

    float typeASetbackClipSdf(const Vec3& p, const TypeAPlotSdf& plotSdf) const
    {
        float clip = -1e9f;
        for (const auto& plane : plotSdf.setbackPlanes) {
            float outsideOffsetBoundary = -dot2d(p - plane.point, plane.inwardNormal);
            clip = std::max(clip, outsideOffsetBoundary);
        }

        return clip;
    }

    float typeBSdf(const Vec3& p) const
    {
        float d = 1e9f;

        for (const auto& plotSdf : m_typeBSdfPlots) {
            float a = wholeGraphOffsetSdf(p, plotSdf.graphSegments, plotSdf.graphJointPoints, plotSdf.graphHalfWidth);

            float clip = typeBSetbackClipSdf(p, plotSdf);
            d = std::min(d, std::max(a, clip));
        }

        return d;
    }

    float typeBSetbackClipSdf(const Vec3& p, const TypeBPlotSdf& plotSdf) const
    {
        float clip = -1e9f;
        for (const auto& plane : plotSdf.setbackPlanes) {
            float outsideOffsetBoundary = -dot2d(p - plane.point, plane.inwardNormal);
            clip = std::max(clip, outsideOffsetBoundary);
        }

        return clip;
    }

    float wholeGraphOffsetSdf(
        const Vec3& p,
        const std::vector<std::pair<Vec3, Vec3>>& graphSegments,
        const std::vector<Vec3>& graphJointPoints,
        float halfWidth
    ) const
    {
        if (graphSegments.empty()) return 1e9f;

        float d = 1e9f;
        for (const auto& segment : graphSegments) {
            Vec3 edgeVector = segment.second - segment.first;
            float edgeLength = std::sqrt(edgeVector.x * edgeVector.x + edgeVector.y * edgeVector.y);
            if (edgeLength < 1e-6f) continue;

            Vec3 tangent = normalized2d(edgeVector);
            Vec3 normal(-tangent.y, tangent.x, 0.0f);
            Vec3 center = (segment.first + segment.second) * 0.5f;
            d = std::min(d, orientedBoxSdf(p, center, tangent, normal, edgeLength * 0.5f, halfWidth));
        }

        for (const auto& jointPoint : graphJointPoints) {
            Vec3 axisX(1.0f, 0.0f, 0.0f);
            for (const auto& segment : graphSegments) {
                if ((segment.first - jointPoint).length() < 1e-6f) {
                    axisX = normalized2d(segment.second - segment.first);
                    break;
                }
                if ((segment.second - jointPoint).length() < 1e-6f) {
                    axisX = normalized2d(segment.second - segment.first);
                    break;
                }
            }

            Vec3 axisY(-axisX.y, axisX.x, 0.0f);
            d = std::min(d, orientedBoxSdf(p, jointPoint, axisX, axisY, halfWidth, halfWidth));
        }

        return d;
    }

    float orientedBoxSdf(const Vec3& p, const Vec3& center, const Vec3& axisX, const Vec3& axisY, float halfX, float halfY) const
    {
        Vec3 rel = p - center;
        float qx = std::abs(dot2d(rel, axisX)) - halfX;
        float qy = std::abs(dot2d(rel, axisY)) - halfY;
        float ox = std::max(qx, 0.0f);
        float oy = std::max(qy, 0.0f);
        float outside = std::sqrt(ox * ox + oy * oy);
        float inside = std::min(std::max(qx, qy), 0.0f);
        return outside + inside;
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
        (void)renderer;
        const Color graphColor(0.0f, 0.0f, 0.0f, 1.0f);
        zDisplayGraphSetting graphDisplay;
        graphDisplay.showEdges = true;
        graphDisplay.showVertices = true;
        graphDisplay.edgeColor = graphColor;
        graphDisplay.vertexColor = graphColor;
        graphDisplay.edgeWidth = 1.0f;
        graphDisplay.vertexSize = 5.0f;

        for (auto& plotData : m_plots) {
            if (plotData.typeABlendWeight <= 0.001f) continue;
            scene().draw(plotData.centerlineGraph, graphDisplay);
        }
    }

    void drawCenterlineGraphDebug(Renderer& renderer)
    {
        (void)renderer;
        zDisplayGraphSetting graphDisplay;
        graphDisplay.showEdges = true;
        graphDisplay.showVertices = true;
        graphDisplay.drawVertexIds = true;
        graphDisplay.edgeColor = Color(0.15f, 0.15f, 0.15f, 1.0f);
        graphDisplay.vertexColor = Color(0.0f, 0.0f, 0.0f, 1.0f);
        graphDisplay.vertexIdColor = Color(0.75f, 0.0f, 0.35f, 1.0f);
        graphDisplay.edgeWidth = 1.0f;
        graphDisplay.vertexSize = 5.0f;
        graphDisplay.vertexIdSize = 0.18f;

        for (auto& plotData : m_plots) {
            scene().draw(plotData.centerlineGraph, graphDisplay);
        }
    }

    void drawTypeBCenterlineGraphs(Renderer& renderer)
    {
        (void)renderer;
        const Color graphColor(0.0f, 0.0f, 0.0f, 1.0f);
        zDisplayGraphSetting graphDisplay;
        graphDisplay.showEdges = true;
        graphDisplay.showVertices = true;
        graphDisplay.edgeColor = graphColor;
        graphDisplay.vertexColor = graphColor;
        graphDisplay.edgeWidth = 1.0f;
        graphDisplay.vertexSize = 6.0f;

        for (auto& plotData : m_plots) {
            if (plotData.typeBBlendWeight <= 0.001f) continue;
            scene().draw(plotData.typeBCenterlineGraph, graphDisplay);
        }
    }

    void drawTypeCCenterlineGraphs(Renderer& renderer)
    {
        (void)renderer;
        const Color graphColor(0.0f, 0.0f, 0.0f, 1.0f);
        zDisplayGraphSetting graphDisplay;
        graphDisplay.showEdges = true;
        graphDisplay.showVertices = true;
        graphDisplay.edgeColor = graphColor;
        graphDisplay.vertexColor = graphColor;
        graphDisplay.edgeWidth = 1.0f;
        graphDisplay.vertexSize = 6.0f;

        for (auto& plotData : m_plots) {
            if (plotData.typeCBlendWeight <= 0.001f) continue;
            scene().draw(plotData.typeCCenterlineGraph, graphDisplay);
        }
    }

    void drawEffectiveTypologyGraphs(Renderer& renderer)
    {
        (void)renderer;
        const Color graphColor(0.0f, 0.0f, 0.0f, 1.0f);
        zDisplayGraphSetting graphDisplay;
        graphDisplay.showEdges = true;
        graphDisplay.showVertices = true;
        graphDisplay.edgeColor = graphColor;
        graphDisplay.vertexColor = graphColor;
        graphDisplay.edgeWidth = 1.0f;
        graphDisplay.vertexSize = 6.0f;

        for (auto& plotData : m_plots) {
            if (plotData.effectiveGraphSegments.empty()) continue;
            scene().draw(plotData.effectiveCenterlineGraph, graphDisplay);
        }
    }

    void drawTypeASdfContour(Renderer& renderer)
    {
        (void)renderer;
        zDisplayGraphSetting contourDisplay;
        contourDisplay.showEdges = true;
        contourDisplay.showVertices = false;
        contourDisplay.edgeColor = Color(1.0f, 0.0f, 1.0f, 1.0f);
        contourDisplay.edgeWidth = 3.0f;
        scene().draw(m_typeAIsoContour, contourDisplay);
    }

    void drawTypeBSdfContour(Renderer& renderer)
    {
        (void)renderer;
        zDisplayGraphSetting contourDisplay;
        contourDisplay.showEdges = true;
        contourDisplay.showVertices = false;
        contourDisplay.edgeColor = Color(1.0f, 0.0f, 1.0f, 1.0f);
        contourDisplay.edgeWidth = 3.0f;
        scene().draw(m_typeBIsoContour, contourDisplay);
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
        contourDisplay.edgeColor = Color(0.5f, 0.5f, 0.5f, 1.0f);
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
