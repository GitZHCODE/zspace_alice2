#define __MAIN__
#ifdef __MAIN__

#include <zspace/interface.h>
#include <zspace/io.h>

#include <alice2.h>
#include <sketches/SketchRegistry.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

using namespace alice2;

class zSpaceUrbanBlendSketch : public ISketch {
public:
    std::string getName() const override { return "zSpace Urban Codex Loop"; }
    std::string getDescription() const override { return "Clean minimal urban streets and buildings sketch for Codex/VLM critique iterations."; }
    std::string getAuthor() const override { return "Codex + alice2 + zspace_core"; }

    void setup() override
    {
        scene().setBackgroundColor(Color(1.0f, 1.0f, 1.0f, 1.0f));
        scene().setShowGrid(false);
        scene().setShowAxes(false);

        m_ui = std::make_unique<SimpleUI>(input());
        m_ui->setTheme(SimpleUI::UITheme::Dark);
        m_ui->addSlider("p", Vec2{14.0f, 82.0f}, 240.0f, 0.0f, 100.0f, m_p);
        m_ui->addSlider("minW", Vec2{14.0f, 112.0f}, 240.0f, 1.0f, 60.0f, m_typeAMinWidthMeters);
        m_ui->addSlider("maxW", Vec2{14.0f, 142.0f}, 240.0f, 1.0f, 80.0f, m_typeAMaxWidthMeters);
        m_ui->addToggle("Base Mesh", UIRect{14.0f, 172.0f, 130.0f, 24.0f}, m_drawBaseMesh);
        m_ui->addToggle("Height Map", UIRect{14.0f, 202.0f, 130.0f, 24.0f}, m_drawHeightFieldMap);
        m_ui->addToggle("Field Mesh", UIRect{14.0f, 232.0f, 130.0f, 24.0f}, m_drawStreetFieldMesh);

        loadMesh();
        if (!m_loaded) return;

        std::cout << "[URBAN BLEND] Clean streets and buildings sketch loaded." << std::endl;
        std::cout << "[URBAN BLEND] Faces: " << zSpace::zFnMesh(m_mesh).numPolygons() << std::endl;
    }

    void update(float) override
    {
        if (!m_loaded || m_screenshotTaken) return;

        sanitizeBuildingWidthControls();
        if (std::abs(m_p - m_lastBuiltP) > 0.001f ||
            std::abs(m_typeAMinWidthMeters - m_lastBuiltMinWidthMeters) > 0.001f ||
            std::abs(m_typeAMaxWidthMeters - m_lastBuiltMaxWidthMeters) > 0.001f) {
            zSpace::zFnMesh fn(m_mesh);
            rebuildUrbanModel(fn);
        }

        m_frameCount++;
        if (m_frameCount == 10) {
            setPlanCamera();
        }
        else if (m_autoCapture && m_frameCount > 30) {
            Application::getInstance()->takeScreenshot();
            m_screenshotTaken = true;
            std::cout << "[URBAN BLEND] Screenshot captured. Exiting." << std::endl;
            exit(0);
        }
    }

    void draw(Renderer& renderer, Camera&) override
    {
        if (!m_loaded) return;

        zSpace::zFnMesh fn(m_mesh);
        if (m_drawBaseMesh) {
            drawNeutralBaseMesh(renderer, fn);
        }
        if (m_drawHeightFieldMap) {
            drawHeightFieldMap(renderer, fn);
        }
        drawStreetSdfGeometry(renderer);
        drawBuildingIsoMeshes(renderer);
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
            std::cout << "[URBAN BLEND] Manual screenshot captured. Exiting." << std::endl;
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
    bool m_drawBaseMesh = true;
    bool m_drawHeightFieldMap = false;
    bool m_drawStreetFieldMesh = false;
    int m_frameCount = 0;

    zSpace::zPoint m_boundsMin;
    zSpace::zPoint m_boundsMax;
    Vec3 m_plotCenterMin;
    Vec3 m_plotCenterMax;

    float m_buildingZ = 0.004f;
    float m_baseZ = -0.004f;
    float m_typeAMinWidthMeters = 15.0f;
    float m_typeAMaxWidthMeters = 25.0f;
    float m_lastBuiltMinWidthMeters = -1.0f;
    float m_lastBuiltMaxWidthMeters = -1.0f;
    float m_typeARoadSetbackMeters = 5.0f;
    float m_typeALocalSetbackMeters = 2.0f;
    float m_p = 12.0f;
    float m_lastBuiltP = -1.0f;
    float m_modelUnitsPerMeter = 1.0f;
    float m_globalParameterScale = 0.1f;
    int m_streetFieldResolution = 320;
    float m_buildingSdfCellSizeMeters = 1.5f;
    float m_buildingSdfCellSizeModelUnits = 0.0f;
    int m_buildingSdfSamplesPerInputCell = 96;
    int m_buildingSdfMinResolution = 32;
    int m_buildingSdfMaxResolution = 512;

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
        TypeC,
        TypeD
    };

    enum class PlotUse {
        Building,
        Green
    };

    struct ShapeParams {
        float typeAWeight = 1.0f;
        float typeBWeight = 0.0f;
        float typeCWeight = 0.0f;
        float typeDWeight = 0.0f;
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
        int plotId = -1;
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

    struct TypeASetbackPlane {
        Vec3 point;
        Vec3 inwardNormal;
    };

    struct TypeBPlotSdf {
        std::vector<std::pair<Vec3, Vec3>> graphSegments;
        std::vector<Vec3> graphJointPoints;
        float graphHalfWidth = 0.0f;
        bool usePolygonSdf = false;
        std::vector<Vec3> polygonVertices;
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
        PlotUse plotUse = PlotUse::Building;
        Vec3 center;
        float typeABlendWeight = 1.0f;
        float typeBBlendWeight = 0.0f;
        float typeCBlendWeight = 0.0f;
        float typeDBlendWeight = 0.0f;
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
            totalWeight += typeDBlendWeight;
            if (totalWeight <= 0.001f) return;

            float aWeight = typeABlendWeight / totalWeight;
            float bWeight = typeBBlendWeight / totalWeight;
            float cWeight = typeCBlendWeight / totalWeight;
            float dWeight = typeDBlendWeight / totalWeight;

            if (dWeight >= aWeight && dWeight >= bWeight && dWeight >= cWeight) {
                effectiveGraphSegments = makeTypeDOverlaySegments(p);
                createGraphFromSegments(effectiveGraphSegments, effectiveCenterlineGraph, graphZ);
                return;
            }

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

        static std::vector<TypeBGraphSegment> makeTypeDOverlaySegments(const std::vector<Vec3>& p)
        {
            std::vector<TypeBGraphSegment> segments;
            const int n = static_cast<int>(p.size());
            if (n < 3) return segments;

            segments.reserve(n);
            for (int i = 0; i < n; ++i) {
                segments.push_back({ p[i], p[(i + 1) % n] });
            }

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
    std::vector<std::vector<std::pair<int, float>>> m_plotAdjacency;
    std::vector<TypologyAnchor> m_typologyAnchors;
    std::vector<std::vector<float>> m_typologyAnchorDistances;
    zSpace::zObjectMeshScalarField m_streetSdfField;
    zSpace::zObjectGraph m_streetIsoContour;
    std::vector<zSpace::zObjectMesh> m_buildingIsoMeshes;
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

    float streetOffsetSdf(const Vec3& p) const
    {
        float d = 1e9f;
        for (const auto& edge : m_streetEdges) {
            d = std::min(d, distanceToSegment2d(p, edge.a, edge.b) - edge.offsetWidth);
        }
        return d;
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
        Vec3 bMin = toVec3(m_boundsMin);
        Vec3 bMax = toVec3(m_boundsMax);
        m_modelUnitsPerMeter = 1.0f;
        std::cout << "[URBAN BLEND] Input units treated as meters | model units per meter: "
                  << m_modelUnitsPerMeter * m_globalParameterScale << std::endl;
        initializeTypologyAnchors(bMin, bMax);

        sanitizeBuildingWidthControls();
        rebuildUrbanModel(fn);

        m_loaded = true;
    }

    void rebuildUrbanModel(zSpace::zFnMesh& fn)
    {
        buildStreetEdges(fn);
        buildPlotRecords(fn);
        buildStreetSdfField();
        buildTypeACenterlineGraphs();
        buildTypeBCenterlineGraphs();
        buildTypeCCenterlineGraphs();
        buildEffectiveTypologyGraphs();
        buildBuildingIsoMeshes();
        m_lastBuiltP = m_p;
        m_lastBuiltMinWidthMeters = m_typeAMinWidthMeters;
        m_lastBuiltMaxWidthMeters = m_typeAMaxWidthMeters;
    }

    void sanitizeBuildingWidthControls()
    {
        m_typeAMinWidthMeters = std::clamp(m_typeAMinWidthMeters, 1.0f, 80.0f);
        m_typeAMaxWidthMeters = std::clamp(m_typeAMaxWidthMeters, 1.0f, 80.0f);
        if (m_typeAMaxWidthMeters < m_typeAMinWidthMeters + 0.5f) {
            m_typeAMaxWidthMeters = std::min(80.0f, m_typeAMinWidthMeters + 0.5f);
        }
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

        ShapeParams topLeftC;
        topLeftC.typeAWeight = 0.0f;
        topLeftC.typeBWeight = 0.0f;
        topLeftC.typeCWeight = 1.0f;
        topLeftC.typeDWeight = 0.0f;
        topLeftC.buildingWidthMeters = 18.0f;
        topLeftC.typeCEdgeFraction = 1.0f;
        topLeftC.typeCOrientationIndex = 1.0f;

        ShapeParams bottomLeftB;
        bottomLeftB.typeAWeight = 0.0f;
        bottomLeftB.typeBWeight = 1.0f;
        bottomLeftB.typeCWeight = 0.0f;
        bottomLeftB.typeDWeight = 0.0f;
        bottomLeftB.buildingWidthMeters = 20.0f;
        bottomLeftB.typeBXFraction = 0.5f;
        bottomLeftB.typeBInternalEdgeFraction = 0.5f;
        bottomLeftB.typeBOrientationIndex = 1.0f;

        ShapeParams bottomRightA;
        bottomRightA.typeAWeight = 1.0f;
        bottomRightA.typeBWeight = 0.0f;
        bottomRightA.typeCWeight = 0.0f;
        bottomRightA.typeDWeight = 0.0f;
        bottomRightA.buildingWidthMeters = 22.0f;
        bottomRightA.typeAEdgeLengthFraction = 1.0f;

        ShapeParams topRightA;
        topRightA.typeAWeight = 1.0f;
        topRightA.typeBWeight = 0.0f;
        topRightA.typeCWeight = 0.0f;
        topRightA.typeDWeight = 0.0f;
        topRightA.buildingWidthMeters = 20.0f;
        topRightA.typeAEdgeLengthFraction = 0.65f;

        m_typologyAnchors.push_back({ topLeft, topLeftC, 1.0f, radius });
        m_typologyAnchors.push_back({ bottomLeft, bottomLeftB, 1.0f, radius });
        m_typologyAnchors.push_back({ bottomRight, bottomRightA, 1.0f, radius });
        m_typologyAnchors.push_back({ topRight, topRightA, 1.0f, radius });

        std::cout << "[URBAN BLEND] Typology anchors: " << m_typologyAnchors.size()
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
        std::cout << "[URBAN BLEND] Street edges: " << m_streetEdges.size() << " | p=" << m_p << std::endl;
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
            plotData.plotUse = PlotUse::Building;
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
            buildPlotAdjacency();
            assignTypologyAnchorPlots();
            buildTypologyAnchorDistances();
            updateBuildingSdfCellSizeFromInputGrid();
            for (auto& plotData : m_plots) {
                applyTypologyGene(plotData);
            }
        }

        std::cout << "[URBAN BLEND] Plot records: " << m_plots.size() << std::endl;
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

    void updateBuildingSdfCellSizeFromInputGrid()
    {
        std::vector<float> cellSizes;
        cellSizes.reserve(m_plots.size());

        for (const auto& plotData : m_plots) {
            float area = std::abs(signedArea2d(plotData.vertices));
            if (area <= 1e-8f) continue;
            cellSizes.push_back(std::sqrt(area));
        }

        if (cellSizes.empty()) {
            m_buildingSdfCellSizeModelUnits = std::max(metersToModelUnits(m_buildingSdfCellSizeMeters), 1e-6f);
            return;
        }

        std::sort(cellSizes.begin(), cellSizes.end());
        float referenceCellSize = cellSizes[cellSizes.size() / 2];
        m_buildingSdfCellSizeModelUnits = std::max(
            referenceCellSize / static_cast<float>(std::max(m_buildingSdfSamplesPerInputCell, 1)),
            1e-6f
        );

        std::cout << "[URBAN BLEND] Building SDF reference cell: " << referenceCellSize
                  << " model units | samples per input cell: " << m_buildingSdfSamplesPerInputCell
                  << " | SDF sample spacing: " << m_buildingSdfCellSizeModelUnits
                  << " model units" << std::endl;
    }

    void buildPlotAdjacency()
    {
        m_plotAdjacency.assign(m_plots.size(), {});

        for (int i = 0; i < static_cast<int>(m_plots.size()); ++i) {
            for (int j = i + 1; j < static_cast<int>(m_plots.size()); ++j) {
                if (!plotsShareBoundaryEdge(m_plots[i], m_plots[j])) continue;

                float cost = std::max((m_plots[i].center - m_plots[j].center).length(), 1e-6f);
                m_plotAdjacency[i].push_back({ j, cost });
                m_plotAdjacency[j].push_back({ i, cost });
            }
        }
    }

    bool plotsShareBoundaryEdge(const plot& a, const plot& b) const
    {
        for (const auto& edgeA : a.boundaryEdges) {
            for (const auto& edgeB : b.boundaryEdges) {
                if (sameUndirectedEdge(edgeA.a, edgeA.b, edgeB.a, edgeB.b)) return true;
            }
        }
        return false;
    }

    bool sameUndirectedEdge(const Vec3& a0, const Vec3& a1, const Vec3& b0, const Vec3& b1) const
    {
        const float eps = 1e-4f;
        bool same = (a0 - b0).length() < eps && (a1 - b1).length() < eps;
        bool reverse = (a0 - b1).length() < eps && (a1 - b0).length() < eps;
        return same || reverse;
    }

    void assignTypologyAnchorPlots()
    {
        if (m_plots.empty() || m_typologyAnchors.size() < 4) return;

        int bottomLeft = extremePlotByScore([](const Vec3& p) { return p.x + p.y; }, false);
        int bottomRight = extremePlotByScore([](const Vec3& p) { return p.x - p.y; }, true);
        int topLeft = extremePlotByScore([](const Vec3& p) { return p.x - p.y; }, false);
        int topRight = extremePlotByScore([](const Vec3& p) { return p.x + p.y; }, true);

        const int defaultAnchors[4] = { bottomLeft, bottomRight, topLeft, topRight };
        for (int i = 0; i < 4; ++i) {
            int plotId = std::clamp(defaultAnchors[i], 0, static_cast<int>(m_plots.size()) - 1);
            m_typologyAnchors[i].plotId = plotId;
            m_typologyAnchors[i].position = m_plots[plotId].center;
            m_plots[plotId].plotUse = PlotUse::Building;
        }

        std::cout << "[URBAN BLEND] Typology anchor plot IDs | BL: " << m_typologyAnchors[0].plotId
                  << " BR: " << m_typologyAnchors[1].plotId
                  << " TL: " << m_typologyAnchors[2].plotId
                  << " TR: " << m_typologyAnchors[3].plotId << std::endl;
    }

    template <typename ScoreFn>
    int extremePlotByScore(ScoreFn scoreFn, bool findMax) const
    {
        int bestIndex = 0;
        float bestScore = findMax ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
        for (int i = 0; i < static_cast<int>(m_plots.size()); ++i) {
            float score = scoreFn(m_plots[i].center);
            if ((findMax && score > bestScore) || (!findMax && score < bestScore)) {
                bestScore = score;
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    void buildTypologyAnchorDistances()
    {
        m_typologyAnchorDistances.clear();
        m_typologyAnchorDistances.reserve(m_typologyAnchors.size());
        for (const auto& anchor : m_typologyAnchors) {
            m_typologyAnchorDistances.push_back(shortestPlotDistances(anchor.plotId));
        }
    }

    std::vector<float> shortestPlotDistances(int sourcePlotId) const
    {
        const float infinity = std::numeric_limits<float>::max();
        std::vector<float> distances(m_plots.size(), infinity);
        if (sourcePlotId < 0 || sourcePlotId >= static_cast<int>(m_plots.size())) return distances;

        using QueueItem = std::pair<float, int>;
        std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> queue;
        distances[sourcePlotId] = 0.0f;
        queue.push({ 0.0f, sourcePlotId });

        while (!queue.empty()) {
            auto [distance, plotId] = queue.top();
            queue.pop();
            if (distance > distances[plotId]) continue;

            if (plotId < 0 || plotId >= static_cast<int>(m_plotAdjacency.size())) continue;
            for (const auto& edge : m_plotAdjacency[plotId]) {
                int neighbor = edge.first;
                float nextDistance = distance + edge.second;
                if (neighbor < 0 || neighbor >= static_cast<int>(distances.size())) continue;
                if (nextDistance >= distances[neighbor]) continue;

                distances[neighbor] = nextDistance;
                queue.push({ nextDistance, neighbor });
            }
        }

        return distances;
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

        std::cout << "[URBAN BLEND] Plot boundary edges | primary: " << primary
                  << " secondary: " << secondary
                  << " tertiary: " << tertiary
                  << " split: " << split << std::endl;
    }

    void logBuildingTypeSummary() const
    {
        int building = 0;
        int green = 0;
        int typeA = 0;
        int typeB = 0;
        int typeC = 0;
        int typeD = 0;
        for (const auto& plotData : m_plots) {
            if (plotData.plotUse == PlotUse::Green) {
                green++;
                continue;
            }

            building++;
            if (plotData.buildingType == BuildingType::TypeA) {
                typeA++;
            }
            else if (plotData.buildingType == BuildingType::TypeB) {
                typeB++;
            }
            else if (plotData.buildingType == BuildingType::TypeC) {
                typeC++;
            }
            else if (plotData.buildingType == BuildingType::TypeD) {
                typeD++;
            }
        }

        std::cout << "[URBAN BLEND] Plot use assignment | building: " << building
                  << " green: " << green << std::endl;
        std::cout << "[URBAN BLEND] Building type assignment | Type A: " << typeA
                  << " Type B: " << typeB
                  << " Type C: " << typeC
                  << " Type D: " << typeD << std::endl;
    }

    ShapeParams computeTypologyGene(const plot& plotData) const
    {
        if (m_typologyAnchors.empty()) {
            return fallbackShapeParams(plotData.id);
        }

        std::vector<float> weights(m_typologyAnchors.size(), 0.0f);
        bool hasGraphDistances = m_typologyAnchorDistances.size() == m_typologyAnchors.size();
        bool exactAnchor = false;
        const float epsilon = 1e-5f;

        if (hasGraphDistances && plotData.id >= 0 && plotData.id < static_cast<int>(m_plots.size())) {
            for (size_t i = 0; i < m_typologyAnchors.size(); ++i) {
                const auto& distances = m_typologyAnchorDistances[i];
                if (plotData.id >= static_cast<int>(distances.size())) {
                    hasGraphDistances = false;
                    break;
                }

                float distance = distances[plotData.id];
                if (distance == std::numeric_limits<float>::max()) continue;

                if (distance <= epsilon) {
                    std::fill(weights.begin(), weights.end(), 0.0f);
                    weights[i] = 1.0f;
                    exactAnchor = true;
                    break;
                }

                float influence = m_typologyAnchors[i].strength / ((distance + 0.05f) * (distance + 0.05f));
                weights[i] = influence;
            }
        }

        float totalWeight = 0.0f;
        for (float weight : weights) totalWeight += weight;
        if (hasGraphDistances && (exactAnchor || totalWeight > 1e-6f)) {
            return blendedShapeParams(weights);
        }

        weights.clear();
        weights.reserve(m_typologyAnchors.size());
        for (const auto& anchor : m_typologyAnchors) {
            float radius = std::max(anchor.radius, 1e-6f);
            float d = (plotData.center - anchor.position).length() / radius;
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
        result.typeDWeight = 0.0f;
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
            result.typeDWeight += anchor.params.typeDWeight * w;
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
        result.typeDWeight = std::clamp(result.typeDWeight / totalWeight, 0.0f, 1.0f);
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
        BuildingType randomType = randomBuildingType(plotId);
        params.typeAWeight = randomType == BuildingType::TypeA ? 1.0f : 0.0f;
        params.typeBWeight = randomType == BuildingType::TypeB ? 1.0f : 0.0f;
        params.typeCWeight = randomType == BuildingType::TypeC ? 1.0f : 0.0f;
        params.typeDWeight = randomType == BuildingType::TypeD ? 1.0f : 0.0f;
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
        ShapeParams gene = computeTypologyGene(plotData);
        float totalWeight = gene.typeAWeight + gene.typeBWeight + gene.typeCWeight + gene.typeDWeight;
        if (totalWeight <= 1e-6f) totalWeight = 1.0f;
        float typeAWeight = std::clamp(gene.typeAWeight / totalWeight, 0.0f, 1.0f);
        float typeBWeight = std::clamp(gene.typeBWeight / totalWeight, 0.0f, 1.0f);
        float typeCWeight = std::clamp(gene.typeCWeight / totalWeight, 0.0f, 1.0f);
        float typeDWeight = std::clamp(gene.typeDWeight / totalWeight, 0.0f, 1.0f);
        plotData.typeABlendWeight = typeAWeight;
        plotData.typeBBlendWeight = typeBWeight;
        plotData.typeCBlendWeight = typeCWeight;
        plotData.typeDBlendWeight = typeDWeight;
        plotData.buildingType = BuildingType::TypeA;
        if (typeDWeight >= typeAWeight && typeDWeight >= typeBWeight && typeDWeight >= typeCWeight) {
            plotData.buildingType = BuildingType::TypeD;
        }
        else if (typeBWeight >= typeAWeight && typeBWeight >= typeCWeight) {
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
        float value = deterministicUnitRandom(plotId, 4);
        if (value < 0.25f) return BuildingType::TypeA;
        if (value < 0.50f) return BuildingType::TypeB;
        if (value < 0.75f) return BuildingType::TypeC;
        return BuildingType::TypeD;
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
            if (plotData.plotUse != PlotUse::Building) continue;
            float width = metersToModelUnits(plotData.typeABuildingWidthMeters);
            if (width < minWidth || width <= 1e-6f) continue;

            plotData.buildCenterlineGraph(
                metersToModelUnits(m_typeARoadSetbackMeters),
                metersToModelUnits(m_typeALocalSetbackMeters),
                width,
                primaryStreetWidth() * 0.5f,
                secondaryStreetWidth() * 0.5f,
                tertiaryStreetWidth() * 0.5f,
                m_buildingZ + 0.003f
            );
            graphEdges += static_cast<int>(plotData.centerlineGraphEdges.size());
        }

        std::cout << "[URBAN BLEND] Type A plot centerline graph edges: " << graphEdges
                  << " | width range " << m_typeAMinWidthMeters << "-" << m_typeAMaxWidthMeters << "m"
                  << " | road setback " << m_typeARoadSetbackMeters << "m"
                  << " | local setback " << m_typeALocalSetbackMeters << "m" << std::endl;
    }

    void buildTypeBCenterlineGraphs()
    {
        int graphCount = 0;
        int graphEdges = 0;
        for (auto& plotData : m_plots) {
            if (plotData.plotUse != PlotUse::Building) continue;
            if (plotData.typeBBlendWeight <= 0.001f) continue;

            plotData.typeBYFraction = 1.0f - plotData.typeBXFraction;
            plotData.buildTypeBSGraph(
                plotData.typeBXFraction,
                plotData.typeBInternalEdgeFraction,
                plotData.typeBOrientationIndex,
                m_buildingZ + 0.009f
            );
            graphCount++;
            graphEdges += static_cast<int>(plotData.typeBGraphSegments.size());
        }

        std::cout << "[URBAN BLEND] Type B S graphs: " << graphCount
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
            if (plotData.plotUse != PlotUse::Building) continue;
            if (plotData.typeCBlendWeight <= 0.001f) continue;

            plotData.buildTypeCParallelGraph(
                plotData.typeCEdgeFraction,
                plotData.typeCOrientationIndex,
                m_buildingZ + 0.009f
            );
            graphCount++;
            graphEdges += static_cast<int>(plotData.typeCGraphSegments.size());
        }

        std::cout << "[URBAN BLEND] Type C parallel graphs: " << graphCount
                  << " | graph edges: " << graphEdges
                  << " | edge random range 0.5-1.0" << std::endl;
    }

    void buildEffectiveTypologyGraphs()
    {
        int graphCount = 0;
        int graphEdges = 0;
        for (auto& plotData : m_plots) {
            if (plotData.plotUse != PlotUse::Building) continue;
            plotData.buildEffectiveGraph(m_buildingZ + 0.011f);
            if (plotData.effectiveGraphSegments.empty()) continue;

            graphCount++;
            graphEdges += static_cast<int>(plotData.effectiveGraphSegments.size());
        }

        std::cout << "[URBAN BLEND] Effective typology transport graphs: " << graphCount
                  << " | graph edges: " << graphEdges
                  << " | shared parametric topology" << std::endl;
    }

    void buildBuildingIsoMeshes()
    {
        m_buildingIsoMeshes.clear();
        m_typeBSdfPlots.clear();

        for (auto& plotData : m_plots) {
            if (plotData.plotUse != PlotUse::Building) continue;
            if (plotData.effectiveGraphSegments.empty()) continue;

            const float buildingWidth = metersToModelUnits(plotData.typeABuildingWidthMeters);
            const float edgeHalfDepth = buildingWidth * 0.5f;
            if (edgeHalfDepth <= 1e-6f || plotData.vertices.empty()) continue;

            TypeBPlotSdf plotSdf;
            addTypeBSetbackClipPlanes(plotData, plotSdf);
            plotSdf.graphHalfWidth = edgeHalfDepth;
            plotSdf.usePolygonSdf = plotData.buildingType == BuildingType::TypeD;
            if (plotSdf.usePolygonSdf) {
                plotSdf.polygonVertices = centerlineGraphPolygon(plotData);
                if (plotSdf.polygonVertices.size() < 3) continue;
            }
            else {
                addGraphSegmentsToSdf(plotData.effectiveGraphSegments, plotSdf);
                if (plotSdf.graphSegments.empty()) continue;
            }

            Vec3 pMin = plotData.vertices[0];
            Vec3 pMax = plotData.vertices[0];
            for (const auto& p : plotData.vertices) {
                pMin.x = std::min(pMin.x, p.x);
                pMin.y = std::min(pMin.y, p.y);
                pMax.x = std::max(pMax.x, p.x);
                pMax.y = std::max(pMax.y, p.y);
            }

            float pad = edgeHalfDepth * 1.5f;
            zSpace::zPoint fieldMin(pMin.x - pad, pMin.y - pad, 0.0f);
            zSpace::zPoint fieldMax(pMax.x + pad, pMax.y + pad, 0.0f);

            zSpace::zObjectMeshScalarField plotField;
            zSpace::zFnMeshScalarField fieldFn(plotField);
            int fieldResX = buildingFieldResolution(fieldMax.x - fieldMin.x);
            int fieldResY = buildingFieldResolution(fieldMax.y - fieldMin.y);
            fieldFn.create(fieldMin, fieldMax, fieldResX, fieldResY, 1, true, false);

            zSpace::zPointArray positions;
            fieldFn.getPositions(positions);

            zSpace::zScalarArray values;
            values.reserve(positions.size());
            for (const auto& p : positions) {
                Vec3 sample = toVec3(p);
                float graphSdf = plotSdf.usePolygonSdf
                    ? polygonSdf(sample, plotSdf.polygonVertices)
                    : wholeGraphOffsetSdf(sample, plotSdf.graphSegments, plotSdf.graphJointPoints, plotSdf.graphHalfWidth);
                float clipSdf = typeBSetbackClipSdf(sample, plotSdf);
                values.push_back(std::max(graphSdf, clipSdf));
            }

            zSpace::zObjectMesh isoMesh;
            zSpace::zFnMesh fieldMeshFn(plotField);
            fieldMeshFn.getIsoMesh(values, 0.0f, true, isoMesh);
            liftBuildingIsoMesh(isoMesh);

            zSpace::zFnMesh isoFn(isoMesh);
            if (isoFn.numPolygons() <= 0) continue;

            m_buildingIsoMeshes.push_back(isoMesh);
            m_typeBSdfPlots.push_back(plotSdf);
        }

        std::cout << "[URBAN BLEND] Building iso meshes: " << m_buildingIsoMeshes.size()
                  << " | per-plot SDF spacing " << m_buildingSdfCellSizeModelUnits << " model units"
                  << " | reference samples/cell " << m_buildingSdfSamplesPerInputCell
                  << " | resolution clamp " << m_buildingSdfMinResolution
                  << "-" << m_buildingSdfMaxResolution << std::endl;
    }

    int buildingFieldResolution(float modelLength) const
    {
        float cellSize = m_buildingSdfCellSizeModelUnits > 1e-6f
            ? m_buildingSdfCellSizeModelUnits
            : std::max(metersToModelUnits(m_buildingSdfCellSizeMeters), 1e-6f);
        int resolution = static_cast<int>(std::ceil(std::max(modelLength, cellSize) / cellSize)) + 1;
        return std::clamp(resolution, m_buildingSdfMinResolution, m_buildingSdfMaxResolution);
    }

    void addGraphSegmentsToSdf(const std::vector<plot::TypeBGraphSegment>& sourceSegments, TypeBPlotSdf& plotSdf) const
    {
        for (const auto& graphSegment : sourceSegments) {
            plotSdf.graphSegments.push_back({ graphSegment.start, graphSegment.end });
            plotSdf.graphJointPoints.push_back(graphSegment.start);
            plotSdf.graphJointPoints.push_back(graphSegment.end);
        }
    }

    std::vector<Vec3> centerlineGraphPolygon(plot& plotData) const
    {
        std::vector<Vec3> polygon;
        zSpace::zFnGraph graphFn(plotData.centerlineGraph);
        zSpace::zPointArray positions;
        graphFn.getVertexPositions(positions);
        polygon.reserve(positions.size());
        for (const auto& position : positions) {
            polygon.push_back(toVec3(position));
        }
        return polygon;
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

    void liftBuildingIsoMesh(zSpace::zObjectMesh& mesh)
    {
        zSpace::zFnMesh meshFn(mesh);
        zSpace::zPointArray positions;
        meshFn.getVertexPositions(positions);
        for (auto& p : positions) {
            p.z = m_buildingZ + 0.006f;
        }
        if (!positions.empty()) {
            meshFn.setVertexPositions(positions);
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
            p.z = m_baseZ + 0.004f;
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

    float typeBSetbackClipSdf(const Vec3& p, const TypeBPlotSdf& plotSdf) const
    {
        float clip = -1e9f;
        for (const auto& plane : plotSdf.setbackPlanes) {
            float outsideOffsetBoundary = -dot2d(p - plane.point, plane.inwardNormal);
            clip = std::max(clip, outsideOffsetBoundary);
        }

        return clip;
    }

    float polygonSdf(const Vec3& p, const std::vector<Vec3>& polygon) const
    {
        if (polygon.size() < 3) return 1e9f;

        float d = 1e9f;
        bool inside = false;
        for (size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
            const Vec3& a = polygon[i];
            const Vec3& b = polygon[j];
            d = std::min(d, distanceToSegment2d(p, a, b));

            bool crosses = ((a.y > p.y) != (b.y > p.y));
            if (crosses) {
                float denominator = b.y - a.y;
                if (std::abs(denominator) < 1e-8f) continue;
                float x = (b.x - a.x) * (p.y - a.y) / denominator + a.x;
                if (p.x < x) inside = !inside;
            }
        }

        return inside ? -d : d;
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

    void drawNeutralBaseMesh(Renderer& renderer, zSpace::zFnMesh& fn)
    {
        for (int i = 0; i < fn.numPolygons(); ++i) {
            zSpace::zItMeshFace face(m_mesh, i);
            if (!face.isActive()) continue;

            std::vector<zSpace::zVector> positions;
            face.getVertexPositions(positions);
            if (positions.size() < 3) continue;

            Vec3 center = withZ(toVec3(face.getCenter()), m_baseZ);
            Color faceColor = plotUseForFaceIndex(i) == PlotUse::Green
                ? Color(0.62f, 0.78f, 0.54f, 1.0f)
                : Color(0.88f, 0.88f, 0.82f, 1.0f);

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

    void drawHeightFieldMap(Renderer& renderer, zSpace::zFnMesh& fn)
    {
        for (int i = 0; i < fn.numPolygons(); ++i) {
            zSpace::zItMeshFace face(m_mesh, i);
            if (!face.isActive()) continue;

            std::vector<zSpace::zVector> positions;
            face.getVertexPositions(positions);
            if (positions.size() < 3) continue;

            Vec3 faceCenter = toVec3(face.getCenter());
            float value = attractorHeightValue(faceCenter);
            Color faceColor = lerpColor(
                Color(0.0f, 0.0f, 1.0f, 1.0f),
                Color(1.0f, 0.0f, 1.0f, 1.0f),
                value
            );

            Vec3 center = withZ(faceCenter, m_baseZ + 0.002f);
            for (size_t j = 0; j < positions.size(); ++j) {
                Vec3 p1 = withZ(toVec3(positions[j]), m_baseZ + 0.002f);
                Vec3 p2 = withZ(toVec3(positions[(j + 1) % positions.size()]), m_baseZ + 0.002f);
                renderer.drawTriangle(center, p1, p2, faceColor);
            }
        }
    }

    float attractorHeightValue(const Vec3& p) const
    {
        Vec3 span = m_plotCenterMax - m_plotCenterMin;
        float radius = std::max(std::max(span.x, span.y), 1e-6f);
        Vec3 attractor(
            m_plotCenterMin.x + span.x * 0.35f,
            m_plotCenterMin.y + span.y * 0.65f,
            0.0f
        );
        float d = (p - attractor).length() / radius;
        return saturate(1.0f - d);
    }

    PlotUse plotUseForFaceIndex(int faceIndex) const
    {
        for (const auto& plotData : m_plots) {
            if (plotData.faceIndex == faceIndex) return plotData.plotUse;
        }
        return PlotUse::Building;
    }

    void drawEffectiveTypologyGraphs(Renderer& renderer)
    {
        (void)renderer;
        const Color graphColor(1.0f, 0.0f, 1.0f, 1.0f);
        zDisplayGraphSetting graphDisplay;
        graphDisplay.showEdges = true;
        graphDisplay.showVertices = true;
        graphDisplay.edgeColor = graphColor;
        graphDisplay.vertexColor = graphColor;
        graphDisplay.edgeWidth = 1.5f;
        graphDisplay.vertexSize = 6.0f;

        for (auto& plotData : m_plots) {
            if (plotData.effectiveGraphSegments.empty()) continue;
            scene().draw(plotData.effectiveCenterlineGraph, graphDisplay);
        }
    }

    void drawBuildingIsoMeshes(Renderer& renderer)
    {
        (void)renderer;
        zDisplayMeshSetting meshDisplay;
        meshDisplay.showFaces = true;
        meshDisplay.showEdges = false;
        meshDisplay.showVertices = false;
        meshDisplay.useVertexColors = false;
        meshDisplay.faceColor = Color(0.0f, 0.0f, 0.0f, 1.0f);

        for (auto& mesh : m_buildingIsoMeshes) {
            scene().draw(mesh, meshDisplay);
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

};

ALICE2_REGISTER_SKETCH_AUTO(zSpaceUrbanBlendSketch)

#endif

