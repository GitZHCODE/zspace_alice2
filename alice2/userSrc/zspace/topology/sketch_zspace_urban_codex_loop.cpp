#define __MAIN__
#ifdef __MAIN__

#include <zspace/interface.h>
#include <zspace/io.h>

#include <alice2.h>
#include <sketches/SketchRegistry.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
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

        loadMesh();
        if (!m_loaded) return;

        std::cout << "[URBAN CODEX LOOP] Clean neutral baseline loaded." << std::endl;
        std::cout << "[URBAN CODEX LOOP] Faces: " << zSpace::zFnMesh(m_mesh).numPolygons() << std::endl;
    }

    void update(float) override
    {
        if (!m_loaded || m_screenshotTaken) return;

        m_frameCount++;
        if (m_frameCount == 10) {
            setPlanCamera();
        }
        else if (m_frameCount > 30) {
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
        drawStreetHierarchySdf(renderer, fn);
        drawOpenSpaceSdf(renderer, fn);
        drawSimpleMassing(renderer, fn);
    }

    bool onKeyPress(unsigned char, int, int) override
    {
        return false;
    }

private:
    zSpace::zObjectMesh m_mesh;
    std::string m_meshPath = "data/input_grid_01.obj";

    bool m_loaded = false;
    bool m_screenshotTaken = false;
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
    float m_openSpaceZ = 0.001f;
    float m_civicSpineWidth = 0.055f;
    float m_civicPlazaRadius = 0.135f;
    float m_primaryStreetWidth = 0.032f;
    float m_secondaryStreetWidth = 0.024f;
    float m_neighborhoodPlazaRadius = 0.105f;
    Vec3 m_civicSpineA;
    Vec3 m_civicSpineB;
    Vec3 m_neighborhoodPlazaA;
    Vec3 m_neighborhoodPlazaB;
    Vec3 m_primaryStreetA;
    Vec3 m_primaryStreetB;
    Vec3 m_secondaryStreetA;
    Vec3 m_secondaryStreetB;

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

    float streetHierarchySdf(const Vec3& p) const
    {
        float primary = distanceToSegment2d(p, m_primaryStreetA, m_primaryStreetB) - m_primaryStreetWidth;
        float secondary = distanceToSegment2d(p, m_secondaryStreetA, m_secondaryStreetB) - m_secondaryStreetWidth;
        return std::min(primary, secondary);
    }

    bool isStreetSpace(const Vec3& p) const
    {
        return streetHierarchySdf(p) < 0.0f;
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
        float spineProximity = 1.0f - smoothstep(0.04f, 0.34f, distanceToSegment2d(p, m_primaryStreetA, m_primaryStreetB));
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
        m_civicSpineA = Vec3(bMin.x + span.x * 0.18f, bMin.y + span.y * 0.45f, 0.0f);
        m_civicSpineB = Vec3(bMax.x - span.x * 0.18f, bMin.y + span.y * 0.58f, 0.0f);
        m_neighborhoodPlazaA = Vec3(bMin.x + span.x * 0.32f, bMin.y + span.y * 0.47f, 0.0f);
        m_neighborhoodPlazaB = Vec3(bMin.x + span.x * 0.74f, bMin.y + span.y * 0.61f, 0.0f);
        m_primaryStreetA = Vec3(bMin.x + span.x * 0.08f, bMin.y + span.y * 0.34f, 0.0f);
        m_primaryStreetB = Vec3(bMax.x - span.x * 0.08f, bMin.y + span.y * 0.66f, 0.0f);
        m_secondaryStreetA = Vec3(bMin.x + span.x * 0.56f, bMin.y + span.y * 0.10f, 0.0f);
        m_secondaryStreetB = Vec3(bMin.x + span.x * 0.60f, bMax.y - span.y * 0.10f, 0.0f);

        zSpace::zPointArray vertices;
        fn.getVertexPositions(vertices);
        m_maxDistance = 0.0f;
        for (const auto& v : vertices) {
            m_maxDistance = std::max(m_maxDistance, (toVec3(v) - toVec3(m_meshCenter)).length());
        }
        if (m_maxDistance < 1e-5f) m_maxDistance = 1.0f;

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

    void drawStreetHierarchySdf(Renderer& renderer, zSpace::zFnMesh& fn)
    {
        for (int i = 0; i < fn.numPolygons(); ++i) {
            zSpace::zItMeshFace face(m_mesh, i);
            if (!face.isActive()) continue;

            Vec3 center = toVec3(face.getCenter());
            if (!isStreetSpace(center)) continue;

            std::vector<zSpace::zVector> positions;
            face.getVertexPositions(positions);
            if (positions.size() < 3) continue;

            Vec3 c = withZ(center, m_openSpaceZ + 0.001f);
            Color streetColor(0.98f, 0.98f, 0.95f, 1.0f);
            for (size_t j = 0; j < positions.size(); ++j) {
                Vec3 p1 = withZ(toVec3(positions[j]), m_openSpaceZ + 0.001f);
                Vec3 p2 = withZ(toVec3(positions[(j + 1) % positions.size()]), m_openSpaceZ + 0.001f);
                renderer.drawTriangle(c, p1, p2, streetColor);
            }
        }
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
