// #define __MAIN__
#ifdef __MAIN__

#include <zspace/interface.h>
#include <zspace/io.h>

#include <alice2.h>
#include <sketches/SketchRegistry.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace alice2;

class zSpaceUrbanEvaluatorSketch : public ISketch {
public:
    std::string getName() const override { return "zSpace Urban Evaluator"; }
    std::string getDescription() const override { return "2D urban grid face offset generator for VLM evaluation."; }
    std::string getAuthor() const override { return "alice2 + zspace_core"; }

    void setup() override
    {
        // Setup B&W plan view background
        scene().setBackgroundColor(Color(1.0f, 1.0f, 1.0f, 1.0f));
        scene().setShowGrid(false);
        scene().setShowAxes(false);

        loadMesh();

        // Print face stats for diagnostics
        zSpace::zFnMesh fn(m_mesh);
        std::cout << "--- FACE CLASSIFICATION ANALYSIS ---" << std::endl;
        int numCore = 0, numSuburban = 0, numPark = 0;
        for (int i = 0; i < fn.numPolygons(); ++i) {
            zSpace::zItMeshFace face(m_mesh, i);
            if (!face.isActive()) continue;
            UrbanZone zone = getZone(face);
            zSpace::zPoint fCenter = face.getCenter();
            double area = face.getPlanarFaceArea();
            double d = (toVec3(fCenter) - toVec3(m_meshCenter)).length();
            std::cout << "Face " << i << " | Center: (" << fCenter.x << ", " << fCenter.y << ") | Area: " << area << " | Dist: " << d;
            if (zone == UrbanZone::Park) {
                std::cout << " | ZONE: Park" << std::endl;
                numPark++;
            } else if (zone == UrbanZone::Core) {
                std::cout << " | ZONE: Core" << std::endl;
                numCore++;
            } else {
                std::cout << " | ZONE: Suburban" << std::endl;
                numSuburban++;
            }
        }
        std::cout << "Summary: Park=" << numPark << ", Core=" << numCore << ", Suburban=" << numSuburban << std::endl;
        std::cout << "------------------------------------" << std::endl;
    }

    void update(float) override
    {
        if (m_loaded && !m_screenshotTaken) {
            m_frameCount++;
            if (m_frameCount == 10) {
                // Position camera directly overhead centered on the mesh
                zSpace::zPoint minBB;
                zSpace::zPoint maxBB;
                zSpace::zFnMesh fn(m_mesh);
                fn.getBounds(minBB, maxBB);
                Vec3 bMin = toVec3(minBB);
                Vec3 bMax = toVec3(maxBB);

                Vec3 center = (bMin + bMax) * 0.5f;
                Vec3 size = bMax - bMin;
                float maxSize = std::max({size.x, size.y, size.z});
                std::cout << "[DEBUG] minBB: " << bMin.x << ", " << bMin.y << ", " << bMin.z 
                          << " | bMax: " << bMax.x << ", " << bMax.y << ", " << bMax.z 
                          << " | size: " << size.x << ", " << size.y << ", " << size.z 
                          << " | maxSize: " << maxSize << std::endl;

                // Synchronize camera controller state for a close top-down orthographic plan view
                float halfSize = maxSize * 0.55f;
                float aspect = camera().getAspectRatio();
                camera().setOrthographic(-halfSize * aspect, halfSize * aspect, -halfSize, halfSize, 0.1f, 1000.0f);

                CameraState planState;
                planState.mode = CameraMode::Orbit;
                planState.orbitCenter = center;
                planState.orbitDistance = maxSize * 2.0f;
                planState.position = Vec3(center.x, center.y, center.z + planState.orbitDistance);
                planState.rotation = alice2::Quaternion::fromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -90.0f * ALICE2_DEG_TO_RAD);
                Application::getInstance()->getCameraController().setCameraState(planState);
            }
            else if (m_frameCount > 30) {
                // // Take screenshot and exit
                // Application::getInstance()->takeScreenshot();
                // m_screenshotTaken = true;
                
                // std::cout << "[URBAN EVALUATOR] Screenshot captured. Exiting." << std::endl;
                // exit(0);
            }
        }
    }

    enum class UrbanZone {
        Core,
        Suburban,
        Park
    };

    void draw(Renderer& renderer, Camera&) override
    {
        if (!m_loaded) return;

        zSpace::zFnMesh fn(m_mesh);
        
        for (int i = 0; i < fn.numPolygons(); ++i) {
            try {
                zSpace::zItMeshFace face(m_mesh, i);
                if (!face.isActive()) continue;

                UrbanZone zone = getZone(face);

                if (zone == UrbanZone::Park) {
                    // Park spaces are rendered as public plazas with tree grids
                    double parkSetback = 0.008;
                    double treeSize = 0.005;
                    double treeSpacing = 0.025;
                    drawPlazaTrees(renderer, face, parkSetback, treeSize, treeSpacing);
                    continue; 
                }

                double setback = calculateOffset(face);

                if (zone == UrbanZone::Core) {
                    // Core: Perimeter Courtyard blocks (O, U, or L shapes)
                    double area = face.getPlanarFaceArea();
                    double depth = 0.055 + area * 0.2;
                    if (depth < 0.055) depth = 0.055;
                    if (depth > 0.100) depth = 0.100;

                    int typology = face.getId() % 3; // 0 = O-courtyard, 1 = U-shape, 2 = L-shape
                    drawPerimeterBlock(renderer, face, setback, depth, typology);
                    drawStreetTrees(renderer, face, setback);
                }
                else if (zone == UrbanZone::Suburban) {
                    // Suburban: Subdivided detached buildings
                    double buildingSize = 0.062;
                    double spacing = 0.078;
                    drawDetachedSubdivision(renderer, face, setback, buildingSize, spacing);
                    drawStreetTrees(renderer, face, setback);
                }
            }
            catch (...) {
                // Safe skip
            }
        }
    }

    bool onKeyPress(unsigned char, int, int) override
    {
        return false;
    }

private:
    zSpace::zObjectMesh m_mesh;
    std::string m_meshPath = "data/input_grid_01.obj";
    bool m_loaded = false;
    int m_frameCount = 0;
    bool m_screenshotTaken = false;

    zSpace::zPoint m_meshCenter;
    double m_maxDistance = 1.0;

    // --- Generator Parameters ---
    double m_minOffset = 0.004;
    double m_maxOffset = 0.009;

    UrbanZone getZone(zSpace::zItMeshFace& face)
    {
        zSpace::zPoint fCenter = face.getCenter();
        Vec3 fc = toVec3(fCenter);

        // 1. Park hubs (Civic plazas and public open space)
        Vec3 parkHub1(0.1f, 0.2f, 0.0f);
        Vec3 parkHub2(1.1f, 0.7f, 0.0f);
        Vec3 centralPlaza = toVec3(m_meshCenter);
        
        if ((fc - parkHub1).length() < 0.09f || 
            (fc - parkHub2).length() < 0.09f || 
            (fc - centralPlaza).length() < 0.09f) {
            return UrbanZone::Park;
        }

        // 2. Core vs Suburban check based on distance to core and block size
        double d = (fc - toVec3(m_meshCenter)).length();
        double area = face.getPlanarFaceArea();
        
        // Downtown core zone contains central blocks and large scale blocks
        if (d < m_maxDistance * 0.45 || area > 0.14) {
            return UrbanZone::Core;
        }
        
        return UrbanZone::Suburban;
    }

    double calculateOffset(zSpace::zItMeshFace& face)
    {
        // Consistent street setback for clean connectivity
        return 0.015;
    }

    bool isPointInPolygon(const Vec3& p, const std::vector<Vec3>& polygon)
    {
        bool inside = false;
        size_t n = polygon.size();
        for (size_t i = 0, j = n - 1; i < n; j = i++) {
            if (((polygon[i].y > p.y) != (polygon[j].y > p.y)) &&
                (p.x < (polygon[j].x - polygon[i].x) * (p.y - polygon[i].y) / (polygon[j].y - polygon[i].y) + polygon[i].x)) {
                inside = !inside;
            }
        }
        return inside;
    }

    Vec3 getClosestEdgeDirection(const Vec3& p, const std::vector<Vec3>& polygon)
    {
        if (polygon.size() < 2) return Vec3(1.0f, 0.0f, 0.0f);
        
        float minDist = 1e9f;
        Vec3 bestDir(1.0f, 0.0f, 0.0f);
        
        size_t n = polygon.size();
        for (size_t i = 0; i < n; ++i) {
            Vec3 a = polygon[i];
            Vec3 b = polygon[(i + 1) % n];
            
            Vec3 ab = b - a;
            float lenSq = ab.lengthSquared();
            if (lenSq < 1e-6f) continue;
            
            Vec3 ap = p - a;
            float t = (ap.x * ab.x + ap.y * ab.y) / lenSq;
            if (t < 0.0f) t = 0.0f;
            if (t > 1.0f) t = 1.0f;
            
            Vec3 closestPoint = a + ab * t;
            float dist = (p - closestPoint).length();
            if (dist < minDist) {
                minDist = dist;
                bestDir = ab.normalized();
            }
        }
        return bestDir;
    }

    void drawPerimeterBlock(Renderer& renderer, zSpace::zItMeshFace& face, double setback, double depth, int typology)
    {
        std::vector<zSpace::zVector> outerPositions;
        face.getOffsetFacePositions(setback, outerPositions);
        if (outerPositions.size() < 3) return;
        
        bool drawn = false;
        std::vector<zSpace::zVector> innerPositions;
        try {
            face.getOffsetFacePositions(setback + depth, innerPositions);
            if (innerPositions.size() == outerPositions.size()) {
                // Draw perimeter building frames
                for (size_t j = 0; j < outerPositions.size(); ++j) {
                    if (typology == 1 && j == 0) continue; // U-shape
                    if (typology == 2 && (j == 0 || j == 1)) continue; // L-shape
                    
                    Vec3 p1 = toVec3(outerPositions[j]);
                    Vec3 p2 = toVec3(outerPositions[(j + 1) % outerPositions.size()]);
                    Vec3 ip2 = toVec3(innerPositions[(j + 1) % innerPositions.size()]);
                    Vec3 ip1 = toVec3(innerPositions[j]);
                    
                    renderer.drawTriangle(p1, p2, ip2, Color(0.0f, 0.0f, 0.0f, 1.0f));
                    renderer.drawTriangle(p1, ip2, ip1, Color(0.0f, 0.0f, 0.0f, 1.0f));
                }
                drawn = true;
            }
        }
        catch (...) {
            // fallback to solid
        }
        
        if (!drawn) {
            // Fallback: draw solid block
            Vec3 center(0.0f, 0.0f, 0.0f);
            for (const auto& p : outerPositions) {
                center += toVec3(p);
            }
            center /= static_cast<float>(outerPositions.size());
            for (size_t j = 0; j < outerPositions.size(); ++j) {
                Vec3 p1 = toVec3(outerPositions[j]);
                Vec3 p2 = toVec3(outerPositions[(j + 1) % outerPositions.size()]);
                renderer.drawTriangle(center, p1, p2, Color(0.0f, 0.0f, 0.0f, 1.0f));
            }
        }

        // Overlay a thin white alley to slice the block along its center (Z = 0.002f to prevent Z-fighting)
        float minX = 1e9f, maxX = -1e9f, minY = 1e9f, maxY = -1e9f;
        for (const auto& p : outerPositions) {
            Vec3 v = toVec3(p);
            if (v.x < minX) minX = v.x;
            if (v.x > maxX) maxX = v.x;
            if (v.y < minY) minY = v.y;
            if (v.y > maxY) maxY = v.y;
        }
        
        float cx = (minX + maxX) * 0.5f;
        float cy = (minY + maxY) * 0.5f;
        float w = maxX - minX;
        float h = maxY - minY;
        float alleyWidth = 0.008f;
        
        if (w > h) {
            // Cut vertically
            Vec3 c1(cx - alleyWidth * 0.5f, minY - 0.01f, 0.002f);
            Vec3 c2(cx + alleyWidth * 0.5f, minY - 0.01f, 0.002f);
            Vec3 c3(cx + alleyWidth * 0.5f, maxY + 0.01f, 0.002f);
            Vec3 c4(cx - alleyWidth * 0.5f, maxY + 0.01f, 0.002f);
            renderer.drawTriangle(c1, c2, c3, Color(1.0f, 1.0f, 1.0f, 1.0f));
            renderer.drawTriangle(c1, c3, c4, Color(1.0f, 1.0f, 1.0f, 1.0f));
        } else {
            // Cut horizontally
            Vec3 c1(minX - 0.01f, cy - alleyWidth * 0.5f, 0.002f);
            Vec3 c2(maxX + 0.01f, cy - alleyWidth * 0.5f, 0.002f);
            Vec3 c3(maxX + 0.01f, cy + alleyWidth * 0.5f, 0.002f);
            Vec3 c4(minX - 0.01f, cy + alleyWidth * 0.5f, 0.002f);
            renderer.drawTriangle(c1, c2, c3, Color(1.0f, 1.0f, 1.0f, 1.0f));
            renderer.drawTriangle(c1, c3, c4, Color(1.0f, 1.0f, 1.0f, 1.0f));
        }
    }

    void drawDetachedSubdivision(Renderer& renderer, zSpace::zItMeshFace& face, double setback, double buildingSize, double spacing)
    {
        std::vector<zSpace::zVector> outerPositions;
        face.getOffsetFacePositions(setback, outerPositions);
        if (outerPositions.size() < 3) return;
        
        std::vector<Vec3> poly;
        float minX = 1e9f, maxX = -1e9f, minY = 1e9f, maxY = -1e9f;
        for (const auto& p : outerPositions) {
            Vec3 v = toVec3(p);
            poly.push_back(v);
            if (v.x < minX) minX = v.x;
            if (v.x > maxX) maxX = v.x;
            if (v.y < minY) minY = v.y;
            if (v.y > maxY) maxY = v.y;
        }
        
        int housesPlaced = 0;
        
        // Loop through the bounding box grid of the setback polygon
        for (float x = minX + spacing * 0.5f; x < maxX; x += spacing) {
            for (float y = minY + spacing * 0.5f; y < maxY; y += spacing) {
                Vec3 gridCenter(x, y, 0.0f);
                
                // Add a deterministic pseudo-random jitter for organic feel
                int hashVal = static_cast<int>(x * 3700.0f + y * 7300.0f);
                float jitterX = ((hashVal % 7) - 3) * 0.005f;
                float jitterY = (((hashVal / 7) % 7) - 3) * 0.005f;
                Vec3 p = gridCenter + Vec3(jitterX, jitterY, 0.0f);
                
                // Add deterministic size variation (+/- 15%) for variety
                float sizeFactor = 1.0f + ((hashVal % 5) - 2) * 0.075f;
                float currentSize = static_cast<float>(buildingSize) * sizeFactor;
                
                float halfLen = currentSize * 0.5f;
                float halfWid = currentSize * 0.38f;
                
                // Orient the house footprint parallel to the closest block edge (street)
                Vec3 E = getClosestEdgeDirection(p, poly);
                Vec3 N(-E.y, E.x, 0.0f);
                
                Vec3 c1 = p - E * halfLen - N * halfWid;
                Vec3 c2 = p + E * halfLen - N * halfWid;
                Vec3 c3 = p + E * halfLen + N * halfWid;
                Vec3 c4 = p - E * halfLen + N * halfWid;
                
                // Ensure the entire building sits inside the block
                if (isPointInPolygon(c1, poly) && isPointInPolygon(c2, poly) && 
                    isPointInPolygon(c3, poly) && isPointInPolygon(c4, poly)) {
                    
                    renderer.drawTriangle(c1, c2, c3, Color(0.0f, 0.0f, 0.0f, 1.0f));
                    renderer.drawTriangle(c1, c3, c4, Color(0.0f, 0.0f, 0.0f, 1.0f));
                    housesPlaced++;
                }
            }
        }
        
        // Fallback: place a single house at center if none fit in the grid
        if (housesPlaced == 0) {
            Vec3 center(0.0f, 0.0f, 0.0f);
            for (const auto& v : poly) {
                center += v;
            }
            center /= static_cast<float>(poly.size());
            
            int hashVal = static_cast<int>(center.x * 3700.0f + center.y * 7300.0f);
            float sizeFactor = 1.0f + ((hashVal % 5) - 2) * 0.075f;
            float currentSize = static_cast<float>(buildingSize) * sizeFactor;
            
            float halfLen = currentSize * 0.5f;
            float halfWid = currentSize * 0.38f;
            Vec3 E = getClosestEdgeDirection(center, poly);
            Vec3 N(-E.y, E.x, 0.0f);
            
            Vec3 c1 = center - E * halfLen - N * halfWid;
            Vec3 c2 = center + E * halfLen - N * halfWid;
            Vec3 c3 = center + E * halfLen + N * halfWid;
            Vec3 c4 = center - E * halfLen + N * halfWid;
            
            renderer.drawTriangle(c1, c2, c3, Color(0.0f, 0.0f, 0.0f, 1.0f));
            renderer.drawTriangle(c1, c3, c4, Color(0.0f, 0.0f, 0.0f, 1.0f));
        }
    }

    void drawPlazaTrees(Renderer& renderer, zSpace::zItMeshFace& face, double setback, double treeSize, double spacing)
    {
        std::vector<zSpace::zVector> outerPositions;
        face.getOffsetFacePositions(setback, outerPositions);
        if (outerPositions.size() < 3) return;
        
        std::vector<Vec3> poly;
        float minX = 1e9f, maxX = -1e9f, minY = 1e9f, maxY = -1e9f;
        for (const auto& p : outerPositions) {
            Vec3 v = toVec3(p);
            poly.push_back(v);
            if (v.x < minX) minX = v.x;
            if (v.x > maxX) maxX = v.x;
            if (v.y < minY) minY = v.y;
            if (v.y > maxY) maxY = v.y;
        }
        
        for (float x = minX + spacing * 0.5f; x < maxX; x += spacing) {
            for (float y = minY + spacing * 0.5f; y < maxY; y += spacing) {
                // Add a deterministic pseudo-random jitter for organic plazas
                int hashVal = static_cast<int>(x * 2900.0f + y * 5700.0f);
                float jitterX = ((hashVal % 5) - 2) * 0.0015f;
                float jitterY = (((hashVal / 5) % 5) - 2) * 0.0015f;
                Vec3 p(x + jitterX, y + jitterY, 0.0f);
                
                // Add minor size variation
                float sizeFactor = 1.0f + ((hashVal % 3) - 1) * 0.15f; // 0.85, 1.0, 1.15
                float halfSize = static_cast<float>(treeSize * 0.5) * sizeFactor;
                
                Vec3 c1 = p + Vec3(-halfSize, -halfSize, 0.0f);
                Vec3 c2 = p + Vec3(halfSize, -halfSize, 0.0f);
                Vec3 c3 = p + Vec3(halfSize, halfSize, 0.0f);
                Vec3 c4 = p + Vec3(-halfSize, halfSize, 0.0f);
                
                if (isPointInPolygon(c1, poly) && isPointInPolygon(c2, poly) && 
                    isPointInPolygon(c3, poly) && isPointInPolygon(c4, poly)) {
                    renderer.drawTriangle(c1, c2, c3, Color(0.0f, 0.0f, 0.0f, 1.0f));
                    renderer.drawTriangle(c1, c3, c4, Color(0.0f, 0.0f, 0.0f, 1.0f));
                }
            }
        }
    }

    void drawStreetTrees(Renderer& renderer, zSpace::zItMeshFace& face, double setback)
    {
        std::vector<zSpace::zVector> outerPositions;
        face.getOffsetFacePositions(setback, outerPositions);
        if (outerPositions.size() < 3) return;
        
        double spacing = 0.035; // Space between trees
        double treeSize = 0.003; // Size of trees
        double streetOffset = -0.004; // Offset into the street space
        
        std::vector<zSpace::zVector> streetPositions;
        try {
            face.getOffsetFacePositions(setback + streetOffset, streetPositions);
        }
        catch (...) {
            return;
        }
        if (streetPositions.size() != outerPositions.size()) return;
        
        for (size_t j = 0; j < streetPositions.size(); ++j) {
            Vec3 p1 = toVec3(streetPositions[j]);
            Vec3 p2 = toVec3(streetPositions[(j + 1) % streetPositions.size()]);
            
            Vec3 dir = p2 - p1;
            float len = dir.length();
            if (len < 1e-4f) continue;
            
            dir.normalize();
            
            int numTrees = static_cast<int>(len / spacing);
            for (int t = 0; t <= numTrees; ++t) {
                float dist = t * spacing;
                if (dist > 0.005f && dist < len - 0.005f) {
                    Vec3 treeCenter = p1 + dir * dist;
                    
                    float halfSize = static_cast<float>(treeSize * 0.5);
                    Vec3 c1 = treeCenter + Vec3(-halfSize, -halfSize, 0.003f);
                    Vec3 c2 = treeCenter + Vec3(halfSize, -halfSize, 0.003f);
                    Vec3 c3 = treeCenter + Vec3(halfSize, halfSize, 0.003f);
                    Vec3 c4 = treeCenter + Vec3(-halfSize, halfSize, 0.003f);
                    
                    renderer.drawTriangle(c1, c2, c3, Color(0.0f, 0.0f, 0.0f, 1.0f));
                    renderer.drawTriangle(c1, c3, c4, Color(0.0f, 0.0f, 0.0f, 1.0f));
                }
            }
        }
    }

    static Vec3 toVec3(const zSpace::zVector& p)
    {
        return Vec3(static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z));
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
        if (fn.numHalfEdges() <= 0) {
            m_loaded = false;
            std::cout << "[ERROR] Loaded mesh, but no halfedges were created." << std::endl;
            return;
        }

        m_loaded = true;
        std::cout << "[URBAN EVALUATOR] Loaded grid mesh successfully." << std::endl;

        // Calculate mesh center and bounds to normalize distance
        zSpace::zPoint minBB;
        zSpace::zPoint maxBB;
        fn.getBounds(minBB, maxBB);
        m_meshCenter = (minBB + maxBB) * 0.5;

        m_maxDistance = 0.0;
        zSpace::zPointArray vertices;
        fn.getVertexPositions(vertices);
        for (const auto& v : vertices) {
            double d = (toVec3(v) - toVec3(m_meshCenter)).length();
            if (d > m_maxDistance) {
                m_maxDistance = d;
            }
        }
        if (m_maxDistance < 1e-5) m_maxDistance = 1.0;
    }
};

ALICE2_REGISTER_SKETCH_AUTO(zSpaceUrbanEvaluatorSketch)

#endif
