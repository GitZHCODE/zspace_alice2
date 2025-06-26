// alice2 Scalar Field Educational Sketch 4: Directional Boolean Operations with Sun Vector
// Demonstrates dynamic boolean operations based on sun direction and exposure calculation

#define __MAIN__
#ifdef __MAIN__

#include "../include/alice2.h"
#include "../src/sketches/SketchRegistry.h"
#include "scalarField.h"
#include <cmath>

using namespace alice2;

class ScalarField04DirectionalSketch : public ISketch {
private:
    // Scalar fields with consistent dimensions: 100x100 grid, bounds (-50, -50) to (50, 50)
    ScalarField2D m_field_lower;    // Rectangle field (lower)
    ScalarField2D m_field_upper;    // Rotated rectangle field (upper, 30 degrees)
    ScalarField2D m_circleField;    // Individual circle field for operations
    ScalarField2D m_resultField;    // Final result field
    
    // Animation and timing
    float m_time;
    
    // Boolean flags for computation controls (prefix with "b_")
    bool b_computeDirectional;
    
    // Boolean flags for visualization controls (prefix with "d_")
    bool d_drawField;
    bool d_drawValues;
    bool d_drawContours;
    bool d_drawTower;
    bool d_animateSun;
    
    // Geometric parameters - consistent with specification
    Vec3 m_rectCenter;      // Rectangle center at (-15, -10)
    Vec3 m_circleCenter;    // Circle center at (15, 10)
    Vec3 m_rectSize;        // Rectangle size
    
    // Sun direction and exposure
    Vec3 m_sunDirection;    // 2D sun direction vector
    bool m_manualSunControl;
    
    // Rectangle face data for exposure calculation
    struct RectFace {
        Vec3 center;
        Vec3 normal;
        float exposure;
        std::vector<Vec3> circlePositions;
        std::vector<float> circleRadii;
        std::vector<bool> isUnion; // true for union, false for subtract
    };
    std::vector<RectFace> m_rectFaces;
    
    // Tower visualization parameters
    std::vector<float> m_towerLevels;
    std::vector<std::vector<std::pair<Vec3, Vec3>>> m_towerContours;

public:
    ScalarField04DirectionalSketch() 
        : m_field_lower(Vec3(-50, -50, 0), Vec3(50, 50, 0), 100, 100)
        , m_field_upper(Vec3(-50, -50, 0), Vec3(50, 50, 0), 100, 100)
        , m_circleField(Vec3(-50, -50, 0), Vec3(50, 50, 0), 100, 100)
        , m_resultField(Vec3(-50, -50, 0), Vec3(50, 50, 0), 100, 100)
        , m_time(0.0f)
        , b_computeDirectional(false)
        , d_drawField(true)
        , d_drawValues(false)
        , d_drawContours(true)
        , d_drawTower(false)
        , d_animateSun(true)
        , m_rectCenter(-15, -10, 0)
        , m_circleCenter(15, 10, 0)
        , m_rectSize(20.0f, 15.0f, 0.0f)
        , m_sunDirection(1.0f, 0.0f, 0.0f)
        , m_manualSunControl(false)
    {
        // Initialize tower levels
        m_towerLevels = {0.0f, 3.0f, 6.0f, 9.0f, 12.0f};
        m_towerContours.resize(m_towerLevels.size());
        
        // Initialize rectangle faces
        initializeRectFaces();
    }
    
    ~ScalarField04DirectionalSketch() = default;

    // Sketch information
    std::string getName() const override {
        return "Scalar Field 04: Directional Boolean";
    }

    std::string getDescription() const override {
        return "Educational sketch demonstrating directional boolean operations with sun exposure";
    }

    std::string getAuthor() const override {
        return "alice2 Educational Series";
    }

    // Sketch lifecycle
    void setup() override {
        scene().setBackgroundColor(Vec3(0.05f, 0.05f, 0.1f));
        scene().setShowGrid(true);
        scene().setGridSize(25.0f);
        scene().setGridDivisions(4);
        scene().setShowAxes(true);
        scene().setAxesLength(10.0f);
        
        std::cout << "Scalar Field 04: Directional Boolean loaded" << std::endl;
        std::cout << "Field dimensions: 100x100 grid, bounds (-50,-50) to (50,50)" << std::endl;
        std::cout << "Rectangle center: (-15, -10), Circle center: (15, 10)" << std::endl;
        
        // Initialize base fields
        generateBaseFields();
    }

    void update(float deltaTime) override {
        m_time += deltaTime;
        
        // Animate sun direction if enabled
        if (d_animateSun && !m_manualSunControl) {
            float angle = m_time * 0.3f;
            m_sunDirection.x = std::cos(angle);
            m_sunDirection.y = std::sin(angle);
            m_sunDirection = m_sunDirection.normalized();
        }
        
        // Update face exposures based on sun direction
        updateFaceExposures();
        
        // Regenerate directional field if computation is enabled
        if (b_computeDirectional) {
            generateDirectionalField();
            if (d_drawTower) {
                generateTowerContours();
            }
        }
    }

    void draw(Renderer& renderer, Camera& camera) override {
        if (d_drawTower) {
            // Side-by-side display: original field + tower
            drawSideBySideView(renderer);
        } else {
            // Standard single view
            drawStandardView(renderer);
        }
        
        // Draw geometric elements
        drawGeometry(renderer);
        
        // Draw sun vector
        drawSunVector(renderer);
        
        // Draw UI and controls
        drawUI(renderer);
    }

    void cleanup() override {
        std::cout << "Scalar Field 04: Directional Boolean cleanup" << std::endl;
    }

private:
    void initializeRectFaces() {
        m_rectFaces.clear();
        
        // Four faces of the rectangle with their normals
        Vec3 rectMin = m_rectCenter - m_rectSize;
        Vec3 rectMax = m_rectCenter + m_rectSize;
        
        // Top face (normal pointing up)
        RectFace topFace;
        topFace.center = Vec3(m_rectCenter.x, rectMax.y, 0);
        topFace.normal = Vec3(0, 1, 0);
        m_rectFaces.push_back(topFace);
        
        // Right face (normal pointing right)
        RectFace rightFace;
        rightFace.center = Vec3(rectMax.x, m_rectCenter.y, 0);
        rightFace.normal = Vec3(1, 0, 0);
        m_rectFaces.push_back(rightFace);
        
        // Bottom face (normal pointing down)
        RectFace bottomFace;
        bottomFace.center = Vec3(m_rectCenter.x, rectMin.y, 0);
        bottomFace.normal = Vec3(0, -1, 0);
        m_rectFaces.push_back(bottomFace);
        
        // Left face (normal pointing left)
        RectFace leftFace;
        leftFace.center = Vec3(rectMin.x, m_rectCenter.y, 0);
        leftFace.normal = Vec3(-1, 0, 0);
        m_rectFaces.push_back(leftFace);
    }
    
    void generateBaseFields() {
        // Generate lower rectangle field
        m_field_lower.clear_field();
        m_field_lower.apply_scalar_rect(m_rectCenter, m_rectSize, 0.0f);
        
        // Generate upper rectangle field (rotated 30 degrees)
        m_field_upper.clear_field();
        m_field_upper.apply_scalar_rect(m_rectCenter, m_rectSize, 0.523599f); // 30 degrees in radians
    }
    
    void updateFaceExposures() {
        // Calculate exposure for each face based on sun direction
        for (auto& face : m_rectFaces) {
            // Dot product gives exposure (higher = more sun exposure)
            face.exposure = std::max(0.0f, face.normal.dot(m_sunDirection));
            
            // Clear previous circles
            face.circlePositions.clear();
            face.circleRadii.clear();
            face.isUnion.clear();
            
            // Add circles based on exposure
            if (face.exposure > 0.7f) {
                // High exposure: more subtract circles
                for (int i = 0; i < 3; ++i) {
                    Vec3 pos = face.center + Vec3((i-1) * 8.0f, 0, 0);
                    face.circlePositions.push_back(pos);
                    face.circleRadii.push_back(4.0f + std::sin(m_time + i) * 2.0f);
                    face.isUnion.push_back(false); // subtract
                }
            } else if (face.exposure > 0.3f) {
                // Medium exposure: mixed operations
                Vec3 pos1 = face.center + Vec3(-6, 0, 0);
                Vec3 pos2 = face.center + Vec3(6, 0, 0);
                face.circlePositions.push_back(pos1);
                face.circlePositions.push_back(pos2);
                face.circleRadii.push_back(5.0f);
                face.circleRadii.push_back(5.0f);
                face.isUnion.push_back(true);  // union
                face.isUnion.push_back(false); // subtract
            } else {
                // Low exposure: more union circles
                for (int i = 0; i < 2; ++i) {
                    Vec3 pos = face.center + Vec3((i-0.5f) * 10.0f, 0, 0);
                    face.circlePositions.push_back(pos);
                    face.circleRadii.push_back(6.0f + std::cos(m_time + i) * 2.0f);
                    face.isUnion.push_back(true); // union
                }
            }
        }
    }
    
    void generateDirectionalField() {
        // Start with blended base fields
        m_resultField = m_field_lower;
        m_resultField.boolean_smin(m_field_upper, 1.5f);
        
        // Apply directional boolean operations for each face
        for (const auto& face : m_rectFaces) {
            for (size_t i = 0; i < face.circlePositions.size(); ++i) {
                // Generate circle field
                m_circleField.clear_field();
                m_circleField.apply_scalar_circle(face.circlePositions[i], face.circleRadii[i]);
                
                // Apply boolean operation based on face exposure
                if (face.isUnion[i]) {
                    m_resultField.boolean_union(m_circleField);
                } else {
                    m_resultField.boolean_subtract(m_circleField);
                }
            }
        }
    }
    
    void generateTowerContours() {
        // Extract contours at each tower level
        for (size_t i = 0; i < m_towerLevels.size(); ++i) {
            float threshold = -5.0f + i * 2.0f;
            ContourData contours = m_resultField.get_contours(threshold);
            m_towerContours[i] = contours.line_segments;
        }
    }
    
    void drawStandardView(Renderer& renderer) {
        // Choose which field to display
        ScalarField2D* displayField = &m_field_lower;
        if (b_computeDirectional) {
            displayField = &m_resultField;
        }
        
        // Draw scalar field visualization
        if (d_drawField) {
            displayField->draw_points(renderer, 2);
        }
        
        // Draw scalar values as text
        if (d_drawValues) {
            displayField->draw_values(renderer, 12);
        }
        
        // Draw contours
        if (d_drawContours) {
            drawContours(renderer, *displayField);
        }
    }
    
    void drawSideBySideView(Renderer& renderer) {
        // Left side: original field
        renderer.pushMatrix();
        Mat4 leftTransform = Mat4::translation(Vec3(-25, 0, 0)) * Mat4::scale(Vec3(0.7f, 0.7f, 0.7f));
        renderer.multMatrix(leftTransform);

        if (d_drawField && b_computeDirectional) {
            m_resultField.draw_points(renderer, 3);
        }
        if (d_drawContours && b_computeDirectional) {
            drawContours(renderer, m_resultField);
        }

        renderer.popMatrix();

        // Right side: tower visualization
        drawTowerVisualization(renderer);
    }
    
    void drawTowerVisualization(Renderer& renderer) {
        renderer.pushMatrix();
        Mat4 towerTransform = Mat4::translation(Vec3(25, 0, 0));
        renderer.multMatrix(towerTransform);
        
        // Draw tower contours at different Z levels
        for (size_t i = 0; i < m_towerLevels.size(); ++i) {
            float z = m_towerLevels[i];
            
            // Color gradient from blue (bottom) to red (top)
            float t = static_cast<float>(i) / (m_towerLevels.size() - 1);
            Vec3 color = Vec3(t, 0.2f, 1.0f - t);
            renderer.setColor(color);
            
            // Draw contours at this level
            for (const auto& segment : m_towerContours[i]) {
                Vec3 start = segment.first + Vec3(0, 0, z);
                Vec3 end = segment.second + Vec3(0, 0, z);
                renderer.drawLine(start, end, color, 2.0f);
            }
            
            // Draw level indicator
            renderer.drawText("Z=" + std::to_string(static_cast<int>(z)), 
                            Vec3(-40, -40, z), 0.8f);
        }
        
        renderer.popMatrix();
    }
    
    void drawContours(Renderer& renderer, const ScalarField2D& field) {
        renderer.setColor(Vec3(1.0f, 1.0f, 1.0f));
        
        for (int i = 0; i < 6; ++i) {
            float threshold = -8.0f + i * 3.0f;
            field.drawIsocontours(renderer, threshold);
        }
    }
    
    void drawGeometry(Renderer& renderer) {
        // Draw rectangle center in blue
        renderer.setColor(Vec3(0.2f, 0.2f, 1.0f));
        renderer.drawPoint(m_rectCenter, Vec3(0.2f, 0.2f, 1.0f), 8.0f);
        renderer.drawText("RECT", m_rectCenter + Vec3(0, 0, 5), 1.0f);
        
        // Draw circle center in red
        renderer.setColor(Vec3(1.0f, 0.2f, 0.2f));
        renderer.drawPoint(m_circleCenter, Vec3(1.0f, 0.2f, 0.2f), 8.0f);
        renderer.drawText("CIRCLE", m_circleCenter + Vec3(0, 0, 5), 1.0f);
        
        // Draw face circles if directional computation is active
        if (b_computeDirectional) {
            for (const auto& face : m_rectFaces) {
                for (size_t i = 0; i < face.circlePositions.size(); ++i) {
                    Vec3 color = face.isUnion[i] ? Vec3(0.2f, 1.0f, 0.2f) : Vec3(1.0f, 0.2f, 0.2f);
                    renderer.setColor(color);
                    renderer.drawPoint(face.circlePositions[i], color, 4.0f);
                }
            }
        }
    }
    
    void drawSunVector(Renderer& renderer) {
        // Draw sun vector as animated arrow
        Vec3 sunStart = Vec3(30, 30, 10);
        Vec3 sunEnd = sunStart + m_sunDirection * 15.0f;
        
        // Sun vector in yellow
        renderer.setColor(Vec3(1.0f, 1.0f, 0.0f));
        renderer.drawLine(sunStart, sunEnd, Vec3(1.0f, 1.0f, 0.0f), 3.0f);
        
        // Arrow head
        Vec3 arrowDir = m_sunDirection * 3.0f;
        Vec3 perpDir = Vec3(-m_sunDirection.y, m_sunDirection.x, 0) * 1.5f;
        renderer.drawLine(sunEnd, sunEnd - arrowDir + perpDir, Vec3(1.0f, 1.0f, 0.0f), 2.0f);
        renderer.drawLine(sunEnd, sunEnd - arrowDir - perpDir, Vec3(1.0f, 1.0f, 0.0f), 2.0f);
        
        // Sun direction text
        renderer.drawText("SUN", sunEnd + Vec3(2, 2, 0), 1.0f);
    }
    
    void drawUI(Renderer& renderer) {
        // Title and description
        renderer.setColor(Vec3(1.0f, 1.0f, 1.0f));
        renderer.drawString(getName(), 10, 30);
        renderer.drawString("Educational sketch: Directional boolean with sun exposure", 10, 50);
        
        // FPS display
        renderer.setColor(Vec3(0.0f, 1.0f, 1.0f));
        renderer.drawString("FPS: " + std::to_string(Application::getInstance()->getFPS()), 10, 80);
        
        // Current mode display
        renderer.setColor(Vec3(1.0f, 1.0f, 0.0f));
        std::string mode = b_computeDirectional ? "DIRECTIONAL ACTIVE" : "BASE FIELDS";
        renderer.drawString("Current Mode: " + mode, 10, 110);
        
        // Sun direction info
        renderer.setColor(Vec3(0.8f, 0.8f, 0.8f));
        renderer.drawString("Sun Dir: (" + std::to_string(m_sunDirection.x).substr(0, 4) + ", " + 
                          std::to_string(m_sunDirection.y).substr(0, 4) + ")", 10, 140);
        
        // Controls
        renderer.setColor(Vec3(0.7f, 0.7f, 0.7f));
        renderer.drawString("Controls:", 10, 180);
        renderer.drawString("'S' - Toggle Sun Animation", 10, 200);
        renderer.drawString("Arrow Keys - Manual Sun Control", 10, 220);
        renderer.drawString("'D' - Toggle Directional Computation", 10, 240);
        renderer.drawString("'T' - Toggle Tower Visualization", 10, 260);
        renderer.drawString("'F' - Toggle Field Visualization", 10, 280);
        renderer.drawString("'C' - Toggle Contours", 10, 300);
        
        // Status indicators
        renderer.setColor(Vec3(0.5f, 1.0f, 0.5f));
        renderer.drawString("Directional: " + std::string(b_computeDirectional ? "ON" : "OFF"), 10, 330);
        renderer.drawString("Sun Anim: " + std::string(d_animateSun ? "ON" : "OFF"), 10, 350);
        renderer.drawString("Tower: " + std::string(d_drawTower ? "ON" : "OFF"), 10, 370);
    }

public:
    // Input handling
    bool onKeyPress(unsigned char key, int x, int y) override {
        switch (key) {
            case 's':
            case 'S': // Toggle sun animation
                d_animateSun = !d_animateSun;
                m_manualSunControl = !d_animateSun;
                return true;
                
            case 'd':
            case 'D': // Toggle directional computation
                b_computeDirectional = !b_computeDirectional;
                if (b_computeDirectional) {
                    generateDirectionalField();
                }
                return true;
                
            case 't':
            case 'T': // Toggle tower visualization
                d_drawTower = !d_drawTower;
                if (d_drawTower && b_computeDirectional) {
                    generateTowerContours();
                }
                return true;
                
            case 'f':
            case 'F': // Toggle field visualization
                d_drawField = !d_drawField;
                return true;
                
            case 'c':
            case 'C': // Toggle contours
                d_drawContours = !d_drawContours;
                return true;
                
            case 'v':
            case 'V': // Toggle value display
                d_drawValues = !d_drawValues;
                return true;
        }
        return false;
    }
    
    // Special key handling for arrow keys (manual sun control)
    bool onSpecialKeyPress(int key, int x, int y) {
        if (!m_manualSunControl) return false;
        
        const float step = 0.1f;
        switch (key) {
            case 101: // Up arrow
                m_sunDirection.y += step;
                break;
            case 103: // Down arrow
                m_sunDirection.y -= step;
                break;
            case 100: // Left arrow
                m_sunDirection.x -= step;
                break;
            case 102: // Right arrow
                m_sunDirection.x += step;
                break;
            default:
                return false;
        }
        
        m_sunDirection = m_sunDirection.normalized();
        return true;
    }
};

// Register the sketch with alice2
ALICE2_REGISTER_SKETCH_AUTO(ScalarField04DirectionalSketch)

#endif // __MAIN__
