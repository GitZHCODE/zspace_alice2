// alice2 Scalar Field Educational Sketch 3: SDF Blending and Tower Visualization
// Demonstrates smooth minimum blending and multi-level contour extraction with tower visualization

#define __MAIN__
#ifdef __MAIN__

#include "../include/alice2.h"
#include "../src/sketches/SketchRegistry.h"
#include "scalarField.h"
#include <cmath>

using namespace alice2;

class ScalarField03BlendingSketch : public ISketch {
private:
    // Scalar fields with consistent dimensions: 100x100 grid, bounds (-50, -50) to (50, 50)
    ScalarField2D m_field_lower;    // Rectangle field (lower)
    ScalarField2D m_field_upper;    // Circle field (upper)
    ScalarField2D m_blended_field;  // Result of smooth blending
    
    // Animation and timing
    float m_time;
    
    // Boolean flags for computation controls (prefix with "b_")
    bool b_computeBlend;
    
    // Boolean flags for visualization controls (prefix with "d_")
    bool d_drawField;
    bool d_drawValues;
    bool d_drawContours;
    bool d_drawTower;
    
    // Geometric parameters - consistent with specification
    Vec3 m_rectCenter;      // Rectangle center at (-15, -10)
    Vec3 m_circleCenter;    // Circle center at (15, 10)
    Vec3 m_rectSize;        // Rectangle size
    float m_circleRadius;   // Circle radius
    
    // Blending parameters
    float m_blendFactor;
    
    // Tower visualization parameters
    std::vector<float> m_towerLevels;  // Z-levels for tower: 0, 3, 6, 9, 12
    std::vector<std::vector<std::pair<Vec3, Vec3>>> m_towerContours; // Contours at each level

public:
    ScalarField03BlendingSketch() 
        : m_field_lower(Vec3(-50, -50, 0), Vec3(50, 50, 0), 100, 100)
        , m_field_upper(Vec3(-50, -50, 0), Vec3(50, 50, 0), 100, 100)
        , m_blended_field(Vec3(-50, -50, 0), Vec3(50, 50, 0), 100, 100)
        , m_time(0.0f)
        , b_computeBlend(false)
        , d_drawField(true)
        , d_drawValues(false)
        , d_drawContours(true)
        , d_drawTower(false)
        , m_rectCenter(0, 0, 0)
        , m_circleCenter(15, 10, 0)
        , m_rectSize(20.0f, 15.0f, 0.0f)
        , m_circleRadius(12.0f)
        , m_blendFactor(2.0f)
    {
        // Initialize tower levels: 20 floors with 3-unit spacing (Z=0 to Z=57)
        m_towerLevels.clear();
        for (int i = 0; i < 20; ++i) {
            m_towerLevels.push_back(i * 3.0f);
        }
        m_towerContours.resize(m_towerLevels.size());
    }
    
    ~ScalarField03BlendingSketch() = default;

    // Sketch information
    std::string getName() const override {
        return "Scalar Field 03: SDF Blending & Tower";
    }

    std::string getDescription() const override {
        return "Educational sketch demonstrating smooth minimum blending and tower visualization";
    }

    std::string getAuthor() const override {
        return "alice2 Educational Series";
    }

    // Sketch lifecycle
    void setup() override {
        scene().setBackgroundColor(Vec3(0.05f, 0.05f, 0.1f));
        scene().setShowGrid(false);
        scene().setGridSize(25.0f);
        scene().setGridDivisions(4);
        scene().setShowAxes(true);
        scene().setAxesLength(10.0f);
        
        std::cout << "Scalar Field 03: SDF Blending & Tower loaded" << std::endl;
        std::cout << "Field dimensions: 100x100 grid, bounds (-50,-50) to (50,50)" << std::endl;
        std::cout << "Rectangle center: (0, 0), Circle center: (15, 10)" << std::endl;
        
        // Initialize base fields
        generateBaseFields();
    }

    void update(float deltaTime) override {
        m_time += deltaTime;
        
        // Animate blend factor
        m_blendFactor = 2.0f + std::sin(m_time * 0.8f) * 1.5f;
        
        // Regenerate blended field if computation is enabled
        if (b_computeBlend) {
            generateBlendedField();
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
        
        // Draw geometric centers
        drawGeometry(renderer);
        
        // Draw UI and controls
        drawUI(renderer);
    }

    void cleanup() override {
        std::cout << "Scalar Field 03: SDF Blending & Tower cleanup" << std::endl;
    }

private:
    void generateBaseFields() {
        // Generate rectangle field (lower)
        m_field_lower.clear_field();
        m_field_lower.apply_scalar_rect(m_rectCenter, m_rectSize, 0.0f);
        
        // Generate circle field (upper)
        m_field_upper.clear_field();
        m_field_upper.apply_scalar_circle(m_circleCenter, m_circleRadius);
    }
    
    void generateBlendedField() {
        // Start with lower field (rectangle)
        m_blended_field = ScalarField2D(m_field_lower);
        
        // Apply smooth minimum blending with upper field (circle)
        m_blended_field.boolean_smin(m_field_upper, m_blendFactor);
    }
    
    void generateTowerContours() {
        // Extract contours at each tower level
        for (size_t i = 0; i < m_towerLevels.size(); ++i) {
            float threshold = 0.1f; // Adjust threshold for each level
            ContourData contours = m_blended_field.get_contours(threshold);
            m_towerContours[i] = contours.line_segments;
        }
    }
    
    void drawStandardView(Renderer& renderer) {
        // Choose which field to display
        ScalarField2D* displayField = &m_field_lower;
        if (b_computeBlend) {
            displayField = &m_blended_field;
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
        // Left side: original blended field (scaled and translated)
        renderer.pushMatrix();
        Mat4 leftTransform = Mat4::translation(Vec3(-25, 0, 0)) * Mat4::scale(Vec3(0.7f, 0.7f, 0.7f));
        renderer.multMatrix(leftTransform);

        if (d_drawField && b_computeBlend) {
            m_blended_field.draw_points(renderer, 3);
        }
        if (d_drawContours && b_computeBlend) {
            drawContours(renderer, m_blended_field);
        }

        renderer.popMatrix();

        // Right side: tower visualization
        drawTowerVisualization(renderer);
    }
    
    void drawTowerVisualization(Renderer& renderer) {
        // Draw tower contours at different Z levels without matrix transformations
        // Position tower further away to prevent overlap
        Vec3 towerOffset(60, 0, 0);

        for (size_t i = 0; i < m_towerLevels.size(); ++i) {
            float z = m_towerLevels[i];

            // Color gradient from magenta (bottom) to purple (top)
            float t = static_cast<float>(i) / (m_towerLevels.size() - 1);
            Vec3 color = Vec3(1.0f, 0.2f * (1.0f - t), 1.0f - 0.3f * t); // Magenta to purple
            renderer.setColor(color);

            // Draw contours at this level with tower offset
            for (const auto& segment : m_towerContours[i]) {
                Vec3 start = segment.first + towerOffset + Vec3(0, 0, z);
                Vec3 end = segment.second + towerOffset + Vec3(0, 0, z);
                renderer.drawLine(start, end, color, 2.0f);
            }

            // Draw level indicator every 5 levels to reduce clutter
            if (i % 5 == 0) {
                renderer.drawText("Z=" + std::to_string(static_cast<int>(z)),
                                towerOffset + Vec3(-40, -40, z), 0.8f);
            }
        }
    }
    
    void drawContours(Renderer& renderer, const ScalarField2D& field) {
        renderer.setColor(Vec3(1.0f, 1.0f, 1.0f));

        // Draw single contour line
        float threshold = 0.0f;
        field.drawIsocontours(renderer, threshold);
    }
    
    void drawGeometry(Renderer& renderer) {
        // Draw rectangle center in blue
        renderer.setColor(Vec3(0.2f, 0.2f, 1.0f));
        renderer.drawPoint(m_rectCenter, Vec3(0.2f, 0.2f, 1.0f), 8.0f);
        renderer.drawText("RECT", m_rectCenter + Vec3(0, 0, 5), 1.0f);
    }
    
    void drawUI(Renderer& renderer) {
        // Title and description
        renderer.setColor(Vec3(1.0f, 1.0f, 1.0f));
        renderer.drawString(getName(), 10, 30);
        renderer.drawString("Educational sketch: SDF blending with tower visualization", 10, 50);
        
        // FPS display
        renderer.setColor(Vec3(0.0f, 1.0f, 1.0f));
        renderer.drawString("FPS: " + std::to_string(Application::getInstance()->getFPS()), 10, 80);
        
        // Current mode display
        renderer.setColor(Vec3(1.0f, 1.0f, 0.0f));
        std::string mode = b_computeBlend ? "BLENDED" : "BASE RECTANGLE";
        renderer.drawString("Current Mode: " + mode, 10, 110);
        
        // Blend factor
        renderer.setColor(Vec3(0.8f, 0.8f, 0.8f));
        renderer.drawString("Blend Factor: " + std::to_string(m_blendFactor).substr(0, 4), 10, 140);
        
        // Tower info
        if (d_drawTower) {
            renderer.drawString("Tower Levels: 5 (Z=0,3,6,9,12)", 10, 170);
        }
        
        // Controls
        renderer.setColor(Vec3(0.7f, 0.7f, 0.7f));
        renderer.drawString("Controls:", 10, 200);
        renderer.drawString("'B' - Toggle Blend Computation", 10, 220);
        renderer.drawString("'T' - Toggle Tower Visualization", 10, 240);
        renderer.drawString("'+'/'-' - Adjust Blend Factor", 10, 260);
        renderer.drawString("'F' - Toggle Field Visualization", 10, 280);
        renderer.drawString("'C' - Toggle Contours", 10, 300);
        renderer.drawString("'V' - Toggle Value Display", 10, 320);
        
        // Status indicators
        renderer.setColor(Vec3(0.5f, 1.0f, 0.5f));
        renderer.drawString("Blend: " + std::string(b_computeBlend ? "ON" : "OFF"), 10, 350);
        renderer.drawString("Tower: " + std::string(d_drawTower ? "ON" : "OFF"), 10, 370);
        renderer.drawString("Field: " + std::string(d_drawField ? "ON" : "OFF"), 10, 390);
    }

public:
    // Input handling
    bool onKeyPress(unsigned char key, int x, int y) override {
        switch (key) {
            case 'b':
            case 'B': // Toggle blend computation
                b_computeBlend = !b_computeBlend;
                if (b_computeBlend) {
                    generateBlendedField();
                }
                return true;
                
            case 't':
            case 'T': // Toggle tower visualization
                d_drawTower = !d_drawTower;
                if (d_drawTower && b_computeBlend) {
                    generateTowerContours();
                }
                return true;
                
            case '+':
            case '=': // Increase blend factor
                m_blendFactor = std::min(m_blendFactor + 0.2f, 5.0f);
                if (b_computeBlend) {
                    generateBlendedField();
                    if (d_drawTower) generateTowerContours();
                }
                return true;
                
            case '-':
            case '_': // Decrease blend factor
                m_blendFactor = std::max(m_blendFactor - 0.2f, 0.2f);
                if (b_computeBlend) {
                    generateBlendedField();
                    if (d_drawTower) generateTowerContours();
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
};

// Register the sketch with alice2
ALICE2_REGISTER_SKETCH_AUTO(ScalarField03BlendingSketch)

#endif // __MAIN__
