// alice2 Bezier Blend Sketch
// Demonstrates Bezier blending of scalar fields with contour extraction and z-offset visualization

#define __MAIN__
#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>
#include <computeGeom/scalarField.h>
#include <vector>
#include <cmath>

#include <map>

using namespace alice2;

class BezierSketch : public ISketch {
private:
    // Scalar fields with consistent dimensions: 100x100 grid, bounds (-50, -50) to (50, 50)
    ScalarField2D m_field_0;        // First field (circles)
    ScalarField2D m_field_1;        // Second field (rectangle)
    ScalarField2D m_field_2;        // Third field (line)
    ScalarField2D m_temp_field;     // Temporary field for operations
    ScalarField2D m_blended_field;  // Result of Bezier blending

    // Tower visualization data
    std::vector<float> m_towerLevels;
    std::map<size_t, std::vector<std::pair<Vec3, Vec3>>> m_towerContours;

    // Animation and timing
    float m_time;
    float m_blendParameter;  // t parameter for Bezier blend [0,1]

    // Boolean flags for computation controls (prefix with "b_")
    bool b_computeBezier;
    bool b_animateBlend;

    // Boolean flags for display controls (prefix with "d_")
    bool d_drawField;
    bool d_drawContours;
    bool d_drawTower;
    bool d_drawValues;

    // Colors
    Vec3 magenta = Vec3(1.0f, 0.0f, 1.0f);
    Vec3 purple = Vec3(0.5f, 0.0f, 1.0f);
    Vec3 cyan = Vec3(0.0f, 1.0f, 1.0f);

public:
    BezierSketch() : m_time(0.0f), m_blendParameter(0.5f),
                     b_computeBezier(false), b_animateBlend(false),
                     d_drawField(true), d_drawContours(true), d_drawTower(false), d_drawValues(false) {
        // Initialize tower levels (20 floors, 3 units apart)
        for (int i = 0; i < 20; ++i) {
            m_towerLevels.push_back(static_cast<float>(i * 3));
        }
    }

    ~BezierSketch() = default;

    // Sketch information
    std::string getName() const override {
        return "Bezier Blend Sketch";
    }

    std::string getDescription() const override {
        return "Bezier blending of scalar fields with contour extraction and z-offset visualization";
    }

    std::string getAuthor() const override {
        return "alice2 User";
    }

    // Sketch lifecycle
    void setup() override {
        scene().setBackgroundColor(Vec3(0.95f, 0.95f, 0.95f));
        scene().setShowGrid(false);
        scene().setGridSize(25.0f);
        scene().setGridDivisions(4);
        scene().setShowAxes(true);
        scene().setAxesLength(10.0f);

        std::cout << "Bezier Blend Sketch loaded" << std::endl;
        std::cout << "Field dimensions: 100x100 grid, bounds (-50,-50) to (50,50)" << std::endl;
        std::cout << "Press 't' to compute Bezier blend and extract contours" << std::endl;
        std::cout << "Press 'a' to toggle blend animation" << std::endl;

        // Initialize fields with consistent dimensions
        generateBaseFields();
    }

    void update(float deltaTime) override {
        m_time += deltaTime;

        // Animate blend parameter if enabled
        if (b_animateBlend) {
            m_blendParameter = 0.5f + 0.5f * std::sin(m_time * 0.5f); // Oscillate between 0 and 1
        }

        // Regenerate blended field if computation is enabled
        if (b_computeBezier) {
            generateBezierBlend();
            generateTowerContours();
            b_computeBezier = false;
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
        std::cout << "Bezier Blend Sketch cleanup" << std::endl;
    }
    // Input handling
    bool onKeyPress(unsigned char key, int x, int y) override {
        switch (key) {
            case 't':
            case 'T':
                b_computeBezier = true;
                std::cout << "Computing Bezier blend with t=" << m_blendParameter << std::endl;
                return true;

            case 'a':
            case 'A':
                b_animateBlend = !b_animateBlend;
                std::cout << "Blend animation: " << (b_animateBlend ? "ON" : "OFF") << std::endl;
                return true;

            case 'f':
            case 'F':
                d_drawField = !d_drawField;
                std::cout << "Field visualization: " << (d_drawField ? "ON" : "OFF") << std::endl;
                return true;

            case 'c':
            case 'C':
                d_drawContours = !d_drawContours;
                std::cout << "Contour visualization: " << (d_drawContours ? "ON" : "OFF") << std::endl;
                return true;

            case 'z':
            case 'Z':
                d_drawTower = !d_drawTower;
                std::cout << "Tower visualization: " << (d_drawTower ? "ON" : "OFF") << std::endl;
                return true;

            case 'v':
            case 'V':
                d_drawValues = !d_drawValues;
                std::cout << "Value display: " << (d_drawValues ? "ON" : "OFF") << std::endl;
                return true;

            case '+':
            case '=':
                if (!b_animateBlend) {
                    m_blendParameter = std::min(1.0f, m_blendParameter + 0.1f);
                    std::cout << "Blend parameter: " << m_blendParameter << std::endl;
                    b_computeBezier = true;
                }
                return true;

            case '-':
            case '_':
                if (!b_animateBlend) {
                    m_blendParameter = std::max(0.0f, m_blendParameter - 0.1f);
                    std::cout << "Blend parameter: " << m_blendParameter << std::endl;
                    b_computeBezier = true;
                }
                return true;
        }
        return false;
    }

    bool onMousePress(int button, int state, int x, int y) override {
        return false;
    }

    bool onMouseMove(int x, int y) override {
        return false;
    }

private:
    // Compute "n choose k" iteratively to avoid overflow
    double binomial(int n, int k) const {
        if (k < 0 || k > n) return 0.0;
        if (k > n - k) k = n - k;
        double c = 1.0;
        for (int i = 1; i <= k; ++i) {
            c *= (n - (k - i));
            c /= i;
        }
        return c;
    }

    // Blend a vector of scalar fields F[0..n] at parameter t in [0,1]
    double bezier_blend(const std::vector<double>& F, double t) const {
        int n = (int)F.size() - 1;
        double one_minus_t = 1.0 - t;
        double result = 0.0;

        for (int i = 0; i <= n; ++i) {
            double B = binomial(n, i)
                     * std::pow(t, i)
                     * std::pow(one_minus_t, n - i);
            result += B * F[i];
        }
        return result;
    }

    void generateBaseFields() {
        // Initialize all fields with consistent dimensions
        m_field_0 = ScalarField2D(Vec3(-50, -50, 0), Vec3(50, 50, 0), 100, 100);
        m_field_1 = ScalarField2D(Vec3(-50, -50, 0), Vec3(50, 50, 0), 100, 100);
        m_field_2 = ScalarField2D(Vec3(-50, -50, 0), Vec3(50, 50, 0), 100, 100);
        m_temp_field = ScalarField2D(Vec3(-50, -50, 0), Vec3(50, 50, 0), 100, 100);
        m_blended_field = ScalarField2D(Vec3(-50, -50, 0), Vec3(50, 50, 0), 100, 100);

        // Field 0: Three circles (triangle pattern)
        m_field_0.clear_field();
        m_temp_field.clear_field();
        m_temp_field.apply_scalar_circle(Vec3(-20, -15, 0), 12.0f);
        m_field_0.boolean_union(m_temp_field);

        m_temp_field.clear_field();
        m_temp_field.apply_scalar_circle(Vec3(20, -15, 0), 12.0f);
        m_field_0.boolean_union(m_temp_field);

        m_temp_field.clear_field();
        m_temp_field.apply_scalar_circle(Vec3(0, 20, 0), 12.0f);
        m_field_0.boolean_union(m_temp_field);

        // Field 1: Rectangle
        m_field_1.clear_field();
        m_field_1.apply_scalar_rect(Vec3(0, 0, 0), Vec3(25, 15, 0), 0.0f);

        // Field 2: Cross pattern (two perpendicular rectangles)
        m_field_2.clear_field();
        m_temp_field.clear_field();
        m_temp_field.apply_scalar_rect(Vec3(0, 0, 0), Vec3(30, 8, 0), 0.0f);
        m_field_2.boolean_union(m_temp_field);

        m_temp_field.clear_field();
        m_temp_field.apply_scalar_rect(Vec3(0, 0, 0), Vec3(8, 30, 0), 0.0f);
        m_field_2.boolean_union(m_temp_field);

        std::cout << "Base fields generated: Circles, Rectangle, Cross" << std::endl;
    }

    void generateBezierBlend() {
        m_blended_field.clear_field();

        // Get field values for all three fields
        std::vector<float> values_0 = m_field_0.get_values();
        std::vector<float> values_1 = m_field_1.get_values();
        std::vector<float> values_2 = m_field_2.get_values();

        // Apply Bezier blending at each grid point
        std::vector<float> blended_values(values_0.size());

        for (size_t i = 0; i < values_0.size(); ++i) {
            // Create vector of field values at this point
            std::vector<double> F = {
                static_cast<double>(values_0[i]),
                static_cast<double>(values_1[i]),
                static_cast<double>(values_2[i])
            };

            // Apply Bezier blend
            double blended = bezier_blend(F, static_cast<double>(m_blendParameter));
            blended_values[i] = static_cast<float>(blended);
        }

        // Set the blended values
        m_blended_field.set_values(blended_values);

        std::cout << "Bezier blend computed with t=" << m_blendParameter << std::endl;
    }

    void generateTowerContours() {
        m_towerContours.clear();

        // Extract contours at each tower level
        for (size_t i = 0; i < m_towerLevels.size(); ++i) {
            // Use different blend parameters for each level
            float level_t = static_cast<float>(i) / (m_towerLevels.size() - 1);

            // Generate blend for this level
            std::vector<float> values_0 = m_field_0.get_values();
            std::vector<float> values_1 = m_field_1.get_values();
            std::vector<float> values_2 = m_field_2.get_values();

            std::vector<float> level_values(values_0.size());
            for (size_t j = 0; j < values_0.size(); ++j) {
                std::vector<double> F = {
                    static_cast<double>(values_0[j]),
                    static_cast<double>(values_1[j]),
                    static_cast<double>(values_2[j])
                };
                double blended = bezier_blend(F, static_cast<double>(level_t));
                level_values[j] = static_cast<float>(blended);
            }

            // Create temporary field for this level
            ScalarField2D level_field = m_blended_field;
            level_field.set_values(level_values);

            // Get contours
            float threshold = 0.1f;
            ContourData contours = level_field.get_contours(threshold);
            m_towerContours[i] = contours.line_segments;
        }

        std::cout << "Tower contours generated for " << m_towerLevels.size() << " levels" << std::endl;
    }

    void drawStandardView(Renderer& renderer) {
        // Draw scalar field visualization
        if (d_drawField) {
            m_blended_field.draw_points(renderer, 2);
        }

        // Draw scalar values as text
        if (d_drawValues) {
            m_blended_field.draw_values(renderer, 12);
        }

        // Draw contours
        if (d_drawContours) {
            renderer.setColor(magenta);
            drawContours(renderer, m_blended_field);
        }
    }

    void drawSideBySideView(Renderer& renderer) {
        // Left side: original blended field
        drawStandardView(renderer);

        // Right side: tower visualization
        drawTowerVisualization(renderer);
    }

    void drawTowerVisualization(Renderer& renderer) {
        Vec3 towerOffset = Vec3(100, 0, 0); // Offset tower to the right

        for (size_t i = 0; i < m_towerLevels.size(); ++i) {
            float z = m_towerLevels[i];

            // Color gradient from magenta (bottom) to purple (top)
            float t = static_cast<float>(i) / (m_towerLevels.size() - 1);
            Vec3 color = Vec3::lerp(magenta, purple, t);
            renderer.setColor(color);

            // Draw contours at this level with tower offset
            for (const auto& segment : m_towerContours[i]) {
                Vec3 start = segment.first + towerOffset + Vec3(0, 0, z);
                Vec3 end = segment.second + towerOffset + Vec3(0, 0, z);
                renderer.drawLine(start, end, color, 2.0f);
            }

            // Draw level indicator every 5 levels to reduce clutter
            if (i % 5 == 0) {
                renderer.drawText("t=" + std::to_string(static_cast<float>(i) / (m_towerLevels.size() - 1)).substr(0, 4),
                                towerOffset + Vec3(-40, -40, z), 0.8f);
            }
        }
    }

    void drawContours(Renderer& renderer, const ScalarField2D& field) {
        // Draw single contour line
        float threshold = 0.0f;
        field.drawIsocontours(renderer, threshold);
    }

    void drawGeometry(Renderer& renderer) {
        // Draw field centers with distinct colors
        renderer.setColor(cyan);
        renderer.drawPoint(Vec3(-20, -15, 0), cyan, 6.0f);
        renderer.drawText("F0", Vec3(-20, -15, 5), 1.0f);

        renderer.drawPoint(Vec3(20, -15, 0), cyan, 6.0f);
        renderer.drawPoint(Vec3(0, 20, 0), cyan, 6.0f);

        renderer.setColor(Vec3(0.2f, 0.2f, 1.0f));
        renderer.drawPoint(Vec3(0, 0, 0), Vec3(0.2f, 0.2f, 1.0f), 6.0f);
        renderer.drawText("F1", Vec3(0, 0, 5), 1.0f);

        renderer.setColor(Vec3(1.0f, 0.5f, 0.0f));
        renderer.drawPoint(Vec3(0, 0, 0), Vec3(1.0f, 0.5f, 0.0f), 4.0f);
        renderer.drawText("F2", Vec3(0, 0, 10), 1.0f);
    }

    void drawUI(Renderer& renderer) {
        // Draw control instructions
        std::vector<std::string> instructions = {
            "Bezier Blend Controls:",
            "T - Compute blend",
            "A - Toggle animation",
            "+/- - Adjust blend parameter",
            "F - Toggle field display",
            "C - Toggle contours",
            "Z - Toggle tower view",
            "V - Toggle values",
            "",
            "Blend parameter: " + std::to_string(m_blendParameter).substr(0, 4),
            "Animation: " + std::string(b_animateBlend ? "ON" : "OFF")
        };

        float y_offset = 20.0f;
        for (const auto& instruction : instructions) {
            renderer.setColor(Vec3(0.2f, 0.2f, 0.2f));
            renderer.drawString(instruction, 20, y_offset);
            y_offset += 20.0f;
        }
    }
};

// Register the sketch with alice2 (both old and new systems)
//ALICE2_REGISTER_SKETCH_AUTO(BezierSketch)

#endif // __MAIN__
