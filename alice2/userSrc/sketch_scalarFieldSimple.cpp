// alice2 Simple Scalar Field Test
// A minimal test to verify scalar field functionality

#define __MAIN__
#ifdef __MAIN__

#include "../include/alice2.h"
#include "../src/sketches/SketchRegistry.h"
#include "scalarField.h"
#include <cmath>

using namespace alice2;

class SimpleScalarFieldSketch : public ISketch {
private:
    ScalarField2D m_scalarField;
    std::vector<std::pair<Vec3, Vec3>> m_isolines;
    float m_time;
    bool m_fieldGenerated;
    bool d_iso;

public:
    SimpleScalarFieldSketch() 
        : m_time(0.0f)
        , m_fieldGenerated(false)
    {}
    ~SimpleScalarFieldSketch() = default;

    // Sketch information
    std::string getName() const override {
        return "Simple Scalar Field Test";
    }

    std::string getDescription() const override {
        return "Minimal scalar field test with basic circle SDF";
    }

    std::string getAuthor() const override {
        return "alice2 Test";
    }

    // Sketch lifecycle
    void setup() override {
        scene().setBackgroundColor(Vec3(0.1f, 0.1f, 0.1f));
        std::cout << "Simple Scalar Field Test loaded" << std::endl;

        scene().setShowGrid(false);
        scene().setGridSize(25.0f);
        scene().setGridDivisions(6);
        scene().setShowAxes(true);
        scene().setAxesLength(5.0f);

        // Generate a simple circle field
        try {
            std::cout << "Creating ScalarField2D..." << std::endl;
            m_scalarField = ScalarField2D();
            std::cout << "ScalarField2D created successfully" << std::endl;
            
            std::cout << "Clearing field..." << std::endl;
            m_scalarField.clearField();
            std::cout << "Field cleared" << std::endl;
            
            std::cout << "Adding circle SDF..." << std::endl;
            m_scalarField.addCircleSDF(Vec3(0, 0, 0), 25.0f);
            std::cout << "Circle SDF added successfully" << std::endl;
            
            m_fieldGenerated = true;
        }
        catch (const std::exception& e) {
            std::cout << "Error in setup: " << e.what() << std::endl;
            m_fieldGenerated = false;
        }
    }

    void update(float deltaTime) override {
        m_time += deltaTime;
    }

    void draw(Renderer& renderer, Camera& camera) override {
        // Draw a test point to verify rendering works
        renderer.drawPoint(Vec3(0, 0, 10), Vec3(1.0f, 0.0f, 0.0f), 10.0f);
        
        if (m_fieldGenerated) {
            try {
                // Draw only a few field points as a test
                m_scalarField.drawFieldPoints(renderer);
            }
            catch (const std::exception& e) {
                std::cout << "Error drawing field: " << e.what() << std::endl;
            }
        }

        if(d_iso)
            m_scalarField.drawIsocontours(renderer, 0.5f);


        // 2D text rendering
        renderer.setColor(Vec3(1.0f, 1.0f, 1.0f));
        renderer.drawString(getName(), 10, 30);
        renderer.drawString(getDescription(), 10, 50);

        renderer.setColor(Vec3(0.0f, 1.0f, 1.0f));
        renderer.drawString("FPS: " + std::to_string((Application::getInstance()->getFPS())), 10, 70);

        renderer.setColor(Vec3(1.0f, 1.0f, 0.0f));
        renderer.drawString("Field Generated: " + std::string(m_fieldGenerated ? "YES" : "NO"), 10, 100);

        renderer.setColor(Vec3(0.75f, 0.75f, 0.75f));
        renderer.drawString("'R' - Regenerate field", 10, 140);
        renderer.drawString("'C' - Regenerate iso contours", 10, 160);
    }

    void cleanup() override {
        std::cout << "Simple Scalar Field Test cleanup" << std::endl;
    }

    // Input handling
    bool onKeyPress(unsigned char key, int x, int y) override {
        switch (key) {
            case 'r':
            case 'R':
                try {
                    std::cout << "Regenerating field..." << std::endl;
                    m_scalarField.clearField();
                    m_scalarField.addCircleSDF(Vec3(0, 0, 0), 25.0f);
                    m_fieldGenerated = true;
                    std::cout << "Field regenerated successfully" << std::endl;
                }
                catch (const std::exception& e) {
                    std::cout << "Error regenerating field: " << e.what() << std::endl;
                    m_fieldGenerated = false;
                }
                return true;
            case 'c':
            case 'C':
            //m_scalarField.computeIsocontours(0.5, m_isolines);
            d_iso = true;
            return true;

        }
        return false;
    }
};

// Register the sketch with alice2
ALICE2_REGISTER_SKETCH_AUTO(SimpleScalarFieldSketch)

#endif // __MAIN__
