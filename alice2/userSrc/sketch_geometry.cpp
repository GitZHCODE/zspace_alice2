// alice2 Geometry Sketch
// Demonstrates basic geometric shapes and transformations

#define __MAIN__
#ifdef __MAIN__

#include "../include/alice2.h"
#include "../src/sketches/SketchRegistry.h"
#include <cmath>

using namespace alice2;

class GeometrySketch : public ISketch {
private:
    float m_time;
    float m_rotationSpeed;

public:
    GeometrySketch() : m_time(0.0f), m_rotationSpeed(1.0f) {}
    ~GeometrySketch() = default;

    // Sketch information
    std::string getName() const override {
        return "Geometry Sketch";
    }

    std::string getDescription() const override {
        return "Demonstrates basic geometric shapes with rotation animations";
    }

    std::string getAuthor() const override {
        return "alice2 Examples";
    }

    // Sketch lifecycle
    void setup() override {
        // Set a dark blue background
        scene().setBackgroundColor(Vec3(0.1f, 0.1f, 0.3f));
        std::cout << "Geometry Sketch loaded - Background set to dark blue" << std::endl;

        // Example: Enable grid
        scene().setShowGrid(true);
        scene().setGridSize(10.0f);
        scene().setGridDivisions(10);

        // Example: Enable axes
        scene().setShowAxes(true);
        scene().setAxesLength(2.0f);
    }

    void update(float deltaTime) override {
        m_time += deltaTime * m_rotationSpeed;
    }

    void draw(Renderer& renderer, Camera& camera) override {
        // Draw rotating cubes
        for (int i = 0; i < 3; i++) {
            renderer.pushMatrix();
            
            // Position cubes in a line
            Vec3 position(i * 3.0f - 3.0f, 0, 1);
            Mat4 translation = Mat4::translation(position);
            
            // Rotate each cube at different speeds
            float rotation = m_time * (1.0f + i * 0.5f);
            Mat4 rotationMatrix = Mat4::rotation(ZUp::UP, rotation);
            
            Mat4 transform = translation * rotationMatrix;
            renderer.multMatrix(transform);
            
            // Different colors for each cube
            Vec3 colors[] = {
                Vec3(1.0f, 0.3f, 0.3f), // Red
                Vec3(0.3f, 1.0f, 0.3f), // Green
                Vec3(0.3f, 0.3f, 1.0f)  // Blue
            };
            
            renderer.drawCube(1.5f, colors[i]);
            renderer.popMatrix();
        }

        // Draw a spiral of points
        int numPoints = 50;
        for (int i = 0; i < numPoints; i++) {
            float t = i / float(numPoints - 1);
            float angle = t * 4.0f * PI + m_time;
            float height = t * 8.0f - 4.0f;
            float radius = 2.0f + std::sin(m_time * 2.0f) * 0.5f;
            
            Vec3 pos(
                radius * std::cos(angle),
                radius * std::sin(angle),
                height
            );
            
            // Color based on height
            Vec3 color(
                0.5f + 0.5f * std::sin(t * PI),
                0.5f + 0.5f * std::cos(t * PI),
                0.8f
            );
            
            renderer.drawPoint(pos, color, 8.0f);
        }

        // Draw connecting lines between cubes
        renderer.setColor(Vec3(1.0f, 1.0f, 0.5f), 0.7f);
        for (int i = 0; i < 2; i++) {
            Vec3 start(i * 3.0f - 3.0f, 0, 1);
            Vec3 end((i + 1) * 3.0f - 3.0f, 0, 1);
            renderer.drawLine(start, end, Vec3(1.0f, 1.0f, 0.5f), 3.0f);
        }
    }

    void cleanup() override {
        std::cout << "Geometry Sketch cleanup" << std::endl;
    }

    // Input handling
    bool onKeyPress(unsigned char key, int x, int y) override {
        switch (key) {
            case '+':
            case '=':
                m_rotationSpeed += 0.2f;
                std::cout << "Rotation speed increased to " << m_rotationSpeed << std::endl;
                return true;
                
            case '-':
            case '_':
                m_rotationSpeed = std::max(0.1f, m_rotationSpeed - 0.2f);
                std::cout << "Rotation speed decreased to " << m_rotationSpeed << std::endl;
                return true;
                
            case '0':
                m_rotationSpeed = 1.0f;
                std::cout << "Rotation speed reset to " << m_rotationSpeed << std::endl;
                return true;
        }
        return false; // Not handled
    }
};

// Register the sketch with alice2
ALICE2_REGISTER_SKETCH_AUTO(GeometrySketch)

#endif // __MAIN__
