 // alice2 Empty Sketch Template
// Minimal template for creating a new user sketch in alice2

#define __MAIN__
#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>

using namespace alice2;

class EmptySketch : public ISketch {
public:
    EmptySketch() = default;
    ~EmptySketch() = default;

    // Sketch information
    std::string getName() const override {
        return "Empty Sketch";
    }

    std::string getDescription() const override {
        return "A minimal empty sketch template";
    }

    std::string getAuthor() const override {
        return "alice2 User";
    }

    // Sketch lifecycle
    void setup() override {
        // Initialize your sketch here
    }

    void update(float deltaTime) override {
        // Update your sketch logic here
    }

    void draw(Renderer& renderer, Camera& camera) override {
        // Draw your custom content here
    }

    // Input handling (optional)
    bool onKeyPress(unsigned char key, int x, int y) override {
        // Handle keyboard input
        return false;
    }

    bool onMousePress(int button, int state, int x, int y) override {
        // Handle mouse button input
        return false;
    }

    bool onMouseMove(int x, int y) override {
        // Handle mouse movement
        return false;
    }
};

// Register the sketch with alice2 (both old and new systems)
//ALICE2_REGISTER_SKETCH_AUTO(EmptySketch)

#endif // __MAIN__
