// GPU-Accelerated Scalar Field Performance Comparison
// Demonstrates GPU vs CPU performance for boolean operations

#include <alice2.h>
#include <sketches/SketchRegistry.h>
#include <computeGeom/scalarField.h>
#include <core/ShaderManager.h>
#include <chrono>
#include <fstream>

using namespace alice2;

class FieldGPUSketch : public ISketch {
private:
    // Scalar fields for testing
    ScalarField2D m_fieldA, m_fieldB, m_resultGPU, m_resultCPU;
    
    // Performance timing
    float m_cpuTime, m_gpuTime;
    int m_testIterations;
    bool m_testRunning;
    
    // Display options (following established patterns)
    bool b_computeGPU;
    bool b_computeCPU;
    bool b_runPerformanceTest;
    bool d_drawField;
    bool d_drawContours;
    bool d_showGPUResult;
    bool d_showCPUResult;
    bool d_showDifference;
    
    // Test parameters
    Vec3 m_circleACenter, m_circleBCenter;
    float m_circleARadius, m_circleBRadius;
    
    // GPU initialization
    std::shared_ptr<ShaderManager> m_shaderManager;
    bool m_gpuInitialized;

public:
    FieldGPUSketch() 
        : m_fieldA(Vec3(-50, -50, 0), Vec3(50, 50, 0), 4000, 4000)
        , m_fieldB(Vec3(-50, -50, 0), Vec3(50, 50, 0), 4000, 4000)
        , m_resultGPU(Vec3(-50, -50, 0), Vec3(50, 50, 0), 4000, 4000)
        , m_resultCPU(Vec3(-50, -50, 0), Vec3(50, 50, 0), 4000, 4000)
        , m_cpuTime(0.0f), m_gpuTime(0.0f), m_testIterations(10)
        , m_testRunning(false), b_computeGPU(true), b_computeCPU(true)
        , b_runPerformanceTest(false), d_drawField(true), d_drawContours(true)
        , d_showGPUResult(true), d_showCPUResult(false), d_showDifference(false)
        , m_circleACenter(-15, 0, 0), m_circleBCenter(15, 0, 0)
        , m_circleARadius(20.0f), m_circleBRadius(18.0f)
        , m_gpuInitialized(false) {
    }

    std::string getName() const override {
        return "GPU Scalar Field Performance";
    }

    std::string getDescription() const override {
        return "Performance comparison between CPU and GPU scalar field boolean operations";
    }

    std::string getAuthor() const override {
        return "alice2 GPU System";
    }

    void setup() override {
        scene().setBackgroundColor(Vec3(0.05f, 0.05f, 0.1f));
        scene().setShowGrid(false);
        scene().setGridSize(25.0f);
        scene().setGridDivisions(4);
        scene().setShowAxes(true);
        scene().setAxesLength(10.0f);

        std::cout << "GPU Scalar Field Performance Comparison loaded" << std::endl;
        std::cout << "Field dimensions: 1000x1000 grid, bounds (-50,-50) to (50,50)" << std::endl;

        // Write debug info to file
        std::ofstream debugFile("gpu_debug.txt");
        debugFile << "GPU Scalar Field Debug Log\n";
        debugFile << "==========================\n";
        debugFile.close();

        // Initialize GPU system
        initializeGPU();

        // Generate initial test fields
        generateTestFields();

        // Run initial boolean operations
        if (b_computeGPU) performGPUOperation();
        if (b_computeCPU) performCPUOperation();

        // Write GPU status to debug file
        std::ofstream debugFile2("gpu_debug.txt", std::ios::app);
        debugFile2 << "GPU Initialized: " << (m_gpuInitialized ? "YES" : "NO") << "\n";
        debugFile2 << "GPU Enabled: " << (ScalarField2D::isGPUEnabled() ? "YES" : "NO") << "\n";
        debugFile2.close();
    }

    void update(float deltaTime) override {
        // Run performance test if requested
        if (b_runPerformanceTest && !m_testRunning) {
            runPerformanceTest();
            b_runPerformanceTest = false;
        }
    }

    void draw(Renderer& renderer, Camera& camera) override {
        // Choose which field to display
        ScalarField2D* displayField = nullptr;
        if (d_showGPUResult && m_gpuInitialized) {
            displayField = &m_resultGPU;
        } else if (d_showCPUResult) {
            displayField = &m_resultCPU;
        } else if (d_showDifference) {
            // Show difference field (would need implementation)
            displayField = &m_resultCPU; // Fallback
        }
        
        // Draw scalar field visualization
        if (displayField && d_drawField) {
            displayField->draw_points(renderer, 2); // Every 2nd point for performance
        }
        
        // Draw contours
        if (displayField && d_drawContours) {
            drawContours(renderer, *displayField);
        }
        
        // Draw test geometry
        drawTestGeometry(renderer);
        
        // Draw UI and performance metrics
        drawUI(renderer);
    }

    void cleanup() override {
        std::cout << "GPU Scalar Field Performance Comparison cleanup" << std::endl;
    }

    bool onKeyPress(unsigned char key, int x, int y) override {
        switch (key) {
            case 'g': case 'G':
                b_computeGPU = !b_computeGPU;
                if (b_computeGPU && m_gpuInitialized) performGPUOperation();
                std::cout << "GPU computation: " << (b_computeGPU ? "ON" : "OFF") << std::endl;
                return true;
                
            case 'c': case 'C':
                b_computeCPU = !b_computeCPU;
                if (b_computeCPU) performCPUOperation();
                std::cout << "CPU computation: " << (b_computeCPU ? "ON" : "OFF") << std::endl;
                return true;
                
            case 't': case 'T':
                b_runPerformanceTest = true;
                std::cout << "Running performance test..." << std::endl;
                return true;
                
            case 'f': case 'F':
                d_drawField = !d_drawField;
                std::cout << "Field visualization: " << (d_drawField ? "ON" : "OFF") << std::endl;
                return true;
                
            case 'o': case 'O':
                d_drawContours = !d_drawContours;
                std::cout << "Contour visualization: " << (d_drawContours ? "ON" : "OFF") << std::endl;
                return true;
                
            case '1':
                d_showGPUResult = true; d_showCPUResult = false; d_showDifference = false;
                std::cout << "Showing GPU result" << std::endl;
                return true;
                
            case '2':
                d_showGPUResult = false; d_showCPUResult = true; d_showDifference = false;
                std::cout << "Showing CPU result" << std::endl;
                return true;
                
            case '3':
                d_showGPUResult = false; d_showCPUResult = false; d_showDifference = true;
                std::cout << "Showing difference" << std::endl;
                return true;
        }
        return false;
    }

private:
    void initializeGPU() {
        m_shaderManager = std::make_shared<ShaderManager>();
        if (!m_shaderManager->initialize()) {
            std::cout << "Failed to initialize ShaderManager" << std::endl;
            return;
        }

        // Initialize GPU acceleration for scalar fields
        ScalarField2D::initializeGPU(m_shaderManager);
        m_gpuInitialized = ScalarField2D::isGPUEnabled();

        if (m_gpuInitialized) {
            std::cout << "GPU acceleration initialized successfully" << std::endl;
        } else {
            std::cout << "GPU acceleration not available - using CPU fallback" << std::endl;
        }
    }

    void generateTestFields() {
        // Generate field A (circle at left)
        m_fieldA.clear_field();
        m_fieldA.apply_scalar_circle(m_circleACenter, m_circleARadius);
        
        // Generate field B (circle at right)
        m_fieldB.clear_field();
        m_fieldB.apply_scalar_circle(m_circleBCenter, m_circleBRadius);
    }

    void performGPUOperation() {
        if (!m_gpuInitialized) return;
        
        m_resultGPU = m_fieldA;
        m_resultGPU.boolean_union(m_fieldB);
    }

    void performCPUOperation() {
        m_resultCPU = m_fieldA;
        m_resultCPU.boolean_union_fallback(m_fieldB);
    }

    void runPerformanceTest() {
        if (m_testRunning) return;
        m_testRunning = true;
        
        std::cout << "Running performance test with " << m_testIterations << " iterations..." << std::endl;
        
        // Test CPU performance
        auto startTime = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < m_testIterations; ++i) {
            ScalarField2D tempResult = m_fieldA;
            tempResult.boolean_union_fallback(m_fieldB);
        }
        auto endTime = std::chrono::high_resolution_clock::now();
        m_cpuTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        
        // Test GPU performance (if available)
        if (m_gpuInitialized) {
            startTime = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < m_testIterations; ++i) {
                ScalarField2D tempResult = m_fieldA;
                tempResult.boolean_union(m_fieldB);
            }
            endTime = std::chrono::high_resolution_clock::now();
            m_gpuTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        }
        
        m_testRunning = false;
        
        std::cout << "Performance test completed:" << std::endl;
        std::cout << "  CPU time: " << m_cpuTime << " ms" << std::endl;
        if (m_gpuInitialized) {
            std::cout << "  GPU time: " << m_gpuTime << " ms" << std::endl;
            std::cout << "  Speedup: " << (m_cpuTime / m_gpuTime) << "x" << std::endl;
        }
    }

    void drawContours(Renderer& renderer, const ScalarField2D& field) {
        renderer.setColor(Vec3(1.0f, 1.0f, 1.0f)); // White contours
        field.drawIsocontours(renderer, 0.0f);
    }

    void drawTestGeometry(Renderer& renderer) {
        // Draw circle centers
        renderer.setColor(Vec3(0.2f, 1.0f, 0.2f)); // Green for circle A
        renderer.drawPoint(m_circleACenter, Vec3(0.2f, 1.0f, 0.2f), 8.0f);
        renderer.drawText("A", m_circleACenter + Vec3(0, 0, 5), 1.0f);
        
        renderer.setColor(Vec3(1.0f, 0.2f, 0.2f)); // Red for circle B
        renderer.drawPoint(m_circleBCenter, Vec3(1.0f, 0.2f, 0.2f), 8.0f);
        renderer.drawText("B", m_circleBCenter + Vec3(0, 0, 5), 1.0f);
    }

    void drawUI(Renderer& renderer) {
        renderer.setColor(Vec3(1.0f, 1.0f, 1.0f));
        renderer.drawString(getName(), 10, 30);
        renderer.drawString(getDescription(), 10, 50);
        
        // Performance metrics
        renderer.setColor(Vec3(0.0f, 1.0f, 1.0f));
        renderer.drawString("FPS: " + std::to_string(Application::getInstance()->getFPS()), 10, 70);
        
        if (m_cpuTime > 0) {
            renderer.drawString("CPU Time: " + std::to_string(m_cpuTime) + " ms", 10, 90);
        }
        if (m_gpuTime > 0 && m_gpuInitialized) {
            renderer.drawString("GPU Time: " + std::to_string(m_gpuTime) + " ms", 10, 110);
            renderer.drawString("Speedup: " + std::to_string(m_cpuTime / m_gpuTime) + "x", 10, 130);
        }
        
        // Status indicators
        renderer.setColor(Vec3(0.75f, 0.75f, 0.75f));
        int yPos = 160;
        renderer.drawString("GPU Status: " + std::string(m_gpuInitialized ? "ENABLED" : "DISABLED"), 10, yPos);
        yPos += 20;
        renderer.drawString("GPU Compute: " + std::string(b_computeGPU ? "ON" : "OFF"), 10, yPos);
        yPos += 20;
        renderer.drawString("CPU Compute: " + std::string(b_computeCPU ? "ON" : "OFF"), 10, yPos);
        yPos += 20;
        
        // Display mode
        std::string displayMode = d_showGPUResult ? "GPU Result" : 
                                 d_showCPUResult ? "CPU Result" : 
                                 d_showDifference ? "Difference" : "None";
        renderer.drawString("Display: " + displayMode, 10, yPos);
        yPos += 40;
        
        // Controls
        renderer.drawString("Controls:", 10, yPos);
        yPos += 20;
        renderer.drawString("'G' - Toggle GPU computation", 10, yPos);
        yPos += 20;
        renderer.drawString("'C' - Toggle CPU computation", 10, yPos);
        yPos += 20;
        renderer.drawString("'T' - Run performance test", 10, yPos);
        yPos += 20;
        renderer.drawString("'F' - Toggle field visualization", 10, yPos);
        yPos += 20;
        renderer.drawString("'O' - Toggle contour visualization", 10, yPos);
        yPos += 20;
        renderer.drawString("'1' - Show GPU result", 10, yPos);
        yPos += 20;
        renderer.drawString("'2' - Show CPU result", 10, yPos);
        yPos += 20;
        renderer.drawString("'3' - Show difference", 10, yPos);
    }
};

// Register the sketch
ALICE2_REGISTER_SKETCH(FieldGPUSketch)
