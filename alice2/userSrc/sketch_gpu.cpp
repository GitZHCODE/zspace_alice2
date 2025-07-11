// alice2 GPU Scalar Field Test
// Demonstrates minimal GPU-based scalar field implementation with performance comparison

#define __MAIN__
#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>
#include <computeGeom/scalarField_gpu.h>
#include <computeGeom/scalarField.h>
#include <core/ShaderManager.h>
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace alice2;
int RES = 1024;

class GPUSketch : public ISketch {
private:
    // GPU Scalar field instances
    ScalarFieldGPU m_fieldA_gpu;
    ScalarFieldGPU m_fieldB_gpu;
    ScalarFieldGPU m_resultGPU;

    // CPU Scalar field instances for comparison
    ScalarField2D m_fieldA_cpu;
    ScalarField2D m_fieldB_cpu;
    ScalarField2D m_resultCPU;

    // Performance timing results
    struct PerformanceResults {
        float initTime_gpu = 0.0f;
        float initTime_cpu = 0.0f;
        float computeTime_gpu = 0.0f;
        float computeTime_cpu = 0.0f;
        float drawTime_gpu = 0.0f;
        float drawTime_cpu = 0.0f;
        int testIterations = 100;
        bool testsCompleted = false;
    } m_performance;

    // Control flags (following established naming conventions)
    bool b_computeGPU;
    bool b_computeCPU;
    bool b_runPerformanceTest;
    bool b_runInitTest;
    bool b_runComputeTest;
    bool b_runDrawTest;

    // Display flags
    bool d_drawField;
    bool d_showGPUResult;  // true = show GPU result, false = show CPU result
    bool d_showPerformanceInfo;

    // Circle parameters for testing
    Vec3 m_circleACenter;
    Vec3 m_circleBCenter;
    float m_circleARadius;
    float m_circleBRadius;

    // GPU initialization state
    bool m_gpuInitialized;

public:
    GPUSketch()
        : m_fieldA_gpu(Vec3(-50, -50, 0), Vec3(50, 50, 0), RES, RES)
        , m_fieldB_gpu(Vec3(-50, -50, 0), Vec3(50, 50, 0), RES, RES)
        , m_resultGPU(Vec3(-50, -50, 0), Vec3(50, 50, 0), RES, RES)
        , m_fieldA_cpu(Vec3(-50, -50, 0), Vec3(50, 50, 0), RES, RES)
        , m_fieldB_cpu(Vec3(-50, -50, 0), Vec3(50, 50, 0), RES, RES)
        , m_resultCPU(Vec3(-50, -50, 0), Vec3(50, 50, 0), RES, RES)
        , b_computeGPU(true), b_computeCPU(false)
        , b_runPerformanceTest(false), b_runInitTest(false)
        , b_runComputeTest(false), b_runDrawTest(false)
        , d_drawField(true), d_showGPUResult(true), d_showPerformanceInfo(true)
        , m_circleACenter(-15, 0, 0), m_circleBCenter(15, 0, 0)
        , m_circleARadius(20.0f), m_circleBRadius(18.0f)
        , m_gpuInitialized(false) {
    }

    ~GPUSketch() = default;

    // Sketch information
    std::string getName() const override {
        return "GPU Scalar Field Test";
    }

    std::string getDescription() const override {
        return "Minimal GPU-based scalar field implementation with performance comparison";
    }

    std::string getAuthor() const override {
        return "alice2 Educational Series";
    }

    // Sketch lifecycle
    void setup() override {
        // Initialize scene settings
        scene().setBackgroundColor(Vec3(0.05f, 0.05f, 0.1f));
        scene().setShowGrid(false);
        scene().setShowAxes(true);
        scene().setAxesLength(10.0f);
        
        camera().setPosition(Vec3(0,0,200));
        camera().setOrbitCenter(Vec3(0,0,0));
        camera().setOrbitDistance(500);

        // Init field resolution
        std::cout<< "Test Field Resolution: \n";
        std::cin >> RES;

        // Initialize GPU acceleration
        initializeGPU();

        // Generate test fields
        generateTestFields();

        std::cout << "GPU Scalar Field Test loaded" << std::endl;
        std::cout << "Field dimensions: " << RES << "," << RES  << " grid, bounds (-50,-50) to (50,50)" << std::endl;
        std::cout << "GPU enabled: " << (m_gpuInitialized ? "Yes" : "No") << std::endl;
    }

    void update(float deltaTime) override {
        // Run performance tests if requested
        if (b_runInitTest) {
            runInitializationTest();
            b_runInitTest = false;
        }
        if (b_runComputeTest) {
            runComputeTest();
            b_runComputeTest = false;
        }
        if (b_runDrawTest) {
            runDrawTest();
            b_runDrawTest = false;
        }
        if (b_runPerformanceTest) {
            runFullPerformanceTest();
            b_runPerformanceTest = false;
        }

        // Update computations based on flags
        if (b_computeGPU && m_gpuInitialized) {
            performGPUOperation();
        }
        if (b_computeCPU) {
            performCPUOperation();
        }
    }

    void draw(Renderer& renderer, Camera& camera) override {
        // Draw scalar field visualization
        if (d_drawField) {
            if (d_showGPUResult && m_gpuInitialized) {
                // Draw GPU result
                m_resultGPU.draw_points(renderer, 2);
            } else {
                // Draw CPU result
                m_resultCPU.draw_points(renderer, 2);
            }
        }

        // Draw test geometry (circles)
        drawTestGeometry(renderer);

        // Draw UI and performance info
        drawUI(renderer);
    }

    // Input handling
    bool onKeyPress(unsigned char key, int x, int y) override {
        switch (key) {
            case 'g': case 'G':
                b_computeGPU = !b_computeGPU;
                b_computeCPU = false;  // Disable CPU when enabling GPU
                std::cout << "GPU computation: " << (b_computeGPU ? "ON" : "OFF") << std::endl;
                return true;

            case 'c': case 'C':
                b_computeCPU = !b_computeCPU;
                b_computeGPU = false;  // Disable GPU when enabling CPU
                std::cout << "CPU computation: " << (b_computeCPU ? "ON" : "OFF") << std::endl;
                return true;

            case 'v': case 'V':
                d_showGPUResult = !d_showGPUResult;
                std::cout << "Showing " << (d_showGPUResult ? "GPU" : "CPU") << " result" << std::endl;
                return true;

            case 'f': case 'F':
                d_drawField = !d_drawField;
                std::cout << "Field visualization: " << (d_drawField ? "ON" : "OFF") << std::endl;
                return true;

            case 'p': case 'P':
                b_runPerformanceTest = true;
                std::cout << "Running full performance comparison..." << std::endl;
                return true;

            case '1':
                b_runInitTest = true;
                std::cout << "Running initialization test..." << std::endl;
                return true;

            case '2':
                b_runComputeTest = true;
                std::cout << "Running compute test..." << std::endl;
                return true;

            case '3':
                b_runDrawTest = true;
                std::cout << "Running draw test..." << std::endl;
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
    void initializeGPU() {
        // Get shader manager from the scene/renderer system
        auto shaderManager = std::make_shared<alice2::ShaderManager>();
        if (shaderManager->initialize()) {
            ScalarFieldGPU::initialize_gpu(shaderManager);
            m_gpuInitialized = ScalarFieldGPU::is_gpu_enabled();
        } else {
            std::cerr << "Failed to initialize shader manager" << std::endl;
            m_gpuInitialized = false;
        }
    }

    void generateTestFields() {
        // Generate GPU fields
        if (m_gpuInitialized) {
            m_fieldA_gpu.clear_field();
            m_fieldA_gpu.apply_scalar_circle(m_circleACenter, m_circleARadius);

            m_fieldB_gpu.clear_field();
            m_fieldB_gpu.apply_scalar_circle(m_circleBCenter, m_circleBRadius);
        }

        // Generate CPU fields
        m_fieldA_cpu.clear_field();
        m_fieldA_cpu.apply_scalar_circle(m_circleACenter, m_circleARadius);

        m_fieldB_cpu.clear_field();
        m_fieldB_cpu.apply_scalar_circle(m_circleBCenter, m_circleBRadius);
    }

    void performGPUOperation() {
        if (!m_gpuInitialized) return;

        m_resultGPU = m_fieldA_gpu;
        m_resultGPU.boolean_union(m_fieldB_gpu);
    }

    void performCPUOperation() {
        m_resultCPU = m_fieldA_cpu;
        m_resultCPU.boolean_union(m_fieldB_cpu);
    }

    void runPerformanceTest() {
        // Legacy method - redirect to full performance test
        runFullPerformanceTest();
    }

    // Individual performance test methods
    void runInitializationTest() {
        std::cout << "\n=== INITIALIZATION PERFORMANCE TEST ===" << std::endl;

        const int iterations = 10;

        // Test GPU initialization
        float gpuInitTime = 0.0f;
        if (m_gpuInitialized) {
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) {
                ScalarFieldGPU tempField(Vec3(-50, -50, 0), Vec3(50, 50, 0), 100, 100);
            }
            auto end = std::chrono::high_resolution_clock::now();
            gpuInitTime = std::chrono::duration<float, std::milli>(end - start).count();
        }

        // Test CPU initialization
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            ScalarField2D tempField(Vec3(-50, -50, 0), Vec3(50, 50, 0), 100, 100);
        }
        auto end = std::chrono::high_resolution_clock::now();
        float cpuInitTime = std::chrono::duration<float, std::milli>(end - start).count();

        // Store results
        m_performance.initTime_gpu = gpuInitTime / iterations;
        m_performance.initTime_cpu = cpuInitTime / iterations;

        // Print results
        std::cout << "Initialization Results (" << iterations << " iterations):" << std::endl;
        std::cout << "  GPU: " << std::fixed << std::setprecision(3) << m_performance.initTime_gpu << " ms/init" << std::endl;
        std::cout << "  CPU: " << std::fixed << std::setprecision(3) << m_performance.initTime_cpu << " ms/init" << std::endl;
        if (m_gpuInitialized && m_performance.initTime_cpu > 0) {
            float speedup = m_performance.initTime_cpu / m_performance.initTime_gpu;
            std::cout << "  GPU Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        }
    }

    void runComputeTest() {
        std::cout << "\n=== COMPUTE PERFORMANCE TEST ===" << std::endl;

        const int iterations = 100;
        generateTestFields();

        // Test GPU compute performance
        float gpuComputeTime = 0.0f;
        if (m_gpuInitialized) {
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) {
                ScalarFieldGPU tempResult = m_fieldA_gpu;
                tempResult.boolean_union(m_fieldB_gpu);
            }
            auto end = std::chrono::high_resolution_clock::now();
            gpuComputeTime = std::chrono::duration<float, std::milli>(end - start).count();
        }

        // Test CPU compute performance
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            ScalarField2D tempResult = m_fieldA_cpu;
            tempResult.boolean_union(m_fieldB_cpu);
        }
        auto end = std::chrono::high_resolution_clock::now();
        float cpuComputeTime = std::chrono::duration<float, std::milli>(end - start).count();

        // Store results
        m_performance.computeTime_gpu = gpuComputeTime / iterations;
        m_performance.computeTime_cpu = cpuComputeTime / iterations;

        // Print results
        std::cout << "Compute Results (" << iterations << " iterations):" << std::endl;
        std::cout << "  GPU: " << std::fixed << std::setprecision(3) << m_performance.computeTime_gpu << " ms/op" << std::endl;
        std::cout << "  CPU: " << std::fixed << std::setprecision(3) << m_performance.computeTime_cpu << " ms/op" << std::endl;
        if (m_gpuInitialized && m_performance.computeTime_cpu > 0) {
            float speedup = m_performance.computeTime_cpu / m_performance.computeTime_gpu;
            std::cout << "  GPU Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        }
    }

    void runDrawTest() {
        std::cout << "\n=== DRAW PERFORMANCE TEST ===" << std::endl;
        std::cout << "Draw performance test requires visual rendering - results approximate" << std::endl;

        // Note: Drawing performance is harder to measure accurately without frame timing
        // This is a placeholder for future implementation
        m_performance.drawTime_gpu = 0.0f;
        m_performance.drawTime_cpu = 0.0f;

        std::cout << "Draw test completed (visual inspection recommended)" << std::endl;
    }

    void runFullPerformanceTest() {
        std::cout << "\n=== FULL PERFORMANCE COMPARISON ===" << std::endl;

        runInitializationTest();
        runComputeTest();
        runDrawTest();

        m_performance.testsCompleted = true;

        std::cout << "\n=== SUMMARY ===" << std::endl;
        std::cout << "Initialization - GPU: " << std::fixed << std::setprecision(3) << m_performance.initTime_gpu << "ms, CPU: " << m_performance.initTime_cpu << "ms" << std::endl;
        std::cout << "Computation    - GPU: " << std::fixed << std::setprecision(3) << m_performance.computeTime_gpu << "ms, CPU: " << m_performance.computeTime_cpu << "ms" << std::endl;

        if (m_gpuInitialized) {
            if (m_performance.initTime_cpu > 0) {
                float initSpeedup = m_performance.initTime_cpu / m_performance.initTime_gpu;
                std::cout << "Init Speedup: " << std::fixed << std::setprecision(2) << initSpeedup << "x" << std::endl;
            }
            if (m_performance.computeTime_cpu > 0) {
                float computeSpeedup = m_performance.computeTime_cpu / m_performance.computeTime_gpu;
                std::cout << "Compute Speedup: " << std::fixed << std::setprecision(2) << computeSpeedup << "x" << std::endl;
            }
        }
    }

    void drawTestGeometry(Renderer& renderer) {
        // Draw circle A center (left)
        renderer.setColor(Vec3(1.0f, 0.0f, 0.0f)); // Red
        renderer.drawPoint(m_circleACenter);

        // Draw circle B center (right)
        renderer.setColor(Vec3(0.0f, 1.0f, 0.0f)); // Green
        renderer.drawPoint(m_circleBCenter);

        // Draw simple line to show the circles' relationship
        renderer.setColor(Vec3(0.5f, 0.5f, 0.5f)); // Gray
        renderer.drawLine(m_circleACenter, m_circleBCenter);
    }

    void drawUI(Renderer& renderer) {
        if (!d_showPerformanceInfo) return;

        renderer.setColor(Vec3(1.0f, 0.0f, 0.5f));
        renderer.drawString("FPS: " + std::to_string(Application::getInstance()->getFPS()), 10, 30);

        // Draw performance information and controls
        renderer.setColor(Vec3(1.0f, 1.0f, 1.0f));

        std::string info = "GPU vs CPU Scalar Field Performance Test\n";
        info += "Controls:\n";
        info += "  G = GPU compute, C = CPU compute\n";
        info += "  V = Switch display (GPU/CPU result)\n";
        info += "  F = Toggle field visualization\n";
        info += "  P = Full performance test\n";
        info += "  1 = Init test, 2 = Compute test, 3 = Draw test\n\n";

        info += "Current Mode:\n";
        info += "  Compute: " + std::string(b_computeGPU ? "GPU" : (b_computeCPU ? "CPU" : "NONE")) + "\n";
        info += "  Display: " + std::string(d_showGPUResult ? "GPU result" : "CPU result") + "\n";
        info += "  GPU Available: " + std::string(m_gpuInitialized ? "Yes" : "No") + "\n\n";

        if (m_performance.testsCompleted) {
            info += "Performance Results:\n";
            info += "  Init - GPU: " + std::to_string(m_performance.initTime_gpu) + "ms, CPU: " + std::to_string(m_performance.initTime_cpu) + "ms\n";
            info += "  Compute - GPU: " + std::to_string(m_performance.computeTime_gpu) + "ms, CPU: " + std::to_string(m_performance.computeTime_cpu) + "ms\n";

            if (m_gpuInitialized && m_performance.computeTime_cpu > 0) {
                float speedup = m_performance.computeTime_cpu / m_performance.computeTime_gpu;
                info += "  GPU Compute Speedup: " + std::to_string(speedup) + "x\n";
            }
        }

        renderer.drawString(info, 10, 50);
    }
};

// Register the sketch with alice2
ALICE2_REGISTER_SKETCH(GPUSketch)

#endif // __MAIN__
