// AutoDecoder CUDA Parity Sketch (drop-in)
// Follows the alice2 Base Sketch pattern (no main defined here).
// Press 'P' (or 'p') to run the CUDA CPU-vs-GPU parity test; results are printed to the console and shown on HUD.

#define __MAIN__
#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>

// ---- Include your ML bits ----
// Adjust include paths to match your project structure if needed.
#include <ML/genericMLP.h>
#include <ML/AutoDecoderTrainer.h>
#include <ML/AutoDecoderTrainerCUDA.h>

#include <sstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>

using namespace alice2;

namespace ad_test {

// generate two 2D circle SDFs on [-1,1]^2 (r=0.4, 0.6), 16x16 each
static void makeDataset(std::vector<alice2::AutoDecoderSample>& out, int& numShapes)
{
    using namespace alice2;
    const int NX=16, NY=16;
    const float xmin=-1.f, xmax=1.f, ymin=-1.f, ymax=1.f;
    const float radii[2] = {0.4f, 0.6f};
    numShapes = 2;

    out.clear();
    out.reserve(NX*NY*numShapes);
    for (int s=0; s<numShapes; ++s) {
        for (int j=0;j<NY;++j){
            float y = ymin + (ymax-ymin)*j/(NY-1);
            for (int i=0;i<NX;++i){
                float x = xmin + (xmax-xmin)*i/(NX-1);
                float d = std::sqrt(x*x+y*y) - radii[s];
                AutoDecoderSample smp;
                smp.shapeIndex = s;
                smp.coordinate = {x,y};
                smp.sdf = d;
                out.push_back(std::move(smp));
            }
        }
    }
}

// Exposed test (no main). Returns 0 on success.
inline int run_autodecoder_cuda_parity(std::string& logOut)
{
    using namespace alice2;

    // Model dims
    int latentDim = 8, coordDim = 2;
    std::vector<int> hidden = {32,32};

    // Two identical initial decoders so CPU & CUDA start from same point
    MLP cpuDec(latentDim + coordDim, hidden, 1);
    MLP cpuDec2 = cpuDec; // copy

    // CPU trainer
    AutoDecoderTrainer cpuTrainer(cpuDec);
    std::vector<AutoDecoderSample> samples;
    int numShapes=0;
    ad_test::makeDataset(samples, numShapes);
    cpuTrainer.initialize(numShapes, latentDim, coordDim);
    cpuTrainer.setSamples(samples);

    // Deterministic, identical latents for CPU & CUDA (no RNG)
    std::vector<std::vector<float>> codes(numShapes, std::vector<float>(latentDim, 0.0f));
    for (int s = 0; s < numShapes; ++s) {
        for (int k = 0; k < latentDim; ++k) {
            codes[s][k] = 0.001f * float(s + k); // any fixed pattern is fine
        }
    }

    AutoDecoderTrainingConfig cfg;
    cfg.epochs = 2;
    cfg.learningRateWeights = 5e-4f;
    cfg.learningRateLatent  = 1e-3f;
    cfg.latentRegularization = 1e-4f;
    cfg.latentInitStd = 0.01f;
    cfg.shuffleSeed = 2025u;

    // CUDA trainer
    AutoDecoderTrainerCUDA gpuTrainer(cpuDec2);
    gpuTrainer.initialize(numShapes, latentDim, coordDim);
    gpuTrainer.setSamples(samples);

    // Set the same starting latents on both trainers BEFORE training
    cpuTrainer.setLatentCodes(codes);
    gpuTrainer.setLatentCodesHost(codes);

    // Train
    AutoDecoderTrainingStats stCPU = cpuTrainer.train(cfg);
    AutoDecoderTrainingStats stGPU = gpuTrainer.train(cfg);

    // Compare weights/biases/latents
    float maxAbsDiffW = 0.0f, maxAbsDiffB = 0.0f, maxAbsDiffZ = 0.0f;

    for (int l=0; l<(int)cpuDec.W.size(); ++l){
        for (int o=0;o<(int)cpuDec.W[l].size();++o){
            for (int j=0;j<(int)cpuDec.W[l][o].size();++j){
                maxAbsDiffW = std::max(maxAbsDiffW, std::abs(cpuDec.W[l][o][j] - cpuDec2.W[l][o][j]));
            }
        }
    }
    for (int l=0; l<(int)cpuDec.b.size(); ++l){
        for (int o=0;o<(int)cpuDec.b[l].size();++o){
            maxAbsDiffB = std::max(maxAbsDiffB, std::abs(cpuDec.b[l][o] - cpuDec2.b[l][o]));
        }
    }

    const auto& Zcpu = cpuTrainer.getLatentCodes();
    const auto& Zgpu = gpuTrainer.getLatentCodesHost();
    for (int s=0; s<numShapes; ++s){
        for (int k=0; k<latentDim; ++k){
            maxAbsDiffZ = std::max(maxAbsDiffZ, std::abs(Zcpu[s][k] - Zgpu[s][k]));
        }
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "CPU avgLoss=" << stCPU.averageLoss << " last=" << stCPU.lastLoss << "\n";
    oss << "GPU avgLoss=" << stGPU.averageLoss << " last=" << stGPU.lastLoss << "\n";
    oss << std::scientific << std::setprecision(3);
    oss << "Max |W_cpu - W_gpu|=" << maxAbsDiffW
        << "  |b|=" << maxAbsDiffB
        << "  |Z|=" << maxAbsDiffZ << "\n";
    logOut = oss.str();

    // Tolerances (B=1 parity)
    if (maxAbsDiffW >= 1e-5f) return 1;
    if (maxAbsDiffB >= 1e-5f) return 2;
    if (maxAbsDiffZ >= 1e-5f) return 3;
    return 0;
}

} // namespace ad_test


class AutoDecoderTestSketch : public ISketch {
public:
    AutoDecoderTestSketch() = default;
    ~AutoDecoderTestSketch() = default;

    std::string getName() const override { return "AutoDecoder CUDA Parity"; }
    std::string getDescription() const override {
        return "Press 'P' to run CPU vs CUDA autodecoder parity test (B=1).";
    }

    void setup() override {
        // Simple UI (optional) — show a hint
    }

    void update(float /*deltaTime*/) override {
        // nothing per-frame
    }

    void draw(Renderer& renderer, Camera& /*camera*/) override {
        renderer.setColor(Color(0.9f, 0.9f, 0.9f));
        renderer.drawString(getName(), 10, 30);
        renderer.drawString(getDescription(), 10, 50);

        // Show last result on HUD
        renderer.setColor(Color(0.75f, 0.85f, 1.0f));
        int y = 80;
        if (!m_lastLog.empty()) {
            std::stringstream ss(m_lastLog);
            std::string line; int lines = 0;
            while (std::getline(ss, line) && lines < 6) {
                renderer.drawString(line, 10, y);
                y += 18; ++lines;
            }
        } else {
            renderer.drawString("Awaiting 'P'...", 10, y);
        }
    }

    // Keyboard handler — trigger on 'p' / 'P'
    bool onKeyPress(unsigned char key, int /*x*/, int /*y*/) override {
        // If your framework uses ASCII: 'p' == 112, 'P' == 80
        if (key == 'p' || key == 'P') {
            std::string log;
            std::cout << "[AutoDecoderTest] Triggered with key 'P'. Running parity test..." << std::endl;
            int rc = ad_test::run_autodecoder_cuda_parity(log);
            std::cout << "[AutoDecoderTest] rc=" << rc << "\n" << log << std::endl;

            m_lastLog = log;
            m_lastRC  = rc;
        }
    }

private:
    std::string m_lastLog;
    int         m_lastRC{-999};
};

// Register the sketch with alice2 (enable whichever macro your project uses)
ALICE2_REGISTER_SKETCH_AUTO(AutoDecoderTestSketch)

#endif // __MAIN__
