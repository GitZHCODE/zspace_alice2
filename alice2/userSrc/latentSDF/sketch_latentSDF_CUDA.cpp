//#define __MAIN__
#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>

#include <ML/DeepSDF/FieldViewer.h>
#include <ML/DeepSDF/LatentSDF_CUDA.h>

#include <fstream>
#include <random>

using namespace alice2;
using namespace DeepSDF;

class Sketch_LatentSDF_CUDA : public ISketch {
public:
    std::string getName()        const override { return "LatentSDF (CUDA)"; }
    std::string getDescription() const override { return "GPU auto-decoder with FieldViewer UI + TrainingDataSet IO."; }
    std::string getAuthor()      const override { return "alice2 User"; }

    void setup() override {
        scene().setBackgroundColor(Color(0.0f, 0.0f, 0.0f));
        scene().setShowGrid(false);
        scene().setShowAxes(false);

        // --- Domain (matches what FieldViewer expects)
        domain_.resX = gridResX_;
        domain_.resY = gridResY_;
        domain_.xMin = xMin_;
        domain_.xMax = xMax_;
        domain_.yMin = yMin_;
        domain_.yMax = yMax_;

        // --- Prepare analytic originals for 3 canonical shapes (0=circle,1=box,2=triangle)
        originals_.resize(numShapes_);
        for (int s = 0; s < numShapes_; ++s) {
            buildAnalyticGrid(s, domain_.resX, domain_.resY,
                              domain_.xMin, domain_.xMax, domain_.yMin, domain_.yMax,
                              originals_[s].values, originals_[s].minValue, originals_[s].maxValue);
        }

        // --- Allocate reconstruction buffers (one per shape for side-by-side view, if desired)
        recon_.resize(numShapes_);
        for (int s = 0; s < numShapes_; ++s) {
            recon_[s].values.assign(size_t(domain_.resX) * size_t(domain_.resY), 0.0f);
            recon_[s].minValue = 0.0f;
            recon_[s].maxValue = 0.0f;
        }

        // --- Bind FieldViewer
        // If your FieldViewer supports multi-shape rows, great. If not, we show one shape at a time (activeShape_).
        viewer_.setDomain(&domain_);
        viewer_.setOriginal(&originals_);
        viewer_.setReconstructed(&recon_);

        // --- Try to load a dataset (optional, mirrors CPU sketch behavior)
        std::ifstream fin(datasetPath_);
        if (fin.good()) {
            if (dataset_.loadJSON(datasetPath_)) {
                std::printf("[CUDA][Dataset] Loaded '%s' (samples=%zu)\n", datasetPath_.c_str(), dataset_.size());
            } else {
                std::printf("[CUDA][Dataset] Found '%s' but failed to parse. Proceeding without it.\n", datasetPath_.c_str());
            }
        } else {
            std::printf("[CUDA][Dataset] No dataset file found at '%s'.\n", datasetPath_.c_str());
        }

        // --- Init CUDA model
        // (We keep maxBatch_ modest; trainMicroBatchGPU handles per-sample Z, batch W updates.)
        decoder_.initialize(/*numShapes*/ numShapes_,
                            /*latentDim*/ latentDim_,
                            /*hidden*/    {64,64,64},
                            /*seed*/      initSeed_,
                            /*maxBatch*/  maxBatch_,
                            /*numFreqs*/  6,
                            /*includeInput*/ true);

        // Training knobs
        decoder_.setLambdaLatent(lambdaLatent_);
        decoder_.setWeightDecayW(weightDecayW_);

        // Prebuild X coordinates per row for forward pass
        buildScanlineXs();
        // Initial reconstruction
        rebuildReconstruction();
    }

    void update(float /*dt*/) override {
        // no-op: we train on keypress to stay deterministic with your workflow
    }

    void draw(Renderer& r, Camera&) override {
        const float startY = 20.0f;
        const float gapY   = 36.0f;

        // If you want to preview all shapes stacked, uncomment the loop; here we just use activeShape_
        viewer_.drawOriginalRow(r, startY);
        viewer_.drawReconstructionRow(r, startY + viewer_.config().tileSize + gapY);
        viewer_.drawHelp(r, startY + 2*viewer_.config().tileSize + gapY + 40.0f);
    }

    bool onKeyPress(unsigned char k, int, int) override {
        switch (k) {
        // --- FieldViewer toggles (match CPU sketch)
        case '1': viewer_.config().debugMode = 0; return true;
        case '2': viewer_.config().debugMode = 1; return true;
        case '3': viewer_.config().debugMode = 2; return true;
        case '4': viewer_.config().debugMode = 3; return true;

        case 'm': case 'M':
            viewer_.config().softMask = !viewer_.config().softMask; return true;

        case '[':
            viewer_.config().tau = std::max(0.005f, viewer_.config().tau * 0.80f); return true;
        case ']':
            viewer_.config().tau = std::min(0.50f,  viewer_.config().tau * 1.25f); return true;

        // --- Train a small number of micro-batches
        case 't': case 'T': {
            std::mt19937 rng(trainSeed_);
            Sampler localSampler(sampleSeed_);
            for (int e = 0; e < trainEpochs_; ++e) {
                for (int step = 0; step < stepsPerEpoch_; ++step) {
                    decoder_.trainMicroBatchGPU(/*B*/ microBatchB_, localSampler, rng, lrW_, lrZ_);
                }
                epochsDone_++;
            }

            // read stats
            double avgLoss = 0.0, meanZ = 0.0;
            decoder_.syncStatsToHost(avgLoss, meanZ, /*reset*/true);
            std::printf("[CUDA][Train] epoch %d  avgLoss=%.6f  mean||z||=%.6f\n", epochsDone_, float(avgLoss), float(meanZ));

            decoder_.syncLatentsToHost();
            rebuildReconstruction();
            return true;
        }

        // --- Rebuild reconstruction without training
        case 'r': case 'R':
            decoder_.syncLatentsToHost();
            rebuildReconstruction();
            return true;

        // --- Save / Load dataset (optional)
        case 'j': case 'J': // save
            if (dataset_.saveJSON(datasetPath_)) {
                std::printf("[CUDA][Dataset] Saved '%s'\n", datasetPath_.c_str());
            } else {
                std::printf("[CUDA][Dataset] Save failed for '%s'\n", datasetPath_.c_str());
            }
            return true;

        case 'k': case 'K': { // load
            DeepSDF::TrainingDataset tmp;
            if (tmp.loadJSON(datasetPath_)) {
                dataset_ = std::move(tmp);
                std::printf("[CUDA][Dataset] Reloaded '%s' (samples=%zu)\n", datasetPath_.c_str(), dataset_.size());
            } else {
                std::printf("[CUDA][Dataset] Load failed for '%s'\n", datasetPath_.c_str());
            }
            return true;
        }

        default:
            break;
        }
        return false;
    }

private:
    // ---- Helpers ----------------------------------------------------------

    void setActiveShape(int s) {
        viewer_.setOriginal(&originals_);
        viewer_.setReconstructed(&recon_);
        rebuildReconstruction();
    }

    void buildScanlineXs() {
        xs_.resize(domain_.resX);
        const float dx = (domain_.resX > 1) ? (domain_.xMax - domain_.xMin) / float(domain_.resX - 1) : 0.0f;
        for (int x = 0; x < domain_.resX; ++x) {
            xs_[x] = domain_.xMin + dx * float(x);
        }
    }

    void rebuildReconstruction() {
        const float dy = (domain_.resY > 1) ? (domain_.yMax - domain_.yMin) / float(domain_.resY - 1) : 0.0f;
        std::vector<float> row; row.resize(domain_.resX);

        for (int s = 0; s < numShapes_; ++s) {
            GridField& g = recon_[s];
            g.values.resize(size_t(domain_.resX) * size_t(domain_.resY));
            g.minValue = +1e9f; g.maxValue = -1e9f;

            for (int y = 0; y < domain_.resY; ++y) {
                const float yy = domain_.yMin + dy * float(y);
                decoder_.forwardRowGPU(s, xs_, yy, row);
                for (int x = 0; x < domain_.resX; ++x) {
                    const float v = row[x];
                    g.values[size_t(y) * size_t(domain_.resX) + size_t(x)] = v;
                    g.minValue = std::min(g.minValue, v);
                    g.maxValue = std::max(g.maxValue, v);
                }
            }
        }
    }

private:
    // --- Viewer + fields
    FieldViewer        viewer_;
    FieldDomain        domain_;
    std::vector<GridField> originals_;
    std::vector<GridField> recon_;

    // --- Model
    TinyAutoDecoderCUDA decoder_;

    // --- Optional dataset (kept for parity with CPU workflow)
    DeepSDF::TrainingDataset    dataset_;
    std::string        datasetPath_ = "latentSDF_dataset.json";

    // --- Params (kept close to CPU defaults)
    int   gridResX_       = 128;
    int   gridResY_       = 128;
    float xMin_           = -1.2f, xMax_ = 1.2f;
    float yMin_           = -1.2f, yMax_ = 1.2f;

    int   numShapes_      = 3;     // circle / box / triangle

    int   latentDim_      = 16;
    int   maxBatch_       = 256;   // upper bound for micro-batching inside the CUDA trainer

    // training
    int   trainEpochs_    = 50;
    int   stepsPerEpoch_  = 200;
    int   microBatchB_    = 16;

    float lrW_            = 5e-2f;
    float lrZ_            = 5e-2f;

    float lambdaLatent_   = 1e-4f;
    float weightDecayW_   = 1e-6f;

    unsigned initSeed_    = 1234;
    unsigned trainSeed_   = 2025;
    unsigned sampleSeed_  = 777;

    int epochsDone_       = 0;

    // precomputed scanline positions for forwardRowGPU
    std::vector<float> xs_;
};

ALICE2_REGISTER_SKETCH_AUTO(Sketch_LatentSDF_CUDA)

#endif // __MAIN__
