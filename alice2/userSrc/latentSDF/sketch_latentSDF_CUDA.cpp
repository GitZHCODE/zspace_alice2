#define __MAIN__
#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>

#include <ML/DeepSDF/LatentSDF_CUDA.h>
#include <ML/DeepSDF/FieldViewer.h>

#include <fstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <cstdio>

using namespace alice2;
using namespace DeepSDF;

class Sketch_LatentSDF_CUDA : public ISketch {
public:
    std::string getName()        const override { return "LatentSDF (CUDA)"; }
    std::string getDescription() const override { return "GPU auto-decoder training with direct field view."; }
    std::string getAuthor()      const override { return "alice2 User"; }

    void setup() override {
        scene().setBackgroundColor(Color(0.0f, 0.0f, 0.0f));
        scene().setShowGrid(false);
        scene().setShowAxes(false);

        domain_.resX = gridResX_;
        domain_.resY = gridResY_;
        domain_.xMin = xMin_;
        domain_.xMax = xMax_;
        domain_.yMin = yMin_;
        domain_.yMax = yMax_;

        originals_.resize(numShapes_);
        for (int s = 0; s < numShapes_; ++s) {
            buildAnalyticGrid(s, domain_.resX, domain_.resY,
                              domain_.xMin, domain_.xMax,
                              domain_.yMin, domain_.yMax,
                              originals_[s].values,
                              originals_[s].minValue,
                              originals_[s].maxValue);
        }

        recon_.assign(numShapes_, GridField{});

        decoder_.initialize(numShapes_,
                            latentDim_,
                            {64, 64, 64},
                            initSeed_,
                            maxBatch_,
                            6,
                            true);
        decoder_.setLambdaLatent(lambdaLatent_);
        decoder_.setWeightDecayW(weightDecayW_);
        numShapes_ = decoder_.numShapes();
        latentDim_ = decoder_.latentDim();

        buildScanlineXs();
        rebuildReconstruction();
    }

    void update(float) override {}

    void draw(Renderer& renderer, Camera&) override {
        const float startY = 20.0f;
        const float gapY   = 36.0f;

        drawFieldRow(renderer, originals_, startY, "Original (analytic labels)");
        drawFieldRow(renderer, recon_,      startY + displayCfg_.tileSize + gapY,
                     "Reconstructed (decoder output)");
        drawHelp(renderer, startY + 2 * displayCfg_.tileSize + gapY + 32.0f);
    }

    bool onKeyPress(unsigned char k, int, int) override {
        switch (k) {
        case '1': displayCfg_.debugMode = 0; return true;
        case '2': displayCfg_.debugMode = 1; return true;
        case '3': displayCfg_.debugMode = 2; return true;
        case '4': displayCfg_.debugMode = 3; return true;

        case 'm': case 'M':
            displayCfg_.softMask = !displayCfg_.softMask; return true;
        case '[':
            displayCfg_.tau = std::max(0.005f, displayCfg_.tau * 0.8f); return true;
        case ']':
            displayCfg_.tau = std::min(0.5f, displayCfg_.tau * 1.25f); return true;

        case 't': case 'T': {
            std::mt19937 rng(trainSeed_);
            Sampler sampler(sampleSeed_);
            for (int e = 0; e < trainEpochs_; ++e) {
                for (int s = 0; s < stepsPerEpoch_; ++s) {
                    decoder_.trainMicroBatchGPU(microBatchB_, sampler, rng, lrW_, lrZ_);
                }
                epochsDone_++;
            }
            double avgLoss = 0.0, meanZ = 0.0;
            decoder_.syncStatsToHost(avgLoss, meanZ, true);
            std::printf("[CUDA][Train] epoch %d  avgLoss=%.6f  mean||z||=%.6f\n",
                        epochsDone_, float(avgLoss), float(meanZ));
            decoder_.syncLatentsToHost();
            rebuildReconstruction();
            return true;
        }

        case 'r': case 'R':
            decoder_.syncLatentsToHost();
            rebuildReconstruction();
            return true;

        case 'j': case 'J':
            decoder_.saveModelJSON(modelPath_, domain_);
            return true;

        case 'l': case 'L': {
            FieldDomain loadedDomain = domain_;
            if (decoder_.loadModelJSON(modelPath_, loadedDomain)) {
                domain_ = loadedDomain;
                gridResX_ = domain_.resX;
                gridResY_ = domain_.resY;
                xMin_ = domain_.xMin; xMax_ = domain_.xMax;
                yMin_ = domain_.yMin; yMax_ = domain_.yMax;

                numShapes_ = decoder_.numShapes();
                latentDim_ = decoder_.latentDim();

                originals_.resize(numShapes_);
                for (int s = 0; s < numShapes_; ++s) {
                    buildAnalyticGrid(s, domain_.resX, domain_.resY,
                                      domain_.xMin, domain_.xMax,
                                      domain_.yMin, domain_.yMax,
                                      originals_[s].values,
                                      originals_[s].minValue,
                                      originals_[s].maxValue);
                }
                recon_.assign(numShapes_, GridField{});
                buildScanlineXs();
                rebuildReconstruction();
                std::printf("[CUDA][Model] Reloaded '%s'\n", modelPath_.c_str());
            }
            return true;
        }

        default:
            break;
        }
        return false;
    }

private:
    struct DisplayConfig {
        bool  softMask  = true;
        float tau       = 0.05f;
        int   debugMode = 0;
        float tileSize  = 220.0f;
    };

    void buildScanlineXs() {
        xs_.resize(domain_.resX);
        const float dx = (domain_.resX > 1)
            ? (domain_.xMax - domain_.xMin) / float(domain_.resX - 1)
            : 0.0f;
        for (int x = 0; x < domain_.resX; ++x) {
            xs_[x] = domain_.xMin + dx * float(x);
        }
    }

    void rebuildReconstruction() {
        const float dy = (domain_.resY > 1)
            ? (domain_.yMax - domain_.yMin) / float(domain_.resY - 1)
            : 0.0f;

        const int shapes = decoder_.numShapes();
        if ((int)recon_.size() != shapes) {
            recon_.assign(shapes, GridField{});
        }

        std::vector<float> row(domain_.resX);

        for (int s = 0; s < shapes; ++s) {
            GridField& field = recon_[s];
            field.values.resize(size_t(domain_.resX) * size_t(domain_.resY));
            field.minValue =  std::numeric_limits<float>::max();
            field.maxValue = -std::numeric_limits<float>::max();

            for (int y = 0; y < domain_.resY; ++y) {
                const float yy = domain_.yMin + dy * float(y);
                decoder_.forwardRowGPU(s, xs_, yy, row);
                for (int x = 0; x < domain_.resX; ++x) {
                    const float v = row[x];
                    const size_t idx = size_t(y) * size_t(domain_.resX) + size_t(x);
                    field.values[idx] = v;
                    field.minValue = std::min(field.minValue, v);
                    field.maxValue = std::max(field.maxValue, v);
                }
            }
        }
    }

    void drawFieldRow(Renderer& renderer,
                      const std::vector<GridField>& grids,
                      float top,
                      const char* label) const
    {
        if (grids.empty() || domain_.resX <= 0 || domain_.resY <= 0) return;

        renderer.setColor(Color(0.9f, 0.9f, 0.9f));
        renderer.drawString(label, 20.0f, top - 8.0f);

        const float gap = 25.0f;
        const float tile = displayCfg_.tileSize;
        const float cellW = tile / float(domain_.resX);
        const float cellH = tile / float(domain_.resY);

        for (size_t i = 0; i < grids.size(); ++i) {
            const float left = 20.0f + float(i) * (tile + gap);
            drawField(renderer, grids[i], left, top, cellW, cellH);
        }
    }

    void drawField(Renderer& renderer,
                   const GridField& field,
                   float left, float top,
                   float cellW, float cellH) const
    {
        const bool forceMask = (displayCfg_.debugMode == 0);
        for (int y = 0; y < domain_.resY; ++y) {
            for (int x = 0; x < domain_.resX; ++x) {
                const size_t idx = size_t(y) * size_t(domain_.resX) + size_t(x);
                const float sdf = field.values[idx];
                float g = 0.5f;
                if (forceMask) {
                    if (displayCfg_.softMask) {
                        const float tau = std::max(displayCfg_.tau, 1e-6f);
                        g = 1.0f / (1.0f + std::exp(-(sdf / tau)));
                    } else {
                        g = sdf < 0.0f ? 0.0f : 1.0f;
                    }
                } else {
                    const float range = (field.maxValue - field.minValue == 0.0f)
                                      ? 1.0f
                                      : (field.maxValue - field.minValue);
                    g = std::clamp((sdf - field.minValue) / range, 0.0f, 1.0f);
                }
                const Color color(g, g, g);
                const float px = left + (float(x) + 0.5f) * cellW;
                const float py = top  + (float(y) + 0.5f) * cellH;
                const float ps = std::max(cellW, cellH);
                renderer.draw2dPoint(Vec2(px, py), color, ps);
            }
        }
    }

    void drawHelp(Renderer& renderer, float y) const {
        renderer.setColor(Color(0.7f, 0.7f, 0.7f));
        renderer.drawString("Keys: T train  R rebuild  M mask  [ ] tau  1-4 debug  J save model  L load model",
                            20.0f, y);
    }

private:
    DisplayConfig displayCfg_;
    FieldDomain domain_;
    std::vector<GridField> originals_;
    std::vector<GridField> recon_;

    TinyAutoDecoderCUDA decoder_;
    std::string modelPath_ = "latent_model.json";

    int   gridResX_ = 128;
    int   gridResY_ = 128;
    float xMin_ = -1.2f, xMax_ = 1.2f;
    float yMin_ = -1.2f, yMax_ = 1.2f;

    int numShapes_ = 3;
    int latentDim_ = 16;
    int maxBatch_ = 256;

    int   trainEpochs_ = 50;
    int   stepsPerEpoch_ = 200;
    int   microBatchB_ = 16;
    float lrW_ = 5e-2f;
    float lrZ_ = 5e-2f;
    float lambdaLatent_ = 1e-4f;
    float weightDecayW_ = 1e-6f;
    unsigned initSeed_ = 1234;
    unsigned trainSeed_ = 2025;
    unsigned sampleSeed_ = 777;
    int epochsDone_ = 0;

    std::vector<float> xs_;
};

ALICE2_REGISTER_SKETCH_AUTO(Sketch_LatentSDF_CUDA)

#endif // __MAIN__
