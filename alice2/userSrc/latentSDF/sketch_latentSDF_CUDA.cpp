#define __MAIN__
#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>

#include <ML/DeepSDF/FieldViewer.h>
#include <ML/DeepSDF/LatentSDF_CUDA.h>

#include <fstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <array>
#include <stdexcept>
#include <string>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace alice2;
using namespace DeepSDF;

namespace {
inline void checkCudaStatus(const char* call, cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("[CUDA] ") + call + " failed: " + cudaGetErrorString(status));
    }
}
} // namespace

class GpuLatentPanel {
public:
    void initialize(TinyAutoDecoderCUDA* decoder,
                    const FieldDomain& domain,
                    int tileRes,
                    int gap,
                    int panelN,
                    float panelSizePixels)
    {
        releaseResources();
        decoder_ = decoder;
        domain_  = domain;
        tileRes_ = std::max(1, tileRes);
        gap_     = std::max(0, gap);
        panelN_  = std::max(1, panelN);
        panelSizePx_ = panelSizePixels;
        computeDimensions();
        ensureResources();
        dirty_ = true;
    }

    void setLatentSource(const std::vector<std::vector<float>>* latents) {
        latents_ = latents;
        dirty_ = true;
    }

    void setCornerIndices(const std::array<int,4>& indices) {
        cornerIdx_ = indices;
        dirty_ = true;
    }

    void markDirty() { dirty_ = true; }

    void rebuildIfNeeded(const FieldRenderConfig& cfg) {
        if (!dirty_) return;
        if (!decoder_ || !latents_) return;
        ensureResources();
        if (!resourcesReady_) return;

        const auto& allLatents = *latents_;
        if (allLatents.empty()) return;

        const int latentDim = decoder_->latentDim();
        scratchLatent_.resize(latentDim);

        std::array<const std::vector<float>*,4> corners{};
        for (int i = 0; i < 4; ++i) {
            int idx = cornerIdx_[i];
            if (idx >= 0 && idx < (int)allLatents.size()) corners[i] = &allLatents[idx];
            else corners[i] = &allLatents.front();
        }
        for (auto* ptr : corners) {
            if (!ptr || (int)ptr->size() != latentDim) return;
        }

        for (int gy = 0; gy < panelN_; ++gy) {
            const float v = (panelN_ == 1) ? 0.0f : float(gy) / float(panelN_ - 1);
            for (int gx = 0; gx < panelN_; ++gx) {
                const float u = (panelN_ == 1) ? 0.0f : float(gx) / float(panelN_ - 1);
                const float a = (1.0f - u) * (1.0f - v);
                const float b = u * (1.0f - v);
                const float c = (1.0f - u) * v;
                const float d = u * v;
                for (int i = 0; i < latentDim; ++i) {
                    scratchLatent_[i] = a * (*corners[0])[i]
                                      + b * (*corners[1])[i]
                                      + c * (*corners[2])[i]
                                      + d * (*corners[3])[i];
                }
                const int offsetX = gx * tileRes_ + std::max(0, gx) * gap_;
                const int offsetY = gy * tileRes_ + std::max(0, gy) * gap_;
                decoder_->decodeLatentGridToDevice(scratchLatent_.data(),
                                                   tileRes_, tileRes_,
                                                   domain_.xMin, domain_.xMax,
                                                   domain_.yMin, domain_.yMax,
                                                   dPanel_, panelW_, offsetX, offsetY);
            }
        }

        decoder_->panelToRGBA(dPanel_, panelW_, panelH_,
                              tileRes_, gap_, panelN_,
                              cfg, dPanelRGBA_);

        if (cudaResource_) {
            checkCudaStatus("cudaGraphicsMapResources", cudaGraphicsMapResources(1, &cudaResource_));
            cudaArray_t array = nullptr;
            checkCudaStatus("cudaGraphicsSubResourceGetMappedArray",
                            cudaGraphicsSubResourceGetMappedArray(&array, cudaResource_, 0, 0));
            checkCudaStatus("cudaMemcpy2DToArray",
                            cudaMemcpy2DToArray(array, 0, 0,
                                                dPanelRGBA_, panelW_ * sizeof(uchar4),
                                                panelW_ * sizeof(uchar4), panelH_,
                                                cudaMemcpyDeviceToDevice));
            checkCudaStatus("cudaGraphicsUnmapResources", cudaGraphicsUnmapResources(1, &cudaResource_));
        }
        dirty_ = false;
    }

    void draw(Renderer& renderer, float left, float top, float size) {
        if (!resourcesReady_ || texture_ == 0) return;

        int vx, vy, vw, vh;
        renderer.getViewport(vx, vy, vw, vh);

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(0, vw, vh, 0, -1, 1);

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, texture_);
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

        const float right  = left + size;
        const float bottom = top  + size;

        glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(left,  top);
            glTexCoord2f(1.0f, 0.0f); glVertex2f(right, top);
            glTexCoord2f(1.0f, 1.0f); glVertex2f(right, bottom);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(left,  bottom);
        glEnd();

        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);

        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
    }

    void shutdown() { releaseResources(); }

    float panelSizePixels() const { return panelSizePx_; }

private:
    void computeDimensions() {
        panelW_ = panelN_ * tileRes_ + (panelN_ - 1) * gap_;
        panelH_ = panelW_;
        scratchLatent_.clear();
        if (decoder_) {
            scratchLatent_.resize(decoder_->latentDim());
        }
    }

    void ensureResources() {
        if (resourcesReady_) return;
        if (!decoder_ || panelW_ <= 0 || panelH_ <= 0) return;

        const size_t pixelCount = size_t(panelW_) * size_t(panelH_);
        checkCudaStatus("cudaMalloc panel", cudaMalloc(&dPanel_, pixelCount * sizeof(float)));
        checkCudaStatus("cudaMalloc panelRGBA", cudaMalloc(&dPanelRGBA_, pixelCount * sizeof(uchar4)));

        glGenTextures(1, &texture_);
        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, panelW_, panelH_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        checkCudaStatus("cudaGraphicsGLRegisterImage",
                        cudaGraphicsGLRegisterImage(&cudaResource_, texture_, GL_TEXTURE_2D,
                                                     cudaGraphicsRegisterFlagsWriteDiscard));

        resourcesReady_ = true;
    }

    void releaseResources() {
        if (cudaResource_) {
            cudaGraphicsUnregisterResource(cudaResource_);
            cudaResource_ = nullptr;
        }
        if (texture_ != 0) {
            glDeleteTextures(1, &texture_);
            texture_ = 0;
        }
        if (dPanel_) {
            cudaFree(dPanel_);
            dPanel_ = nullptr;
        }
        if (dPanelRGBA_) {
            cudaFree(dPanelRGBA_);
            dPanelRGBA_ = nullptr;
        }
        resourcesReady_ = false;
    }

private:
    TinyAutoDecoderCUDA* decoder_ = nullptr;
    FieldDomain          domain_{};
    int                  tileRes_ = 56;
    int                  gap_ = 4;
    int                  panelN_ = 5;
    int                  panelW_ = 0;
    int                  panelH_ = 0;
    float                panelSizePx_ = 400.0f;

    bool                 resourcesReady_ = false;
    bool                 dirty_ = true;

    GLuint               texture_ = 0;
    cudaGraphicsResource* cudaResource_ = nullptr;
    float*               dPanel_ = nullptr;
    uchar4*              dPanelRGBA_ = nullptr;

    const std::vector<std::vector<float>>* latents_ = nullptr;
    std::array<int,4>    cornerIdx_{0,1,2,3};
    std::vector<float>   scratchLatent_;
};

class Sketch_LatentSDF_CUDA : public ISketch {
public:
    ~Sketch_LatentSDF_CUDA() override { panel_.shutdown(); }

    std::string getName()        const override { return "LatentSDF (CUDA)"; }
    std::string getDescription() const override { return "GPU auto-decoder with direct renderer + TrainingDataSet IO."; }
    std::string getAuthor()      const override { return "alice2 User"; }

    void setup() override {
        scene().setBackgroundColor(Color(0.0f, 0.0f, 0.0f));
        scene().setShowGrid(false);
        scene().setShowAxes(false);

        // --- Domain (shared with CPU sketch defaults)
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

        // --- Optional dataset load (mirrors CPU sketch behavior)
        std::ifstream fin(datasetPath_);
        if (fin.good()) {
            if (dataset_.loadJSON(datasetPath_)) {
                std::printf("[CUDA][Dataset] Loaded '%s' (samples=%zu)\n",
                            datasetPath_.c_str(), dataset_.size());
            } else {
                std::printf("[CUDA][Dataset] Found '%s' but failed to parse. Proceeding without it.\n",
                            datasetPath_.c_str());
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

        panel_.initialize(&decoder_, domain_, kPanelTile, kPanelGap, kPanelN, kPanelSize);
        panel_.setLatentSource(&decoder_.latents());
        panel_.setCornerIndices({0, 1, 2, 3});
        panel_.markDirty();

        // Prebuild X coordinates per row for forward pass
        buildScanlineXs();
        // Initial reconstruction
        rebuildReconstruction();
    }

    void update(float /*dt*/) override {
        // no-op: we train on keypress to stay deterministic with your workflow
    }

    void draw(Renderer& r, Camera&) override {
        const float startY   = 20.0f;
        const float gapY     = 36.0f;
        const float tileSize = displayCfg_.tileSize;

        FieldRenderConfig cfg{};
        cfg.debugMode = displayCfg_.debugMode;
        cfg.softMask  = displayCfg_.softMask ? 1 : 0;
        cfg.tau       = displayCfg_.tau;
        panel_.rebuildIfNeeded(cfg);

        drawFieldRow(r, originals_, startY, "Original (analytic labels)");
        drawFieldRow(r, recon_,      startY + tileSize + gapY, "Reconstructed (decoder output)");
        r.setColor(Color(0.85f, 0.85f, 0.95f));
        r.drawString("GPU Latent Panel", kPanelLeft, kPanelTop - 20.0f);
        panel_.draw(r, kPanelLeft, kPanelTop, kPanelSize);
        drawHelp(r, startY + 2 * tileSize + gapY + 40.0f);
    }

    bool onKeyPress(unsigned char k, int, int) override {
        switch (k) {
        // --- Display toggles (match CPU sketch)
        case '1':
            displayCfg_.debugMode = 0;
            panel_.markDirty();
            return true;
        case '2':
            displayCfg_.debugMode = 1;
            panel_.markDirty();
            return true;
        case '3':
            displayCfg_.debugMode = 2;
            panel_.markDirty();
            return true;
        case '4':
            displayCfg_.debugMode = 3;
            panel_.markDirty();
            return true;

        case 'm': case 'M':
            displayCfg_.softMask = !displayCfg_.softMask;
            panel_.markDirty();
            return true;

        case '[':
            displayCfg_.tau = std::max(0.005f, displayCfg_.tau * 0.80f);
            panel_.markDirty();
            return true;
        case ']':
            displayCfg_.tau = std::min(0.50f,  displayCfg_.tau * 1.25f);
            panel_.markDirty();
            return true;

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
            panel_.markDirty();
            return true;
        }

        // --- Rebuild reconstruction without training
        case 'r': case 'R':
            decoder_.syncLatentsToHost();
            rebuildReconstruction();
            panel_.markDirty();
            return true;

        case 'j': case 'J':
            if (dataset_.saveJSON(datasetPath_)) {
                std::printf("[CUDA][Dataset] Saved '%s'\n", datasetPath_.c_str());
            } else {
                std::printf("[CUDA][Dataset] Save failed for '%s'\n", datasetPath_.c_str());
            }
            return true;

        case 'k': case 'K': {
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
        panel_.markDirty();
    }

    void drawFieldRow(Renderer& renderer,
                      const std::vector<GridField>& grids,
                      float top,
                      const char* label) {
        if (grids.empty() || domain_.resX <= 0 || domain_.resY <= 0) {
            return;
        }

        renderer.setColor(Color(0.9f, 0.9f, 0.9f));
        renderer.drawString(label, 20.0f, top - 8.0f);

        const float gap   = 25.0f;
        const float tile  = displayCfg_.tileSize;
        const float cellW = (domain_.resX > 0) ? tile / float(domain_.resX) : tile;
        const float cellH = (domain_.resY > 0) ? tile / float(domain_.resY) : tile;

        for (size_t i = 0; i < grids.size(); ++i) {
            const float left = 20.0f + float(i) * (tile + gap);
            drawFieldHeatmap(renderer, grids[i], left, top, cellW, cellH);
        }
    }

    void drawFieldHeatmap(Renderer& renderer,
                          const GridField& field,
                          float left,
                          float top,
                          float cellW,
                          float cellH) const {
        const int resX = domain_.resX;
        const int resY = domain_.resY;
        if (resX <= 0 || resY <= 0) {
            return;
        }
        if (field.values.size() < size_t(resX) * size_t(resY)) {
            return;
        }

        for (int y = 0; y < resY; ++y) {
            for (int x = 0; x < resX; ++x) {
                const size_t idx = size_t(y) * size_t(resX) + size_t(x);
                const float sdf = field.values[idx];
                const float g   = sampleToGray(field, sdf);
                const Color color(g, g, g);

                const float px = left + (float(x) + 0.5f) * cellW;
                const float py = top  + (float(y) + 0.5f) * cellH;
                const float ps = std::max(cellW, cellH);
                renderer.draw2dPoint(Vec2(px, py), color, ps);
            }
        }
    }

    float sampleToGray(const GridField& field, float sdf) const {
        const bool forceMask = (displayCfg_.debugMode == 0);
        if (forceMask) {
            if (displayCfg_.softMask) {
                const float tau = std::max(displayCfg_.tau, 1e-6f);
                const float t   = 1.0f / (1.0f + std::exp(-(sdf / tau)));
                return std::clamp(t, 0.0f, 1.0f);
            }
            return sdf < 0.0f ? 0.0f : 1.0f;
        }

        const float range = (field.maxValue - field.minValue == 0.0f)
                                ? 1.0f
                                : (field.maxValue - field.minValue);
        const float norm = (sdf - field.minValue) / range;
        return std::clamp(norm, 0.0f, 1.0f);
    }

    void drawHelp(Renderer& renderer, float y) const {
        renderer.setColor(Color(0.7f, 0.7f, 0.7f));
        renderer.drawString("Keys: T train burst  R rebuild  M mask toggle  [ ] tau  1-4 debug  J save  K load", 20.0f, y);
        renderer.drawString("GPU latent panel blends shapes {0,1,2,3}; toggles affect both rows and panel.", 20.0f, y + 20.0f);
    }

private:
    static constexpr float kPanelLeft  = 10.0f;
    static constexpr float kPanelTop   = 540.0f;
    static constexpr float kPanelSize  = 400.0f;
    static constexpr int   kPanelN     = 5;
    static constexpr int   kPanelTile  = 56;
    static constexpr int   kPanelGap   = 4;

    struct DisplayConfig {
        bool  softMask  = true;
        float tau       = 0.05f;
        int   debugMode = 0;
        float tileSize  = 220.0f;
    };

    // --- Display + fields
    GpuLatentPanel     panel_;
    DisplayConfig      displayCfg_;
    FieldDomain        domain_;
    std::vector<GridField> originals_;
    std::vector<GridField> recon_;

    // --- Model
    TinyAutoDecoderCUDA decoder_;
    DeepSDF::TrainingDataset dataset_;
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
