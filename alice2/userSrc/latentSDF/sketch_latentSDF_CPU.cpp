// alice2 Empty Sketch Template
// Minimal template for creating a new user sketch in alice2

#define __MAIN__
#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>
// keep your own headers if needed; we won't use the external trainer in this sketch
// #include <ML/autodecoder.h>

using namespace alice2;

#pragma once
#include <vector>
#include <tuple>
#include <random>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <limits>
#include <memory>

// ----------------- Utility -----------------
inline float sigmoid01(float x) {
    // map to [0,1]
    return 1.0f / (1.0f + std::exp(-x));
}
inline float softMask01(float sdf, float tau) {
    // center at 0, sharpen by tau, then map to [0,1]
    return sigmoid01(sdf / tau);
}

inline float clampSDF(float d, float beta = 0.1f) {
    // softly scale by beta, then clamp to [-1,1]
    float v = d / beta;
    if (v < -1.f) v = -1.f;
    if (v >  1.f) v =  1.f;
    return v;
}

inline Color valueToGray(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return Color(t, t, t);
}

struct GridField {
    std::vector<float> values;
    float minValue = 0.0f;
    float maxValue = 0.0f;
};

struct FieldDomain {
    int   resX = 128, resY = 128;
    float xMin = -1.2f, xMax = 1.2f;
    float yMin = -1.2f, yMax = 1.2f;
};

// ---------- Analytic SDFs (negative inside, positive outside, 0 on boundary) ----------
inline float sdCircle(float x, float y, float r = 0.6f) {
    return std::sqrt(x*x + y*y) - r;
}
inline float sdBox(float x, float y, float hx = 0.55f, float hy = 0.55f) {
    float ax = std::fabs(x) - hx;
    float ay = std::fabs(y) - hy;
    float ox = std::max(ax, 0.0f);
    float oy = std::max(ay, 0.0f);
    float outside = std::sqrt(ox*ox + oy*oy);
    float inside  = std::max(ax, ay); // negative inside
    return (inside <= 0.0f) ? inside : outside;
}
inline float sdTriangleUp(float x, float y, float s = 1.1f) {
    const float k = std::sqrt(3.0f);
    x = std::fabs(x);
    float d1 = (k*x + y) - s;
    float d2 = (k*x - y) - s;
    float d3 = -y - s * 0.3f;
    float outside = std::max(std::max(d1, d2), d3);
    if (outside > 0.0f) return outside;
    float de1 = (k*x + y) - s;
    float de2 = (k*x - y) - s;
    float de3 = -y - s * 0.3f;
    float m = std::max(std::max(de1, de2), de3);
    return m;
}

// Convert true SDF to sign label in {-1, 0, +1} with band eps around boundary
inline float labelFromSDF(float d, float eps = 0.02f) {
    if (d < -eps) return -1.0f;
    if (d >  eps) return +1.0f;
    return 0.0f;
}

// ---------- Sampler over shapes ----------
struct Sampler {
    float range = 1.2f;
    float boundaryFrac = 0.5f;
    float boundaryBand = 0.02f;
    float cornerFrac   = 0.15f; // 15% of samples try to hit corners

    std::mt19937 rng;
    std::uniform_real_distribution<float> U;

    Sampler(unsigned seed = 999) : rng(seed), U(-1.f, 1.f) {}

    static float sdf(int shapeIdx, float x, float y) {
        switch (shapeIdx) {
            case 0: return sdCircle(x, y, 0.6f);
            case 1: return sdBox(x, y, 0.55f, 0.55f);
            case 2: return sdTriangleUp(x, y, 1.1f);
            default: return 1e9f;
        }
    }

 std::tuple<float,float,float> sampleForShape(int shapeIdx) {
        std::uniform_real_distribution<float> Udom(-range, range);
        std::uniform_real_distribution<float> U01(0.f, 1.f);

        auto emit = [&](float X, float Y){
            float d = sdf(shapeIdx, X, Y);
            return std::tuple<float,float,float>{X, Y, clampSDF(d)};
        };

        float r = U01(rng);
        if (r < cornerFrac) {
            // crude “corner attractor”: propose from a tighter box and accept if within boundary band
            for (int tries=0; tries<200; ++tries){
                float x = Udom(rng)*0.8f;
                float y = Udom(rng)*0.8f;
                float d = sdf(shapeIdx, x, y);
                // encourage places where multiple linear constraints meet (|d| small AND |x| and |y| moderate)
                if (std::fabs(d) < boundaryBand*1.5f && (std::fabs(x)+std::fabs(y) > 0.6f))
                    return emit(x,y);
            }
        }

        if (r < cornerFrac + boundaryFrac) {
            for (int tries = 0; tries < 200; ++tries) {
                float x = Udom(rng), y = Udom(rng);
                float d = sdf(shapeIdx, x, y);
                if (std::fabs(d) < boundaryBand) return emit(x,y);
            }
        }

        float x = Udom(rng), y = Udom(rng);
        return emit(x,y);
    }
};

// ---------- Minimal MLP + AutoDecoder (dependency-free) ----------
struct TinyMLP {
    int inDim=0, outDim=1;
    std::vector<int> hidden;
    std::vector<int> layerIn, layerOut;

    // params & buffers
    std::vector<std::vector<float>> W, b, gW, gb, pre, gPre, act, gAct;

    bool linearOutput = true; // NEW: final layer is linear by default

    void initialize(int inputDim, const std::vector<int>& hiddenLayers, int outputDim=1, unsigned seed=42) {
        inDim = inputDim; hidden = hiddenLayers; outDim = outputDim;
        const int L = (int)hidden.size() + 1;
        layerIn.resize(L); layerOut.resize(L);
        int prev = inDim;
        for (int l=0; l<L; ++l) {
            int width = (l < (int)hidden.size()) ? hidden[l] : outDim;
            layerIn[l] = prev; layerOut[l] = width; prev = width;
        }
        W.resize(L); b.resize(L); gW.resize(L); gb.resize(L);
        pre.resize(L); gPre.resize(L);
        act.resize(L+1); gAct.resize(L+1);

        std::mt19937 rng(seed); std::normal_distribution<float> N(0.f,1.f);
        for (int l=0; l<L; ++l) {
            int rows = layerOut[l], cols = layerIn[l];
            W[l].resize(rows*cols);
            b[l].assign(rows, 0.f);
            gW[l].assign(rows*cols, 0.f);
            gb[l].assign(rows, 0.f);
            pre[l].assign(rows, 0.f);
            gPre[l].assign(rows, 0.f);

            // Xavier: N(0, 1/sqrt(cols)). For the last layer, a touch smaller helps.
            float scale = (l == L-1) ? (0.5f/std::sqrt((float)cols)) : (1.0f/std::sqrt((float)cols));
            for (int i=0;i<rows*cols;i++) W[l][i] = scale * N(rng);
        }
        act[0].assign(inDim, 0.f);
        gAct[0].assign(inDim, 0.f);
        act[L].assign(outDim, 0.f);
        gAct[L].assign(outDim, 0.f);
    }

    // Forward: tanh on hidden; final either tanh or linear
    float forward(const std::vector<float>& x) {
        act[0] = x;
        const int L = (int)layerIn.size();
        for (int l=0; l<L; ++l) {
            int rows = layerOut[l], cols = layerIn[l];
            auto& out = act[l+1]; auto& z = pre[l];
            z.assign(rows, 0.f); out.assign(rows, 0.f);
            const auto& Wl = W[l]; const auto& bl = b[l];
            for (int i=0;i<rows;i++){
                float s = bl[i];
                const float* wrow = &Wl[i*cols];
                for (int j=0;j<cols;j++) s += wrow[j]*act[l][j];
                z[i]=s;
            }
            if (l == L-1 && linearOutput) {
                // identity
                for (int i=0;i<rows;i++) out[i] = z[i];
            } else {
                // tanh
                for (int i=0;i<rows;i++) out[i] = std::tanh(z[i]);
            }
        }
        return act[L][0];
    }

    // Backward: derivative is 1 for linear output, (1-a^2) for tanh
    std::vector<float> backward_and_accumulate(float target, float weightDecay=0.f) {
        const int L = (int)layerIn.size();
        for (int l=0;l<L;l++){
            std::fill(gW[l].begin(), gW[l].end(), 0.f);
            std::fill(gb[l].begin(), gb[l].end(), 0.f);
            std::fill(gPre[l].begin(), gPre[l].end(), 0.f);
        }
        for (int l=0;l<=L;l++){
            if ((int)gAct[l].size()!=(int)act[l].size()) gAct[l].assign(act[l].size(),0.f);
            else std::fill(gAct[l].begin(), gAct[l].end(), 0.f);
        }

        const float y = act[L][0];
        gAct[L][0] = (y - target); // d(1/2 (y-t)^2)/dy

        for (int l=L-1; l>=0; --l){
            int rows = layerOut[l], cols = layerIn[l];
            if (l == L-1 && linearOutput) {
                for (int i=0;i<rows;i++) gPre[l][i] = gAct[l+1][i]; // derivative 1
            } else {
                for (int i=0;i<rows;i++){
                    float a = act[l+1][i];
                    gPre[l][i] = gAct[l+1][i]*(1.f - a*a); // tanh'
                }
            }

            // bias & weights
            for (int i=0;i<rows;i++){
                gb[l][i] += gPre[l][i];
                float* gWrow = &gW[l][i*cols];
                for (int j=0;j<cols;j++) {
                    gWrow[j] += gPre[l][i]*act[l][j];
                }
            }

            // backprop to previous activations
            for (int j=0;j<cols;j++){
                float acc=0.f;
                for (int i=0;i<rows;i++) acc += W[l][i*cols+j]*gPre[l][i];
                gAct[l][j] += acc;
            }

            // optional L2 weight decay (on W, not b)
            if (weightDecay > 0.f) {
                float* gWptr = gW[l].data();
                const float* Wptr = W[l].data();
                const int n = rows*cols;
                for (int k=0;k<n;k++) gWptr[k] += weightDecay * Wptr[k];
            }
        }
        return gAct[0]; // gradient wrt input (z and coords)
    }

    void sgd(float lr) {
        const int L=(int)layerIn.size();
        for (int l=0;l<L;l++){
            int rows=layerOut[l], cols=layerIn[l];
            for (int i=0;i<rows;i++){
                b[l][i] -= lr*gb[l][i];
                float* Wrow = &W[l][i*cols];
                const float* gWrow=&gW[l][i*cols];
                for (int j=0;j<cols;j++) Wrow[j] -= lr*gWrow[j];
            }
        }
    }
};

struct PosEnc2D {
    int   numFreqs = 2;     // try 4–8
    bool  includeInput = true;
    float twoPi = 6.283185307179586f;

    int encodedDim() const {
        int base = includeInput ? 2 : 0;
        return base + 2 /*sin,cos*/ * 2 /*x,y*/ * numFreqs;
    }

    // enc = [x,y] (optional) + {sin(2^k*2π*x), cos(...), sin(...y), cos(...y)} for k=0..numFreqs-1
    void encode(float x, float y, std::vector<float>& out) const {
        out.clear();
        out.reserve(encodedDim());
        if (includeInput) { out.push_back(x); out.push_back(y); }
        float freq = 1.0f;
        for (int k=0;k<numFreqs;k++){
            float ax = freq * twoPi * x;
            float ay = freq * twoPi * y;
            out.push_back(std::sinf(ax));
            out.push_back(std::cosf(ax));
            out.push_back(std::sinf(ay));
            out.push_back(std::cosf(ay));
            freq *= 2.0f;
        }
    }
};

struct TinyAutoDecoder {
    int numShapes=0, latentDim=16, coordDim=2;
    float lambdaLatent = 1e-4f;
    float weightDecayW = 1e-6f; // tiny L2 on weights (optional)
    TinyMLP decoder;
    std::vector<std::vector<float>> Z;

    PosEnc2D enc;
    int coordEncDim = 0;
    std::vector<float> encBuf; // reuse buffer

    void initialize(int nShapes, int zDim, const std::vector<int>& hidden, unsigned seed=1234){
        numShapes=nShapes; latentDim=zDim;

        enc.numFreqs = 6;         // tweak 4–8
        enc.includeInput = true;  // keep raw (x,y) too
        coordEncDim = enc.encodedDim();

        decoder.linearOutput = true;
        decoder.initialize(latentDim + coordEncDim, hidden, 1, seed);

        Z.assign(numShapes, std::vector<float>(latentDim, 0.f));
        std::mt19937 rng(seed); std::normal_distribution<float> N(0.f,0.01f);
        for (int i=0;i<numShapes;i++) for (int j=0;j<latentDim;j++) Z[i][j]=N(rng);

        encBuf.reserve(coordEncDim);
    }

    std::vector<float> buildInput(int shapeIdx, float x, float y) const {
        enc.encode(x, y, const_cast<std::vector<float>&>(encBuf)); // fill encBuf
        std::vector<float> in(latentDim + coordEncDim);
        // copy latent
        for (int j=0;j<latentDim;j++) in[j]=Z[shapeIdx][j];
        // copy encoded coords
        for (int j=0;j<coordEncDim;j++) in[latentDim + j] = encBuf[j];
        return in;
    }

    std::tuple<float,float,float> trainSample(int shapeIdx, float x, float y, float target, float lrW, float lrZ){
        auto in = buildInput(shapeIdx, x, y);
        float ypred = decoder.forward(in);

        float diff=ypred-target;
        float mse=0.5f*diff*diff;

        float l2z=0.f; for (int j=0;j<latentDim;j++) l2z += Z[shapeIdx][j]*Z[shapeIdx][j];
        float loss = mse + 0.5f*lambdaLatent*l2z;

        auto dL_din = decoder.backward_and_accumulate(target, /*weightDecayW*/weightDecayW);
        for (int j=0;j<latentDim;j++) dL_din[j] += lambdaLatent*Z[shapeIdx][j];

        decoder.sgd(lrW);
        for (int j=0;j<latentDim;j++) Z[shapeIdx][j] -= lrZ*dL_din[j];

        return {loss, ypred, target};
    }
};

// ---------- Build an analytic label field on a grid ----------
inline void buildAnalyticGrid(int shapeIdx,
                              int resX, int resY,
                              float xMin, float xMax, float yMin, float yMax,
                              std::vector<float>& out, float& vmin, float& vmax)
{
    out.resize(size_t(resX)*size_t(resY));
    vmin =  std::numeric_limits<float>::max();
    vmax = -std::numeric_limits<float>::max();

    const float xStep = (resX > 1) ? (xMax - xMin) / float(resX - 1) : 0.f;
    const float yStep = (resY > 1) ? (yMax - yMin) / float(resY - 1) : 0.f;

    for (int y = 0; y < resY; ++y) {
        const float yy = yMin + yStep * float(y);
        for (int x = 0; x < resX; ++x) {
            const float xx = xMin + xStep * float(x);
            const float d  = Sampler::sdf(shapeIdx, xx, yy);
            const float v  = labelFromSDF(d, 0.02f); // -1 / 0 / +1
            const size_t idx = size_t(y) * size_t(resX) + size_t(x);
            out[idx] = v;
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }
    }
}


// ========================== SKETCH ==========================
class LatentSDFSketch : public ISketch {
public:
    LatentSDFSketch() = default;
    ~LatentSDFSketch() = default;

    std::string getName() const override { return "LatentSDFSketch"; }
    std::string getDescription() const override { return "Original vs Reconstructed latent SDFs"; }
    std::string getAuthor() const override { return "alice2 User"; }

    void setup() override {

        scene().setBackgroundColor(Color(0.0f, 0.0f, 0.0f));
        scene().setShowGrid(false);
        scene().setShowAxes(false);

        // Domain/grid
        m_domain.resX = (m_gridResolutionX > 0) ? m_gridResolutionX : 128;
        m_domain.resY = (m_gridResolutionY > 0) ? m_gridResolutionY : 128;
        m_domain.xMin = (m_xMin < m_xMax) ? m_xMin : -1.2f;
        m_domain.xMax = (m_xMin < m_xMax) ? m_xMax :  1.2f;
        m_domain.yMin = (m_yMin < m_yMax) ? m_yMin : -1.2f;
        m_domain.yMax = (m_yMin < m_yMax) ? m_yMax :  1.2f;

        // Mirror back for convenience
        m_gridResolutionX = m_domain.resX;
        m_gridResolutionY = m_domain.resY;
        m_xMin = m_domain.xMin; m_xMax = m_domain.xMax;
        m_yMin = m_domain.yMin; m_yMax = m_domain.yMax;

        // Build analytic originals (0 circle, 1 square, 2 triangle)
        m_originalGridFields.assign(3, GridField{});
        for (int s = 0; s < 3; ++s) {
            buildAnalyticGrid(s, m_domain.resX, m_domain.resY,
                              m_domain.xMin, m_domain.xMax, m_domain.yMin, m_domain.yMax,
                              m_originalGridFields[s].values,
                              m_originalGridFields[s].minValue,
                              m_originalGridFields[s].maxValue);
        }
        m_hasOriginal = true;

        // Init minimal auto-decoder
        const std::vector<int> hidden = {64,64,64};
        m_ad.initialize(/*numShapes*/3, /*latentDim*/m_latentDim, hidden, /*seed*/1234);
        m_ad.lambdaLatent = 1e-4f;

        // Warm-start: do a tiny bit of training so recon isn't random
        trainBurst(/*epochs*/1, /*stepsPerEpoch*/5000);

        // Build reconstructed grids
        generateReconstruction();
    }

    void update(float) override { /* no-op */ }

    void draw(Renderer& renderer, Camera&) override {
        const float startY = 20.0f;
        const float gapY   = 40.0f;

        renderer.setColor(Color(0.9f, 0.9f, 0.9f));
        renderer.drawString("Original (analytic labels)", 20.0f, startY - 8.0f);
        drawFieldRow(renderer, m_originalGridFields, startY);

        const float reconTop = startY + m_tileSize + gapY;
        renderer.setColor(Color(0.9f, 0.9f, 0.9f));
        renderer.drawString("Reconstructed (decoder output)", 20.0f, reconTop - 8.0f);
        drawFieldRow(renderer, m_reconstructedGridFields, reconTop);

        renderer.setColor(Color(0.7f, 0.7f, 0.7f));
        renderer.drawString("Hotkeys: [T] train burst   [R] refresh recon",
                            20.0f, reconTop + m_tileSize + 40.0f);
    }

    bool onKeyPress(unsigned char key, int, int) override {
        switch (key) {
            case 't': case 'T':
                trainBurst(50, 10000);
                generateReconstruction();
                return true;
            case 'r': case 'R':
                generateReconstruction();
                return true;

            // NEW: mask view toggles
            case 'm': case 'M':
                m_showMask = !m_showMask;   // toggle heatmap ↔ mask
                return true;
            case 's': case 'S':
                m_softMask = !m_softMask;   // toggle soft ↔ hard
                return true;
            case '[':
                m_tau = std::max(0.005f, m_tau * 0.8f); // sharper
                return true;
            case ']':
                m_tau = std::min(0.5f, m_tau * 1.25f);  // softer
                return true;
        }
        return false;
    }

private:
    // -------- minimal training burst (per-sample schedule) --------
    void trainBurst(int epochs, int stepsPerEpoch) {
        Sampler samp(777);
        std::uniform_int_distribution<int> pick(0, 2);
        std::mt19937 rng(2025);

        const float lrW = 5e-4f;
        const float lrZ = 1e-3f;

        for (int e = 1; e <= epochs; ++e) {
            double runLoss = 0.0;
            for (int s = 0; s < stepsPerEpoch; ++s) {
                int idx = pick(rng);
                auto [x, y, t] = samp.sampleForShape(idx); // t is now (soft) SDF in [-1,1]
                auto [L, ypred, tgt] = m_ad.trainSample(idx, x, y, t, lrW, lrZ);
                runLoss += (double)L;
            }
            double avgLoss = runLoss / (double)stepsPerEpoch;

            double meanZ = 0.0;
            for (int i = 0; i < 3; ++i) {
                double nz = 0.0;
                for (int j = 0; j < m_latentDim; ++j) nz += (double)m_ad.Z[i][j] * m_ad.Z[i][j];
                meanZ += std::sqrt(nz + 1e-12);
            }
            meanZ /= 3.0;
            std::printf("[TrainBurst] epoch=%d  avgLoss=%.6f  mean||z||=%.6f\n", m_numEpoch, (float)avgLoss, (float)meanZ);
            m_numEpoch++;
        }
    }

    // -------- build reconstructed grids from current decoder+latents --------
    void generateReconstruction() {
        if (m_latentDim <= 0 || m_coordDim != 2) return;
        if (m_gridResolutionX <= 0 || m_gridResolutionY <= 0) return;

        const auto& latentCodes = m_ad.Z;
        if (latentCodes.empty()) return;

        m_reconstructedGridFields.assign(latentCodes.size(), GridField{});

        const float xStep = (m_gridResolutionX > 1) ? (m_xMax - m_xMin) / float(m_gridResolutionX - 1) : 0.0f;
        const float yStep = (m_gridResolutionY > 1) ? (m_yMax - m_yMin) / float(m_gridResolutionY - 1) : 0.0f;

        for (size_t shape = 0; shape < latentCodes.size(); ++shape) {
            const auto& latent = latentCodes[shape];
            if (latent.size() != size_t(m_latentDim)) continue;

            GridField& grid = m_reconstructedGridFields[shape];
            grid.values.resize(size_t(m_gridResolutionX) * size_t(m_gridResolutionY));
            grid.minValue =  std::numeric_limits<float>::max();
            grid.maxValue = -std::numeric_limits<float>::max();

        std::vector<float> in(size_t(m_latentDim + m_ad.coordEncDim), 0.f);
        for (int y = 0; y < m_gridResolutionY; ++y) {
            const float yy = m_yMin + yStep * float(y);
            for (int x = 0; x < m_gridResolutionX; ++x) {
                const float xx = m_xMin + xStep * float(x);
                const size_t idx = size_t(y) * size_t(m_gridResolutionX) + size_t(x);

                std::copy(latent.begin(), latent.end(), in.begin());
                m_ad.enc.encode(xx, yy, m_ad.encBuf);
                for (int j=0;j<m_ad.coordEncDim;j++) in[m_latentDim + j] = m_ad.encBuf[j];

                const float v = m_ad.decoder.forward(in);
                grid.values[idx] = v;
                grid.minValue = std::min(grid.minValue, v);
                grid.maxValue = std::max(grid.maxValue, v);
            }
        }

            if (grid.minValue ==  std::numeric_limits<float>::max()) grid.minValue = 0.f;
            if (grid.maxValue == -std::numeric_limits<float>::max()) grid.maxValue = 0.f;
        }
        m_hasReconstruction = true;
    }

    void drawFieldRow(Renderer& renderer, const std::vector<GridField>& grids, float top) const {
        if (grids.empty() || m_gridResolutionX <= 0 || m_gridResolutionY <= 0) return;

        const float gap = 25.0f;
        const float cellW = m_tileSize / float(m_gridResolutionX);
        const float cellH = m_tileSize / float(m_gridResolutionY);

        for (size_t fieldIdx = 0; fieldIdx < grids.size(); ++fieldIdx) {
            const float left = 20.0f + float(fieldIdx) * (m_tileSize + gap);
            drawFieldHeatmap(renderer, grids[fieldIdx], left, top, cellW, cellH);
            renderer.setColor(Color(0.7f, 0.7f, 0.9f));
            renderer.drawString("#" + std::to_string(fieldIdx), left, top + m_tileSize + 16.0f);
        }
    }

    void drawFieldHeatmap(Renderer& renderer, const GridField& field,
                        float left, float top, float cellW, float cellH) const
    {
        if (!m_showMask) {
            // original continuous heatmap
            const float safeMin = field.minValue;
            const float safeMax = field.maxValue;
            const float range   = (safeMax - safeMin == 0.0f) ? 1.0f : (safeMax - safeMin);

            for (int y = 0; y < m_gridResolutionY; ++y) {
                for (int x = 0; x < m_gridResolutionX; ++x) {
                    const size_t idx = size_t(y) * size_t(m_gridResolutionX) + size_t(x);
                    const float value = field.values[idx];
                    const float norm  = (value - safeMin) / range;
                    const Color color = valueToGray(norm);

                    const float px = left + (float(x) + 0.5f) * cellW;
                    const float py = top  + (float(y) + 0.5f) * cellH;
                    const float pointSize = std::max(cellW, cellH) * 0.8f;
                    renderer.draw2dPoint(Vec2(px, py), color, pointSize);
                }
            }
            return;
        }

        // MASK MODE: inside=black, outside=white (hard or soft)
        for (int y = 0; y < m_gridResolutionY; ++y) {
            for (int x = 0; x < m_gridResolutionX; ++x) {
                const size_t idx = size_t(y) * size_t(m_gridResolutionX) + size_t(x);
                const float sdf  = field.values[idx];

                float v01;
                if (m_softMask) {
                    // soft edge controlled by m_tau
                    v01 = softMask01(sdf, m_tau);  // ~0 inside, ~1 outside
                } else {
                    // hard step at 0
                    v01 = (sdf < 0.0f) ? 0.0f : 1.0f;
                }
                const Color color = valueToGray(v01);

                const float px = left + (float(x) + 0.5f) * cellW;
                const float py = top  + (float(y) + 0.5f) * cellH;
                const float pointSize = std::max(cellW, cellH) * 0.8f;
                renderer.draw2dPoint(Vec2(px, py), color, pointSize);
            }
        }

        // Optional: thin contour overlay (zero-crossings) for extra “crisp”
        renderer.setColor(Color(1.0f, 0.2f, 0.2f));
        const float ps = std::max(cellW, cellH) * 0.9f;
        for (int y = 0; y < m_gridResolutionY - 1; ++y) {
            for (int x = 0; x < m_gridResolutionX - 1; ++x) {
                const size_t i00 = size_t(y)   * size_t(m_gridResolutionX) + size_t(x);
                const size_t i10 = size_t(y)   * size_t(m_gridResolutionX) + size_t(x+1);
                const size_t i01 = size_t(y+1) * size_t(m_gridResolutionX) + size_t(x);
                const size_t i11 = size_t(y+1) * size_t(m_gridResolutionX) + size_t(x+1);

                const float s00 = field.values[i00];
                const float s10 = field.values[i10];
                const float s01 = field.values[i01];
                const float s11 = field.values[i11];

                // if any sign change in the cell, draw a dot at its center
                const bool signMix = (s00<0)!=(s10<0) || (s10<0)!=(s11<0) || (s11<0)!=(s01<0) || (s01<0)!=(s00<0);
                if (signMix) {
                    const float cx = left + (float(x)+1.0f) * cellW;
                    const float cy = top  + (float(y)+1.0f) * cellH;
                    renderer.draw2dPoint(Vec2(cx, cy), Color(1.0f, 0.2f, 0.2f), ps*0.25f);
                }
            }
        }
    }

private:
    // domain & layout
    FieldDomain m_domain;
    float m_tileSize = 220.0f;
    int m_numEpoch = 0;

    // flags
    bool  m_hasOriginal       = false;
    bool  m_hasReconstruction = false;

    // fields
    std::vector<GridField> m_originalGridFields;
    std::vector<GridField> m_reconstructedGridFields;

    // minimal model
    TinyAutoDecoder m_ad;

    // UI/config
    int   m_gridResolutionX   = 128;
    int   m_gridResolutionY   = 128;
    int   m_latentDim         = 16;
    int   m_coordDim          = 2;
    float m_xMin              = -1.2f, m_xMax = 1.2f;
    float m_yMin              = -1.2f, m_yMax = 1.2f;

    bool  m_showMask          = true;   // toggle heatmap vs mask
    bool  m_softMask          = true;   // soft (tanh) vs hard step
    float m_tau               = 0.05f;  // soft threshold sharpness
};

// Register the sketch with alice2
ALICE2_REGISTER_SKETCH_AUTO(LatentSDFSketch)

#endif // __MAIN__
