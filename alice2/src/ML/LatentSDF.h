#pragma once
#include <vector>
#include <tuple>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstring>

// ---------- Analytic SDFs (host) ----------
inline float sdCircle(float x, float y, float r = 0.6f) { return std::sqrt(x*x + y*y) - r; }
inline float sdBox(float x, float y, float hx = 0.55f, float hy = 0.55f) {
    float ax = std::fabs(x) - hx, ay = std::fabs(y) - hy;
    float ox = std::max(ax, 0.0f), oy = std::max(ay, 0.0f);
    float outside = std::sqrt(ox*ox + oy*oy);
    float inside  = std::max(ax, ay);
    return (inside <= 0.0f) ? inside : outside;
}
inline float sdTriangleUp(float x, float y, float s = 1.1f) {
    const float k = std::sqrt(3.0f); x = std::fabs(x);
    float d1 = (k*x + y) - s, d2 = (k*x - y) - s, d3 = -y - s*0.3f;
    float outside = std::max(std::max(d1,d2), d3);
    if (outside > 0.0f) return outside;
    return std::max(std::max(d1,d2), d3);
}

// clamp true SDF into [-1,1] band (β=0.1)
inline float clampSDF(float d, float beta = 0.1f) {
    float v = d / beta; if (v<-1.f) v=-1.f; if (v>1.f) v=1.f; return v;
}

struct GridField { std::vector<float> values; float minValue=0.f, maxValue=0.f; };
struct FieldDomain { int resX=128, resY=128; float xMin=-1.2f,xMax=1.2f,yMin=-1.2f,yMax=1.2f; };

// ---------- CPU sampler (boundary-biased) ----------
struct Sampler {
    float range=1.2f, boundaryFrac=0.6f, boundaryBand=0.02f, cornerFrac=0.15f;
    std::mt19937 rng; std::uniform_real_distribution<float> U;
    Sampler(unsigned seed=999):rng(seed),U(-1.f,1.f){}
    static float sdf(int shapeIdx,float x,float y){
        switch(shapeIdx){ case 0: return sdCircle(x,y,0.6f); case 1: return sdBox(x,y,0.55f,0.55f); case 2: return sdTriangleUp(x,y,1.1f); default: return 1e9f; }
    }
    std::tuple<float,float,float> sampleForShape(int shapeIdx){
        std::uniform_real_distribution<float> Udom(-range,range), U01(0.f,1.f);
        auto emit=[&](float X,float Y){ return std::tuple<float,float,float>{X,Y, clampSDF(sdf(shapeIdx,X,Y))}; };
        float r=U01(rng);
        if (r < cornerFrac) {
            for (int t=0;t<200;++t){ float x=Udom(rng)*0.8f, y=Udom(rng)*0.8f; float d=sdf(shapeIdx,x,y);
                if (std::fabs(d)<boundaryBand*1.5f && (std::fabs(x)+std::fabs(y)>0.6f)) return emit(x,y); }
        }
        if (r < cornerFrac+boundaryFrac) {
            for (int t=0;t<200;++t){ float x=Udom(rng), y=Udom(rng); float d=sdf(shapeIdx,x,y);
                if (std::fabs(d)<boundaryBand) return emit(x,y); }
        }
        float x=Udom(rng), y=Udom(rng); return emit(x,y);
    }
};

inline void buildAnalyticGrid(int shapeIdx,int resX,int resY,float xMin,float xMax,float yMin,float yMax,
                              std::vector<float>& out,float& vmin,float& vmax){
    out.resize(size_t(resX)*size_t(resY));
    vmin= std::numeric_limits<float>::max(); vmax=-vmin;
    float xStep=(resX>1)?(xMax-xMin)/float(resX-1):0.f, yStep=(resY>1)?(yMax-yMin)/float(resY-1):0.f;
    for(int y=0;y<resY;++y){ float yy=yMin+yStep*float(y);
        for(int x=0;x<resX;++x){ float xx=xMin+xStep*float(x);
            float v=clampSDF(Sampler::sdf(shapeIdx,xx,yy),0.1f);
            size_t i=size_t(y)*size_t(resX)+size_t(x);
            out[i]=v; vmin=std::min(vmin,v); vmax=std::max(vmax,v);
        }
    }
}

// ---------- CUDA auto-decoder (opaque API; implemented in .cu) ----------
class TinyAutoDecoderCUDA {
public:
    TinyAutoDecoderCUDA();
    ~TinyAutoDecoderCUDA();

    void initialize(int numShapes, int latentDim,
                    const std::vector<int>& hidden,
                    unsigned seed, int maxBatch,
                    int numFreqs = 6, bool includeInput = true);

    // micro-batch training step (GPU): batch W update, per-sample Z update
    void trainMicroBatchGPU(int B, Sampler& sampler, std::mt19937& rng,
                            float lrW, float lrZ);

    // row forward for visualization
    void forwardRowGPU(int shapeIdx, const std::vector<float>& xs, float y,
                       std::vector<float>& outY) const;

    // sync device latents to host (e.g., before visualization)
    void syncLatentsToHost();

    // pull running stats (no training sync unless you call this)
    // If reset=true, running loss & sample counters are zeroed after read.
    void syncStatsToHost(double& avgLoss, double& meanZ, bool reset = true);

    // accessors / knobs
    const std::vector<std::vector<float>>& latents() const { return Z_; }
    std::vector<std::vector<float>>&       latents()       { return Z_; }
    int  latentDim()   const { return latentDim_; }
    int  coordEncDim() const { return coordEncDim_; }
    int  numShapes()   const { return numShapes_; }

    void setLambdaLatent(float v){ lambdaLatent_ = v; }
    void setWeightDecayW(float v){ weightDecayW_ = v; }

private:
    struct Impl; Impl* impl_ = nullptr;

    int   numShapes_    = 0;
    int   latentDim_    = 0;
    int   coordEncDim_  = 0;
    float lambdaLatent_ = 1e-4f;
    float weightDecayW_ = 1e-6f;

    std::vector<std::vector<float>> Z_; // host mirror of latents (used only when you request)
};
