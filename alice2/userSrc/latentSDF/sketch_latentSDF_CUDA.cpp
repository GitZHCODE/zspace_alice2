//#define __MAIN__
#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>
#include "ML/LatentSDF.h"

using namespace alice2;

inline Color valueToGray(float t){ t=std::clamp(t,0.0f,1.0f); return Color(t,t,t); }

class LatentSDFSketch_CUDA : public ISketch {
public:
    std::string getName() const override { return "LatentSDFSketch_CUDA_Split"; }
    std::string getDescription() const override { return "GPU-batched auto-decoder (Fourier, row-batched recon)"; }
    std::string getAuthor() const override { return "alice2 User"; }

    void setup() override {
        scene().setBackgroundColor(Color(0,0,0));
        scene().setShowGrid(false); scene().setShowAxes(false);

        // Domain
        m_domain.resX = m_gridResolutionX;
        m_domain.resY = m_gridResolutionY;
        m_domain.xMin = m_xMin; m_domain.xMax = m_xMax;
        m_domain.yMin = m_yMin; m_domain.yMax = m_yMax;

        // Analytic originals
        m_original.assign(3, GridField{});
        for (int s=0;s<3;++s){
            buildAnalyticGrid(s, m_domain.resX, m_domain.resY,
                              m_domain.xMin, m_domain.xMax,
                              m_domain.yMin, m_domain.yMax,
                              m_original[s].values, m_original[s].minValue, m_original[s].maxValue);
        }

        // CUDA AD
        ad_.setLambdaLatent(lambdaLatent);   // L2 on latents
        ad_.setWeightDecayW(weitDecayW);   // weight decay on MLP weights

        const std::vector<int> hidden = {64,64,64};
        ad_.initialize(/*numShapes*/3, /*latentDim*/m_latentDim, hidden, /*seed*/1234, /*maxBatch*/1024, /*numFreqs*/6, /*includeInput*/true);

        trainBurst(1, 200, m_batchSize);
        generateReconstruction();
    }

    void update(float /*time*/) override {}

    void draw(Renderer& r, Camera&) override {
        const float startY=20.f, gapY=40.f;
        r.setColor(Color(0.9f,0.9f,0.9f));
        r.drawString("Original (analytic scaled SDF)", 20.f, startY-8.f);
        drawRow(r, m_original, startY);

        const float reconTop = startY + m_tile + gapY;
        r.setColor(Color(0.9f,0.9f,0.9f));
        r.drawString(m_mask? (m_soft? "Reconstructed (soft mask; CUDA)":"Reconstructed (hard mask; CUDA)")
                            : "Reconstructed (continuous SDF; CUDA)", 20.f, reconTop-8.f);
        drawRow(r, m_recon, reconTop);

        r.setColor(Color(0.7f,0.7f,0.7f));
        char buf[256];
        std::snprintf(buf,sizeof(buf),"Hotkeys: [T] train x1  [B/b] ++/-- batch  [M] mask  [S] soft  [[]/[]] tau=%.3f  B=%d",
                      m_tau, m_batchSize);
        r.drawString(buf, 20.f, reconTop + m_tile + 40.f);
    }

    bool onKeyPress(unsigned char k, int, int) override {
        switch(k){
            case 't': case 'T': 
            {
                std::printf("\nRunning CUDA\n");
                trainBurst(50, 200, m_batchSize); 
                generateReconstruction(); 
                m_numEpoch+=50;

                double avgLoss=0.0, meanZ=0.0;
                ad_.syncStatsToHost(avgLoss, meanZ, /*reset=*/true);
                std::printf("[TrainCUDA] epoch=%d  avgLoss=%.6f  mean||z||=%.6f  (B=%d)\n",
                            m_numEpoch, float(avgLoss), float(meanZ), m_batchSize);

                if(lrW > 5e-3f && avgLoss < 1e-2f)
                {
                    lrW = 5e-3f;
                    lrZ = 1e-2f;
                }

                return true;
            }
            case 'r': case 'R': generateReconstruction(); return true;
            case 'b': m_batchSize = std::max(16, m_batchSize/2); return true;
            case 'B': m_batchSize = std::min(1024, m_batchSize*2); return true;
            case 'm': case 'M': m_mask=!m_mask; return true;
            case 's': case 'S': m_soft=!m_soft; return true;
            case '[': m_tau=std::max(0.005f, m_tau*0.8f); return true;
            case ']': m_tau=std::min(0.5f, m_tau*1.25f); return true;
        }
        return false;
    }

private:
    void trainBurst(int epochs,int microBatches,int B){
        Sampler samp(777); std::mt19937 rng(2025);
        for (int e=1;e<=epochs;++e){
            double epochLoss=0.0, meanZ=0.0;
            for (int mb=0;mb<microBatches;++mb){
                double L=0.0, mZ=0.0; ad_.trainMicroBatchGPU(B, samp, rng, lrW, lrZ);
                epochLoss += L; meanZ += mZ;
            }
        }
    }

    void generateReconstruction(){
        const auto& Z = ad_.latents(); if (Z.empty()) return;
        m_recon.assign(Z.size(), GridField{});
        const int W=m_gridResolutionX, H=m_gridResolutionY;
        const float xStep=(W>1)?(m_xMax-m_xMin)/float(W-1):0.f;
        const float yStep=(H>1)?(m_yMax-m_yMin)/float(H-1):0.f;

        std::vector<float> xs(W), yrow;
        for (size_t s=0;s<Z.size();++s){
            auto& g = m_recon[s];
            g.values.resize(size_t(W)*size_t(H));
            g.minValue= std::numeric_limits<float>::max(); g.maxValue=-g.minValue;

            for (int y=0;y<H;++y){
                float yy = m_yMin + yStep*float(y);
                for (int x=0;x<W;++x) xs[x] = m_xMin + xStep*float(x);
                ad_.forwardRowGPU((int)s, xs, yy, yrow);
                for (int x=0;x<W;++x){
                    size_t idx = size_t(y)*size_t(W)+size_t(x);
                    float v=yrow[x];
                    g.values[idx]=v; g.minValue=std::min(g.minValue,v); g.maxValue=std::max(g.maxValue,v);
                }
            }
        }
    }

    inline float sigmoid01(float x) const { return 1.f/(1.f+std::exp(-x)); }
    inline float softMask01(float v,float t) const { return sigmoid01(v/t); }

    void drawRow(Renderer& r, const std::vector<GridField>& grids, float top) const {
        const float gap=25.f, cellW=m_tile/float(m_gridResolutionX), cellH=m_tile/float(m_gridResolutionY);
        for (size_t i=0;i<grids.size();++i){
            float left=20.f + float(i)*(m_tile+gap);
            const auto& g=grids[i];
            if (!m_mask){
                float mn=g.minValue, mx=g.maxValue, rg=(mx-mn)==0.f?1.f:(mx-mn);
                for (int y=0;y<m_gridResolutionY;++y) for (int x=0;x<m_gridResolutionX;++x){
                    float norm=(g.values[size_t(y)*size_t(m_gridResolutionX)+size_t(x)]-mn)/rg;
                    float px=left+(float(x)+0.5f)*cellW, py=top+(float(y)+0.5f)*cellH;
                    r.draw2dPoint(Vec2(px,py), valueToGray(norm), std::max(cellW,cellH)*0.8f);
                }
            } else {
                for (int y=0;y<m_gridResolutionY;++y) for (int x=0;x<m_gridResolutionX;++x){
                    float v=g.values[size_t(y)*size_t(m_gridResolutionX)+size_t(x)];
                    float v01 = m_soft? softMask01(v,m_tau) : (v<0.f?0.f:1.f);
                    float px=left+(float(x)+0.5f)*cellW, py=top+(float(y)+0.5f)*cellH;
                    r.draw2dPoint(Vec2(px,py), valueToGray(v01), std::max(cellW,cellH)*0.8f);
                }
            }
            r.setColor(Color(0.7f,0.7f,0.9f));
            r.drawString("#"+std::to_string(i), left, top+m_tile+16.f);

            // Optional: thin contour overlay (zero-crossings) for extra “crisp”
            r.setColor(Color(1.0f, 0.2f, 0.2f));
            const float ps = std::max(cellW, cellH) * 0.25f; // small dot

            for (int y = 0; y < m_gridResolutionY - 1; ++y) {
                for (int x = 0; x < m_gridResolutionX - 1; ++x) {
                    const size_t i00 = size_t(y)   * size_t(m_gridResolutionX) + size_t(x);
                    const size_t i10 = size_t(y)   * size_t(m_gridResolutionX) + size_t(x+1);
                    const size_t i01 = size_t(y+1) * size_t(m_gridResolutionX) + size_t(x);
                    const size_t i11 = size_t(y+1) * size_t(m_gridResolutionX) + size_t(x+1);

                    const float s00 = g.values[i00];
                    const float s10 = g.values[i10];
                    const float s01 = g.values[i01];
                    const float s11 = g.values[i11];

                    const bool signMix =
                        ((s00 < 0) != (s10 < 0)) ||
                        ((s10 < 0) != (s11 < 0)) ||
                        ((s11 < 0) != (s01 < 0)) ||
                        ((s01 < 0) != (s00 < 0));

                    if (signMix) {
                        const float cx = left + (float(x) + 1.0f) * cellW;
                        const float cy = top  + (float(y) + 1.0f) * cellH;
                        r.draw2dPoint(Vec2(cx, cy), Color(1.0f, 0.2f, 0.2f), ps);
                    }
                }
            }
        }
    }

private:
    FieldDomain m_domain;
    float m_tile = 220.f;

    bool m_mask=true, m_soft=true; float m_tau=0.05f;

    std::vector<GridField> m_original, m_recon;

    TinyAutoDecoderCUDA ad_;

    int m_gridResolutionX = 128, m_gridResolutionY = 128, m_latentDim = 16;
    float m_xMin = -1.2f, m_xMax = 1.2f, m_yMin = -1.2f, m_yMax = 1.2f;
    int m_batchSize = 16;
    int m_numEpoch = 0;

    float lrW = 5e-2f;
    float lrZ = 5e-2f;
    float lambdaLatent = 1e-4f;
    float weitDecayW = 1e-6f;
};

ALICE2_REGISTER_SKETCH_AUTO(LatentSDFSketch_CUDA)

#endif // __MAIN__
