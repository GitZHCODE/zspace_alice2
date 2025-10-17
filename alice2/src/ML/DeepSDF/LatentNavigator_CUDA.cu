#define ALICE2_USE_CUDA

#ifdef ALICE2_USE_CUDA

#include "LatentNavigator_CUDA.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

namespace DeepSDF {

namespace {
inline void checkCuda(const char* call, cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "[LatentNavigator_CUDA] %s failed: %s\n",
                     call, cudaGetErrorString(status));
        std::abort();
    }
}
}

LatentNavigator_CUDA::LatentNavigator_CUDA() = default;
LatentNavigator_CUDA::~LatentNavigator_CUDA() { shutdown(); }

void LatentNavigator_CUDA::initialize(TinyAutoDecoderCUDA* decoder,
                                      const FieldDomain* domain,
                                      const std::vector<std::vector<float>>* latentSource)
{
    decoder_ = decoder;
    domain_  = domain;
    latents_ = latentSource;
    markPanelDirty();
}

void LatentNavigator_CUDA::shutdown()
{
    if (panelResource_) {
        cudaGraphicsUnregisterResource(panelResource_);
        panelResource_ = nullptr;
    }
    if (panelTexture_ != 0) {
        glDeleteTextures(1, &panelTexture_);
        panelTexture_ = 0;
    }
    if (dPanelField_)   { checkCuda("cudaFree(panelField)", cudaFree(dPanelField_));   dPanelField_ = nullptr; }
    if (dPanelRGBA_)    { checkCuda("cudaFree(panelRGBA)",  cudaFree(dPanelRGBA_));    dPanelRGBA_ = nullptr; }
    if (dScratchLatent_){ checkCuda("cudaFree(scratchLatent)", cudaFree(dScratchLatent_)); dScratchLatent_ = nullptr; }
    if (dTileMin_)      { checkCuda("cudaFree(tileMin)", cudaFree(dTileMin_)); dTileMin_ = nullptr; }
    if (dTileMax_)      { checkCuda("cudaFree(tileMax)", cudaFree(dTileMax_)); dTileMax_ = nullptr; }
    if (dFieldScratch_) { checkCuda("cudaFree(fieldScratch)", cudaFree(dFieldScratch_)); dFieldScratch_ = nullptr; }
    fieldScratchRes_ = 0;
    decoder_ = nullptr;
    domain_  = nullptr;
    latents_ = nullptr;
}

void LatentNavigator_CUDA::setPanelResolution(int N, int tileRes, int gap)
{
    panelN_  = std::max(1, N);
    tileRes_ = std::max(4, tileRes);
    tileGap_ = std::max(0, gap);
    markPanelDirty();
}

void LatentNavigator_CUDA::setCornerIndices(const std::array<int,4>& indices)
{
    cornerIdx_ = indices;
    markPanelDirty();
}

void LatentNavigator_CUDA::setLatentSource(const std::vector<std::vector<float>>* latentSource)
{
    latents_ = latentSource;
    markPanelDirty();
}

void LatentNavigator_CUDA::setDetailResolution(int /*res*/)
{
    // reserved for future
}

void LatentNavigator_CUDA::markPanelDirty()
{
    panelDirty_ = true;
}

bool LatentNavigator_CUDA::ensurePanelResources()
{
    if (!decoder_ || !domain_) return false;

    const int newW = panelN_ * tileRes_ + (panelN_ - 1) * tileGap_;
    const int newH = newW;
    if (newW <= 0 || newH <= 0) return false;

    const size_t fieldBytes = size_t(newW) * size_t(newH) * sizeof(float);
    if (!dPanelField_ || panelW_ != newW || panelH_ != newH) {
        if (dPanelField_) checkCuda("cudaFree(panelField)", cudaFree(dPanelField_));
        checkCuda("cudaMalloc(panelField)", cudaMalloc(&dPanelField_, fieldBytes));
        panelDirty_ = true;
    }
    if (!dPanelRGBA_ || panelW_ != newW || panelH_ != newH) {
        if (dPanelRGBA_) checkCuda("cudaFree(panelRGBA)", cudaFree(dPanelRGBA_));
        checkCuda("cudaMalloc(panelRGBA)", cudaMalloc(&dPanelRGBA_, size_t(newW)*size_t(newH)*sizeof(uchar4)));
        panelDirty_ = true;
    }

    if (!dScratchLatent_) {
        checkCuda("cudaMalloc(scratchLatent)", cudaMalloc(&dScratchLatent_, size_t(decoder_->latentDim()) * sizeof(float)));
    }

    const int tileCount = panelN_ * panelN_;
    if (!dTileMin_) {
        checkCuda("cudaMalloc(tileMin)", cudaMalloc(&dTileMin_, size_t(tileCount) * sizeof(float)));
    }
    if (!dTileMax_) {
        checkCuda("cudaMalloc(tileMax)", cudaMalloc(&dTileMax_, size_t(tileCount) * sizeof(float)));
    }

    if (panelTexture_ == 0 || panelW_ != newW || panelH_ != newH) {
        if (panelResource_) {
            cudaGraphicsUnregisterResource(panelResource_);
            panelResource_ = nullptr;
        }
        if (panelTexture_ != 0) {
            glDeleteTextures(1, &panelTexture_);
            panelTexture_ = 0;
        }

        glGenTextures(1, &panelTexture_);
        glBindTexture(GL_TEXTURE_2D, panelTexture_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, newW, newH, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        checkCuda("cudaGraphicsGLRegisterImage",
                  cudaGraphicsGLRegisterImage(&panelResource_, panelTexture_, GL_TEXTURE_2D,
                                              cudaGraphicsRegisterFlagsWriteDiscard));
    }

    panelW_ = newW;
    panelH_ = newH;

    if (scratchLatentHost_.size() != size_t(decoder_->latentDim())) {
        scratchLatentHost_.assign(size_t(decoder_->latentDim()), 0.0f);
    }

    return true;
}

bool LatentNavigator_CUDA::rebuildPanel(const FieldRenderConfig& cfg)
{
    if (!ensurePanelResources()) return false;
    if (!latents_ || latents_->empty()) return false;

    const auto& allLatents = *latents_;
    auto pickLatent = [&](int idx) -> const std::vector<float>& {
        if (idx >= 0 && idx < (int)allLatents.size()) return allLatents[idx];
        return allLatents.front();
    };

    const auto& z00 = pickLatent(cornerIdx_[0]);
    const auto& z10 = pickLatent(cornerIdx_[1]);
    const auto& z01 = pickLatent(cornerIdx_[2]);
    const auto& z11 = pickLatent(cornerIdx_[3]);

    if ((int)z00.size() != decoder_->latentDim()) return false;

    for (int gy = 0; gy < panelN_; ++gy) {
        const float v = (panelN_ == 1) ? 0.0f : float(gy) / float(panelN_ - 1);
        for (int gx = 0; gx < panelN_; ++gx) {
            const float u = (panelN_ == 1) ? 0.0f : float(gx) / float(panelN_ - 1);
            const float a = (1.0f - u) * (1.0f - v);
            const float b = u * (1.0f - v);
            const float c = (1.0f - u) * v;
            const float d = u * v;

            for (int i = 0; i < decoder_->latentDim(); ++i) {
                scratchLatentHost_[i] = a * z00[i] + b * z10[i] + c * z01[i] + d * z11[i];
            }

            const int offsetX = gx * tileRes_ + std::max(0, gx) * tileGap_;
            const int offsetY = gy * tileRes_ + std::max(0, gy) * tileGap_;

            decoder_->decodeLatentGridToDevice(scratchLatentHost_.data(),
                                               tileRes_, tileRes_,
                                               domain_->xMin, domain_->xMax,
                                               domain_->yMin, domain_->yMax,
                                               dPanelField_,
                                               panelW_,
                                               offsetX,
                                               offsetY,
                                               dScratchLatent_);
        }
    }

    decoder_->panelToRGBA(dPanelField_, panelW_, panelH_,
                          tileRes_, tileGap_, panelN_,
                          cfg, dTileMin_, dTileMax_, dPanelRGBA_);

    checkCuda("cudaGraphicsMapResources", cudaGraphicsMapResources(1, &panelResource_));
    cudaArray_t array = nullptr;
    checkCuda("cudaGraphicsSubResourceGetMappedArray",
              cudaGraphicsSubResourceGetMappedArray(&array, panelResource_, 0, 0));

    checkCuda("cudaMemcpy2DToArray",
              cudaMemcpy2DToArray(array, 0, 0,
                                  dPanelRGBA_, panelW_ * sizeof(uchar4),
                                  panelW_ * sizeof(uchar4), panelH_,
                                  cudaMemcpyDeviceToDevice));

    checkCuda("cudaGraphicsUnmapResources", cudaGraphicsUnmapResources(1, &panelResource_));

    lastPanelCfg_ = cfg;
    panelDirty_ = false;
    return true;
}

bool LatentNavigator_CUDA::drawPanel(alice2::Renderer& renderer,
                                     float left, float top, float size,
                                     const FieldRenderConfig& cfg)
{
    if (!decoder_ || !domain_ || !latents_ || latents_->empty()) return false;
    if (!ensurePanelResources()) return false;
    if (panelDirty_ || cfg.debugMode != lastPanelCfg_.debugMode ||
        cfg.softMask != lastPanelCfg_.softMask ||
        std::fabs(cfg.tau - lastPanelCfg_.tau) > 1e-6f) {
        if (!rebuildPanel(cfg)) return false;
    }

    int vx, vy, vw, vh;
    renderer.getViewport(vx, vy, vw, vh);

    const GLboolean depthTestWasEnabled = glIsEnabled(GL_DEPTH_TEST);
    GLboolean depthMaskWasEnabled = GL_TRUE;
    glGetBooleanv(GL_DEPTH_WRITEMASK, &depthMaskWasEnabled);

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, vw, vh, 0, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, panelTexture_);
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

    glDepthMask(depthMaskWasEnabled);
    if (depthTestWasEnabled) {
        glEnable(GL_DEPTH_TEST);
    } else {
        glDisable(GL_DEPTH_TEST);
    }

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    return true;
}

bool LatentNavigator_CUDA::decodeLatent(const std::vector<float>& latent, GridField& out)
{
    if (!decoder_ || !domain_) return false;
    if ((int)latent.size() != decoder_->latentDim()) return false;
    if (!dScratchLatent_) {
        checkCuda("cudaMalloc(scratchLatent)", cudaMalloc(&dScratchLatent_, size_t(decoder_->latentDim()) * sizeof(float)));
    }
    if (!dFieldScratch_ || fieldScratchRes_ != domain_->resX * domain_->resY) {
        if (dFieldScratch_) checkCuda("cudaFree(fieldScratch)", cudaFree(dFieldScratch_));
        fieldScratchRes_ = domain_->resX * domain_->resY;
        checkCuda("cudaMalloc(fieldScratch)", cudaMalloc(&dFieldScratch_, size_t(fieldScratchRes_) * sizeof(float)));
    }

    decoder_->decodeLatentGridToDevice(latent.data(),
                                       domain_->resX,
                                       domain_->resY,
                                       domain_->xMin, domain_->xMax,
                                       domain_->yMin, domain_->yMax,
                                       dFieldScratch_,
                                       domain_->resX,
                                       0, 0,
                                       dScratchLatent_);

    std::vector<float> host(fieldScratchRes_);
    checkCuda("cudaMemcpy(fieldScratch)", cudaMemcpy(host.data(), dFieldScratch_,
                                                    size_t(fieldScratchRes_) * sizeof(float),
                                                    cudaMemcpyDeviceToHost));

    out.values = host;
    out.minValue = *std::min_element(host.begin(), host.end());
    out.maxValue = *std::max_element(host.begin(), host.end());
    return true;
}

bool LatentNavigator_CUDA::getFieldAt(const std::vector<float>& latent, GridField& out)
{
    return decodeLatent(latent, out);
}

bool LatentNavigator_CUDA::getBlendedField(float u, float v, GridField& out)
{
    if (!latents_ || latents_->empty()) return false;
    const auto& allLatents = *latents_;
    auto pickLatent = [&](int idx) -> const std::vector<float>& {
        if (idx >= 0 && idx < (int)allLatents.size()) return allLatents[idx];
        return allLatents.front();
    };

    const auto& z00 = pickLatent(cornerIdx_[0]);
    const auto& z10 = pickLatent(cornerIdx_[1]);
    const auto& z01 = pickLatent(cornerIdx_[2]);
    const auto& z11 = pickLatent(cornerIdx_[3]);

    if ((int)z00.size() != decoder_->latentDim()) return false;

    scratchLatentHost_.resize(decoder_->latentDim());

    const float a = (1.0f - u) * (1.0f - v);
    const float b = u * (1.0f - v);
    const float c = (1.0f - u) * v;
    const float d = u * v;

    for (int i = 0; i < decoder_->latentDim(); ++i) {
        scratchLatentHost_[i] = a * z00[i] + b * z10[i] + c * z01[i] + d * z11[i];
    }
    return decodeLatent(scratchLatentHost_, out);
}

void LatentNavigator_CUDA::drawField(alice2::Renderer& renderer,
                                     const GridField& field,
                                     float left, float top, float size,
                                     const FieldRenderConfig& cfg) const
{
    if (!domain_) return;
    const int resX = domain_->resX;
    const int resY = domain_->resY;
    if (resX <= 0 || resY <= 0) return;
    if (field.values.size() < size_t(resX) * size_t(resY)) return;

    const float cellW = size / float(resX);
    const float cellH = size / float(resY);

    for (int y = 0; y < resY; ++y) {
        for (int x = 0; x < resX; ++x) {
            const size_t idx = size_t(y) * size_t(resX) + size_t(x);
            const float sdf = field.values[idx];
            float g = 0.5f;
            if (cfg.debugMode == 0) {
                if (cfg.softMask) {
                    const float tau = std::max(cfg.tau, 1e-6f);
                    g = 1.0f / (1.0f + std::exp(-(sdf / tau)));
                } else {
                    g = sdf < 0.0f ? 0.0f : 1.0f;
                }
            } else {
                const float range = (field.maxValue - field.minValue == 0.0f) ? 1.0f : (field.maxValue - field.minValue);
                g = std::clamp((sdf - field.minValue) / range, 0.0f, 1.0f);
            }
            const alice2::Color color(g, g, g);
            const float px = left + (float(x) + 0.5f) * cellW;
            const float py = top  + (float(y) + 0.5f) * cellH;
            const float ps = std::max(cellW, cellH);
            renderer.draw2dPoint(alice2::Vec2(px, py), color, ps);
        }
    }
}

} // namespace DeepSDF

#endif // ALICE2_USE_CUDA
