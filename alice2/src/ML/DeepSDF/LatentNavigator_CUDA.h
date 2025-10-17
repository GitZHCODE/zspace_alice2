#pragma once

#define ALICE2_USE_CUDA
#ifdef ALICE2_USE_CUDA

#include <vector>
#include <array>
#include <memory>
#include <optional>

#include <alice2.h>
#include <cuda_gl_interop.h>

#include "LatentSDF_CUDA.h"
#include "FieldViewer.h"

namespace DeepSDF {

class LatentNavigator_CUDA {
public:
    LatentNavigator_CUDA();
    ~LatentNavigator_CUDA();

    void initialize(TinyAutoDecoderCUDA* decoder,
                    const FieldDomain* domain,
                    const std::vector<std::vector<float>>* latentSource);
    void shutdown();

    void setPanelResolution(int N, int tileRes, int gap);
    void setCornerIndices(const std::array<int,4>& indices);
    void setLatentSource(const std::vector<std::vector<float>>* latentSource);

    void setDetailResolution(int res);

    void markPanelDirty();

    bool drawPanel(alice2::Renderer& renderer,
                   float left, float top, float size,
                   const FieldRenderConfig& cfg);

    bool getFieldAt(const std::vector<float>& latent, GridField& out);
    bool getBlendedField(float u, float v, GridField& out);

    void drawField(alice2::Renderer& renderer,
                   const GridField& field,
                   float left, float top, float size,
                   const FieldRenderConfig& cfg) const;

    const std::array<int,4>& cornerIndices() const { return cornerIdx_; }

private:
    bool ensurePanelResources();
    bool rebuildPanel(const FieldRenderConfig& cfg);
    bool decodeLatent(const std::vector<float>& latent, GridField& out);

    TinyAutoDecoderCUDA* decoder_ = nullptr;
    const FieldDomain* domain_ = nullptr;
    const std::vector<std::vector<float>>* latents_ = nullptr;

    // Panel configuration
    int panelN_    = 5;
    int tileRes_   = 56;
    int tileGap_   = 4;
    std::array<int,4> cornerIdx_{0,1,2,3};

    // Device resources
    float* dPanelField_   = nullptr;
    uchar4* dPanelRGBA_   = nullptr;
    float* dScratchLatent_= nullptr;
    float* dTileMin_      = nullptr;
    float* dTileMax_      = nullptr;

    GLuint panelTexture_ = 0;
    cudaGraphicsResource* panelResource_ = nullptr;

    int   panelW_ = 0;
    int   panelH_ = 0;
    int   fieldScratchRes_ = 0;

    bool panelDirty_ = true;
    FieldRenderConfig lastPanelCfg_{};

    float* dFieldScratch_ = nullptr;
    std::vector<float> scratchLatentHost_;
};

} // namespace DeepSDF

#endif // ALICE2_USE_CUDA
