#pragma once
// AutoDecoderTrainerCUDA.h
// GPU trainer for Auto-Decoder with strict per-sample SGD schedule.
// Parallel persistent-kernel implementation (one kernel per epoch).

#include <vector>
#include <cstdint>
#include <string>
#include <stdexcept>

// ---------------- Public data structures ----------------
struct ADTSample {
    int32_t shapeIndex;
    std::vector<float> coord; // length = coordDim
    float target;
};

struct ADTConfig {
    int   epochs  = 1;
    float lrW     = 1e-3f;
    float lrZ     = 1e-2f;
    float lambda  = 1e-3f;
    uint64_t shuffleSeed = 1234ull;
};

struct ADTStats {
    int   epochsRun = 0;
    float avgLoss   = 0.f;
    float lastLoss  = 0.f;
};

// Minimal MLP descriptor used by GPU trainer.
// All weights are row-major per layer: W[out][in].
struct SimpleMLP {
    int inputDim = 0;
    int outputDim = 1;
    std::vector<int> hidden; // e.g. {64,64,64}

    // weights[l]: out x in, row-major; biases[l]: out
    std::vector<std::vector<float>> weights;
    std::vector<std::vector<float>> biases;
};

// ---------------- Trainer ----------------
class AutoDecoderTrainerCUDA {
public:
    AutoDecoderTrainerCUDA();
    ~AutoDecoderTrainerCUDA();

    // Setup
    void setNetwork(const SimpleMLP& mlp, int numShapes, int latentDim, int coordDim);
    void setSamples(const std::vector<ADTSample>& samples);

    // Train (strict per-sample SGD)
    ADTStats train(const ADTConfig& cfg);

    // Access
    const SimpleMLP&       getTrainedMLP() const { return hostMLP_; }
    const std::vector<float>& getLatents() const { return h_Z_; } // flattened [numShapes * latentDim]

private:
    // host-side
    SimpleMLP hostMLP_;
    int numShapes_ = 0;
    int latentDim_ = 0;
    int coordDim_  = 0;

    std::vector<ADTSample> dataset_;
    std::vector<int32_t>   order_;

    // layer dims & offsets (host)
    int numLayers_ = 0; // hidden.size() + 1
    std::vector<int> h_layerIn_, h_layerOut_;
    std::vector<int> h_wOffsets_, h_bOffsets_;

    // device pointers
    float* d_W_ = nullptr;     // flattened weights
    float* d_B_ = nullptr;     // flattened biases
    int*   d_layerIn_  = nullptr;
    int*   d_layerOut_ = nullptr;
    int*   d_wOffsets_ = nullptr;
    int*   d_bOffsets_ = nullptr;

    float*   d_Z_ = nullptr;         // [numShapes, latentDim]
    float*   d_coords_ = nullptr;    // [N, coordDim]
    float*   d_targets_ = nullptr;   // [N]
    int32_t* d_shapeIdx_ = nullptr;  // [N]
    int32_t* d_order_ = nullptr;     // [N]

    std::vector<float> h_Z_;

    // device allocs
    void allocDevice_();
    void freeDevice_();
    void uploadModel_();
    void uploadData_();
    void downloadModel_();
};
