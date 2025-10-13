
#pragma once
// AutoDecoderTrainerCUDA.h
// Persistent-kernel per-sample SGD trainer for an auto-decoder (MLP + latent codes).
// Mirrors CPU schedule: forward -> update W/b -> compute dInput -> update latent Z[s].
// Hidden: tanh; Output: linear. Loss: L = (y - f)^2 + lambda * ||z_s||^2.
//
// This header is self-contained and does not depend on the user's CPU trainer headers.
// If you want it as a drop-in, you can replace the small MLP definition below with your own.

#include <vector>
#include <cstdint>
#include <random>
#include <string>
#include <stdexcept>

struct ADTSample {
    int32_t shapeIndex;
    std::vector<float> coord; // length = coordDim
    float target;
};

struct ADTConfig {
    int epochs = 1;
    float lrW = 1e-3f;
    float lrZ = 1e-2f;
    float lambda = 1e-3f;
    uint64_t shuffleSeed = 1234ull;
    float latentInitStd = 0.01f;
};

struct ADTStats {
    int epochsRun = 0;
    float avgLoss = 0.f;
    float lastLoss = 0.f;
};

// A minimal MLP layout we also use on device (row-major: out x in)
struct SimpleMLP {
    int inputDim = 0;
    int outputDim = 1;
    std::vector<int> hidden; // e.g. {64,64,64}

    // weights[l]: out x in, row-major; biases[l]: out
    std::vector<std::vector<float>> weights;
    std::vector<std::vector<float>> biases;
};

class AutoDecoderTrainerCUDA {
public:
    AutoDecoderTrainerCUDA();
    ~AutoDecoderTrainerCUDA();

    // Model/latent/dataset spec
    void setNetwork(const SimpleMLP& mlp, int numShapes, int latentDim, int coordDim);
    void setSamples(const std::vector<ADTSample>& samples);

    // Train with strict per-sample SGD schedule on GPU
    ADTStats train(const ADTConfig& cfg);

    // Read back trained parameters
    const SimpleMLP& getTrainedMLP() const { return hostMLP_; }
    const std::vector<float>& getLatents() const { return h_Z_; } // numShapes * latentDim

private:
    // host-side storage
    SimpleMLP hostMLP_;
    int numShapes_ = 0;
    int latentDim_ = 0;
    int coordDim_ = 0;

    std::vector<ADTSample> dataset_;
    std::vector<int32_t> order_;

    // device pointers
    void* d_W_ = nullptr;    // flattened weights for all layers
    void* d_B_ = nullptr;    // flattened biases   for all layers
    int*  d_layerIn_  = nullptr;
    int*  d_layerOut_ = nullptr;
    int   numLayers_ = 0; // including output layer

    float* d_Z_ = nullptr;   // [numShapes, latentDim]
    float* d_coords_ = nullptr; // [N, coordDim]
    float* d_targets_ = nullptr; // [N]
    int32_t* d_shapeIdx_ = nullptr; // [N]
    int32_t* d_order_ = nullptr;    // [N]

    std::vector<float> h_Z_; // for reading back

    void allocDevice_();
    void freeDevice_();
    void uploadModel_();
    void uploadData_();
    void downloadModel_();
};
