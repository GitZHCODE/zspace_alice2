#pragma once
#include <vector>
#include <random>
#include <cstdint>
#include <string>

// Use the same types as the CPU trainer to avoid duplication:
#include "ML/AutoDecoderTrainer.h" // brings in alice2::AutoDecoderSample, AutoDecoderTrainingConfig, AutoDecoderTrainingStats
#include "ML/genericMLP.h"         // for MLP

namespace alice2 {

// A minimal GPU trainer that matches the CPU SGD semantics (solverBatch=1).
// It copies weights from the CPU MLP on construction and writes updated weights
// back into the CPU MLP after training. Hidden activations use tanh.
class AutoDecoderTrainerCUDA {
public:
    explicit AutoDecoderTrainerCUDA(MLP& cpuDecoder);
    ~AutoDecoderTrainerCUDA();

    void initialize(int numShapes, int latentDim, int coordinateDim);
    void setSamples(const std::vector<AutoDecoderSample>& samples);

    AutoDecoderTrainingStats train(const AutoDecoderTrainingConfig& cfg);

    // Accessors
    const std::vector<std::vector<float>>& getLatentCodesHost() const { return h_latents_; }
    
    // Set initial latents from host (deterministic parity with CPU), uploads to device.
    void setLatentCodesHost(const std::vector<std::vector<float>>& Z);

private:
    // CPU model reference
    MLP& model_;

    // Problem dims
    int coordDim_ = 0;
    int latentDim_ = 0;
    int inputDim_ = 0;
    int outputDim_ = 0;
    std::vector<int> layerIn_, layerOut_; // per-layer dims

    // Host copies
    std::vector<std::vector<float>> h_latents_;  // [numShapes][latentDim]
    std::vector<AutoDecoderSample> dataset_;
    std::mt19937 rng_;
    bool latentsInit_ = false;

    int epochsRun_ = 0;   // track cumulative epochs across calls

    // Device buffers
    float** d_W_ = nullptr;   // per-layer pointers to row-major [out x in]
    float** d_b_ = nullptr;   // per-layer pointers [out]
    float** d_act_ = nullptr; // per-layer+input activations (A_0..A_L), pointers to [dim]
    float** d_delta_ = nullptr; // per-layer deltas [dim]
    int numLayers_ = 0; // excludes input layer; equals number of weight matrices

    // storage of pointers and raw buffers
    std::vector<float*> dW_raw_, db_raw_, dAct_raw_, dDelta_raw_;

    // other device buffers
    float* d_inputConcat_ = nullptr; // [latentDim+coordDim]

    // Latent table
    float* d_Z_ = nullptr; // [numShapes * latentDim], row-major
    int numShapes_ = 0;

    // Utility
    void ensureLatentInit_(float stddev);
    void syncWeightsFromCPU_();
    void syncWeightsToCPU_();
    void allocDevice_();
    void freeDevice_();

    // One-sample SGD step on device
    void stepSample_(int shapeIndex, const float* h_coord, float target,
                     float lrW, float lrZ, float lambda, float& outLoss);
};

} // namespace alice2
