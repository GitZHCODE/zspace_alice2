#pragma once

#ifdef ALICE2_USE_OPENGL_COMPUTE

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <GL/glew.h>

namespace alice2 {

struct MLPComputeSpec {
    std::vector<int> layerDims; // includes input (index 0) and output (last)
    int latentDim = 0;
    int coordDim = 0;
    int numShapes = 0;
    int numSamples = 0;
    int maxBatchSize = 0;
};

struct MLPComputeConfig {
    int epochs = 1;
    int batchSize = 1;
    float learningRateWeights = 1e-3f;
    float learningRateLatent = 1e-3f;
    float latentRegularization = 1e-4f;
};

class MLPComputeContext {
public:
    MLPComputeContext();
    ~MLPComputeContext();

    bool initialise(const MLPComputeSpec& spec);
    bool uploadWeights(const std::vector<float>& weights, const std::vector<float>& biases);
    bool uploadLatents(const std::vector<float>& latents);
    bool uploadDataset(const std::vector<float>& coords,
                       const std::vector<float>& targets,
                       const std::vector<int>& shapes);

    bool train(const MLPComputeConfig& cfg);

    bool downloadWeights(std::vector<float>& weightsOut, std::vector<float>& biasesOut) const;
    bool downloadLatents(std::vector<float>& latentsOut) const;

    void release();

    int getLayerCount() const { return layerCount; }
    int getInputDim() const { return inputDim; }
    int getOutputDim() const { return outputDim; }

private:
    struct LayerInfo {
        int weightOffset;  // float offset into weights buffer
        int biasOffset;    // float offset into biases buffer
        int actInputOffset; // float offset into activations buffer for input layer
        int actOutputOffset; // float offset into activations buffer for output layer
        int deltaOffset;   // float offset into delta buffer for this layer's outputs
        int inputSize;
        int outputSize;
        int pad0;
    };

    bool ensurePrograms();
    bool allocateBuffers();
    void destroyPrograms();
    void destroyBuffers();

    bool dispatchPrepareInput(int batchStart, int batchSize);
    bool dispatchForward(int layerIndex, int batchSize);
    bool dispatchOutputDelta(int batchStart, int batchSize);
    bool dispatchBackwardGradients(int layerIndex, int batchSize);
    bool dispatchBackwardDelta(int layerIndex, int batchSize);
    bool dispatchBackwardInput(int batchSize);
    bool dispatchLatentGrad(int batchStart, int batchSize, float latentReg);
    bool dispatchApplyUpdate(int totalSamples, const MLPComputeConfig& cfg);

    bool zeroBuffer(GLuint buffer, GLsizeiptr sizeBytes);

    bool checkContextAvailable() const;

    GLuint createProgram(const char* src, const std::string& label);

private:
    MLPComputeSpec spec;

    int layerCount = 0;
    int inputDim = 0;
    int outputDim = 0;
    int maxLayerWidth = 0;

    std::vector<LayerInfo> layerInfos;

    GLsizeiptr weightsFloatCount = 0;
    GLsizeiptr biasesFloatCount = 0;
    GLsizeiptr activationFloatCount = 0;
    GLsizeiptr deltaFloatCount = 0;

    GLuint ssboWeights = 0;
    GLuint ssboBiases = 0;
    GLuint ssboActivations = 0;
    GLuint ssboDeltas = 0;
    GLuint ssboLatents = 0;
    GLuint ssboDatasetCoords = 0;
    GLuint ssboDatasetTargets = 0;
    GLuint ssboDatasetShapes = 0;
    GLuint ssboGradWeights = 0;
    GLuint ssboGradBiases = 0;
    GLuint ssboGradLatents = 0;
    GLuint ssboDeltaInput = 0;

    GLuint ssboLayerInfo = 0;

    GLuint programPrepareInput = 0;
    GLuint programForward = 0;
    GLuint programOutputDelta = 0;
    GLuint programBackwardGrad = 0;
    GLuint programBackwardDelta = 0;
    GLuint programBackwardInput = 0;
    GLuint programLatentGrad = 0;
    GLuint programApplyUpdate = 0;

    bool programsReady = false;
    bool buffersReady = false;

    std::vector<float> weightsHost;
    std::vector<float> biasesHost;
    std::vector<float> latentsHost;

    int datasetSize = 0;
};

} // namespace alice2

#endif // ALICE2_USE_OPENGL_COMPUTE

