#include "MLPComputeContext.h"

#ifdef ALICE2_USE_OPENGL_COMPUTE

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

namespace alice2 {

namespace {
constexpr GLuint kBindingWeights = 0;
constexpr GLuint kBindingBiases = 1;
constexpr GLuint kBindingActivations = 2;
constexpr GLuint kBindingLayerInfo = 3;
constexpr GLuint kBindingDatasetCoords = 4;
constexpr GLuint kBindingDatasetTargets = 5;
constexpr GLuint kBindingDatasetShapes = 6;
constexpr GLuint kBindingDeltas = 7;
constexpr GLuint kBindingGradWeights = 8;
constexpr GLuint kBindingGradBiases = 9;
constexpr GLuint kBindingLatents = 10;
constexpr GLuint kBindingGradLatents = 11;
constexpr GLuint kBindingDeltaInput = 12;

std::string buildPrepareInputShader()
{
    return R"(#version 460 core
layout(local_size_x = 64) in;

struct LayerInfo {
    int weightOffset;
    int biasOffset;
    int actInputOffset;
    int actOutputOffset;
    int deltaOffset;
    int inputSize;
    int outputSize;
    int pad0;
};

layout(std430, binding = 2) buffer Activations { float activations[]; };
layout(std430, binding = 3) readonly buffer Layers { LayerInfo infos[]; };
layout(std430, binding = 4) readonly buffer DatasetCoords { float coords[]; };
layout(std430, binding = 5) readonly buffer DatasetTargets { float targets[]; };
layout(std430, binding = 6) readonly buffer DatasetShapes { int shapes[]; };
layout(std430, binding = 10) readonly buffer Latents { float latents[]; };

uniform int batchStart;
uniform int batchSize;
uniform int latentDim;
uniform int coordDim;
uniform int numSamples;

void main()
{
    uint batchIdx = gl_GlobalInvocationID.x;
    if (batchIdx >= uint(batchSize)) return;
    int sampleIdx = batchStart + int(batchIdx);
    if (sampleIdx >= numSamples) return;

    LayerInfo info0 = infos[0];
    int actOffset = info0.actInputOffset + int(batchIdx) * info0.inputSize;

    int shapeIdx = shapes[sampleIdx];
    int latentOffset = shapeIdx * latentDim;
    for (int i = 0; i < latentDim; ++i)
        activations[actOffset + i] = latents[latentOffset + i];

    int coordOffset = sampleIdx * coordDim;
    for (int i = 0; i < coordDim; ++i)
        activations[actOffset + latentDim + i] = coords[coordOffset + i];
}
)";
}

std::string buildForwardShader()
{
    return R"(#version 460 core
layout(local_size_x = 16, local_size_y = 16) in;

struct LayerInfo {
    int weightOffset;
    int biasOffset;
    int actInputOffset;
    int actOutputOffset;
    int deltaOffset;
    int inputSize;
    int outputSize;
    int pad0;
};

layout(std430, binding = 0) readonly buffer Weights { float weights[]; };
layout(std430, binding = 1) readonly buffer Biases { float biases[]; };
layout(std430, binding = 2) buffer Activations { float activations[]; };
layout(std430, binding = 3) readonly buffer Layers { LayerInfo infos[]; };

uniform int layerIndex;
uniform int batchSize;
uniform int totalLayers;

float activationForward(float x, int idx, int total)
{
    if (idx == total - 1) return x;
    return tanh(x);
}

void main()
{
    uint batchIdx = gl_GlobalInvocationID.x;
    uint neuronIdx = gl_GlobalInvocationID.y;
    if (batchIdx >= uint(batchSize)) return;

    LayerInfo info = infos[layerIndex];
    if (neuronIdx >= uint(info.outputSize)) return;

    int inOffset = info.actInputOffset + int(batchIdx) * info.inputSize;
    int outOffset = info.actOutputOffset + int(batchIdx) * info.outputSize;
    int weightOffset = info.weightOffset + int(neuronIdx) * info.inputSize;
    int biasOffset = info.biasOffset + int(neuronIdx);

    float sum = biases[biasOffset];
    for (int k = 0; k < info.inputSize; ++k)
        sum += weights[weightOffset + k] * activations[inOffset + k];

    activations[outOffset + int(neuronIdx)] = activationForward(sum, layerIndex, totalLayers);
}
)";
}

std::string buildOutputDeltaShader()
{
    return R"(#version 460 core
layout(local_size_x = 64) in;

struct LayerInfo {
    int weightOffset;
    int biasOffset;
    int actInputOffset;
    int actOutputOffset;
    int deltaOffset;
    int inputSize;
    int outputSize;
    int pad0;
};

layout(std430, binding = 2) readonly buffer Activations { float activations[]; };
layout(std430, binding = 3) readonly buffer Layers { LayerInfo infos[]; };
layout(std430, binding = 5) readonly buffer DatasetTargets { float targets[]; };
layout(std430, binding = 7) buffer Deltas { float deltas[]; };

uniform int batchStart;
uniform int batchSize;
uniform int outputLayerIndex;
uniform int numSamples;

void main()
{
    uint batchIdx = gl_GlobalInvocationID.x;
    if (batchIdx >= uint(batchSize)) return;
    int sampleIdx = batchStart + int(batchIdx);
    if (sampleIdx >= numSamples) return;

    LayerInfo info = infos[outputLayerIndex];
    int actOffset = info.actOutputOffset + int(batchIdx) * info.outputSize;
    int deltaOffset = info.deltaOffset + int(batchIdx) * info.outputSize;

    float prediction = activations[actOffset];
    float target = targets[sampleIdx];
    float diff = prediction - target;
    deltas[deltaOffset] = 2.0 * diff;
}
)";
}

std::string buildBackwardGradShader()
{
    return R"(#version 460 core
layout(local_size_x = 16, local_size_y = 16) in;

struct LayerInfo {
    int weightOffset;
    int biasOffset;
    int actInputOffset;
    int actOutputOffset;
    int deltaOffset;
    int inputSize;
    int outputSize;
    int pad0;
};

layout(std430, binding = 2) readonly buffer Activations { float activations[]; };
layout(std430, binding = 3) readonly buffer Layers { LayerInfo infos[]; };
layout(std430, binding = 7) readonly buffer Deltas { float deltas[]; };
layout(std430, binding = 8) buffer GradWeights { float gradWeights[]; };
layout(std430, binding = 9) buffer GradBiases { float gradBiases[]; };

uniform int layerIndex;
uniform int batchSize;

void main()
{
    uint outIdx = gl_GlobalInvocationID.x;
    uint inIdx = gl_GlobalInvocationID.y;

    LayerInfo info = infos[layerIndex];
    if (outIdx >= uint(info.outputSize) || inIdx >= uint(info.inputSize)) return;

    int gradWeightOffset = info.weightOffset + int(outIdx) * info.inputSize + int(inIdx);
    int gradBiasOffset = info.biasOffset + int(outIdx);

    float accumW = 0.0;
    float accumB = 0.0;
    for (int b = 0; b < batchSize; ++b)
    {
        int deltaOffset = info.deltaOffset + b * info.outputSize + int(outIdx);
        int actOffset = info.actInputOffset + b * info.inputSize + int(inIdx);
        float d = deltas[deltaOffset];
        accumW += d * activations[actOffset];
        if (inIdx == 0)
            accumB += d;
    }

    gradWeights[gradWeightOffset] += accumW;
    if (inIdx == 0)
        gradBiases[gradBiasOffset] += accumB;
}
)";
}

std::string buildBackwardDeltaShader()
{
    return R"(#version 460 core
layout(local_size_x = 16, local_size_y = 16) in;

struct LayerInfo {
    int weightOffset;
    int biasOffset;
    int actInputOffset;
    int actOutputOffset;
    int deltaOffset;
    int inputSize;
    int outputSize;
    int pad0;
};

layout(std430, binding = 0) readonly buffer Weights { float weights[]; };
layout(std430, binding = 2) readonly buffer Activations { float activations[]; };
layout(std430, binding = 3) readonly buffer Layers { LayerInfo infos[]; };
layout(std430, binding = 7) buffer Deltas { float deltas[]; };

uniform int layerIndex;
uniform int batchSize;

float activationDerivative(float x)
{
    float t = tanh(x);
    return 1.0 - t * t;
}

void main()
{
    uint batchIdx = gl_GlobalInvocationID.x;
    uint neuronIdx = gl_GlobalInvocationID.y;
    if (batchIdx >= uint(batchSize)) return;
    if (layerIndex == 0) return;

    LayerInfo currentInfo = infos[layerIndex];
    LayerInfo prevInfo = infos[layerIndex - 1];
    if (neuronIdx >= uint(prevInfo.outputSize)) return;

    float sum = 0.0;
    for (int j = 0; j < currentInfo.outputSize; ++j)
    {
        int weightOffset = currentInfo.weightOffset + j * currentInfo.inputSize + int(neuronIdx);
        int deltaOffset = currentInfo.deltaOffset + int(batchIdx) * currentInfo.outputSize + j;
        sum += weights[weightOffset] * deltas[deltaOffset];
    }

    int actPrevOffset = prevInfo.actOutputOffset + int(batchIdx) * prevInfo.outputSize + int(neuronIdx);
    int deltaPrevOffset = prevInfo.deltaOffset + int(batchIdx) * prevInfo.outputSize + int(neuronIdx);
    float deriv = activationDerivative(activations[actPrevOffset]);
    deltas[deltaPrevOffset] = sum * deriv;
}
)";
}

std::string buildBackwardInputShader()
{
    return R"(#version 460 core
layout(local_size_x = 16, local_size_y = 16) in;

struct LayerInfo {
    int weightOffset;
    int biasOffset;
    int actInputOffset;
    int actOutputOffset;
    int deltaOffset;
    int inputSize;
    int outputSize;
    int pad0;
};

layout(std430, binding = 0) readonly buffer Weights { float weights[]; };
layout(std430, binding = 3) readonly buffer Layers { LayerInfo infos[]; };
layout(std430, binding = 7) readonly buffer Deltas { float deltas[]; };
layout(std430, binding = 12) buffer DeltaInput { float deltaInput[]; };

uniform int batchSize;
uniform int inputDim;

void main()
{
    uint batchIdx = gl_GlobalInvocationID.x;
    uint neuronIdx = gl_GlobalInvocationID.y;
    if (batchIdx >= uint(batchSize) || neuronIdx >= uint(inputDim)) return;

    LayerInfo firstInfo = infos[0];
    LayerInfo secondInfo = infos[1];

    float accum = 0.0;
    for (int j = 0; j < secondInfo.outputSize; ++j)
    {
        int weightOffset = secondInfo.weightOffset + j * secondInfo.inputSize + int(neuronIdx);
        int deltaOffset = secondInfo.deltaOffset + int(batchIdx) * secondInfo.outputSize + j;
        accum += weights[weightOffset] * deltas[deltaOffset];
    }

    int dest = int(batchIdx) * inputDim + int(neuronIdx);
    deltaInput[dest] = accum;
}
)";
}

std::string buildLatentGradShader()
{
    return R"(#version 460 core
layout(local_size_x = 64) in;

layout(std430, binding = 6) readonly buffer DatasetShapes { int shapes[]; };
layout(std430, binding = 10) readonly buffer Latents { float latents[]; };
layout(std430, binding = 11) buffer GradLatents { float gradLatents[]; };
layout(std430, binding = 12) readonly buffer DeltaInput { float deltaInput[]; };

uniform int batchStart;
uniform int batchSize;
uniform int numSamples;
uniform int latentDim;
uniform float latentReg;

void main()
{
    uint batchIdx = gl_GlobalInvocationID.x;
    if (batchIdx >= uint(batchSize)) return;
    int sampleIdx = batchStart + int(batchIdx);
    if (sampleIdx >= numSamples) return;

    int shapeIdx = shapes[sampleIdx];
    int latentOffset = shapeIdx * latentDim;
    int deltaOffset = int(batchIdx) * latentDim;

    for (int i = 0; i < latentDim; ++i)
    {
        float gradVal = deltaInput[deltaOffset + i] + 2.0 * latentReg * latents[latentOffset + i];
        atomicAdd(gradLatents[latentOffset + i], gradVal);
    }
}
)";
}

std::string buildApplyUpdateShader()
{
    return R"(#version 460 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer Weights { float weights[]; };
layout(std430, binding = 1) buffer Biases { float biases[]; };
layout(std430, binding = 8) buffer GradWeights { float gradWeights[]; };
layout(std430, binding = 9) buffer GradBiases { float gradBiases[]; };
layout(std430, binding = 10) buffer Latents { float latents[]; };
layout(std430, binding = 11) buffer GradLatents { float gradLatents[]; };

uniform float weightScale;
uniform float latentScale;
uniform int totalWeights;
uniform int totalBiases;
uniform int totalLatents;

void main()
{
    uint idx = gl_GlobalInvocationID.x;

    if (idx < uint(totalWeights))
    {
        weights[idx] -= weightScale * gradWeights[idx];
        gradWeights[idx] = 0.0;
    }
    else if (idx < uint(totalWeights + totalBiases))
    {
        uint biasIdx = idx - uint(totalWeights);
        biases[biasIdx] -= weightScale * gradBiases[biasIdx];
        gradBiases[biasIdx] = 0.0;
    }
    else if (idx < uint(totalWeights + totalBiases + totalLatents))
    {
        uint latentIdx = idx - uint(totalWeights + totalBiases);
        latents[latentIdx] -= latentScale * gradLatents[latentIdx];
        gradLatents[latentIdx] = 0.0;
    }
}
)";
}

} // namespace

MLPComputeContext::MLPComputeContext() = default;
MLPComputeContext::~MLPComputeContext() { release(); }

bool MLPComputeContext::ensurePrograms()
{
    if (programsReady) return true;
    auto make = [&](const std::string& src, const std::string& label) -> GLuint {
        GLuint prog = glCreateProgram();
        GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
        const char* csrc = src.c_str();
        glShaderSource(shader, 1, &csrc, nullptr);
        glCompileShader(shader);
        GLint ok = GL_FALSE;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (ok != GL_TRUE)
        {
            GLint logLen = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLen);
            std::string log(logLen > 1 ? logLen : 1, '\0');
            glGetShaderInfoLog(shader, logLen, nullptr, log.data());
            std::cerr << "[MLPComputeContext] compile failed for " << label << "\n" << log << std::endl;
            glDeleteShader(shader);
            glDeleteProgram(prog);
            return 0;
        }
        glAttachShader(prog, shader);
        glLinkProgram(prog);
        glDeleteShader(shader);
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if (ok != GL_TRUE)
        {
            GLint logLen = 0;
            glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &logLen);
            std::string log(logLen > 1 ? logLen : 1, '\0');
            glGetProgramInfoLog(prog, logLen, nullptr, log.data());
            std::cerr << "[MLPComputeContext] link failed for " << label << "\n" << log << std::endl;
            glDeleteProgram(prog);
            return 0;
        }
        return prog;
    };

    programPrepareInput = make(buildPrepareInputShader(), "prepare_input"); if (!programPrepareInput) return false;
    programForward = make(buildForwardShader(), "forward"); if (!programForward) return false;
    programOutputDelta = make(buildOutputDeltaShader(), "output_delta"); if (!programOutputDelta) return false;
    programBackwardGrad = make(buildBackwardGradShader(), "backward_grad"); if (!programBackwardGrad) return false;
    programBackwardDelta = make(buildBackwardDeltaShader(), "backward_delta"); if (!programBackwardDelta) return false;
    programBackwardInput = make(buildBackwardInputShader(), "backward_input"); if (!programBackwardInput) return false;
    programLatentGrad = make(buildLatentGradShader(), "latent_grad"); if (!programLatentGrad) return false;
    programApplyUpdate = make(buildApplyUpdateShader(), "apply_update"); if (!programApplyUpdate) return false;

    programsReady = true;
    return true;
}


bool MLPComputeContext::dispatchBackwardInput(int batchSize)
{
    glUseProgram(programBackwardInput);
    glUniform1i(glGetUniformLocation(programBackwardInput, "batchSize"), batchSize);
    glUniform1i(glGetUniformLocation(programBackwardInput, "inputDim"), spec.layerDims.front());

    int inputSize = spec.layerDims.front();
    GLuint groupsX = (batchSize + 15) / 16;
    GLuint groupsY = (inputSize + 15) / 16;
    glDispatchCompute(groupsX, groupsY, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    return true;
}

bool MLPComputeContext::dispatchLatentGrad(int batchStart, int batchSize, float latentReg)
{
    glUseProgram(programLatentGrad);
    glUniform1i(glGetUniformLocation(programLatentGrad, "batchStart"), batchStart);
    glUniform1i(glGetUniformLocation(programLatentGrad, "batchSize"), batchSize);
    glUniform1i(glGetUniformLocation(programLatentGrad, "numSamples"), spec.numSamples);
    glUniform1i(glGetUniformLocation(programLatentGrad, "latentDim"), spec.latentDim);
    glUniform1f(glGetUniformLocation(programLatentGrad, "latentReg"), latentReg);

    GLuint groups = (batchSize + 63) / 64;
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    return true;
}


bool MLPComputeContext::initialise(const MLPComputeSpec& specIn)
{
    spec = specIn;
    if (spec.layerDims.size() < 2) return false;
    layerCount = static_cast<int>(spec.layerDims.size()) - 1;
    inputDim = spec.layerDims.front();
    outputDim = spec.layerDims.back();

    maxLayerWidth = 0;
    for (int dim : spec.layerDims)
        maxLayerWidth = std::max(maxLayerWidth, dim);

    std::vector<int> actOffsets(spec.layerDims.size());
    int running = 0;
    for (size_t i = 0; i < spec.layerDims.size(); ++i)
    {
        actOffsets[i] = running;
        running += spec.maxBatchSize * spec.layerDims[i];
    }
    activationFloatCount = running;

    std::vector<int> deltaOffsets(layerCount);
    running = 0;
    for (int i = 0; i < layerCount; ++i)
    {
        deltaOffsets[i] = running;
        running += spec.maxBatchSize * spec.layerDims[i + 1];
    }
    deltaFloatCount = running;

    weightsFloatCount = 0;
    biasesFloatCount = 0;
    layerInfos.resize(layerCount);
    for (int i = 0; i < layerCount; ++i)
    {
        int inDim = spec.layerDims[i];
        int outDim = spec.layerDims[i + 1];
        LayerInfo info{};
        info.weightOffset = weightsFloatCount;
        info.biasOffset = biasesFloatCount;
        info.actInputOffset = actOffsets[i];
        info.actOutputOffset = actOffsets[i + 1];
        info.deltaOffset = deltaOffsets[i];
        info.inputSize = inDim;
        info.outputSize = outDim;
        info.pad0 = 0;
        layerInfos[i] = info;
        weightsFloatCount += inDim * outDim;
        biasesFloatCount += outDim;
    }

    datasetSize = spec.numSamples;

    if (!ensurePrograms()) return false;

    destroyBuffers();
    auto createSSBO = [](GLuint& buf, GLsizeiptr bytes)
    {
        glGenBuffers(1, &buf);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
        glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, nullptr, GL_DYNAMIC_DRAW);
        return buf != 0;
    };

    if (!createSSBO(ssboWeights, weightsFloatCount * sizeof(float))) return false;
    if (!createSSBO(ssboBiases, biasesFloatCount * sizeof(float))) return false;
    if (!createSSBO(ssboActivations, activationFloatCount * sizeof(float))) return false;
    if (!createSSBO(ssboDeltas, deltaFloatCount * sizeof(float))) return false;
    if (!createSSBO(ssboGradWeights, weightsFloatCount * sizeof(float))) return false;
    if (!createSSBO(ssboGradBiases, biasesFloatCount * sizeof(float))) return false;
    if (!createSSBO(ssboLatents, spec.numShapes * spec.latentDim * sizeof(float))) return false;
    if (!createSSBO(ssboGradLatents, spec.numShapes * spec.latentDim * sizeof(float))) return false;
    if (!createSSBO(ssboDatasetCoords, spec.numSamples * spec.coordDim * sizeof(float))) return false;
    if (!createSSBO(ssboDatasetTargets, spec.numSamples * sizeof(float))) return false;
    if (!createSSBO(ssboDatasetShapes, spec.numSamples * sizeof(int))) return false;
    if (!createSSBO(ssboDeltaInput, spec.maxBatchSize * inputDim * sizeof(float))) return false;
    if (!createSSBO(ssboLayerInfo, layerInfos.size() * sizeof(LayerInfo))) return false;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboLayerInfo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, layerInfos.size() * sizeof(LayerInfo), layerInfos.data());

    auto bindBase = [](GLuint buf, GLuint binding)
    {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buf);
    };

    bindBase(ssboWeights, kBindingWeights);
    bindBase(ssboBiases, kBindingBiases);
    bindBase(ssboActivations, kBindingActivations);
    bindBase(ssboLayerInfo, kBindingLayerInfo);
    bindBase(ssboDatasetCoords, kBindingDatasetCoords);
    bindBase(ssboDatasetTargets, kBindingDatasetTargets);
    bindBase(ssboDatasetShapes, kBindingDatasetShapes);
    bindBase(ssboDeltas, kBindingDeltas);
    bindBase(ssboGradWeights, kBindingGradWeights);
    bindBase(ssboGradBiases, kBindingGradBiases);
    bindBase(ssboLatents, kBindingLatents);
    bindBase(ssboGradLatents, kBindingGradLatents);
    bindBase(ssboDeltaInput, kBindingDeltaInput);

    buffersReady = true;
    return true;
}

bool MLPComputeContext::uploadWeights(const std::vector<float>& weights, const std::vector<float>& biases)
{
    if (!buffersReady) return false;
    if (weights.size() != static_cast<size_t>(weightsFloatCount) || biases.size() != static_cast<size_t>(biasesFloatCount))
        return false;

    weightsHost = weights;
    biasesHost = biases;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboWeights);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, weights.size() * sizeof(float), weights.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboBiases);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, biases.size() * sizeof(float), biases.data());
    return true;
}

bool MLPComputeContext::uploadLatents(const std::vector<float>& latents)
{
    if (!buffersReady) return false;
    if (latents.size() != static_cast<size_t>(spec.numShapes * spec.latentDim))
        return false;
    latentsHost = latents;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboLatents);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, latents.size() * sizeof(float), latents.data());
    return true;
}

bool MLPComputeContext::uploadDataset(const std::vector<float>& coords,
                                      const std::vector<float>& targets,
                                      const std::vector<int>& shapes)
{
    if (!buffersReady) return false;
    if (coords.size() != static_cast<size_t>(spec.numSamples * spec.coordDim)) return false;
    if (targets.size() != static_cast<size_t>(spec.numSamples)) return false;
    if (shapes.size() != static_cast<size_t>(spec.numSamples)) return false;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboDatasetCoords);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, coords.size() * sizeof(float), coords.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboDatasetTargets);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, targets.size() * sizeof(float), targets.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboDatasetShapes);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, shapes.size() * sizeof(int), shapes.data());
    return true;
}

bool MLPComputeContext::train(const MLPComputeConfig& cfg)
{
    if (!programsReady || !buffersReady) return false;
    if (cfg.batchSize <= 0 || cfg.batchSize > spec.maxBatchSize) return false;

    for (int epoch = 0; epoch < cfg.epochs; ++epoch)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboGradWeights);
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboGradBiases);
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboGradLatents);
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);

        for (int batchStart = 0; batchStart < spec.numSamples; batchStart += cfg.batchSize)
        {
            int currentBatch = std::min(cfg.batchSize, spec.numSamples - batchStart);

            dispatchPrepareInput(batchStart, currentBatch);

            for (int layer = 0; layer < layerCount; ++layer)
                dispatchForward(layer, currentBatch);

            dispatchOutputDelta(batchStart, currentBatch);

            for (int layer = layerCount - 1; layer >= 0; --layer)
            {
                dispatchBackwardGradients(layer, currentBatch);
                if (layer > 0)
                    dispatchBackwardDelta(layer, currentBatch);
                else
                    dispatchBackwardInput(currentBatch);
            }

            dispatchLatentGrad(batchStart, currentBatch, cfg.latentRegularization);
        }

        dispatchApplyUpdate(spec.numSamples, cfg);
    }

    return true;
}

bool MLPComputeContext::downloadWeights(std::vector<float>& weightsOut, std::vector<float>& biasesOut) const
{
    if (!buffersReady) return false;
    weightsOut.resize(static_cast<size_t>(weightsFloatCount));
    biasesOut.resize(static_cast<size_t>(biasesFloatCount));
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboWeights);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, weightsOut.size() * sizeof(float), weightsOut.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboBiases);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, biasesOut.size() * sizeof(float), biasesOut.data());
    return true;
}

bool MLPComputeContext::downloadLatents(std::vector<float>& latentsOut) const
{
    if (!buffersReady) return false;
    latentsOut.resize(static_cast<size_t>(spec.numShapes) * static_cast<size_t>(spec.latentDim));
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboLatents);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, latentsOut.size() * sizeof(float), latentsOut.data());
    return true;
}

void MLPComputeContext::release()
{
    destroyBuffers();
    destroyPrograms();
}

} // namespace alice2

#endif // ALICE2_USE_OPENGL_COMPUTE
