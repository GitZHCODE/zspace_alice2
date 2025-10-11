#pragma once
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>
#include <fstream>
#include <stdexcept>
#include "alice2.h"
#ifdef ALICE2_USE_OPENGL_COMPUTE
#include "../utils/OpenGL.h"
#endif

using namespace alice2;

//------------------------------------------------------------------ MLP base class

class MLP
{
public:
    int inputDim = 2;
    int outputDim = 1;
    std::vector<int> hiddenDims = { 8, 8 };

    std::vector<std::vector<std::vector<float>>> W;
    std::vector<std::vector<float>> b;
    std::vector<std::vector<float>> activations;

    MLP(){}

    ~MLP()
    {
#ifdef ALICE2_USE_OPENGL_COMPUTE
        if (computeProgram != 0)
        {
            glDeleteProgram(computeProgram);
            computeProgram = 0;
        }
#endif
    }

    MLP(int inDim, std::vector<int> hidden, int outDim)
    {
        initialize(inDim, hidden, outDim);
    }

    void initialize(int inDim, std::vector<int> hidden, int outDim)
    {
        inputDim = inDim;
        hiddenDims = hidden;
        outputDim = outDim;

        std::vector<int> layerDims = { inputDim };
        layerDims.insert(layerDims.end(), hiddenDims.begin(), hiddenDims.end());
        layerDims.push_back(outputDim);

        W.clear(); b.clear();
        for (int l = 0; l < (int)layerDims.size() - 1; ++l)
        {
            int inSize = layerDims[l];
            int outSize = layerDims[l + 1];
            W.push_back(std::vector<std::vector<float>>(outSize, std::vector<float>(inSize)));
            b.push_back(std::vector<float>(outSize));
            for (auto& w_row : W[l])
                for (auto& w : w_row)
                    w = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }

    std::vector<float> forward(std::vector<float>& x)
    {
#ifdef ALICE2_USE_OPENGL_COMPUTE
        if (useComputeShader)
        {
            if (!initializeComputeShader())
            {
                useComputeShader = false;
            }
            else
            {
                std::vector<float> gpuResult;
                if (forwardGPU(x, gpuResult))
                    return gpuResult;
                useComputeShader = false;
            }
        }
#endif
        return forwardCPU(x);
    }

    virtual float computeLoss(std::vector<float>& y_pred, std::vector<float>& y_true)
    {
        float loss = 0.0f;
        for (int i = 0; i < (int)y_pred.size(); ++i)
        {
            float err = y_pred[i] - y_true[i];
            loss += err * err;
        }
        return loss / (float)y_pred.size();
    }

    virtual void computeGradient(std::vector<float>& x, std::vector<float>& y_true, std::vector<float>& gradOut)
    {
        std::vector<float> y_pred = forward(x);
        gradOut.assign(outputDim, 0.0f);
        for (int i = 0; i < outputDim; ++i)
        {
            gradOut[i] = 2.0f * (y_pred[i] - y_true[i]) / (float)outputDim;
        }
    }

    void backward(std::vector<float>& gradOut, float lr)
    {
        std::vector<float> delta = gradOut;

        for (int l = (int)W.size() - 1; l >= 0; --l)
        {
            std::vector<float> prev = activations[l];
            std::vector<float> newDelta(prev.size(), 0.0f);

            for (int i = 0; i < (int)W[l].size(); ++i)
            {
                for (int j = 0; j < (int)W[l][i].size(); ++j)
                {
                    newDelta[j] += delta[i] * W[l][i][j];
                    W[l][i][j] -= lr * delta[i] * prev[j];
                }
                b[l][i] -= lr * delta[i];
            }

            if (l > 0)
            {
                for (int i = 0; i < (int)newDelta.size(); ++i)
                {
                    float a = activations[l][i];
                    newDelta[i] *= (1 - a * a); // tanh derivative
                }
                delta = newDelta;
            }
        }
    }

    std::vector<float> backwardWithInputGrad(const std::vector<float>& gradOut, float lr)
    {
        if (activations.empty())
            throw std::runtime_error("MLP::backwardWithInputGrad requires a prior forward pass.");

        if (gradOut.size() != (size_t)outputDim)
            throw std::runtime_error("MLP::backwardWithInputGrad received gradOut with unexpected size.");

        std::vector<float> delta(gradOut);
        std::vector<float> inputGrad(inputDim, 0.0f);

        for (int l = (int)W.size() - 1; l >= 0; --l)
        {
            const std::vector<float>& prev = activations[l];
            std::vector<float> newDelta(prev.size(), 0.0f);

            for (int i = 0; i < (int)W[l].size(); ++i)
            {
                for (int j = 0; j < (int)W[l][i].size(); ++j)
                {
                    newDelta[j] += delta[i] * W[l][i][j];
                    W[l][i][j] -= lr * delta[i] * prev[j];
                }
                b[l][i] -= lr * delta[i];
            }

            if (l == 0)
            {
                inputGrad = newDelta;
            }

            if (l > 0)
            {
                for (int i = 0; i < (int)newDelta.size(); ++i)
                {
                    float a = activations[l][i];
                    newDelta[i] *= (1 - a * a); // tanh derivative
                }
                delta = newDelta;
            }
        }

        return inputGrad;
    }

    void setUseComputeShader(bool enable)
    {
#ifdef ALICE2_USE_OPENGL_COMPUTE
        useComputeShader = enable;
        if (useComputeShader && !initializeComputeShader())
        {
            useComputeShader = false;
            std::cerr << "MLP: compute shaders unavailable, falling back to CPU." << std::endl;
        }
#else
        if (enable)
            std::cout << "MLP: OpenGL compute shaders not supported. Using CPU path." << std::endl;
#endif
    }

    /**
     * Visualize MLP network structure with nodes and connections
     */
    void visualize(Renderer& renderer, Camera& camera, const Vec3& topLeft = Vec3(400, 200, 0), float bboxWidth = 300.0f, float bboxHeight = 200.0f)
    {
        if (activations.empty()) return; // No data to visualize

        int numLayers = activations.size();
        float nodeRadius = 3.0f;

        // Compute max nodes per layer for vertical spacing
        int maxNodesPerLayer = 0;
        for (const auto& layer : activations) {
            maxNodesPerLayer = std::max(maxNodesPerLayer, (int)layer.size());
        }

        // Ensure reasonable spacing
        float layerSpacing = (numLayers > 1) ? bboxWidth / (numLayers - 1) : 150.0f;
        float verticalSpacing = (maxNodesPerLayer > 1) ? std::min(bboxHeight / (maxNodesPerLayer - 1), 30.0f) : 20.0f;

        std::vector<std::vector<Vec3>> nodePositions(numLayers);

        // Compute node positions with better centering
        for (int l = 0; l < numLayers; l++) {
            int numNodes = activations[l].size();
            if (numNodes == 0) continue;

            float totalHeight = (numNodes - 1) * verticalSpacing;
            float yStart = topLeft.y + (bboxHeight - totalHeight) * 0.5f;

            for (int i = 0; i < numNodes; i++) {
                float x = topLeft.x + l * layerSpacing;
                float y = yStart + i * verticalSpacing;
                nodePositions[l].push_back(Vec3(x, y, topLeft.z));
            }
        }

        for (int l = 0; l < numLayers - 1; l++) {
            if (l >= W.size()) continue;

            int fromSize = activations[l].size();
            int toSize = activations[l + 1].size();

            for (int i = 0; i < fromSize && i < nodePositions[l].size(); i++) {
                for (int j = 0; j < toSize && j < nodePositions[l + 1].size(); j++) {
                    if (j >= W[l].size() || i >= W[l][j].size()) continue;

                    float w = W[l][j][i];
                    float absW = fabs(w);

                    if (absW < 0.05f) continue;

                    float val = std::clamp(w * 3.0f, -1.0f, 1.0f);
                    float r, g, b;
                    get_jet_color(val, r, g, b);

                    Color color(r, g, b);
                    float width = std::clamp(absW * 3.0f, 0.5f, 1.0f);
                    width = 1.0f;

                    Vec2 start = Vec2(nodePositions[l][i].x, nodePositions[l][i].y);
                    Vec2 end = Vec2(nodePositions[l + 1][j].x, nodePositions[l + 1][j].y);
                    renderer.draw2dLine(start, end, color, width);
                }
            }
        }

        for (int l = 0; l < numLayers; l++) {
            for (int i = 0; i < activations[l].size() && i < nodePositions[l].size(); i++) {
                float act = activations[l][i];

                float clampedAct = std::clamp(act, -1.0f, 1.0f);
                float r, g, b;
                get_jet_color(clampedAct, r, g, b);

                Color color(r, g, b);
                Vec2 pos = Vec2(nodePositions[l][i].x, nodePositions[l][i].y);

                float size = 2.0f + 2.0f * fabs(clampedAct);
                renderer.draw2dPoint(pos, color, size);
            }
        }

        renderer.setColor(Color(0.8f, 0.8f, 0.8f));
        for (int l = 0; l < numLayers && !nodePositions[l].empty(); l++) {
            std::string label = (l == 0) ? "Input" : (l == numLayers - 1) ? "Output" : "Hidden";
            float x = nodePositions[l][0].x;
            float y = topLeft.y - 20;
            renderer.drawString(label, x - 15, y);
        }
    }

    inline void get_jet_color(float value, float& r, float& g, float& b) {
        value = clamp(value, -1.0f, 1.0f);
        float normalized = (value + 1.0f) * 0.5f;
        float fourValue = 4.0f * normalized;

        r = clamp(std::min(fourValue - 1.5f, -fourValue + 4.5f), 0.0f, 1.0f);
        g = clamp(std::min(fourValue - 0.5f, -fourValue + 3.5f), 0.0f, 1.0f);
        b = clamp(std::min(fourValue + 0.5f, -fourValue + 2.5f), 0.0f, 1.0f);
    }

private:
    bool useComputeShader = false;
#ifdef ALICE2_USE_OPENGL_COMPUTE
    GLuint computeProgram = 0;
    GLint uniformInputSize = -1;
    GLint uniformOutputSize = -1;
    GLint uniformApplyActivation = -1;
#endif

    std::vector<float> forwardCPU(const std::vector<float>& input)
    {
        activations.clear();
        activations.push_back(input);
        std::vector<float> a = input;

        for (int l = 0; l < (int)W.size(); ++l)
        {
            std::vector<float> z(b[l]);
            for (int i = 0; i < (int)W[l].size(); ++i)
                for (int j = 0; j < (int)W[l][i].size(); ++j)
                    z[i] += W[l][i][j] * a[j];

            if (l < (int)W.size() - 1)
                for (auto& val : z) val = std::tanh(val);

            activations.push_back(z);
            a = z;
        }
        return a;
    }

#ifdef ALICE2_USE_OPENGL_COMPUTE
    bool initializeComputeShader()
    {
        if (computeProgram != 0)
            return true;

        const char* shaderSrc = R"(
#version 430
layout(local_size_x = 64) in;

layout(std430, binding = 0) buffer Weights { float weights[]; };
layout(std430, binding = 1) buffer Inputs { float inputs[]; };
layout(std430, binding = 2) buffer Outputs { float outputs[]; };
layout(std430, binding = 3) buffer Biases { float bias[]; };

uniform int inputSize;
uniform int outputSize;
uniform int applyActivation;

void main()
{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(outputSize)) return;

    float sum = bias[idx];
    uint base = idx * uint(inputSize);
    for (int i = 0; i < inputSize; ++i)
    {
        sum += weights[base + uint(i)] * inputs[i];
    }

    if (applyActivation != 0)
        sum = tanh(sum);

    outputs[idx] = sum;
}
)";

        GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
        glShaderSource(shader, 1, &shaderSrc, nullptr);
        glCompileShader(shader);

        GLint status = GL_FALSE;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
        if (status != GL_TRUE)
        {
            char buffer[1024];
            glGetShaderInfoLog(shader, sizeof(buffer), nullptr, buffer);
            std::cerr << "MLP compute shader compile error: " << buffer << std::endl;
            glDeleteShader(shader);
            return false;
        }

        computeProgram = glCreateProgram();
        glAttachShader(computeProgram, shader);
        glLinkProgram(computeProgram);
        glDeleteShader(shader);

        glGetProgramiv(computeProgram, GL_LINK_STATUS, &status);
        if (status != GL_TRUE)
        {
            char buffer[1024];
            glGetProgramInfoLog(computeProgram, sizeof(buffer), nullptr, buffer);
            std::cerr << "MLP compute shader link error: " << buffer << std::endl;
            glDeleteProgram(computeProgram);
            computeProgram = 0;
            return false;
        }

        uniformInputSize = glGetUniformLocation(computeProgram, "inputSize");
        uniformOutputSize = glGetUniformLocation(computeProgram, "outputSize");
        uniformApplyActivation = glGetUniformLocation(computeProgram, "applyActivation");
        return true;
    }

    bool forwardGPU(const std::vector<float>& input, std::vector<float>& result)
    {
        activations.clear();
        activations.push_back(input);
        std::vector<float> a = input;

        for (int l = 0; l < (int)W.size(); ++l)
        {
            std::vector<float> z;
            bool applyActivation = (l < (int)W.size() - 1);
            if (!runLayerGPU(W[l], b[l], a, applyActivation, z))
                return false;

            activations.push_back(z);
            a = z;
        }

        result = activations.back();
        return true;
    }

    bool runLayerGPU(const std::vector<std::vector<float>>& weights,
                     const std::vector<float>& bias,
                     const std::vector<float>& input,
                     bool applyActivation,
                     std::vector<float>& output)
    {
        if (weights.empty())
            return false;
        int outputSize = (int)weights.size();
        int inputSize = (int)weights[0].size();
        if (inputSize != (int)input.size())
            return false;
        if (bias.size() != (size_t)outputSize)
            return false;

        std::vector<float> flatWeights((size_t)outputSize * (size_t)inputSize);
        for (int i = 0; i < outputSize; ++i)
            for (int j = 0; j < inputSize; ++j)
                flatWeights[(size_t)i * (size_t)inputSize + (size_t)j] = weights[i][j];

        output.assign(outputSize, 0.0f);

        GLuint buffers[4];
        glGenBuffers(4, buffers);

        auto uploadBuffer = [](GLuint buffer, GLuint binding, const std::vector<float>& data)
        {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * data.size(), data.data(), GL_DYNAMIC_DRAW);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buffer);
        };

        uploadBuffer(buffers[0], 0, flatWeights);
        uploadBuffer(buffers[1], 1, input);
        uploadBuffer(buffers[2], 2, output);
        uploadBuffer(buffers[3], 3, bias);

        glUseProgram(computeProgram);
        glUniform1i(uniformInputSize, inputSize);
        glUniform1i(uniformOutputSize, outputSize);
        glUniform1i(uniformApplyActivation, applyActivation ? 1 : 0);

        GLuint groups = (GLuint)((outputSize + 63) / 64);
        if (groups == 0) groups = 1;
        glDispatchCompute(groups, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[2]);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float) * output.size(), output.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        for (int i = 0; i < 4; ++i)
        {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
            glDeleteBuffers(1, &buffers[i]);
        }

        glUseProgram(0);
        return true;
    }
#endif
};
