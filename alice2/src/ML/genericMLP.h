#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>

#include "alice2.h"
#ifdef ALICE2_USE_OPENGL_COMPUTE
#include "MLPComputeContext.h"
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
        releaseGPUTraining();
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
        for (int l = 0; l < static_cast<int>(layerDims.size()) - 1; ++l)
        {
            int inSize = layerDims[l];
            int outSize = layerDims[l + 1];
            W.push_back(std::vector<std::vector<float>>(outSize, std::vector<float>(inSize)));
            b.push_back(std::vector<float>(outSize));
            for (auto& w_row : W[l])
                for (auto& wv : w_row)
                    wv = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }

    std::vector<float> forward(std::vector<float>& x)
    {
        activations.clear();
        activations.push_back(x);
        std::vector<float> a = x;

        for (int l = 0; l < static_cast<int>(W.size()); ++l)
        {
            std::vector<float> z(b[l]);
            for (int i = 0; i < static_cast<int>(W[l].size()); ++i)
                for (int j = 0; j < static_cast<int>(W[l][i].size()); ++j)
                    z[i] += W[l][i][j] * a[j];

            if (l < static_cast<int>(W.size()) - 1)
                for (auto& val : z) val = std::tanh(val);

            activations.push_back(z);
            a = z;
        }
        return a;
    }

    virtual float computeLoss(std::vector<float>& y_pred, std::vector<float>& y_true)
    {
        float loss = 0.0f;
        for (int i = 0; i < static_cast<int>(y_pred.size()); ++i)
        {
            float err = y_pred[i] - y_true[i];
            loss += err * err;
        }
        return loss / static_cast<float>(y_pred.size());
    }

    virtual void computeGradient(std::vector<float>& x, std::vector<float>& y_true, std::vector<float>& gradOut)
    {
        std::vector<float> y_pred = forward(x);
        gradOut.assign(outputDim, 0.0f);
        for (int i = 0; i < outputDim; ++i)
        {
            gradOut[i] = 2.0f * (y_pred[i] - y_true[i]) / static_cast<float>(outputDim);
        }
    }

    void backward(std::vector<float>& gradOut, float lr)
    {
        std::vector<float> delta = gradOut;

        for (int l = static_cast<int>(W.size()) - 1; l >= 0; --l)
        {
            std::vector<float> prev = activations[l];
            std::vector<float> newDelta(prev.size(), 0.0f);

            for (int i = 0; i < static_cast<int>(W[l].size()); ++i)
            {
                for (int j = 0; j < static_cast<int>(W[l][i].size()); ++j)
                {
                    newDelta[j] += delta[i] * W[l][i][j];
                    W[l][i][j] -= lr * delta[i] * prev[j];
                }
                b[l][i] -= lr * delta[i];
            }

            if (l > 0)
            {
                for (int i = 0; i < static_cast<int>(newDelta.size()); ++i)
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

        if (gradOut.size() != static_cast<size_t>(outputDim))
            throw std::runtime_error("MLP::backwardWithInputGrad received gradOut with unexpected size.");

        std::vector<float> delta(gradOut);
        std::vector<float> inputGrad(inputDim, 0.0f);

        for (int l = static_cast<int>(W.size()) - 1; l >= 0; --l)
        {
            const std::vector<float>& prev = activations[l];
            std::vector<float> newDelta(prev.size(), 0.0f);

            for (int i = 0; i < static_cast<int>(W[l].size()); ++i)
            {
                for (int j = 0; j < static_cast<int>(W[l][i].size()); ++j)
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
                for (int i = 0; i < static_cast<int>(newDelta.size()); ++i)
                {
                    float a = activations[l][i];
                    newDelta[i] *= (1 - a * a); // tanh derivative
                }
                delta = newDelta;
            }
        }

        return inputGrad;
    }

    /**
     * Visualize MLP network structure with nodes and connections
     */
    void visualize(Renderer& renderer, Camera& camera, const Vec3& topLeft = Vec3(400, 200, 0), float bboxWidth = 300.0f, float bboxHeight = 200.0f)
    {
        if (activations.empty()) return; // No data to visualize

        int numLayers = static_cast<int>(activations.size());
        float nodeRadius = 3.0f;

        int maxNodesPerLayer = 0;
        for (const auto& layer : activations)
            maxNodesPerLayer = std::max(maxNodesPerLayer, static_cast<int>(layer.size()));

        float layerSpacing = (numLayers > 1) ? bboxWidth / (numLayers - 1) : 150.0f;
        float verticalSpacing = (maxNodesPerLayer > 1) ? std::min(bboxHeight / (maxNodesPerLayer - 1), 30.0f) : 20.0f;

        std::vector<std::vector<Vec3>> nodePositions(numLayers);

        for (int l = 0; l < numLayers; l++) {
            int numNodes = static_cast<int>(activations[l].size());
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
            if (l >= static_cast<int>(W.size())) continue;
            int fromSize = static_cast<int>(activations[l].size());
            int toSize = static_cast<int>(activations[l + 1].size());

            for (int i = 0; i < fromSize && i < static_cast<int>(nodePositions[l].size()); i++) {
                for (int j = 0; j < toSize && j < static_cast<int>(nodePositions[l + 1].size()); j++) {
                    if (j >= static_cast<int>(W[l].size()) || i >= static_cast<int>(W[l][j].size())) continue;

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
            for (int i = 0; i < static_cast<int>(activations[l].size()) && i < static_cast<int>(nodePositions[l].size()); i++) {
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

private:
#ifdef ALICE2_USE_OPENGL_COMPUTE
    std::unique_ptr<MLPComputeContext> gpuContext;
#endif

    inline void get_jet_color(float value, float& r, float& g, float& b) {
        value = clamp(value, -1.0f, 1.0f);
        float normalized = (value + 1.0f) * 0.5f;
        float fourValue = 4.0f * normalized;

        r = clamp(std::min(fourValue - 1.5f, -fourValue + 4.5f), 0.0f, 1.0f);
        g = clamp(std::min(fourValue - 0.5f, -fourValue + 3.5f), 0.0f, 1.0f);
        b = clamp(std::min(fourValue + 0.5f, -fourValue + 2.5f), 0.0f, 1.0f);
    }

public:
#ifdef ALICE2_USE_OPENGL_COMPUTE
    bool enableGPUTraining(const MLPComputeSpec& spec)
    {
        if (!gpuContext)
            gpuContext = std::make_unique<MLPComputeContext>();
        if (!gpuContext->initialise(spec))
        {
            gpuContext.reset();
            return false;
        }
        return uploadWeightsToGPU();
    }

    bool uploadWeightsToGPU()
    {
        if (!gpuContext)
            return false;
        std::vector<float> flatWeights;
        std::vector<float> flatBiases;
        flattenWeights(flatWeights, flatBiases);
        return gpuContext->uploadWeights(flatWeights, flatBiases);
    }

    bool uploadLatentsToGPU(const std::vector<std::vector<float>>& latents)
    {
        if (!gpuContext)
            return false;
        std::vector<float> flat;
        flat.reserve(latents.size() * (latents.empty() ? 0 : latents[0].size()));
        for (const auto& row : latents)
            flat.insert(flat.end(), row.begin(), row.end());
        return gpuContext->uploadLatents(flat);
    }

    bool uploadDatasetToGPU(const std::vector<float>& coords,
                            const std::vector<float>& targets,
                            const std::vector<int>& shapes)
    {
        if (!gpuContext)
            return false;
        return gpuContext->uploadDataset(coords, targets, shapes);
    }

    bool trainOnGPU(const MLPComputeConfig& cfg)
    {
        if (!gpuContext)
            return false;
        return gpuContext->train(cfg);
    }

    bool downloadWeightsFromGPU()
    {
        if (!gpuContext)
            return false;
        std::vector<float> flatWeights;
        std::vector<float> flatBiases;
        if (!gpuContext->downloadWeights(flatWeights, flatBiases))
            return false;
        unflattenWeights(flatWeights, flatBiases);
        return true;
    }

    bool downloadLatentsFromGPU(std::vector<std::vector<float>>& latentsOut, int latentDim)
    {
        if (!gpuContext)
            return false;
        std::vector<float> flat;
        if (!gpuContext->downloadLatents(flat))
            return false;
        size_t numShapes = latentsOut.size();
        if (numShapes * static_cast<size_t>(latentDim) != flat.size())
        {
            if (latentDim <= 0) return false;
            numShapes = flat.size() / static_cast<size_t>(latentDim);
            latentsOut.resize(numShapes);
        }
        for (size_t i = 0; i < numShapes; ++i)
        {
            latentsOut[i].assign(flat.begin() + static_cast<ptrdiff_t>(i * latentDim),
                                 flat.begin() + static_cast<ptrdiff_t>((i + 1) * latentDim));
        }
        return true;
    }

    void releaseGPUTraining()
    {
        if (gpuContext)
            gpuContext->release();
        gpuContext.reset();
    }

    bool hasGPUContext() const { return gpuContext != nullptr; }

private:
    void flattenWeights(std::vector<float>& flatWeights, std::vector<float>& flatBiases)
    {
        int layerCount = static_cast<int>(hiddenDims.size()) + 1;
        flatWeights.clear();
        flatBiases.clear();
        for (int l = 0; l < layerCount; ++l)
        {
            const auto& weightsLayer = W[l];
            const auto& biasLayer = b[l];
            for (const auto& row : weightsLayer)
                flatWeights.insert(flatWeights.end(), row.begin(), row.end());
            flatBiases.insert(flatBiases.end(), biasLayer.begin(), biasLayer.end());
        }
    }

    void unflattenWeights(const std::vector<float>& flatWeights, const std::vector<float>& flatBiases)
    {
        int layerCount = static_cast<int>(hiddenDims.size()) + 1;
        size_t wIndex = 0;
        size_t bIndex = 0;
        for (int l = 0; l < layerCount; ++l)
        {
            int outDim = static_cast<int>(W[l].size());
            int inDim = static_cast<int>(W[l][0].size());
            for (int i = 0; i < outDim; ++i)
            {
                for (int j = 0; j < inDim; ++j)
                {
                    W[l][i][j] = flatWeights[wIndex++];
                }
                b[l][i] = flatBiases[bIndex++];
            }
        }
    }
#endif // ALICE2_USE_OPENGL_COMPUTE
};
