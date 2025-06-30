#pragma once
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdio>

#include <vector>
#include <cmath>
#include <fstream>
#include "alice2.h"

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
        for (int l = 0; l < layerDims.size() - 1; ++l)
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
        activations.clear();
        activations.push_back(x);
        std::vector<float> a = x;

        for (int l = 0; l < W.size(); ++l)
        {
            std::vector<float> z(b[l]);
            for (int i = 0; i < W[l].size(); ++i)
                for (int j = 0; j < W[l][i].size(); ++j)
                    z[i] += W[l][i][j] * a[j];

            if (l < W.size() - 1)
                for (auto& val : z) val = std::tanh(val);

            activations.push_back(z);
            a = z;
        }
        return a;
    }

    virtual float computeLoss(std::vector<float>& y_pred, std::vector<float>& y_true)
    {
        float loss = 0.0f;
        for (int i = 0; i < y_pred.size(); ++i)
        {
            float err = y_pred[i] - y_true[i];
            loss += err * err;
        }
        return loss / y_pred.size();
    }

    virtual void computeGradient(std::vector<float>& x, std::vector<float>& y_true, std::vector<float>& gradOut)
    {
        std::vector<float> y_pred = forward(x);
        gradOut.assign(outputDim, 0.0f);
        for (int i = 0; i < outputDim; ++i)
        {
            gradOut[i] = 2.0f * (y_pred[i] - y_true[i]) / outputDim;
        }
    }

    void backward(std::vector<float>& gradOut, float lr)
    {
        std::vector<float> delta = gradOut;

        for (int l = W.size() - 1; l >= 0; --l)
        {
            std::vector<float> prev = activations[l];
            std::vector<float> newDelta(prev.size(), 0.0f);

            for (int i = 0; i < W[l].size(); ++i)
            {
                for (int j = 0; j < W[l][i].size(); ++j)
                {
                    newDelta[j] += delta[i] * W[l][i][j];
                    W[l][i][j] -= lr * delta[i] * prev[j];
                }
                b[l][i] -= lr * delta[i];
            }

            if (l > 0)
            {
                for (int i = 0; i < newDelta.size(); ++i)
                {
                    float a = activations[l][i];
                    newDelta[i] *= (1 - a * a); // tanh'
                }
                delta = newDelta;
            }
        }
    }

    void visualize(Renderer& renderer, Camera& camera, const Vec3& topLeft = Vec3(50, 450, 0), float bboxWidth = 400.0f, float bboxHeight = 300.0f)
    {
        int numLayers = activations.size();
        float nodeRadius = 3.0f;

        // Compute max nodes per layer for vertical spacing
        int maxNodesPerLayer = 0;
        for (auto& layer : activations)
            maxNodesPerLayer = std::max(maxNodesPerLayer, (int)layer.size());

        float layerSpacing = (numLayers > 1) ? bboxWidth / (numLayers - 1) : 150.0f;
        float verticalSpacing = (maxNodesPerLayer > 1) ? bboxHeight / (maxNodesPerLayer - 1) : 30.0f;

        std::vector<std::vector<Vec3>> nodePositions(numLayers);

        // Compute node positions
        for (int l = 0; l < numLayers; l++)
        {
            int numNodes = activations[l].size();
            float yStart = topLeft.y - 0.5f * (numNodes - 1) * verticalSpacing;

            for (int i = 0; i < numNodes; i++)
            {
                float x = topLeft.x + l * layerSpacing;
                float y = yStart + i * verticalSpacing;
                nodePositions[l].push_back(Vec3(x, y, topLeft.z));
            }
        }

        // --- Draw weight connections
        for (int l = 0; l < numLayers - 1; l++)
        {
            int fromSize = activations[l].size();
            int toSize = activations[l + 1].size();

            for (int i = 0; i < fromSize; i++)
            {
                for (int j = 0; j < toSize; j++)
                {
                    float w = W[l][j][i];
                    float absW = fabs(w);

                    if (absW < 0.02f) continue;  // skip very weak connections

                    float val = std::clamp(w * 5.0f, -1.0f, 1.0f);
                    float r, g, b;
                    get_jet_color(val, r, g, b);

                    Vec3 color(r, g, b);
                    float width = std::clamp(absW * 5.0f, 0.5f, 2.0f);

                    Vec2 start = Vec2(nodePositions[l][i].x, nodePositions[l][i].y);
                    Vec2 end = Vec2(nodePositions[l + 1][j].x, nodePositions[l + 1][j].y);
                    renderer.draw2dLine(start, end, color, width);
                }
            }
        }

        // --- Draw nodes
        for (int l = 0; l < numLayers; l++)
        {
            for (int i = 0; i < activations[l].size(); i++)
            {
                float act = activations[l][i];
                float r, g, b;

                get_jet_color(act, r, g, b);

                Vec3 color(r, g, b);
                renderer.setColor(color);
                //drawSolidCircle(renderer, nodePositions[l][i], nodeRadius, 12);
                Vec2 pos = Vec2(nodePositions[l][i].x, nodePositions[l][i].y);
                renderer.draw2dPoint(pos,color,6);
            }
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
};