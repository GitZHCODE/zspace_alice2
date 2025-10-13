#define __MAIN__
#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>

#include <ML/genericMLP.h>
#include <ML/AutoDecoderTrainer.h>
#include <ML/AutoDecoderTrainerCUDA.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

using namespace alice2;
using json = nlohmann::json;

namespace {
std::pair<float, float> readRange(const json& j, const std::string& key)
{
    const auto& arr = j.at(key);
    if (!arr.is_array() || arr.size() != 2)
        throw std::runtime_error("Expected array[2] for key: " + key);
    return { static_cast<float>(arr[0].get<double>()), static_cast<float>(arr[1].get<double>()) };
}

std::vector<float> readScalarArray(const json& j)
{
    std::vector<float> values;
    values.reserve(j.size());
    for (const auto& entry : j) values.push_back(static_cast<float>(entry.get<double>()));
    return values;
}

inline Color valueToGray(float t){ t = std::clamp(t, 0.0f, 1.0f); return Color(t, t, t); }
}

// -----------------------------------------------------------------------------
class AutoDecoderParitySketch : public ISketch {
public:
    AutoDecoderParitySketch() : m_status("Press 'P' to train both (CPU+GPU)") {}

    std::string getName()  const override { return "AutoDecoder CPU-GPU Parity (JSON)"; }
    std::string getDescription() const override { return "Train CPU & GPU auto-decoder on JSON fields and compare"; }
    std::string getAuthor() const override { return "alice2 User"; }

    void setup() override
    {
        scene().setBackgroundColor(Color(0.0f, 0.0f, 0.0f));
        scene().setShowAxes(false);
        scene().setShowGrid(false);

        const std::filesystem::path defaultPath = std::filesystem::path("inFieldStack.json");
        loadDataset(defaultPath.string());
    }

    void update(float) override {}

    void draw(Renderer& r, Camera&) override
    {
        r.setColor(Color(0.9f, 0.9f, 0.9f));
        r.drawString(getName(), 15, 30);
        r.drawString(m_status, 15, 50);
        r.drawString("Loaded fields: " + std::to_string(m_originalGridFields.size()), 15, 70);
        r.drawString("Samples: " + std::to_string(m_samples.size()), 15, 90);
        r.drawString("Epochs run: " + std::to_string(m_totalEpochs), 15, 110);
        r.drawString("Press 'P' to train (" + std::to_string(m_epochsPerTrigger) + " epochs)", 15, 130);
        r.drawString("Press 'R' to reload dataset", 15, 150);

        char buf[256];
        std::snprintf(buf, sizeof(buf), "CPU %.1f ms | GPU %.1f ms | Epochs %d",
                      m_lastCpuTimeMs, m_lastGpuTimeMs, m_lastEpochs);
        r.drawString(buf, 15, 170);

        const float topOriginal = 210.0f;
        const float topCPU      = topOriginal + m_tileSize + 80.0f;
        const float topGPU      = topCPU + m_tileSize + 80.0f;
        const float topDiff     = topGPU + m_tileSize + 80.0f;

        if (!m_originalGridFields.empty()) {
            r.setColor(Color(0.85f, 0.8f, 0.95f));
            r.drawString("Original Fields", 15, topOriginal - 20.0f);
            drawFieldRow(r, m_originalGridFields, topOriginal);
        }
        if (m_hasCPU && !m_cpuGridFields.empty()) {
            r.setColor(Color(0.8f, 0.9f, 1.0f));
            r.drawString("CPU Reconstruction", 15, topCPU - 20.0f);
            drawFieldRow(r, m_cpuGridFields, topCPU);
        }
        if (m_hasGPU && !m_gpuGridFields.empty()) {
            r.setColor(Color(0.9f, 0.85f, 0.8f));
            r.drawString("GPU Reconstruction", 15, topGPU - 20.0f);
            drawFieldRow(r, m_gpuGridFields, topGPU);
        }
        if (m_hasCPU && m_hasGPU && !m_diffGridFields.empty()) {
            r.setColor(Color(1.0f, 0.8f, 0.8f));
            r.drawString("|CPU - GPU| Difference", 15, topDiff - 20.0f);
            drawFieldRow(r, m_diffGridFields, topDiff);
        }
    }

    bool onKeyPress(unsigned char key, int, int) override
    {
        switch (key) {
        case 'p':
        case 'P':
            runTraining();
            return true;
        case 'r':
        case 'R':
            reloadDataset();
            return true;
        default: break;
        }
        return false;
    }

private:
    struct GridField { std::vector<float> values; float minValue=0.f, maxValue=0.f; };

    bool loadDataset(const std::string& path)
    {
        std::ifstream file(path);
        if (!file.is_open()) { m_status = "Failed to open: " + path; return false; }

        json data;
        try { file >> data; }
        catch (const std::exception& e) { m_status = std::string("JSON parse error: ") + e.what(); return false; }

        try {
            m_gridResolutionX = data.at("scalar_field_XCount").get<int>();
            m_gridResolutionY = data.at("scalar_field_YCount").get<int>();
            std::tie(m_xMin, m_xMax) = readRange(data, "scalar_field_XSize");
            std::tie(m_yMin, m_yMax) = readRange(data, "scalar_field_YSize");

            if (m_gridResolutionX <= 0 || m_gridResolutionY <= 0) throw std::runtime_error("Counts must be positive");
            if (m_xMax <= m_xMin || m_yMax <= m_yMin) throw std::runtime_error("Invalid bounds in JSON");

            const std::size_t expectedSize = (std::size_t)m_gridResolutionX * (std::size_t)m_gridResolutionY;

            struct FieldEntry { int index = 0; std::vector<float> values; };
            std::vector<FieldEntry> fieldEntries; fieldEntries.reserve(data.size());

            for (const auto& [key, value] : data.items()) {
                constexpr const char* prefix = "scalar_field";
                constexpr const char* suffix = "_data";
                if (key.size() <= std::strlen(prefix) + std::strlen(suffix)) continue;
                if (!value.is_array()) continue;
                if (key.rfind(prefix, 0) != 0) continue;
                if (key.find(suffix) != key.size() - std::strlen(suffix)) continue;

                const std::string indexString = key.substr(std::strlen(prefix), key.size() - std::strlen(prefix) - std::strlen(suffix));
                if (indexString.empty() || !std::all_of(indexString.begin(), indexString.end(), ::isdigit)) continue;

                FieldEntry entry;
                entry.index  = std::stoi(indexString);
                entry.values = readScalarArray(value);
                if (entry.values.size() != expectedSize) throw std::runtime_error("Field " + key + " has unexpected sample count");
                fieldEntries.emplace_back(std::move(entry));
            }

            if (fieldEntries.empty()) throw std::runtime_error("No scalar_field*_data entries found in JSON");

            std::sort(fieldEntries.begin(), fieldEntries.end(),
                      [](const FieldEntry& a, const FieldEntry& b) { return a.index < b.index; });

            // Store originals and make samples
            m_originalGridFields.clear();
            m_samples.clear();
            m_originalGridFields.reserve(fieldEntries.size());
            m_samples.reserve(fieldEntries.size() * expectedSize);

            const float xStep = (m_gridResolutionX > 1) ? (m_xMax - m_xMin) / (float)(m_gridResolutionX - 1) : 0.0f;
            const float yStep = (m_gridResolutionY > 1) ? (m_yMax - m_yMin) / (float)(m_gridResolutionY - 1) : 0.0f;

            for (std::size_t idx = 0; idx < fieldEntries.size(); ++idx) {
                m_originalGridFields.emplace_back();
                GridField& grid = m_originalGridFields.back();
                grid.values = std::move(fieldEntries[idx].values);

                auto [minIt, maxIt] = std::minmax_element(grid.values.begin(), grid.values.end());
                grid.minValue = *minIt; grid.maxValue = *maxIt;

                const int shapeIndex = (int)idx;
                for (int y = 0; y < m_gridResolutionY; ++y) {
                    const float yCoord = m_yMin + yStep * (float)y;
                    for (int x = 0; x < m_gridResolutionX; ++x) {
                        const float xCoord = m_xMin + xStep * (float)x;
                        const std::size_t linearIndex = (std::size_t)y * (std::size_t)m_gridResolutionX + (std::size_t)x;

                        AutoDecoderSample sample;
                        sample.shapeIndex = shapeIndex;
                        sample.coordinate = { xCoord, yCoord };
                        sample.sdf = grid.values[linearIndex];
                        m_samples.push_back(std::move(sample));
                    }
                }
            }

            setupTrainers((int)m_originalGridFields.size());
            m_status = "Dataset loaded: " + path;
            m_datasetPath = path;
            return true;
        } catch (const std::exception& e) {
            m_status = std::string("Load failed: ") + e.what();
            m_originalGridFields.clear();
            m_samples.clear();
            m_trainerCPU.reset();
            m_trainerGPU.reset();
            m_hasCPU = m_hasGPU = false;
        }
        return false;
    }

    void reloadDataset(){ if (!m_datasetPath.empty()) loadDataset(m_datasetPath); }

    void setupTrainers(int numShapes)
    {
        if (numShapes <= 0) return;
        m_coordDim = 2;
        m_latentDim = std::max(8, std::min(32, numShapes * 4));

        // CPU
        const std::vector<int> hidden = {64,64,64};
        m_decoderCPU.initialize(m_latentDim + m_coordDim, hidden, 1);
        m_trainerCPU = std::make_unique<AutoDecoderTrainer>(m_decoderCPU);
        m_trainerCPU->initialize(numShapes, m_latentDim, m_coordDim);
        m_trainerCPU->setSamples(m_samples);

        m_cfgCPU.epochs = 1;
        m_cfgCPU.learningRateWeights = 5e-4f;
        m_cfgCPU.learningRateLatent  = 1e-3f;
        m_cfgCPU.latentRegularization= 1e-4f;
        m_cfgCPU.latentInitStd       = 0.01f;
        m_cfgCPU.shuffleSeed         = 2025u;

        // GPU: SimpleMLP with same topology; weights init with fixed seed
        SimpleMLP net;
        net.inputDim  = m_latentDim + m_coordDim;
        net.outputDim = 1;
        net.hidden    = hidden;
        {
            std::mt19937_64 rng(2025ull);
            std::normal_distribution<float> N01(0.f, 0.02f);
            int inDim = net.inputDim;
            for (size_t i=0;i<net.hidden.size();++i){
                int outDim = net.hidden[i];
                net.weights.emplace_back((size_t)outDim*(size_t)inDim);
                net.biases.emplace_back((size_t)outDim, 0.0f);
                for (auto& w : net.weights.back()) w = N01(rng);
                inDim = outDim;
            }
            net.weights.emplace_back((size_t)net.outputDim*(size_t)inDim);
            net.biases.emplace_back((size_t)net.outputDim, 0.0f);
            for (auto& w : net.weights.back()) w = N01(rng);
        }

        m_trainerGPU = std::make_unique<AutoDecoderTrainerCUDA>();
        m_trainerGPU->setNetwork(net, numShapes, m_latentDim, m_coordDim);
        m_trainerGPU->setSamples(toADTSamples(m_samples));

        m_cfgGPU.epochs = 1;
        m_cfgGPU.lrW    = m_cfgCPU.learningRateWeights;
        m_cfgGPU.lrZ    = m_cfgCPU.learningRateLatent;
        m_cfgGPU.lambda = m_cfgCPU.latentRegularization;
        m_cfgGPU.shuffleSeed = m_cfgCPU.shuffleSeed;

        m_totalEpochs = 0;
        m_hasCPU = m_hasGPU = false;
        m_cpuGridFields.clear();
        m_gpuGridFields.clear();
        m_diffGridFields.clear();
    }

    static std::vector<ADTSample> toADTSamples(const std::vector<AutoDecoderSample>& in)
    {
        std::vector<ADTSample> out;
        out.reserve(in.size());
        for (const auto& s : in) {
            ADTSample t;
            t.shapeIndex = s.shapeIndex;
            t.coord      = s.coordinate;
            t.target     = s.sdf;
            out.push_back(std::move(t));
        }
        return out;
    }

    void runTraining()
    {
        if (!m_trainerCPU || !m_trainerGPU) { m_status = "Trainers not ready"; return; }

        std::cout << "[AutoDecoder CPU+GPU] Training start\n";
        for (int i = 0; i < m_epochsPerTrigger; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            AutoDecoderTrainingStats cs = m_trainerCPU->train(m_cfgCPU);
            auto t1 = std::chrono::high_resolution_clock::now();
            ADTStats gs = m_trainerGPU->train(m_cfgGPU);
            auto t2 = std::chrono::high_resolution_clock::now();

            m_lastCpuTimeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
            m_lastGpuTimeMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
            m_lastEpochs = m_cfgCPU.epochs;
            ++m_totalEpochs;

            std::cout << "  Epoch " << m_totalEpochs
                      << ": CPU(avg=" << cs.averageLoss << ", last=" << cs.lastLoss
                      << ", " << m_lastCpuTimeMs << " ms)"
                      << " | GPU(avg=" << gs.avgLoss << ", last=" << gs.lastLoss
                      << ", " << m_lastGpuTimeMs << " ms)"
                      << "  samples=" << cs.totalSamples << std::endl;
        }

        generateReconstructionCPU();
        generateReconstructionGPU();
        buildDiff();

        std::cout << "[AutoDecoder] Total epochs run = " << m_totalEpochs << std::endl;
        m_status = "Training complete (Original / CPU / GPU / |CPU-GPU|)";
    }

    void generateReconstructionCPU()
    {
        if (!m_trainerCPU || m_coordDim != 2 || m_latentDim <= 0) return;
        if (m_gridResolutionX <= 0 || m_gridResolutionY <= 0) return;
        const auto& latents = m_trainerCPU->getLatentCodes();
        if (latents.empty()) return;

        m_cpuGridFields.clear();
        m_cpuGridFields.resize(latents.size());
        std::vector<float> input((size_t)m_latentDim + (size_t)m_coordDim);

        const float xStep = (m_gridResolutionX > 1) ? (m_xMax - m_xMin) / (float)(m_gridResolutionX - 1) : 0.0f;
        const float yStep = (m_gridResolutionY > 1) ? (m_yMax - m_yMin) / (float)(m_gridResolutionY - 1) : 0.0f;

        for (size_t shape = 0; shape < latents.size(); ++shape) {
            const auto& z = latents[shape];
            GridField& grid = m_cpuGridFields[shape];
            grid.values.resize((size_t)m_gridResolutionX * (size_t)m_gridResolutionY);
            grid.minValue = +std::numeric_limits<float>::max();
            grid.maxValue = -std::numeric_limits<float>::max();

            for (int y = 0; y < m_gridResolutionY; ++y) {
                const float yCoord = m_yMin + yStep * (float)y;
                for (int x = 0; x < m_gridResolutionX; ++x) {
                    const float xCoord = m_xMin + xStep * (float)x;
                    const size_t idx = (size_t)y * (size_t)m_gridResolutionX + (size_t)x;

                    std::copy(z.begin(), z.end(), input.begin());
                    input[(size_t)m_latentDim + 0] = xCoord;
                    input[(size_t)m_latentDim + 1] = yCoord;

                    const std::vector<float> prediction = m_decoderCPU.forward(input);
                    const float value = prediction.empty() ? 0.0f : prediction[0];

                    grid.values[idx] = value;
                    grid.minValue = std::min(grid.minValue, value);
                    grid.maxValue = std::max(grid.maxValue, value);
                }
            }
        }
        m_hasCPU = true;
    }

    static float forwardSimple(const SimpleMLP& mlp, const std::vector<float>& x)
    {
        std::vector<float> cur = x, nxt;
        int in = mlp.inputDim;
        for (size_t l=0;l<mlp.hidden.size();++l){
            const int rows = mlp.hidden[l];
            const int cols = in;
            nxt.assign(rows, 0.0f);
            const auto& W = mlp.weights[l];
            const auto& B = mlp.biases[l];
            for (int r=0;r<rows;++r){
                float acc = B[r];
                for (int c=0;c<cols;++c) acc += W[(size_t)r*(size_t)cols + (size_t)c] * cur[(size_t)c];
                nxt[(size_t)r] = std::tanh(acc);
            }
            cur.swap(nxt);
            in = rows;
        }
        // output
        {
            const int rows = mlp.outputDim;
            const int cols = in;
            nxt.assign(rows, 0.0f);
            const auto& W = mlp.weights.back();
            const auto& B = mlp.biases.back();
            for (int r=0;r<rows;++r){
                float acc = B[r];
                for (int c=0;c<cols;++c) acc += W[(size_t)r*(size_t)cols + (size_t)c] * cur[(size_t)c];
                nxt[(size_t)r] = acc;
            }
            cur.swap(nxt);
        }
        return cur[0];
    }

    void generateReconstructionGPU()
    {
        if (!m_trainerGPU || m_coordDim != 2 || m_latentDim <= 0) return;
        if (m_gridResolutionX <= 0 || m_gridResolutionY <= 0) return;

        const std::vector<float>& Zflat = m_trainerGPU->getLatents();
        if (Zflat.empty()) return;
        const int numShapes = (int)(Zflat.size() / (size_t)m_latentDim);

        m_gpuGridFields.clear();
        m_gpuGridFields.resize((size_t)numShapes);

        const float xStep = (m_gridResolutionX > 1) ? (m_xMax - m_xMin) / (float)(m_gridResolutionX - 1) : 0.0f;
        const float yStep = (m_gridResolutionY > 1) ? (m_yMax - m_yMin) / (float)(m_gridResolutionY - 1) : 0.0f;

        std::vector<float> input((size_t)m_latentDim + (size_t)m_coordDim);
        const SimpleMLP& dec = m_trainerGPU->getTrainedMLP();

        for (int s = 0; s < numShapes; ++s) {
            GridField& grid = m_gpuGridFields[(size_t)s];
            grid.values.resize((size_t)m_gridResolutionX * (size_t)m_gridResolutionY);
            grid.minValue = +std::numeric_limits<float>::max();
            grid.maxValue = -std::numeric_limits<float>::max();

            for (int j=0;j<m_latentDim;++j) input[(size_t)j] = Zflat[(size_t)s*(size_t)m_latentDim + (size_t)j];

            for (int y = 0; y < m_gridResolutionY; ++y) {
                const float yCoord = m_yMin + yStep * (float)y;
                for (int x = 0; x < m_gridResolutionX; ++x) {
                    const float xCoord = m_xMin + xStep * (float)x;
                    const size_t idx = (size_t)y * (size_t)m_gridResolutionX + (size_t)x;

                    input[(size_t)m_latentDim + 0] = xCoord;
                    input[(size_t)m_latentDim + 1] = yCoord;

                    const float value = forwardSimple(dec, input);
                    grid.values[idx] = value;
                    grid.minValue = std::min(grid.minValue, value);
                    grid.maxValue = std::max(grid.maxValue, value);
                }
            }
        }
        m_hasGPU = true;
    }

    void buildDiff()
    {
        if (!m_hasCPU || !m_hasGPU) { m_diffGridFields.clear(); return; }
        const size_t n = std::min(m_cpuGridFields.size(), m_gpuGridFields.size());
        m_diffGridFields.resize(n);
        for (size_t i=0;i<n;++i){
            const auto& A = m_cpuGridFields[i];
            const auto& B = m_gpuGridFields[i];
            GridField& D = m_diffGridFields[i];
            const size_t sz = std::min(A.values.size(), B.values.size());
            D.values.resize(sz);
            float mn = +std::numeric_limits<float>::max();
            float mx = -std::numeric_limits<float>::max();
            for (size_t k=0;k<sz;++k){
                float v = std::fabs(A.values[k] - B.values[k]);
                D.values[k] = v;
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            D.minValue = (mn==+std::numeric_limits<float>::max()) ? 0.0f : mn;
            D.maxValue = (mx==-std::numeric_limits<float>::max()) ? 0.0f : mx;
        }
    }

    void drawFieldRow(Renderer& r, const std::vector<GridField>& grids, float top) const
    {
        if (grids.empty() || m_gridResolutionX <= 0 || m_gridResolutionY <= 0) return;
        const float gap = 25.0f;
        const float cellWidth  = m_tileSize / (float)m_gridResolutionX;
        const float cellHeight = m_tileSize / (float)m_gridResolutionY;

        for (size_t fieldIdx=0; fieldIdx<grids.size(); ++fieldIdx) {
            const float left = 20.0f + (float)fieldIdx * (m_tileSize + gap);
            drawFieldHeatmap(r, grids[fieldIdx], left, top, cellWidth, cellHeight);
            r.setColor(Color(0.7f,0.7f,0.9f));
            r.drawString("#" + std::to_string(fieldIdx), left, top + m_tileSize + 16.0f);
        }
    }

    void drawFieldHeatmap(Renderer& r, const GridField& field, float left, float top, float cellWidth, float cellHeight) const
    {
        const float safeMin = field.minValue;
        const float safeMax = field.maxValue;
        const float range = (safeMax - safeMin == 0.0f) ? 1.0f : (safeMax - safeMin);
        for (int y=0;y<m_gridResolutionY;++y){
            for (int x=0;x<m_gridResolutionX;++x){
                const size_t idx = (size_t)y*(size_t)m_gridResolutionX + (size_t)x;
                const float value = field.values[idx];
                const float norm = (value - safeMin) / range;
                const Color color = valueToGray(norm);

                const float px = left + ((float)x + 0.5f) * cellWidth;
                const float py = top  + ((float)y + 0.5f) * cellHeight;
                const float pointSize = std::max(cellWidth, cellHeight) * 0.8f;
                r.draw2dPoint(Vec2(px, py), color, pointSize);
            }
        }
    }

private:
    // CPU
    MLP m_decoderCPU;
    std::unique_ptr<AutoDecoderTrainer> m_trainerCPU;
    AutoDecoderTrainingConfig m_cfgCPU{};

    // GPU
    std::unique_ptr<AutoDecoderTrainerCUDA> m_trainerGPU;
    ADTConfig m_cfgGPU{};

    // dataset
    std::vector<AutoDecoderSample> m_samples;
    std::vector<GridField> m_originalGridFields;

    // reconstructions
    std::vector<GridField> m_cpuGridFields;
    std::vector<GridField> m_gpuGridFields;
    std::vector<GridField> m_diffGridFields;
    bool m_hasCPU=false, m_hasGPU=false;

    // status
    std::string m_status;
    std::string m_datasetPath;
    int m_totalEpochs=0;
    int m_epochsPerTrigger=5;
    double m_lastCpuTimeMs=0.0, m_lastGpuTimeMs=0.0;
    int m_lastEpochs=0;

    // grid dims
    int m_gridResolutionX=0, m_gridResolutionY=0;
    float m_xMin=0.0f, m_xMax=0.0f;
    float m_yMin=0.0f, m_yMax=0.0f;

    // model dims
    int m_latentDim=0, m_coordDim=0;

    float m_tileSize=120.0f;
};

ALICE2_REGISTER_SKETCH_AUTO(AutoDecoderParitySketch)
#endif // __MAIN__
