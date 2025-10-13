#define __MAIN__
#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>

#include <ML/genericMLP.h>
#include <ML/AutoDecoderTrainer.h>
#include <ML/AutoDecoderTrainerCUDA.h>
#include <cuda_runtime.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include <chrono>

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
    for (const auto& entry : j)
    {
        values.push_back(static_cast<float>(entry.get<double>()));
    }
    return values;
}

inline Color valueToGray(float t)
{
    t = std::clamp(t, 0.0f, 1.0f);
    return Color(t, t, t);
}
}

class AutoDecoderSDFSketch : public ISketch {
public:
    AutoDecoderSDFSketch()
        : m_status("Press 'P' to train (CPU+GPU)"),
          m_modelOutputPathCPU("auto_decoder_model_cpu.json"),
          m_modelOutputPathGPU("auto_decoder_model_gpu.json")
    {
    }

    std::string getName() const override { return "SDF AutoDecoder Trainer (CPU + GPU)"; }
    std::string getDescription() const override { return "Train latent SDF decoder from JSON grid on CPU & GPU; P=train, R=reload"; }
    std::string getAuthor() const override { return "alice2 User"; }

    void setup() override
    {
        scene().setBackgroundColor(Color(0.0f, 0.0f, 0.0f));
        scene().setShowAxes(false);
        scene().setShowGrid(false);

        const std::filesystem::path defaultPath = std::filesystem::path("inFieldStack.json");
        loadDataset(defaultPath.string());
    }

    void update(float /*time*/) override {}

    void draw(Renderer& renderer, Camera& /*camera*/) override
    {
        renderer.setColor(Color(0.9f, 0.9f, 0.9f));
        renderer.drawString(getName(), 15, 30);
        renderer.drawString(m_status, 15, 50);
        renderer.drawString("Loaded fields: " + std::to_string(m_originalGridFields.size()), 15, 70);
        renderer.drawString("Samples: " + std::to_string(m_samples.size()), 15, 90);
        renderer.drawString("Epochs run: " + std::to_string(m_totalEpochs), 15, 110);
        renderer.drawString("Press 'P' to train (" + std::to_string(m_epochsPerTrigger) + " epochs, CPU+GPU)", 15, 130);
        renderer.drawString("Press 'R' to reload dataset", 15, 150);

        const float topOriginal = 200.0f;
        const float topReconCPU = topOriginal + m_tileSize + 80.0f;
        const float topReconGPU = topReconCPU + m_tileSize + 80.0f;

        if (!m_originalGridFields.empty())
        {
            renderer.setColor(Color(0.85f, 0.8f, 0.95f));
            renderer.drawString("Original Fields", 15, topOriginal - 20.0f);
            drawFieldRow(renderer, m_originalGridFields, topOriginal);
        }

        if (m_hasReconstructionCPU && !m_reconstructedGridFieldsCPU.empty())
        {
            renderer.setColor(Color(0.75f, 0.9f, 0.75f));
            renderer.drawString("CPU Reconstructed Fields", 15, topReconCPU - 20.0f);
            drawFieldRow(renderer, m_reconstructedGridFieldsCPU, topReconCPU);
        }

        if (m_hasReconstructionGPU && !m_reconstructedGridFieldsGPU.empty())
        {
            renderer.setColor(Color(0.75f, 0.85f, 1.0f));
            renderer.drawString("GPU Reconstructed Fields", 15, topReconGPU - 20.0f);
            drawFieldRow(renderer, m_reconstructedGridFieldsGPU, topReconGPU);
        }
    }

    bool onKeyPress(unsigned char key, int /*x*/, int /*y*/) override
    {
        switch (key)
        {
        case 'p':
        case 'P':
            runTraining();
            return true;
        case 'r':
        case 'R':
            reloadDataset();
            return true;
        default:
            break;
        }
        return false;
    }

private:
    struct GridField
    {
        std::vector<float> values;
        float minValue = 0.0f;
        float maxValue = 0.0f;
    };

    bool loadDataset(const std::string& path)
    {
        std::ifstream file(path);
        if (!file.is_open())
        {
            m_status = "Failed to open: " + path;
            return false;
        }

        json data;
        try
        {
            file >> data;
        }
        catch (const std::exception& e)
        {
            m_status = std::string("JSON parse error: ") + e.what();
            return false;
        }

        try
        {
            m_gridResolutionX = data.at("scalar_field_XCount").get<int>();
            m_gridResolutionY = data.at("scalar_field_YCount").get<int>();
            std::tie(m_xMin, m_xMax) = readRange(data, "scalar_field_XSize");
            std::tie(m_yMin, m_yMax) = readRange(data, "scalar_field_YSize");

            if (m_gridResolutionX <= 0 || m_gridResolutionY <= 0)
                throw std::runtime_error("Counts must be positive");
            if (m_xMax <= m_xMin || m_yMax <= m_yMin)
                throw std::runtime_error("Invalid bounds in JSON");

            const std::size_t expectedSize = static_cast<std::size_t>(m_gridResolutionX) * static_cast<std::size_t>(m_gridResolutionY);

            struct FieldEntry
            {
                int index = 0;
                std::vector<float> values;
            };

            std::vector<FieldEntry> fieldEntries;
            fieldEntries.reserve(data.size());

            for (const auto& [key, value] : data.items())
            {
                constexpr const char* prefix = "scalar_field";
                constexpr const char* suffix = "_data";
                if (key.size() <= std::strlen(prefix) + std::strlen(suffix))
                    continue;
                if (!value.is_array())
                    continue;
                if (key.rfind(prefix, 0) != 0)
                    continue;
                if (key.find(suffix) != key.size() - std::strlen(suffix))
                    continue;

                const std::string indexString = key.substr(std::strlen(prefix), key.size() - std::strlen(prefix) - std::strlen(suffix));
                if (indexString.empty() || !std::all_of(indexString.begin(), indexString.end(), ::isdigit))
                    continue;

                FieldEntry entry;
                entry.index = std::stoi(indexString);
                entry.values = readScalarArray(value);
                if (entry.values.size() != expectedSize)
                    throw std::runtime_error("Field " + key + " has unexpected sample count");

                fieldEntries.emplace_back(std::move(entry));
            }

            if (fieldEntries.empty())
                throw std::runtime_error("No scalar_field*_data entries found in JSON");

            std::sort(fieldEntries.begin(), fieldEntries.end(),
                      [](const FieldEntry& a, const FieldEntry& b) { return a.index < b.index; });

            m_originalGridFields.clear();
            m_originalGridFields.reserve(fieldEntries.size());
            m_reconstructedGridFieldsCPU.clear();
            m_reconstructedGridFieldsGPU.clear();
            m_hasReconstructionCPU = false;
            m_hasReconstructionGPU = false;
            m_samples.clear();
            m_samples.reserve(fieldEntries.size() * expectedSize);

            const float xStep = (m_gridResolutionX > 1)
                                    ? (m_xMax - m_xMin) / static_cast<float>(m_gridResolutionX - 1)
                                    : 0.0f;
            const float yStep = (m_gridResolutionY > 1)
                                    ? (m_yMax - m_yMin) / static_cast<float>(m_gridResolutionY - 1)
                                    : 0.0f;

            for (std::size_t idx = 0; idx < fieldEntries.size(); ++idx)
            {
                m_originalGridFields.emplace_back();
                GridField& grid = m_originalGridFields.back();
                grid.values = std::move(fieldEntries[idx].values);

                auto [minIt, maxIt] = std::minmax_element(grid.values.begin(), grid.values.end());
                grid.minValue = *minIt;
                grid.maxValue = *maxIt;

                const int shapeIndex = static_cast<int>(idx);
                for (int y = 0; y < m_gridResolutionY; ++y)
                {
                    const float yCoord = m_yMin + yStep * static_cast<float>(y);
                    for (int x = 0; x < m_gridResolutionX; ++x)
                    {
                        const float xCoord = m_xMin + xStep * static_cast<float>(x);
                        const std::size_t linearIndex = static_cast<std::size_t>(y) * static_cast<std::size_t>(m_gridResolutionX) + static_cast<std::size_t>(x);

                        AutoDecoderSample sample;
                        sample.shapeIndex = shapeIndex;
                        sample.coordinate = { xCoord, yCoord };
                        sample.sdf = grid.values[linearIndex];
                        m_samples.push_back(std::move(sample));
                    }
                }
            }

            setupTrainers(static_cast<int>(m_originalGridFields.size()));
            m_status = "Dataset loaded: " + path;
            m_datasetPath = path;
            return true;
        }
        catch (const std::exception& e)
        {
            m_status = std::string("Load failed: ") + e.what();
            m_originalGridFields.clear();
            m_reconstructedGridFieldsCPU.clear();
            m_reconstructedGridFieldsGPU.clear();
            m_samples.clear();
            m_trainerCPU.reset();
            m_trainerGPU.reset();
        }

        return false;
    }

    void setupTrainers(int numShapes)
    {
        if (numShapes <= 0)
            return;

        m_coordDim = 2;
        m_latentDim = std::max(8, std::min(32, numShapes * 4));

        const std::vector<int> hidden = { 64, 64, 64 };
        m_decoderCPU.initialize(m_latentDim + m_coordDim, hidden, 1);
        m_decoderGPU = m_decoderCPU; // identical clone

        // Trainers
        m_trainerCPU = std::make_unique<AutoDecoderTrainer>(m_decoderCPU);
        m_trainerCPU->initialize(numShapes, m_latentDim, m_coordDim);
        m_trainerCPU->setSamples(m_samples);

        m_trainerGPU = std::make_unique<AutoDecoderTrainerCUDA>(m_decoderGPU);
        m_trainerGPU->initialize(numShapes, m_latentDim, m_coordDim);
        m_trainerGPU->setSamples(m_samples);

        // Deterministic identical latents for parity
        std::vector<std::vector<float>> codes(static_cast<std::size_t>(numShapes),
                                              std::vector<float>(static_cast<std::size_t>(m_latentDim), 0.0f));
        for (int s = 0; s < numShapes; ++s)
            for (int k = 0; k < m_latentDim; ++k)
                codes[static_cast<std::size_t>(s)][static_cast<std::size_t>(k)] = 0.001f * float(s + k);

        m_trainerCPU->setLatentCodes(codes);
        m_trainerGPU->setLatentCodesHost(codes);

        m_trainingConfig.epochs = 1;
        m_trainingConfig.learningRateWeights = 5e-4f;
        m_trainingConfig.learningRateLatent = 1e-3f;
        m_trainingConfig.latentRegularization = 1e-4f;
        m_trainingConfig.latentInitStd = 0.01f;
        m_trainingConfig.shuffleSeed = 2025u;

        m_totalEpochs = 0;
    }

    void reloadDataset()
    {
        if (m_datasetPath.empty())
            return;
        loadDataset(m_datasetPath);
    }

    void runTraining()
    {
        using namespace std::chrono;

        if (!m_trainerCPU || !m_trainerGPU)
        {
            m_status = "Trainers not ready";
            return;
        }

        std::cout << "[AutoDecoder CPU+GPU] Training start" << std::endl;
        AutoDecoderTrainingStats lastCPU{}, lastGPU{};

        auto totalStart = high_resolution_clock::now();

        for (int i = 0; i < m_epochsPerTrigger; ++i)
        {
            // ---- CPU timing ----
            auto cpuStart = high_resolution_clock::now();
            lastCPU = m_trainerCPU->train(m_trainingConfig);
            auto cpuEnd = high_resolution_clock::now();
            double cpuMs = duration<double, std::milli>(cpuEnd - cpuStart).count();

            // ---- GPU timing ----
            auto gpuStart = high_resolution_clock::now();
            lastGPU = m_trainerGPU->train(m_trainingConfig);
            // Important: sync CUDA to get accurate timing
            cudaDeviceSynchronize();
            auto gpuEnd = high_resolution_clock::now();
            double gpuMs = duration<double, std::milli>(gpuEnd - gpuStart).count();

            ++m_totalEpochs;
            std::cout << "  Epoch " << m_totalEpochs
                    << ": CPU(avg=" << lastCPU.averageLoss << ", last=" << lastCPU.lastLoss
                    << ", " << cpuMs << " ms)"
                    << " | GPU(avg=" << lastGPU.averageLoss << ", last=" << lastGPU.lastLoss
                    << ", " << gpuMs << " ms)"
                    << "  samples=" << lastCPU.totalSamples << std::endl;

            m_trainingConfig.shuffleSeed += 1;
        }

        auto totalEnd = high_resolution_clock::now();
        double totalMs = duration<double, std::milli>(totalEnd - totalStart).count();
        std::cout << "[AutoDecoder] Total training time for " << m_epochsPerTrigger
                << " epochs = " << totalMs << " ms" << std::endl;

        // ---- Rebuild reconstructions for both ----
        const auto& Zcpu = m_trainerCPU->getLatentCodes();
        const auto& Zgpu = m_trainerGPU->getLatentCodesHost();

        generateReconstructionFor(m_decoderCPU, Zcpu, m_reconstructedGridFieldsCPU, m_hasReconstructionCPU);
        generateReconstructionFor(m_decoderGPU, Zgpu, m_reconstructedGridFieldsGPU, m_hasReconstructionGPU);

        // ---- Save models ----
        const bool savedCPU = m_trainerCPU->saveToJson(m_modelOutputPathCPU);
        const bool savedGPU = m_trainerGPU->saveToJson(m_modelOutputPathGPU);

        if (savedCPU && savedGPU)
            m_status = "Training done. Saved CPU+GPU models.";
        else if (savedCPU)
            m_status = "Training done. Saved CPU only.";
        else if (savedGPU)
            m_status = "Training done. Saved GPU only.";
        else
            m_status = "Training done. Save failed.";
    }

    void generateReconstructionFor(MLP& decoder,
                                   const std::vector<std::vector<float>>& latentCodes,
                                   std::vector<GridField>& outGrids,
                                   bool& outHasRecon)
    {
        if (latentCodes.empty() || m_latentDim <= 0 || m_coordDim != 2)
            return;
        if (m_gridResolutionX <= 0 || m_gridResolutionY <= 0)
            return;

        outGrids.clear();
        outGrids.resize(latentCodes.size());

        const float xStep = (m_gridResolutionX > 1)
                                ? (m_xMax - m_xMin) / static_cast<float>(m_gridResolutionX - 1)
                                : 0.0f;
        const float yStep = (m_gridResolutionY > 1)
                                ? (m_yMax - m_yMin) / static_cast<float>(m_gridResolutionY - 1)
                                : 0.0f;

        std::vector<float> input(static_cast<std::size_t>(m_latentDim + m_coordDim));

        for (std::size_t shape = 0; shape < latentCodes.size(); ++shape)
        {
            const auto& latent = latentCodes[shape];
            if (latent.size() != static_cast<std::size_t>(m_latentDim))
                continue;

            GridField& grid = outGrids[shape];
            grid.values.resize(static_cast<std::size_t>(m_gridResolutionX) * static_cast<std::size_t>(m_gridResolutionY));
            grid.minValue = std::numeric_limits<float>::max();
            grid.maxValue = -std::numeric_limits<float>::max();

            for (int y = 0; y < m_gridResolutionY; ++y)
            {
                const float yCoord = m_yMin + yStep * static_cast<float>(y);
                for (int x = 0; x < m_gridResolutionX; ++x)
                {
                    const float xCoord = m_xMin + xStep * static_cast<float>(x);
                    const std::size_t idx = static_cast<std::size_t>(y) * static_cast<std::size_t>(m_gridResolutionX) + static_cast<std::size_t>(x);

                    std::copy(latent.begin(), latent.end(), input.begin());
                    input[m_latentDim + 0] = xCoord;
                    input[m_latentDim + 1] = yCoord;

                    const std::vector<float> prediction = decoder.forward(input);
                    const float value = prediction.empty() ? 0.0f : prediction[0];

                    grid.values[idx] = value;
                    grid.minValue = std::min(grid.minValue, value);
                    grid.maxValue = std::max(grid.maxValue, value);
                }
            }

            if (grid.minValue == std::numeric_limits<float>::max())
            {
                grid.minValue = 0.0f;
                grid.maxValue = 0.0f;
            }
        }

        outHasRecon = true;
    }

    void drawFieldRow(Renderer& renderer, const std::vector<GridField>& grids, float top) const
    {
        if (grids.empty() || m_gridResolutionX <= 0 || m_gridResolutionY <= 0)
            return;

        const float gap = 25.0f;
        const float cellWidth = m_tileSize / static_cast<float>(m_gridResolutionX);
        const float cellHeight = m_tileSize / static_cast<float>(m_gridResolutionY);

        for (std::size_t fieldIdx = 0; fieldIdx < grids.size(); ++fieldIdx)
        {
            const float left = 20.0f + static_cast<float>(fieldIdx) * (m_tileSize + gap);
            drawFieldHeatmap(renderer, grids[fieldIdx], left, top, cellWidth, cellHeight);
            renderer.setColor(Color(0.7f, 0.7f, 0.9f));
            renderer.drawString("#" + std::to_string(fieldIdx), left, top + m_tileSize + 16.0f);
        }
    }

    void drawFieldHeatmap(Renderer& renderer, const GridField& field, float left, float top, float cellWidth, float cellHeight) const
    {
        const float safeMin = field.minValue;
        const float safeMax = field.maxValue;
        const float range = (safeMax - safeMin == 0.0f) ? 1.0f : (safeMax - safeMin);

        const float pointSize = std::max(cellWidth, cellHeight) * 0.8f;

        for (int y = 0; y < m_gridResolutionY; ++y)
        {
            const float py = top + (static_cast<float>(y) + 0.5f) * cellHeight;
            for (int x = 0; x < m_gridResolutionX; ++x)
            {
                const std::size_t idx = static_cast<std::size_t>(y) * static_cast<std::size_t>(m_gridResolutionX) + static_cast<std::size_t>(x);
                const float value = field.values[idx];
                const float norm = (value - safeMin) / range;
                const Color color = valueToGray(norm);

                const float px = left + (static_cast<float>(x) + 0.5f) * cellWidth;
                renderer.draw2dPoint(Vec2(px, py), color, pointSize);
            }
        }
    }

    // Models & trainers
    MLP m_decoderCPU;
    MLP m_decoderGPU;
    std::unique_ptr<AutoDecoderTrainer>     m_trainerCPU;
    std::unique_ptr<AutoDecoderTrainerCUDA> m_trainerGPU;
    AutoDecoderTrainingConfig m_trainingConfig;

    std::vector<GridField> m_originalGridFields;
    std::vector<GridField> m_reconstructedGridFieldsCPU;
    std::vector<GridField> m_reconstructedGridFieldsGPU;
    std::vector<AutoDecoderSample> m_samples;

    std::string m_status;
    std::string m_datasetPath;
    std::string m_modelOutputPathCPU;
    std::string m_modelOutputPathGPU;

    int m_totalEpochs = 0;
    int m_epochsPerTrigger = 5;

    int m_gridResolutionX = 0;
    int m_gridResolutionY = 0;
    float m_xMin = 0.0f;
    float m_xMax = 0.0f;
    float m_yMin = 0.0f;
    float m_yMax = 0.0f;

    int m_latentDim = 0;
    int m_coordDim = 0;

    bool m_hasReconstructionCPU = false;
    bool m_hasReconstructionGPU = false;
    float m_tileSize = 120.0f;
};

ALICE2_REGISTER_SKETCH_AUTO(AutoDecoderSDFSketch)

#endif // __MAIN__
