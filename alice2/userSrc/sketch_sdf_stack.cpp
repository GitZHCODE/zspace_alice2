#define __MAIN__
#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>
#include <computeGeom/scalarField.h>
#include <computeGeom/scalarField3D.h>
#include <objects/MeshObject.h>

#include <nlohmann/json.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace alice2;
using json = nlohmann::json;

class FieldStackFromJson : public ISketch {
public:
    FieldStackFromJson() = default;

    std::string getName() const override        { return "JSON Field Stack"; }
    std::string getDescription() const override { return "Visualize multiple scalar-field slices from JSON"; }
    std::string getAuthor() const override      { return "alice2 User"; }

    void setup() override {
        scene().setBackgroundColor(Color(0.9f, 0.9f, 0.9f));
        scene().setShowAxes(false);
        scene().setShowGrid(false);

        m_meshObject = std::make_shared<MeshObject>("StackIsoMesh");
        m_meshObject->setVisible(false);
        m_meshObject->setRenderMode(MeshRenderMode::NormalShaded);
        m_meshObject->setShowEdges(false);
        m_meshObject->setShowVertices(false);
        scene().addObject(m_meshObject);

        bool loaded = loadFields(m_inputJsonName);
        if (!loaded) {
            const std::filesystem::path fallback = std::filesystem::path("alice2") / "data" / "inFieldStack.json";
            loadFields(fallback.string());
        }
    }

    void update(float /*dt*/) override {}

    void draw(Renderer& renderer, Camera& /*camera*/) override {
        // renderer.setColor(Color(0.9f, 0.9f, 0.9f));
        renderer.setColor(Color(0.1f, 0.1f, 0.1f));
        renderer.drawString("Slices: " + std::to_string(m_contours.size()), 10, 30);
        renderer.drawString("Spacing (+/-): " + std::to_string(m_sliceSpacing), 10, 50);
        renderer.drawString("J: smooth slices, K: stack Laplacian", 10, 70);
        renderer.drawString("P: generate mesh, D: toggle mesh", 10, 90);
        renderer.drawString("E: export fields + mesh", 10, 110);
        renderer.drawString("Mesh visible: " + std::string(m_meshVisible ? "yes" : "no"), 10, 130);
        renderer.drawString(m_statusMessage, 10, 150);

        for (size_t idx = 0; idx < m_contours.size(); ++idx) {
            const float z = static_cast<float>(idx) * m_sliceSpacing;
            const float hue = 360.0f * (static_cast<float>(idx) / std::max<size_t>(1, m_contours.size()));
            float r, g, b;
            ScalarFieldUtils::get_hsv_color(hue / 360.0f * 2.0f - 1.0f, r, g, b);
            renderer.setColor(Color(r, g, b));

            for (const auto& seg : m_contours[idx]) {
                renderer.drawLine(seg.first + Vec3(0, 0, z),
                                  seg.second + Vec3(0, 0, z),
                                  Color(r, g, b),
                                  1.0f);
            }
        }
    }

    bool onKeyPress(unsigned char key, int /*x*/, int /*y*/) override {
        if (key == '+' || key == '=') {
            m_sliceSpacing = std::min(m_sliceSpacing + 0.05f, 2.0f);
            invalidateVolumeMesh();
            return true;
        }
        if (key == '-' || key == '_') {
            m_sliceSpacing = std::max(m_sliceSpacing - 0.05f, 0.05f);
            invalidateVolumeMesh();
            return true;
        }
        if (key == 'j' || key == 'J') {
            smooth();
            return true;
        }
        if (key == 'k' || key == 'K') {
            applyStackLaplacian();
            return true;
        }
        if (key == 'p' || key == 'P') {
            buildVolumeMeshFromStack();
            return true;
        }
        if (key == 'd' || key == 'D') {
            toggleMeshVisibility();
            return true;
        }
        if (key == 'e' || key == 'E') {
            exportFieldsAndMesh();
            return true;
        }
        return false;
    }

private:
    bool loadFields(const std::string& path) {
        m_jsonLoaded = false;

        std::ifstream file(path);
        if (!file.is_open()) {
            m_statusMessage = "Failed to open " + path;
            return false;
        }

        json j;
        file >> j;

        const int resX = j.value("scalar_field_XCount", 0);
        const int resY = j.value("scalar_field_YCount", 0);
        const auto xSizes = j.value("scalar_field_XSize", std::vector<double>{-1.0, 1.0});
        const auto ySizes = j.value("scalar_field_YSize", std::vector<double>{-1.0, 1.0});

        if (resX <= 0 || resY <= 0 || xSizes.size() != 2 || ySizes.size() != 2) {
            m_statusMessage = "Invalid grid metadata in " + path;
            return false;
        }

        const Vec3 bmin(static_cast<float>(xSizes[0]),
                        static_cast<float>(ySizes[0]),
                        0.0f);
        const Vec3 bmax(static_cast<float>(xSizes[1]),
                        static_cast<float>(ySizes[1]),
                        0.0f);
        m_boundsMin = bmin;
        m_boundsMax = bmax;

        std::vector<std::pair<std::string, std::vector<float>>> fieldData;
        for (auto& [key, value] : j.items()) {
            if ((key.rfind("out_scalar_field_", 0) == 0) && value.is_array()) {
                std::vector<float> samples = value.get<std::vector<float>>();
                if (static_cast<int>(samples.size()) != resX * resY) {
                    m_statusMessage = "Size mismatch in " + key;
                    continue;
                }
                fieldData.emplace_back(key, std::move(samples));
            }
        }

        if (fieldData.empty()) {
            m_statusMessage = "No scalar fields found in " + path;
            invalidateVolumeMesh();
            return false;
        }

        std::sort(fieldData.begin(), fieldData.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        m_fields.clear();
        m_fieldKeys.clear();
        m_contours.clear();
        for (auto& entry : fieldData) {
            ScalarField2D slice(bmin, bmax, resX, resY);
            slice.set_values(entry.second);
            m_fieldKeys.emplace_back(entry.first);
            m_fields.emplace_back(std::move(slice));
        }

        regenerateContours();
        invalidateVolumeMesh();

        m_loadedJson = j;
        m_jsonLoaded = true;
        m_loadedJsonPath = path;
        m_statusMessage = "Loaded " + std::to_string(m_fields.size()) + " slices";
        return true;
    }

    void applyStackLaplacian() {
        if (m_fields.size() < 3) {
            m_statusMessage = "Need at least three slices for Laplacian smoothing";
            return;
        }

        const size_t sliceCount = m_fields.size();
        std::vector<std::vector<float>> smoothed(sliceCount);

        for (size_t i = 0; i < sliceCount; ++i) {
            smoothed[i] = m_fields[i].get_values();
        }

        for (size_t i = 1; i + 1 < sliceCount; ++i) {
            const auto& prev = m_fields[i - 1].get_values();
            const auto& next = m_fields[i + 1].get_values();
            auto& dest = smoothed[i];

            for (size_t v = 0; v < dest.size(); ++v) {
                dest[v] = 0.5f * (prev[v] + next[v]);
            }
        }

        for (size_t i = 0; i < sliceCount; ++i) {
            m_fields[i].set_values(smoothed[i]);
        }

        regenerateContours();
        invalidateVolumeMesh();
        m_statusMessage = "Stack Laplacian applied";
    }

    void smooth() {
        if (m_fields.empty()) {
            m_statusMessage = "No slices loaded";
            return;
        }

        for (auto& field : m_fields) {
            smoothField(field);
        }

        regenerateContours();
        invalidateVolumeMesh();
        m_statusMessage = "In-plane smoothing applied";
    }

    void smoothField(ScalarField2D& field) {
        const auto& values = field.get_values();
        if (values.empty()) {
            return;
        }

        const auto resolution = field.get_resolution();
        const int resX = resolution.first;
        const int resY = resolution.second;
        if (resX <= 0 || resY <= 0) {
            return;
        }

        std::vector<float> filtered(values.size(), 0.0f);
        auto index = [resX](int x, int y) { return y * resX + x; };

        for (int y = 0; y < resY; ++y) {
            for (int x = 0; x < resX; ++x) {
                float sum = 0.0f;
                int count = 0;
                const int yStart = std::max(0, y - 1);
                const int yEnd = std::min(resY - 1, y + 1);
                const int xStart = std::max(0, x - 1);
                const int xEnd = std::min(resX - 1, x + 1);

                for (int ny = yStart; ny <= yEnd; ++ny) {
                    for (int nx = xStart; nx <= xEnd; ++nx) {
                        sum += values[index(nx, ny)];
                        ++count;
                    }
                }

                filtered[index(x, y)] = count > 0 ? sum / static_cast<float>(count) : values[index(x, y)];
            }
        }

        field.set_values(filtered);
    }

    bool buildVolumeMeshFromStack() {
        if (m_fields.empty()) {
            m_statusMessage = "Load fields before generating mesh";
            return false;
        }

        const auto resolution = m_fields.front().get_resolution();
        const int resX = resolution.first;
        const int resY = resolution.second;
        const int resZ = static_cast<int>(m_fields.size());
        if (resX <= 0 || resY <= 0 || resZ <= 0) {
            m_statusMessage = "Invalid field resolution";
            return false;
        }

        Vec3 minBound = Vec3(m_boundsMin.x, m_boundsMin.y, 0.0f);
        Vec3 maxBound = Vec3(m_boundsMax.x, m_boundsMax.y, resZ > 1 ? m_sliceSpacing * static_cast<float>(resZ - 1) : m_sliceSpacing);

        m_volumeField = std::make_unique<ScalarField3D>(minBound, maxBound, resX, resY, resZ);

        const size_t layerSize = static_cast<size_t>(resX) * static_cast<size_t>(resY);
        std::vector<float> volumeValues(layerSize * static_cast<size_t>(resZ), 0.0f);
        for (int z = 0; z < resZ; ++z) {
            const auto& sliceValues = m_fields[static_cast<size_t>(z)].get_values();
            std::copy(sliceValues.begin(), sliceValues.end(), volumeValues.begin() + layerSize * static_cast<size_t>(z));
        }

        m_volumeField->set_values(volumeValues);

        auto meshData = m_volumeField->generate_mesh(m_isoLevel);
        if (!meshData || meshData->vertices.empty()) {
            m_statusMessage = "3D mesh generation yielded no geometry";
            m_meshGenerated = false;
            m_meshVisible = false;
            if (m_meshObject) {
                m_meshObject->setVisible(false);
            }
            return false;
        }

        if (!m_meshObject) {
            m_meshObject = std::make_shared<MeshObject>("StackIsoMesh");
            m_meshObject->setVisible(false);
            m_meshObject->setShowEdges(false);
            m_meshObject->setShowVertices(false);
            scene().addObject(m_meshObject);
        }

        m_meshObject->setMeshData(meshData);
        m_meshObject->setRenderMode(MeshRenderMode::NormalShaded);
        m_meshObject->setNormalShadingColors(Color(0.1f, 0.1f, 0.1f), Color(1.0f, 1.0f, 1.0f)); // White to magenta

        m_meshGenerated = true;
        m_meshVisible = true;
        m_meshObject->setVisible(true);
        m_statusMessage = "3D mesh generated";
        return true;
    }

    void toggleMeshVisibility() {
        if (!m_meshGenerated || !m_meshObject) {
            m_statusMessage = "Generate the mesh first (press P)";
            return;
        }

        m_meshVisible = !m_meshVisible;
        m_meshObject->setVisible(m_meshVisible);
        m_statusMessage = m_meshVisible ? "Mesh visible" : "Mesh hidden";
    }

    void exportFieldsAndMesh() {
        if (!m_jsonLoaded || m_fields.empty()) {
            m_statusMessage = "Nothing to export";
            return;
        }

        updateJsonWithCurrentFields();

        std::ofstream outJson(m_outputJsonName);
        if (!outJson.is_open()) {
            m_statusMessage = "Failed to write " + m_outputJsonName;
            return;
        }
        outJson << m_loadedJson.dump(2);

        if (!m_meshGenerated) {
            if (!buildVolumeMeshFromStack()) {
                m_statusMessage = "Fields saved but mesh generation failed";
                return;
            }
        }

        if (!m_meshObject) {
            m_statusMessage = "Fields saved but mesh object missing";
            return;
        }

        m_meshObject->writeToObj(m_outputMeshName);
        m_statusMessage = "Exported fields -> " + m_outputJsonName + ", mesh -> " + m_outputMeshName;
    }

    void updateJsonWithCurrentFields() {
        const auto resolution = m_fields.front().get_resolution();
        m_loadedJson["scalar_field_XCount"] = resolution.first;
        m_loadedJson["scalar_field_YCount"] = resolution.second;
        m_loadedJson["scalar_field_XSize"] = { m_boundsMin.x, m_boundsMax.x };
        m_loadedJson["scalar_field_YSize"] = { m_boundsMin.y, m_boundsMax.y };

        for (size_t i = 0; i < m_fields.size(); ++i) {
            std::string key;
            if (i < m_fieldKeys.size()) {
                key = m_fieldKeys[i];
            } else {
                key = fieldKeyForIndex(i);
                m_fieldKeys.push_back(key);
            }
            m_loadedJson[key] = m_fields[i].get_values();
        }
    }

    std::string fieldKeyForIndex(size_t idx) const {
        std::ostringstream oss;
        oss << "out_scalar_field_" << std::setw(2) << std::setfill('0') << idx;
        return oss.str();
    }

    void regenerateContours() {
        m_contours.clear();
        m_contours.reserve(m_fields.size());
        for (auto& field : m_fields) {
            m_contours.emplace_back(field.get_contours(m_isoLevel).line_segments);
        }
    }

    void invalidateVolumeMesh() {
        m_volumeField.reset();
        m_meshGenerated = false;
        m_meshVisible = false;
        if (m_meshObject) {
            m_meshObject->setVisible(false);
        }
    }

    const std::string m_inputJsonName = "inFieldStack.json";
    const std::string m_outputJsonName = "outFieldStack.json";
    const std::string m_outputMeshName = "outMesh.obj";

    std::vector<ScalarField2D> m_fields;
    std::vector<std::vector<std::pair<Vec3, Vec3>>> m_contours;
    std::vector<std::string> m_fieldKeys;
    float m_isoLevel = 0.01f;
    float m_sliceSpacing = 3.5f;
    std::string m_statusMessage = "Awaiting load";

    Vec3 m_boundsMin = Vec3(-1, -1, 0);
    Vec3 m_boundsMax = Vec3(1, 1, 0);

    std::unique_ptr<ScalarField3D> m_volumeField;
    std::shared_ptr<MeshObject> m_meshObject;
    bool m_meshVisible = false;
    bool m_meshGenerated = false;

    json m_loadedJson;
    bool m_jsonLoaded = false;
    std::string m_loadedJsonPath;
};

ALICE2_REGISTER_SKETCH_AUTO(FieldStackFromJson)

#endif // __MAIN__
