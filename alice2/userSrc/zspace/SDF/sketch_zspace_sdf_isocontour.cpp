// #define __MAIN__
#ifdef __MAIN__

#include <zspace/interface.h>

#include <alice2.h>
#include <sketches/SketchRegistry.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace alice2;

namespace {

float sdCircle(const zSpace::zVector& p, const zSpace::zVector& center, float radius)
{
    const float dx = p.x - center.x;
    const float dy = p.y - center.y;
    return std::sqrt(dx * dx + dy * dy) - radius;
}

std::string formatFloat(float value, int precision = 3)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

} // namespace

class zSpaceSdfIsocontourSketch : public ISketch {
public:
    std::string getName() const override { return "zSpace SDF: Isocontour Threshold"; }
    std::string getDescription() const override { return "Simple circle SDF in a 2D zObjectMeshField with field-helper isocontours."; }
    std::string getAuthor() const override { return "alice2 + zspace_core"; }

    void setup() override
    {
        scene().setBackgroundColor(Color(0.04f, 0.045f, 0.055f, 1.0f));
        scene().setShowGrid(false);
        scene().setShowAxes(true);
        scene().setAxesLength(3.0f);

        buildField();

        m_ui = std::make_unique<SimpleUI>(input());
        m_ui->setTheme(SimpleUI::UITheme::Dark);
        m_ui->addSlider("Iso Threshold", Vec2{10.0f, 128.0f}, 260.0f, m_fieldMin, m_fieldMax, m_threshold);
        m_ui->addToggle("Draw Field", UIRect{10.0f, 158.0f, 110.0f, 24.0f}, m_drawField);
        m_ui->addToggle("Draw Contour", UIRect{130.0f, 158.0f, 130.0f, 24.0f}, m_drawContour);

        refreshIsocontour();
    }

    void update(float) override
    {
        if (std::abs(m_threshold - m_lastThreshold) > 1e-5f) {
            refreshIsocontour();
        }
    }

    void draw(Renderer& renderer, Camera&) override
    {
        if (m_drawField) {
            zDisplayMeshSetting fieldDisplay;
            fieldDisplay.showFaces = true;
            fieldDisplay.showEdges = false;
            fieldDisplay.showVertices = false;
            fieldDisplay.useVertexColors = true;
            scene().draw(m_field, fieldDisplay);
        }

        if (m_drawContour) {
            zDisplayGraphSetting contourDisplay;
            contourDisplay.showEdges = true;
            contourDisplay.showVertices = false;
            contourDisplay.edgeColor = Color(1.0f, 0.48f, 0.0f, 1.0f);
            contourDisplay.edgeWidth = 2.0f;
            scene().draw(m_contour, contourDisplay);
        }

        drawOverlay(renderer);
        if (m_ui) {
            m_ui->draw(renderer);
        }
    }

    bool onMousePress(int button, int state, int x, int y) override
    {
        return m_ui && m_ui->onMousePress(button, state, x, y);
    }

    bool onMouseMove(int x, int y) override
    {
        return m_ui && m_ui->onMouseMove(x, y);
    }

    bool onKeyPress(unsigned char key, int, int) override
    {
        if (key == '0') {
            m_threshold = 0.0f;
            refreshIsocontour();
            return true;
        }
        if (key == 'c' || key == 'C') {
            refreshIsocontour();
            return true;
        }

        return false;
    }

private:
    void buildField()
    {
        zSpace::zFnMeshScalarField fn(m_field);
        fn.create(zSpace::zPoint(-5.0f, -5.0f, 0.0f),
                  zSpace::zPoint(5.0f, 5.0f, 0.0f),
                  m_resolutionX,
                  m_resolutionY,
                  1,
                  true,
                  false);

        zSpace::zPointArray positions;
        fn.getPositions(positions);

        std::vector<zSpace::zScalar> values;
        values.reserve(positions.size());

        m_fieldMin = std::numeric_limits<float>::max();
        m_fieldMax = std::numeric_limits<float>::lowest();

        for (const auto& p : positions) {
            const float sdf = sdCircle(p, zSpace::zVector(0.0f, 0.0f, 0.0f), m_circleRadius);

            values.push_back(sdf);
            m_fieldMin = std::min(m_fieldMin, sdf);
            m_fieldMax = std::max(m_fieldMax, sdf);
        }

        if (m_fieldMin > 0.0f || m_fieldMax < 0.0f) {
            m_threshold = 0.5f * (m_fieldMin + m_fieldMax);
        } else {
            m_threshold = 0.0f;
        }

        fn.setFieldValues(values, zSpace::zFieldSDF);
        fn.updateColors(zSpace::zFieldSDF);
    }

    void refreshIsocontour()
    {
        zSpace::zFnMeshScalarField fn(m_field);
        fn.getIsocontour(m_contour, m_threshold);
        liftContourAboveField();
        updateContourStats();

        m_lastThreshold = m_threshold;
    }

    void liftContourAboveField()
    {
        zSpace::zFnGraph fn(m_contour);
        zSpace::zPointArray positions;
        fn.getVertexPositions(positions);

        for (auto& p : positions) {
            p.z = 0.05f;
        }

        if (!positions.empty()) {
            fn.setVertexPositions(positions);
        }
    }

    void drawOverlay(Renderer& renderer)
    {
        renderer.setColor(Color(1.0f, 1.0f, 1.0f));
        renderer.drawString(getName(), 10, 28);
        renderer.drawString("Simple circle SDF: zFnMeshScalarField::getIsocontour", 10, 50);

        renderer.setColor(Color(0.72f, 0.78f, 0.86f));
        renderer.drawString("Field min/max: " + formatFloat(m_fieldMin) + " / " + formatFloat(m_fieldMax), 10, 78);
        renderer.drawString("Threshold: " + formatFloat(m_threshold) + "   Press 'C' to rebuild, '0' for zero", 10, 100);
        renderer.drawString("Contour vertices/edges: " + std::to_string(m_contourVertexCount) + " / " + std::to_string(m_contourEdgeCount), 10, 122);
        renderer.drawString("Contour source: zSpace only", 10, 144);
    }

    void updateContourStats()
    {
        zSpace::zFnGraph fn(m_contour);
        m_contourVertexCount = fn.numVertices();
        m_contourEdgeCount = fn.numEdges();
    }

    zSpace::zObjectMeshScalarField m_field;
    zSpace::zObjectGraph m_contour;

    std::unique_ptr<SimpleUI> m_ui;
    int m_resolutionX{120};
    int m_resolutionY{120};
    float m_circleRadius{2.5f};
    float m_fieldMin{-1.0f};
    float m_fieldMax{1.0f};
    float m_threshold{0.0f};
    float m_lastThreshold{std::numeric_limits<float>::quiet_NaN()};
    int m_contourVertexCount{0};
    int m_contourEdgeCount{0};
    bool m_drawField{true};
    bool m_drawContour{true};
};

ALICE2_REGISTER_SKETCH_AUTO(zSpaceSdfIsocontourSketch)

#endif // __MAIN__
