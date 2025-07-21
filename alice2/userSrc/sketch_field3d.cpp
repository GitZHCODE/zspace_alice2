// sketch_field3d.cpp
// alice2 3D Scalar Field with Marching Cubes Sketch
// Now demonstrates the 6-term nodal surface
#define __MAIN__
#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>
#include <computeGeom/scalarField3D.h>
#include <objects/MeshObject.h>
#include <memory>
#include <iostream>
#include <vector>
#include <cmath>

using namespace alice2;

class ScalarField3DSketch : public ISketch {
private:
    // Core field & mesh
    std::unique_ptr<ScalarField3D> m_scalarField;
    std::shared_ptr<MeshObject> m_meshObject;

    // Which field to compute
    bool b_computeSphere;
    bool b_computeBox;
    bool b_computeTorus;
    bool b_computeNoise;   // now “nodal” surface
    bool b_recreateField;

    // Draw toggles
    bool d_drawField;
    bool d_drawMesh;
    bool d_drawWireframe;
    bool d_drawSlice;

    // Parameters
    float m_sphereRadius;
    float m_boxSize;
    float m_torusRadius;
    float m_isolevel;
    int   m_currentSlice;
    int   m_fieldResolution;

    // Animation timer
    float m_time;

    // Nodal-surface coefficients
    float m_a1, m_a2, m_a3, m_a4, m_a5, m_a6;

public:
    ScalarField3DSketch()
        : m_scalarField(nullptr)
        , m_meshObject(nullptr)
        , b_computeSphere(true)
        , b_computeBox(false)
        , b_computeTorus(false)
        , b_computeNoise(false)
        , b_recreateField(true)
        , d_drawField(false)
        , d_drawMesh(true)
        , d_drawWireframe(false)
        , d_drawSlice(false)
        , m_sphereRadius(20.0f)
        , m_boxSize(15.0f)
        , m_torusRadius(15.0f)
        , m_isolevel(0.0f)
        , m_currentSlice(25)
        , m_fieldResolution(16)
        , m_time(0.0f)
        // set your desired nodal-surface parameters here:
        , m_a1(0.2f)
        , m_a2(0.05f)
        , m_a3(0.13f)
        , m_a4(0.32f)
        , m_a5(0.01f)
        , m_a6(0.65f)
    {}

    // Sketch information
    std::string getName() const override {
        return "Mesh Sketch";
    }

    std::string getDescription() const override {
        return "Basic mesh rendering demonstration";
    }

    std::string getAuthor() const override {
        return "alice2";
    }

    void setup() override {
        scene().setBackgroundColor(Color(0.1f, 0.1f, 0.1f));
        scene().setShowGrid(true);
        scene().setGridSize(50.0f);
        scene().setGridDivisions(10);
        scene().setShowAxes(true);
        scene().setAxesLength(25.0f);

        // Create the scalar field
        m_scalarField = std::make_unique<ScalarField3D>(
            Vec3(-25, -25, -25), Vec3(25, 25, 25),
            m_fieldResolution, m_fieldResolution, m_fieldResolution);

        // Create mesh object
        m_meshObject = std::make_shared<MeshObject>();
        scene().addObject(m_meshObject);

        generateField();

        std::cout << "3D Scalar Field: Setup complete" << std::endl;
    }

    void update(float deltaTime) override {
        m_time += deltaTime;

        // If in “nodal” mode, update periodically
        if (b_computeNoise) {
            m_a1 = 0.5f * std::sin(m_time + 0.0f);
            m_a2 = 0.5f * std::sin(m_time + 1.0f);
            m_a3 = 0.5f * std::sin(m_time + 2.0f);
            m_a4 = 0.5f * std::sin(m_time + 3.0f);
            m_a5 = 0.5f * std::sin(m_time + 4.0f);
            m_a6 = 0.5f * std::sin(m_time + 5.0f);
            generateField();
        }
    }

    void draw(Renderer& renderer, Camera& camera) override {
        if (d_drawField && m_scalarField) {
            m_scalarField->draw_points(renderer, 4);
        }
        if (d_drawSlice && m_scalarField) {
            m_scalarField->draw_slice(renderer, m_currentSlice, 3.0f);
        }

        if (m_meshObject) {
            m_meshObject->setShowFaces(d_drawMesh);
            m_meshObject->setShowEdges(d_drawWireframe);
            m_meshObject->setRenderMode(MeshRenderMode::NormalShaded);
            m_meshObject->setNormalShadingColors(Color(0.8f, 0.2f, 0.8f),Color(1.0f, 1.0f, 1.0f)); // Magentq to white
        }

        drawUI(renderer);
    }

    void drawUI(Renderer& renderer) {
        float y = 20.0f;
        renderer.drawString("3D Scalar Field Controls:", 10.0f, y); y += 20;
        renderer.drawString("1: Sphere   2: Box   3: Torus   4: Nodal Surface", 10.0f, y); y += 20;
        renderer.drawString("F: Toggle field points   M: Toggle mesh   W: Toggle wireframe", 10.0f, y); y += 20;
        renderer.drawString("S: Toggle slice   l/k: Adjust isolevel (" + std::to_string(m_isolevel) + ")", 10.0f, y); y += 20;
        renderer.drawString("[/]: Change slice (" + std::to_string(m_currentSlice) + ")", 10.0f, y); y += 30;

        // Show which mode we’re in
        const char* mode = b_computeSphere ? "Sphere"
                        : b_computeBox    ? "Box"
                        : b_computeTorus  ? "Torus"
                        : "Nodal Surface";
        renderer.drawString(std::string("Mode: ") + mode, 10.0f, y); y += 30;

        // If nodal, display parameter values
        if (b_computeNoise) {
            renderer.drawString("a1=" + std::to_string(m_a1), 10.0f, y); y += 15;
            renderer.drawString("a2=" + std::to_string(m_a2), 10.0f, y); y += 15;
            renderer.drawString("a3=" + std::to_string(m_a3), 10.0f, y); y += 15;
            renderer.drawString("a4=" + std::to_string(m_a4), 10.0f, y); y += 15;
            renderer.drawString("a5=" + std::to_string(m_a5), 10.0f, y); y += 15;
            renderer.drawString("a6=" + std::to_string(m_a6), 10.0f, y); y += 15;
        }
    }

    bool onKeyPress(unsigned char key, int x, int y) override {
        switch (key) {
            case '1': b_computeSphere = true;  b_computeBox = b_computeTorus = b_computeNoise = false; generateField(); return true;
            case '2': b_computeBox    = true;  b_computeSphere = b_computeTorus = b_computeNoise = false; generateField(); return true;
            case '3': b_computeTorus  = true;  b_computeSphere = b_computeBox = b_computeNoise = false; generateField(); return true;
            case '4': b_computeNoise  = true;  b_computeSphere = b_computeBox = b_computeTorus = false; generateField(); return true;

            case 'f': case 'F': d_drawField     = !d_drawField;     return true;
            case 'm': case 'M': d_drawMesh      = !d_drawMesh;      return true;
            case 'w': case 'W': d_drawWireframe = !d_drawWireframe; return true;
            case 's': case 'S': d_drawSlice     = !d_drawSlice;     return true;

            case 'l': m_isolevel += 0.05f; generateField(); return true;
            case 'k': m_isolevel -= 0.05f; generateField(); return true;
            case '[': m_currentSlice = std::max(0, m_currentSlice - 1); return true;
            case ']': m_currentSlice = std::min(m_fieldResolution-1, m_currentSlice + 1); return true;
        }
        return false;
    }

    void generateMesh() {
        auto meshData = m_scalarField->generate_mesh(m_isolevel);
        m_meshObject->setMeshData(meshData);
    }

    void generateField() {
        if (!m_scalarField) return;
        m_scalarField->clear_field();

        m_scalarField = std::make_unique<ScalarField3D>(
        Vec3(-25, -25, -25), Vec3(25, 25, 25),
        m_fieldResolution, m_fieldResolution, m_fieldResolution);
        b_recreateField = true;

        if (b_computeSphere) {
            m_scalarField->apply_scalar_sphere(Vec3(0, 0, 0), m_sphereRadius);
        }
        else if (b_computeBox) {
            m_scalarField->apply_scalar_box(Vec3(0, 0, 0), Vec3(m_boxSize, m_boxSize, m_boxSize));
        }
        else if (b_computeTorus) {
            m_scalarField->apply_scalar_torus(Vec3(0, 0, 0), m_torusRadius, m_torusRadius * 0.3f);
        }
        else if (b_computeNoise) {
            if(b_recreateField){
                m_scalarField = std::make_unique<ScalarField3D>(
                Vec3(0, 0, 0), Vec3(4, 4, 4),
                m_fieldResolution, m_fieldResolution, m_fieldResolution);
                b_recreateField = false;
            }

            // --- Nodal Surface Formula ---
            // a1*sin(x)*sin(2y)*sin(3z)
            // + a2*sin(2x)*sin(y)*sin(3z)
            // + a3*sin(2x)*sin(3y)*sin(z)
            // + a4*sin(3x)*sin(y)*sin(2z)
            // + a5*sin(x)*sin(3y)*sin(2z)
            // + a6*sin(3x)*sin(2y)*sin(z) = 0
            const auto& pts = m_scalarField->get_points();
            std::vector<float> vals(pts.size());
            for (size_t i = 0; i < pts.size(); ++i) {
                float x = pts[i].x;
                float y = pts[i].y;
                float z = pts[i].z;
                float v = m_a1 * std::sin(    x) * std::sin(2*y) * std::sin(3*z)
                        + m_a2 * std::sin(2*x) * std::sin(   y) * std::sin(3*z)
                        + m_a3 * std::sin(2*x) * std::sin(3*y) * std::sin(   z)
                        + m_a4 * std::sin(3*x) * std::sin(   y) * std::sin(2*z)
                        + m_a5 * std::sin(   x) * std::sin(3*y) * std::sin(2*z)
                        + m_a6 * std::sin(3*x) * std::sin(2*y) * std::sin(   z);
                vals[i] = v;
            }
            m_scalarField->set_values(vals);
        }

        generateMesh();
    }
};

// Register the sketch
ALICE2_REGISTER_SKETCH_AUTO(ScalarField3DSketch)

#endif // __MAIN__
