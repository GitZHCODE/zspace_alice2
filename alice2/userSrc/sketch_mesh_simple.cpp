// Simple Triangle Test - Testing basic mesh rendering with position output
// This sketch creates a simple triangle and outputs transformed positions

#include "../include/alice2.h"
#include <memory>

using namespace alice2;

class SimpleMeshSketch : public ISketch {
private:
    std::shared_ptr<Scene> m_scene;
    std::shared_ptr<MeshObject> m_triangle;
    std::shared_ptr<PrimitiveObject> m_referenceCube;
    float m_time;
    bool m_useImmediateMode;

public:
    void setup() override {
        m_time = 0.0f;
        m_useImmediateMode = false;

        // Create scene
        m_scene = std::make_shared<Scene>();

        // Create a simple triangle mesh
        m_triangle = std::make_shared<MeshObject>("TestTriangle");
        createSimpleTriangle();
        m_triangle->setRenderMode(MeshRenderMode::Lit);
        m_triangle->setColor(Vec3(1.0f, 0.0f, 0.0f)); // Bright red
        m_triangle->getTransform().setPosition(Vec3(0.0f, 0.0f, 0.0f)); // At origin
        m_scene->addObject(m_triangle);

        // REFERENCE: Add a primitive cube for comparison
        m_referenceCube = std::make_shared<PrimitiveObject>(PrimitiveType::Cube, "ReferenceCube");
        m_referenceCube->setColor(Vec3(1.0f, 1.0f, 0.0f)); // Yellow
        m_referenceCube->getTransform().setPosition(Vec3(3.0f, 0.0f, 0.0f));
        m_referenceCube->getTransform().setScale(Vec3(1.0f, 1.0f, 1.0f));
        m_scene->addObject(m_referenceCube);
    }

    void createSimpleTriangle() {
        if (!m_triangle) return;

        // Create mesh data manually
        auto meshData = std::make_shared<MeshData>();

        // Create 3 vertices for a triangle (moved to z = -5 to be clearly in front of camera)
        meshData->vertices = {
            MeshVertex(Vec3(-2.0f, -2.0f, -5.0f), Vec3(0, 0, 1), Vec3(1, 0, 0)), // Red vertex
            MeshVertex(Vec3( 2.0f, -2.0f, -5.0f), Vec3(0, 0, 1), Vec3(0, 1, 0)), // Green vertex
            MeshVertex(Vec3( 0.0f,  2.0f, -5.0f), Vec3(0, 0, 1), Vec3(0, 0, 1))  // Blue vertex
        };

        // Create one triangular face
        meshData->faces = {
            MeshFace({0, 1, 2}, Vec3(0, 0, 1), Vec3(1, 1, 1)) // White face
        };

        // Set the mesh data
        m_triangle->setMeshData(meshData);

        // Print original vertex positions
        std::cout << "\n=== TRIANGLE VERTEX POSITIONS (LOCAL SPACE) ===" << std::endl;
        for (size_t i = 0; i < meshData->vertices.size(); i++) {
            const Vec3& pos = meshData->vertices[i].position;
            const Vec3& color = meshData->vertices[i].color;
            std::cout << "Vertex " << i << ": pos(" << pos.x << ", " << pos.y << ", " << pos.z
                      << ") color(" << color.x << ", " << color.y << ", " << color.z << ")" << std::endl;
        }
    }

    void update(float deltaTime) override {
        m_time += deltaTime;

        // Rotate triangle for visual confirmation
        if (m_triangle) {
            m_triangle->getTransform().setRotation(Quaternion::fromAxisAngle(Vec3(0, 0, 1), m_time * 0.5f));
        }

        // Update scene
        if (m_scene) {
            m_scene->update(deltaTime);
        }
    }

    void draw(Renderer& renderer, Camera& camera) override {
        // Print transformation matrices for debugging
        static int frameCount = 0;
        if (frameCount < 3) {
            printTransformationInfo(renderer, camera);
            frameCount++;
        }

        // Test immediate mode rendering if enabled
        if (m_useImmediateMode) {
            drawImmediateModeTriangle(renderer, camera);
        }

        // Render the scene
        if (m_scene) {
            m_scene->render(renderer, camera);
        }

        // Draw diagnostic info
        renderer.drawString("SIMPLE TRIANGLE TEST", 10, 10);
        renderer.drawString("Red triangle (mesh) - Yellow cube (primitive reference)", 10, 30);

        // Camera position info
        Vec3 camPos = camera.getPosition();
        std::string posStr = "Camera pos: (" + std::to_string(camPos.x) + ", " +
                            std::to_string(camPos.y) + ", " + std::to_string(camPos.z) + ")";
        renderer.drawString(posStr, 10, 50);

        // Toggle instructions
        renderer.drawString("Press 'I' to toggle immediate mode rendering", 10, 70);
        renderer.drawString("Immediate mode: " + std::string(m_useImmediateMode ? "ON" : "OFF"), 10, 90);
    }

    void printTransformationInfo(Renderer& renderer, Camera& camera) {
        std::cout << "\n=== TRANSFORMATION MATRICES DEBUG ===" << std::endl;

        // Get camera matrices
        Mat4 viewMatrix = camera.getViewMatrix();
        Mat4 projMatrix = camera.getProjectionMatrix();

        std::cout << "Camera position: (" << camera.getPosition().x << ", "
                  << camera.getPosition().y << ", " << camera.getPosition().z << ")" << std::endl;

        // Get triangle transform
        if (m_triangle) {
            Mat4 worldMatrix = m_triangle->getTransform().getWorldMatrix();
            Mat4 mvpMatrix = projMatrix * viewMatrix * worldMatrix;

            std::cout << "Triangle world matrix[0-3]: " << worldMatrix.m[0] << ", " << worldMatrix.m[1]
                      << ", " << worldMatrix.m[2] << ", " << worldMatrix.m[3] << std::endl;
            std::cout << "MVP matrix[0-3]: " << mvpMatrix.m[0] << ", " << mvpMatrix.m[1]
                      << ", " << mvpMatrix.m[2] << ", " << mvpMatrix.m[3] << std::endl;

            // Transform triangle vertices to clip space
            std::cout << "\n=== TRANSFORMED VERTEX POSITIONS ===" << std::endl;
            Vec3 vertices[3] = {
                Vec3(-2.0f, -2.0f, -5.0f),
                Vec3( 2.0f, -2.0f, -5.0f),
                Vec3( 0.0f,  2.0f, -5.0f)
            };

            for (int i = 0; i < 3; i++) {
                Vec3 worldPos = worldMatrix.transformPoint(vertices[i]);
                Vec3 clipPos = mvpMatrix.transformPoint(vertices[i]);

                std::cout << "Vertex " << i << ":" << std::endl;
                std::cout << "  Local: (" << vertices[i].x << ", " << vertices[i].y << ", " << vertices[i].z << ")" << std::endl;
                std::cout << "  World: (" << worldPos.x << ", " << worldPos.y << ", " << worldPos.z << ")" << std::endl;
                std::cout << "  Clip:  (" << clipPos.x << ", " << clipPos.y << ", " << clipPos.z << ")" << std::endl;
            }
        }
    }

    void drawImmediateModeTriangle(Renderer& renderer, Camera& camera) {
        // Draw triangle using immediate mode OpenGL for comparison
        renderer.pushMatrix();

        glBegin(GL_TRIANGLES);
        glColor3f(1.0f, 0.0f, 0.0f); glVertex3f(-2.0f, -2.0f, -5.0f);
        glColor3f(0.0f, 1.0f, 0.0f); glVertex3f( 2.0f, -2.0f, -5.0f);
        glColor3f(0.0f, 0.0f, 1.0f); glVertex3f( 0.0f,  2.0f, -5.0f);
        glEnd();

        renderer.popMatrix();
    }

    bool onKeyPress(unsigned char key, int /*x*/, int /*y*/) override {
        if (key == 'I' || key == 'i') {
            m_useImmediateMode = !m_useImmediateMode;
            std::cout << "Immediate mode: " << (m_useImmediateMode ? "ON" : "OFF") << std::endl;
            return true;
        }
        return false;
    }

    std::string getName() const override {
        return "Simple Triangle Test";
    }
};

ALICE2_REGISTER_SKETCH_AUTO(SimpleMeshSketch)
