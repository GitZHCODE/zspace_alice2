#include "Renderer.h"
#include "Camera.h"
#include "FontRenderer.h"
#include "ShaderManager.h"
#include <iostream>
#include <cmath>

// Debug logging flag - set to true to enable detailed renderer logging
#define DEBUG_RENDERER_LOGGING false

namespace alice2 {

    Renderer::Renderer()
        : m_initialized(false)
        , m_viewportX(0)
        , m_viewportY(0)
        , m_viewportWidth(800)
        , m_viewportHeight(600)
        , m_currentColor(1.0f, 1.0f, 1.0f)
        , m_currentAlpha(1.0f)
        , m_wireframeMode(false)
        , m_pointSize(1.0f)
        , m_lineWidth(1.0f)
        , m_renderMode(RenderMode::Triangles)
        , m_lightingEnabled(true)
        , m_ambientLight(0.2f, 0.2f, 0.2f)
        , m_lightDirection(0.0f, -1.0f, -1.0f)
        , m_lightColor(1.0f, 1.0f, 1.0f)
        , m_fontRenderer(std::make_unique<FontRenderer>())
        , m_shaderManager(std::make_unique<ShaderManager>())
        , m_currentCamera(nullptr)
    {
    }

    Renderer::~Renderer() {
        shutdown();
    }

    bool Renderer::initialize() {
        if (m_initialized) return true;

        // Initialize OpenGL state
        setupOpenGL();

        // Initialize font renderer
        if (!m_fontRenderer->initialize()) {
            std::cerr << "Renderer: Failed to initialize FontRenderer" << std::endl;
            return false;
        }

        // Load default font
        if (!m_fontRenderer->loadDefaultFont(16.0f)) {
            std::cerr << "Renderer: Warning - Failed to load default font" << std::endl;
            // Continue anyway - text rendering will just be disabled
        }

        // Initialize shader manager
        if (!m_shaderManager->initialize()) {
            std::cerr << "Renderer: Failed to initialize ShaderManager" << std::endl;
            return false;
        }

        m_initialized = true;
        return true;
    }

    void Renderer::shutdown() {
        m_initialized = false;
    }

    void Renderer::beginFrame() {
        if (!m_initialized) return;
        
        // Clear matrix stack
        while (!m_matrixStack.empty()) {
            m_matrixStack.pop();
        }
        
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
    }

    void Renderer::endFrame() {
        if (!m_initialized) return;

        // Note: Buffer swapping is now handled by GLFW in the main loop
        checkErrors();
    }

    void Renderer::clear() {
        GLState::clear();
    }

    void Renderer::setViewport(int x, int y, int width, int height) {
        m_viewportX = x;
        m_viewportY = y;
        m_viewportWidth = width;
        m_viewportHeight = height;
        GLState::setViewport(x, y, width, height);
    }

    void Renderer::getViewport(int& x, int& y, int& width, int& height) const {
        x = m_viewportX;
        y = m_viewportY;
        width = m_viewportWidth;
        height = m_viewportHeight;
    }

    void Renderer::setCamera(Camera& camera) {
        if (DEBUG_RENDERER_LOGGING) {
            Vec3 pos = camera.getPosition();
            std::cout << "[RENDERER] setCamera: position=(" << pos.x << ", " << pos.y << ", " << pos.z << ")" << std::endl;
        }
        m_currentCamera = &camera;
        setupProjection(camera);
        setupView(camera);
    }

    void Renderer::setupProjection(Camera& camera) {
        if (DEBUG_RENDERER_LOGGING) {
            std::cout << "[RENDERER] setupProjection: Loading projection matrix" << std::endl;
        }
        glMatrixMode(GL_PROJECTION);
        GLMatrix::loadMatrix(camera.getProjectionMatrix());
    }

    void Renderer::setupView(Camera& camera) {
        if (DEBUG_RENDERER_LOGGING) {
            std::cout << "[RENDERER] setupView: Loading view matrix" << std::endl;
        }
        glMatrixMode(GL_MODELVIEW);
        GLMatrix::loadMatrix(camera.getViewMatrix());
        if (DEBUG_RENDERER_LOGGING) {
            std::cout << "[RENDERER] setupView: View matrix loaded" << std::endl;
        }
    }

    void Renderer::pushMatrix() {
        glPushMatrix();
        
        // Also maintain our own stack for queries
        GLfloat matrix[16];
        glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
        Mat4 currentMatrix;
        for (int i = 0; i < 16; i++) {
            currentMatrix.m[i] = matrix[i];
        }
        m_matrixStack.push(currentMatrix);
    }

    void Renderer::popMatrix() {
        glPopMatrix();
        
        if (!m_matrixStack.empty()) {
            m_matrixStack.pop();
        }
    }

    void Renderer::loadMatrix(const Mat4& matrix) {
        GLMatrix::loadMatrix(matrix);
    }

    void Renderer::multMatrix(const Mat4& matrix) {
        GLMatrix::multMatrix(matrix);
    }

    void Renderer::loadIdentity() {
        glLoadIdentity();
    }

    Mat4 Renderer::getCurrentModelViewMatrix() const {
        GLfloat matrix[16];
        glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
        Mat4 result;
        for (int i = 0; i < 16; i++) {
            result.m[i] = matrix[i];
        }
        return result;
    }

    Mat4 Renderer::getCurrentProjectionMatrix() const {
        GLfloat matrix[16];
        glGetFloatv(GL_PROJECTION_MATRIX, matrix);
        Mat4 result;
        for (int i = 0; i < 16; i++) {
            result.m[i] = matrix[i];
        }
        return result;
    }

    Mat4 Renderer::getCurrentModelViewProjectionMatrix() const {
        return getCurrentProjectionMatrix() * getCurrentModelViewMatrix();
    }

    void Renderer::setColor(const Vec3& color, float alpha) {
        m_currentColor = color;
        m_currentAlpha = alpha;
        glColor4f(color.x, color.y, color.z, alpha);
    }

    void Renderer::setWireframe(bool wireframe) {
        m_wireframeMode = wireframe;
        if (wireframe) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        } else {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }
    }

    void Renderer::setPointSize(float size) {
        m_pointSize = size;
        GLState::setPointSize(size);
    }

    void Renderer::setLineWidth(float width) {
        m_lineWidth = width;
        GLState::setLineWidth(width);
    }

    void Renderer::enableLighting(bool enable) {
        m_lightingEnabled = enable;
        if (enable) {
            glEnable(GL_LIGHTING);
            glEnable(GL_LIGHT0);
            
            // Set ambient light
            GLfloat ambient[] = { m_ambientLight.x, m_ambientLight.y, m_ambientLight.z, 1.0f };
            glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient);
            
            // Set directional light
            GLfloat lightPos[] = { -m_lightDirection.x, -m_lightDirection.y, -m_lightDirection.z, 0.0f };
            GLfloat lightColor[] = { m_lightColor.x, m_lightColor.y, m_lightColor.z, 1.0f };
            glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
            glLightfv(GL_LIGHT0, GL_DIFFUSE, lightColor);
            glLightfv(GL_LIGHT0, GL_SPECULAR, lightColor);
        } else {
            glDisable(GL_LIGHTING);
        }
    }

    void Renderer::setAmbientLight(const Vec3& color) {
        m_ambientLight = color;
        if (m_lightingEnabled) {
            GLfloat ambient[] = { color.x, color.y, color.z, 1.0f };
            glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient);
        }
    }

    void Renderer::setDirectionalLight(const Vec3& direction, const Vec3& color) {
        m_lightDirection = direction.normalized();
        m_lightColor = color;
        if (m_lightingEnabled) {
            GLfloat lightPos[] = { -m_lightDirection.x, -m_lightDirection.y, -m_lightDirection.z, 0.0f };
            GLfloat lightColor[] = { m_lightColor.x, m_lightColor.y, m_lightColor.z, 1.0f };
            glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
            glLightfv(GL_LIGHT0, GL_DIFFUSE, lightColor);
            glLightfv(GL_LIGHT0, GL_SPECULAR, lightColor);
        }
    }

    void Renderer::setRenderMode(RenderMode mode) {
        m_renderMode = mode;
        applyRenderMode();
    }

    void Renderer::drawPoint(const Vec3& position) {
        GLDraw::drawPoint(position, m_pointSize);
    }

    void Renderer::drawPoint(const Vec3& position, const Vec3& color, float size) {
        Vec3 oldColor = m_currentColor;
        float oldSize = m_pointSize;
        setColor(color);
        setPointSize(size);
        GLDraw::drawPoint(position, size);
        setColor(oldColor);
        setPointSize(oldSize);
    }

    void Renderer::drawLine(const Vec3& start, const Vec3& end) {
        GLDraw::drawLine(start, end);
    }

    void Renderer::drawLine(const Vec3& start, const Vec3& end, const Vec3& color, float width) {
        Vec3 oldColor = m_currentColor;
        float oldWidth = m_lineWidth;
        setColor(color);
        setLineWidth(width);
        GLDraw::drawLine(start, end);
        setColor(oldColor);
        setLineWidth(oldWidth);
    }

    void Renderer::drawTriangle(const Vec3& v1, const Vec3& v2, const Vec3& v3) {
        glBegin(GL_TRIANGLES);
        glVertex3f(v1.x, v1.y, v1.z);
        glVertex3f(v2.x, v2.y, v2.z);
        glVertex3f(v3.x, v3.y, v3.z);
        glEnd();
    }

    void Renderer::drawTriangle(const Vec3& v1, const Vec3& v2, const Vec3& v3, const Vec3& color) {
        Vec3 oldColor = m_currentColor;
        setColor(color);
        glBegin(GL_TRIANGLES);
        glVertex3f(v1.x, v1.y, v1.z);
        glVertex3f(v2.x, v2.y, v2.z);
        glVertex3f(v3.x, v3.y, v3.z);
        glEnd();
        setColor(oldColor);
    }

    void Renderer::drawQuad(const Vec3& v1, const Vec3& v2, const Vec3& v3, const Vec3& v4) {
        glBegin(GL_QUADS);
        glVertex3f(v1.x, v1.y, v1.z);
        glVertex3f(v2.x, v2.y, v2.z);
        glVertex3f(v3.x, v3.y, v3.z);
        glVertex3f(v4.x, v4.y, v4.z);
        glEnd();
    }

    void Renderer::drawQuad(const Vec3& v1, const Vec3& v2, const Vec3& v3, const Vec3& v4, const Vec3& color) {
        Vec3 oldColor = m_currentColor;
        setColor(color);
        glBegin(GL_QUADS);
        glVertex3f(v1.x, v1.y, v1.z);
        glVertex3f(v2.x, v2.y, v2.z);
        glVertex3f(v3.x, v3.y, v3.z);
        glVertex3f(v4.x, v4.y, v4.z);
        glEnd();
        setColor(oldColor);
    }

    void Renderer::drawCube(float size) {
        GLDraw::drawWireCube(size);
    }

    void Renderer::drawCube(float size, const Vec3& color) {
        Vec3 oldColor = m_currentColor;
        setColor(color);
        GLDraw::drawWireCube(size);
        setColor(oldColor);
    }

    void Renderer::drawSphere(float radius, int segments) {
        // TODO: Implement custom sphere rendering without GLUT
        // For now, draw a simple wireframe cube as placeholder
        drawCube(radius * 2.0f);
    }

    void Renderer::drawSphere(float radius, int segments, const Vec3& color) {
        Vec3 oldColor = m_currentColor;
        setColor(color);
        // TODO: Implement custom sphere rendering without GLUT
        // For now, draw a simple wireframe cube as placeholder
        GLDraw::drawWireCube(radius * 2.0f);
        setColor(oldColor);
    }

    void Renderer::drawCylinder(float radius, float height, int segments) {
        // Simple cylinder implementation using GLUT
        GLUquadric* quad = gluNewQuadric();
        if (quad) {
            glPushMatrix();
            glTranslatef(0, -height * 0.5f, 0);
            gluCylinder(quad, radius, radius, height, segments, 1);
            gluDeleteQuadric(quad);
            glPopMatrix();
        }
    }

    void Renderer::drawCylinder(float radius, float height, int segments, const Vec3& color) {
        Vec3 oldColor = m_currentColor;
        setColor(color);
        // Simple cylinder implementation using GLUT
        GLUquadric* quad = gluNewQuadric();
        if (quad) {
            glPushMatrix();
            glTranslatef(0, -height * 0.5f, 0);
            gluCylinder(quad, radius, radius, height, segments, 1);
            gluDeleteQuadric(quad);
            glPopMatrix();
        }
        setColor(oldColor);
    }

    void Renderer::drawGrid(float size, int divisions, const Vec3& color) {
        Vec3 oldColor = m_currentColor;
        setColor(color);
        GLDraw::drawGrid(size, divisions, color);
        setColor(oldColor);
    }

    void Renderer::drawAxes(float length) {
        GLDraw::drawAxes(length);
    }

    void Renderer::drawAxes(float length, const Vec3& color) {
        Vec3 oldColor = m_currentColor;
        setColor(color);
        GLDraw::drawAxes(length);
        setColor(oldColor);
    }

    void Renderer::setupOpenGL() {
        // Enable depth testing
        GLState::enableDepthTest();
        
        // Enable blending for transparency
        GLState::enableBlending();
        
        // Enable multisampling for anti-aliasing
        GLState::enableMultisampling();
        
        // Set default clear color
        GLState::setClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        
        // Enable smooth points and lines
        glEnable(GL_POINT_SMOOTH);
        glEnable(GL_LINE_SMOOTH);
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
        
        // Set default material properties
        GLfloat mat_ambient[] = { 0.2f, 0.2f, 0.2f, 1.0f };
        GLfloat mat_diffuse[] = { 0.8f, 0.8f, 0.8f, 1.0f };
        GLfloat mat_specular[] = { 0.5f, 0.5f, 0.5f, 1.0f };
        GLfloat mat_shininess[] = { 50.0f };
        
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient);
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);
        
        // Enable color material
        glEnable(GL_COLOR_MATERIAL);
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

        // Enable V-sync 1 -> Disable 0
        glfwSwapInterval(1);
    }

    void Renderer::applyRenderMode() {
        switch (m_renderMode) {
            case RenderMode::Points:
                glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
                break;
            case RenderMode::Lines:
            case RenderMode::Wireframe:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                break;
            case RenderMode::Triangles:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                break;
        }
    }

    void Renderer::checkErrors() const {
        checkGLError("Renderer");
    }

    void Renderer::drawText(const std::string& text, const Vec3& position, float size) {
        if (!m_initialized || !m_fontRenderer || !m_fontRenderer->isInitialized()) {
            return;
        }

        m_fontRenderer->drawText(text, position, size, m_currentColor, m_currentAlpha);
    }



    void Renderer::drawString(const std::string& text, float x, float y) {
        if (!m_initialized || !m_fontRenderer || !m_fontRenderer->isInitialized()) {
            return;
        }

        m_fontRenderer->drawString(text, x, y, m_currentColor, m_currentAlpha);
    }

    void Renderer::drawPoints(const Vec3* points, int count) {
        if (!points || count <= 0) return;

        glBegin(GL_POINTS);
        for (int i = 0; i < count; i++) {
            glVertex3f(points[i].x, points[i].y, points[i].z);
        }
        glEnd();
    }

    void Renderer::drawLines(const Vec3* points, int count) {
        if (!points || count <= 0) return;

        glBegin(GL_LINES);
        for (int i = 0; i < count; i++) {
            glVertex3f(points[i].x, points[i].y, points[i].z);
        }
        glEnd();
    }

    void Renderer::drawTriangles(const Vec3* vertices, int vertexCount, const int* indices, int indexCount) {
        if (!vertices || vertexCount <= 0) return;

        glBegin(GL_TRIANGLES);

        if (indices && indexCount > 0) {
            // Use indexed rendering
            for (int i = 0; i < indexCount; i++) {
                int index = indices[i];
                if (index >= 0 && index < vertexCount) {
                    glVertex3f(vertices[index].x, vertices[index].y, vertices[index].z);
                }
            }
        } else {
            // Use direct vertex array
            for (int i = 0; i < vertexCount; i++) {
                glVertex3f(vertices[i].x, vertices[i].y, vertices[i].z);
            }
        }

        glEnd();
    }

    void Renderer::drawMesh(const Vec3* vertices, const Vec3* normals, const Vec3* colors, int vertexCount,
                           const int* indices, int indexCount, bool enableLighting) {
        if (!vertices || vertexCount <= 0 || !m_shaderManager) return;

        auto meshShader = m_shaderManager->getMeshShader();
        if (!meshShader || !meshShader->isValid()) {
            // Fallback to immediate mode rendering
            drawTriangles(vertices, vertexCount, indices, indexCount);
            return;
        }

        meshShader->use();

        // Get current matrices from OpenGL state
        Mat4 modelViewMatrix = getCurrentModelViewMatrix();
        Mat4 projectionMatrix = getCurrentProjectionMatrix();
        Mat4 mvpMatrix = projectionMatrix * modelViewMatrix;

        // DIAGNOSTIC: Print matrix values (limited output)
        static int debugCallCount = 0;
        if (debugCallCount < 1) {
            std::cout << "[RENDERER] Matrix diagnostic: MVP[0]=" << mvpMatrix.m[0] << std::endl;
            debugCallCount++;
        }

        // Calculate normal matrix (inverse transpose of upper-left 3x3 of model-view matrix)
        // For uniform scaling, we can use the model-view matrix directly
        Mat4 normalMatrix = modelViewMatrix;

        // Set uniforms
        meshShader->setUniform("u_modelViewProjectionMatrix", mvpMatrix);
        meshShader->setUniform("u_modelViewMatrix", modelViewMatrix);
        meshShader->setUniform("u_normalMatrix", normalMatrix);

        meshShader->setUniform("u_lightDirection", m_lightDirection);
        meshShader->setUniform("u_lightColor", m_lightColor);
        meshShader->setUniform("u_ambientLight", m_ambientLight);
        meshShader->setUniform("u_enableLighting", enableLighting);

        // Get attribute locations
        GLint positionLoc = meshShader->getAttributeLocation("a_position");
        GLint normalLoc = meshShader->getAttributeLocation("a_normal");
        GLint colorLoc = meshShader->getAttributeLocation("a_color");

        // DIAGNOSTIC: Print attribute locations (limited output)
        static int attrDebugCount = 0;
        if (attrDebugCount < 1) {
            std::cout << "[RENDERER] Attribute locations: position=" << positionLoc
                      << ", normal=" << normalLoc << ", color=" << colorLoc << std::endl;
            attrDebugCount++;
        }

        // Enable vertex attributes
        if (positionLoc != -1) {
            glEnableVertexAttribArray(positionLoc);
            glVertexAttribPointer(positionLoc, 3, GL_FLOAT, GL_FALSE, 0, vertices);
        }

        if (normalLoc != -1 && normals) {
            glEnableVertexAttribArray(normalLoc);
            glVertexAttribPointer(normalLoc, 3, GL_FLOAT, GL_FALSE, 0, normals);
        }

        if (colorLoc != -1 && colors) {
            glEnableVertexAttribArray(colorLoc);
            glVertexAttribPointer(colorLoc, 3, GL_FLOAT, GL_FALSE, 0, colors);
        }

        // Draw
        std::cout << "[RENDERER] Drawing mesh: vertexCount=" << vertexCount << ", indexCount=" << indexCount << std::endl;
        if (indices && indexCount > 0) {
            std::cout << "[RENDERER] Using glDrawElements with " << indexCount << " indices" << std::endl;
            glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, indices);
        } else {
            std::cout << "[RENDERER] Using glDrawArrays with " << vertexCount << " vertices" << std::endl;
            glDrawArrays(GL_TRIANGLES, 0, vertexCount);
        }

        // Check for OpenGL errors
        GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            std::cout << "[RENDERER] OpenGL error after draw: " << error << std::endl;
        }

        // Disable vertex attributes
        if (positionLoc != -1) glDisableVertexAttribArray(positionLoc);
        if (normalLoc != -1) glDisableVertexAttribArray(normalLoc);
        if (colorLoc != -1) glDisableVertexAttribArray(colorLoc);

        meshShader->unuse();
    }

    void Renderer::drawMeshWireframe(const Vec3* vertices, const Vec3* colors, int vertexCount,
                                    const int* indices, int indexCount) {
        // Debug output (limited)
        static int wireframeCallCount = 0;
        if (wireframeCallCount < 1) {
            std::cout << "[RENDERER] drawMeshWireframe called: vertexCount=" << vertexCount << ", indexCount=" << indexCount << std::endl;
            wireframeCallCount++;
        }

        if (!vertices || vertexCount <= 0 || !m_shaderManager) {
            std::cout << "[RENDERER] drawMeshWireframe: Early return - vertices=" << (vertices ? "valid" : "null")
                      << ", vertexCount=" << vertexCount << ", shaderManager=" << (m_shaderManager ? "valid" : "null") << std::endl;
            return;
        }

        auto wireframeShader = m_shaderManager->getWireframeShader();
        if (!wireframeShader || !wireframeShader->isValid()) {
            std::cout << "[RENDERER] drawMeshWireframe: Wireframe shader invalid, using fallback" << std::endl;
            // Fallback to immediate mode rendering
            setWireframe(true);
            drawTriangles(vertices, vertexCount, indices, indexCount);
            return;
        }

        std::cout << "[RENDERER] drawMeshWireframe: Using wireframe shader" << std::endl;

        wireframeShader->use();

        // Get current matrices from OpenGL state
        Mat4 modelViewMatrix = getCurrentModelViewMatrix();
        Mat4 projectionMatrix = getCurrentProjectionMatrix();
        Mat4 mvpMatrix = projectionMatrix * modelViewMatrix;

        // Set uniforms
        wireframeShader->setUniform("u_modelViewProjectionMatrix", mvpMatrix);

        // Get attribute locations
        GLint positionLoc = wireframeShader->getAttributeLocation("a_position");
        GLint colorLoc = wireframeShader->getAttributeLocation("a_color");

        // DIAGNOSTIC: Print attribute locations
        static int wireframeAttrDebugCount = 0;
        if (wireframeAttrDebugCount < 3) {
            std::cout << "[RENDERER] Wireframe attribute locations: position=" << positionLoc
                      << ", color=" << colorLoc << std::endl;
            wireframeAttrDebugCount++;
        }

        // Enable vertex attributes
        if (positionLoc != -1) {
            glEnableVertexAttribArray(positionLoc);
            glVertexAttribPointer(positionLoc, 3, GL_FLOAT, GL_FALSE, 0, vertices);
        }

        if (colorLoc != -1 && colors) {
            glEnableVertexAttribArray(colorLoc);
            glVertexAttribPointer(colorLoc, 3, GL_FLOAT, GL_FALSE, 0, colors);
        }

        // Set wireframe mode
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        // Draw
        std::cout << "[RENDERER] drawMeshWireframe: Drawing..." << std::endl;
        if (indices && indexCount > 0) {
            std::cout << "[RENDERER] drawMeshWireframe: Using glDrawElements with " << indexCount << " indices" << std::endl;
            glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, indices);
        } else {
            std::cout << "[RENDERER] drawMeshWireframe: Using glDrawArrays with " << vertexCount << " vertices" << std::endl;
            glDrawArrays(GL_TRIANGLES, 0, vertexCount);
        }

        // Check for OpenGL errors
        GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            std::cout << "[RENDERER] drawMeshWireframe: OpenGL error after draw: " << error << std::endl;
        }

        // Restore fill mode
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        // Disable vertex attributes
        if (positionLoc != -1) glDisableVertexAttribArray(positionLoc);
        if (colorLoc != -1) glDisableVertexAttribArray(colorLoc);

        wireframeShader->unuse();
    }

} // namespace alice2
