#include "ShaderManager.h"
#include <iostream>
#include <fstream>
#include <sstream>

namespace alice2 {

    // ShaderProgram implementation
    ShaderProgram::ShaderProgram(GLuint programId) : m_programId(programId) {}

    ShaderProgram::~ShaderProgram() {
        if (m_programId != 0) {
            glDeleteProgram(m_programId);
        }
    }

    ShaderProgram::ShaderProgram(ShaderProgram&& other) noexcept 
        : m_programId(other.m_programId), m_uniformCache(std::move(other.m_uniformCache)) {
        other.m_programId = 0;
    }

    ShaderProgram& ShaderProgram::operator=(ShaderProgram&& other) noexcept {
        if (this != &other) {
            if (m_programId != 0) {
                glDeleteProgram(m_programId);
            }
            m_programId = other.m_programId;
            m_uniformCache = std::move(other.m_uniformCache);
            other.m_programId = 0;
        }
        return *this;
    }

    void ShaderProgram::use() const {
        if (m_programId != 0) {
            glUseProgram(m_programId);
        }
    }

    void ShaderProgram::setUniform(const std::string& name, int value) const {
        GLint location = getUniformLocation(name);
        if (location != -1) {
            glUniform1i(location, value);
        }
    }

    void ShaderProgram::setUniform(const std::string& name, float value) const {
        GLint location = getUniformLocation(name);
        if (location != -1) {
            glUniform1f(location, value);
        }
    }

    void ShaderProgram::setUniform(const std::string& name, const float* values, int count) const {
        GLint location = getUniformLocation(name);
        if (location != -1) {
            glUniform1fv(location, count, values);
        }
    }

    GLint ShaderProgram::getUniformLocation(const std::string& name) const {
        auto it = m_uniformCache.find(name);
        if (it != m_uniformCache.end()) {
            return it->second;
        }
        
        GLint location = glGetUniformLocation(m_programId, name.c_str());
        m_uniformCache[name] = location;
        return location;
    }

    void ShaderProgram::dispatch(GLuint numGroupsX, GLuint numGroupsY, GLuint numGroupsZ) const {
        if (m_programId != 0) {
            glDispatchCompute(numGroupsX, numGroupsY, numGroupsZ);
        }
    }

    void ShaderProgram::memoryBarrier(GLbitfield barriers) const {
        glMemoryBarrier(barriers);
    }

    // ShaderManager implementation
    ShaderManager::ShaderManager() : m_initialized(false) {}

    ShaderManager::~ShaderManager() {
        shutdown();
    }

    bool ShaderManager::initialize() {
        if (m_initialized) return true;

        // Check OpenGL version and extensions
        if (!GLEW_VERSION_3_0) {
            logError("OpenGL 3.0 or higher required for shader support");
            return false;
        }

        std::cout << "ShaderManager: OpenGL Version: " << getOpenGLVersion() << std::endl;
        std::cout << "ShaderManager: GLSL Version: " << getGLSLVersion() << std::endl;

        if (isComputeShaderSupported()) {
            std::cout << "ShaderManager: Compute shaders supported" << std::endl;
        } else {
            std::cout << "ShaderManager: Warning - Compute shaders not supported" << std::endl;
        }

        m_initialized = true;
        return true;
    }

    void ShaderManager::shutdown() {
        clearCache();
        m_initialized = false;
    }

    std::shared_ptr<ShaderProgram> ShaderManager::loadComputeShader(const std::string& name, const std::string& filePath) {
        if (!m_initialized) {
            logError("ShaderManager not initialized");
            return nullptr;
        }

        std::string source = loadShaderFile(filePath);
        if (source.empty()) {
            logError("Failed to load compute shader file: " + filePath);
            return nullptr;
        }

        auto shader = createComputeShader(name, source);
        if (shader) {
            m_shaderPaths[name] = filePath;
        }
        return shader;
    }

    std::shared_ptr<ShaderProgram> ShaderManager::createComputeShader(const std::string& name, const std::string& source) {
        if (!isComputeShaderSupported()) {
            logError("Compute shaders not supported on this system");
            return nullptr;
        }

        GLuint computeShader = compileShader(source, ShaderType::Compute);
        if (computeShader == 0) {
            return nullptr;
        }

        GLuint program = linkProgram({computeShader});
        glDeleteShader(computeShader);

        if (program == 0) {
            return nullptr;
        }

        auto shaderProgram = std::make_shared<ShaderProgram>(program);
        m_shaderCache[name] = shaderProgram;
        
        std::cout << "ShaderManager: Successfully created compute shader: " << name << std::endl;
        return shaderProgram;
    }

    std::shared_ptr<ShaderProgram> ShaderManager::getShader(const std::string& name) const {
        auto it = m_shaderCache.find(name);
        return (it != m_shaderCache.end()) ? it->second : nullptr;
    }

    bool ShaderManager::hasShader(const std::string& name) const {
        return m_shaderCache.find(name) != m_shaderCache.end();
    }

    void ShaderManager::removeShader(const std::string& name) {
        m_shaderCache.erase(name);
        m_shaderPaths.erase(name);
    }

    void ShaderManager::clearCache() {
        m_shaderCache.clear();
        m_shaderPaths.clear();
    }

    std::shared_ptr<ShaderProgram> ShaderManager::loadVertexFragmentShader(const std::string& name,
                                                                           const std::string& vertexPath,
                                                                           const std::string& fragmentPath) {
        if (!m_initialized) {
            logError("ShaderManager not initialized");
            return nullptr;
        }

        std::string vertexSource = loadShaderFile(vertexPath);
        std::string fragmentSource = loadShaderFile(fragmentPath);

        if (vertexSource.empty() || fragmentSource.empty()) {
            logError("Failed to load vertex/fragment shader files");
            return nullptr;
        }

        auto shader = createVertexFragmentShader(name, vertexSource, fragmentSource);
        if (shader) {
            m_shaderPaths[name] = vertexPath + ";" + fragmentPath; // Store both paths
        }
        return shader;
    }

    std::shared_ptr<ShaderProgram> ShaderManager::createVertexFragmentShader(const std::string& name,
                                                                             const std::string& vertexSource,
                                                                             const std::string& fragmentSource) {
        GLuint vertexShader = compileShader(vertexSource, ShaderType::Vertex);
        GLuint fragmentShader = compileShader(fragmentSource, ShaderType::Fragment);

        if (vertexShader == 0 || fragmentShader == 0) {
            if (vertexShader != 0) glDeleteShader(vertexShader);
            if (fragmentShader != 0) glDeleteShader(fragmentShader);
            return nullptr;
        }

        GLuint program = linkProgram({vertexShader, fragmentShader});
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        if (program == 0) {
            return nullptr;
        }

        auto shaderProgram = std::make_shared<ShaderProgram>(program);
        m_shaderCache[name] = shaderProgram;

        std::cout << "ShaderManager: Successfully created vertex/fragment shader: " << name << std::endl;
        return shaderProgram;
    }

    void ShaderManager::reloadShader(const std::string& name) {
        auto pathIt = m_shaderPaths.find(name);
        if (pathIt == m_shaderPaths.end()) {
            logError("Cannot reload shader - no path stored for: " + name);
            return;
        }

        // Remove old shader
        removeShader(name);

        // Check if it's a compute shader or vertex/fragment shader
        const std::string& paths = pathIt->second;
        if (paths.find(';') != std::string::npos) {
            // Vertex/Fragment shader
            size_t pos = paths.find(';');
            std::string vertexPath = paths.substr(0, pos);
            std::string fragmentPath = paths.substr(pos + 1);
            loadVertexFragmentShader(name, vertexPath, fragmentPath);
        } else {
            // Compute shader
            loadComputeShader(name, paths);
        }
    }

    void ShaderManager::reloadAllShaders() {
        std::vector<std::string> shaderNames;
        for (const auto& pair : m_shaderPaths) {
            shaderNames.push_back(pair.first);
        }

        for (const std::string& name : shaderNames) {
            reloadShader(name);
        }
    }

    bool ShaderManager::isComputeShaderSupported() {
        return GLEW_VERSION_4_3 || GLEW_ARB_compute_shader;
    }

    std::string ShaderManager::getOpenGLVersion() {
        const char* version = reinterpret_cast<const char*>(glGetString(GL_VERSION));
        return version ? std::string(version) : "Unknown";
    }

    std::string ShaderManager::getGLSLVersion() {
        const char* version = reinterpret_cast<const char*>(glGetString(GL_SHADING_LANGUAGE_VERSION));
        return version ? std::string(version) : "Unknown";
    }

    GLuint ShaderManager::compileShader(const std::string& source, ShaderType type) const {
        GLuint shader = glCreateShader(static_cast<GLenum>(type));
        const char* sourcePtr = source.c_str();
        glShaderSource(shader, 1, &sourcePtr, nullptr);
        glCompileShader(shader);

        if (!checkShaderCompileErrors(shader, "SHADER")) {
            glDeleteShader(shader);
            return 0;
        }

        return shader;
    }

    GLuint ShaderManager::linkProgram(const std::vector<GLuint>& shaders) const {
        GLuint program = glCreateProgram();
        
        for (GLuint shader : shaders) {
            glAttachShader(program, shader);
        }
        
        glLinkProgram(program);

        if (!checkProgramLinkErrors(program)) {
            glDeleteProgram(program);
            return 0;
        }

        return program;
    }

    std::string ShaderManager::loadShaderFile(const std::string& filePath) const {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            logError("Cannot open shader file: " + filePath);
            return "";
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    bool ShaderManager::checkShaderCompileErrors(GLuint shader, const std::string& type) const {
        GLint success;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        
        if (!success) {
            GLchar infoLog[1024];
            glGetShaderInfoLog(shader, 1024, nullptr, infoLog);
            logError("Shader compilation error (" + type + "): " + std::string(infoLog));
            return false;
        }
        return true;
    }

    bool ShaderManager::checkProgramLinkErrors(GLuint program) const {
        GLint success;
        glGetProgramiv(program, GL_LINK_STATUS, &success);
        
        if (!success) {
            GLchar infoLog[1024];
            glGetProgramInfoLog(program, 1024, nullptr, infoLog);
            logError("Program linking error: " + std::string(infoLog));
            return false;
        }
        return true;
    }

    void ShaderManager::logError(const std::string& message) const {
        std::cerr << "ShaderManager Error: " << message << std::endl;
    }

} // namespace alice2
