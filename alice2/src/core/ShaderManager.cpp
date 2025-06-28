#include "ShaderManager.h"
#include <iostream>
#include <fstream>
#include <sstream>

namespace alice2 {

    // ShaderProgram implementation
    ShaderProgram::ShaderProgram()
        : m_programId(0)
        , m_vertexShaderId(0)
        , m_fragmentShaderId(0)
    {
    }

    ShaderProgram::~ShaderProgram() {
        if (m_programId != 0) {
            glDeleteProgram(m_programId);
        }
        if (m_vertexShaderId != 0) {
            glDeleteShader(m_vertexShaderId);
        }
        if (m_fragmentShaderId != 0) {
            glDeleteShader(m_fragmentShaderId);
        }
    }

    bool ShaderProgram::loadFromFiles(const std::string& vertexPath, const std::string& fragmentPath) {
        std::string vertexSource = loadShaderFile(vertexPath);
        std::string fragmentSource = loadShaderFile(fragmentPath);
        
        if (vertexSource.empty() || fragmentSource.empty()) {
            return false;
        }
        
        return loadFromStrings(vertexSource, fragmentSource);
    }

    bool ShaderProgram::loadFromStrings(const std::string& vertexSource, const std::string& fragmentSource) {
        // Create shader objects
        m_vertexShaderId = glCreateShader(GL_VERTEX_SHADER);
        m_fragmentShaderId = glCreateShader(GL_FRAGMENT_SHADER);
        
        if (m_vertexShaderId == 0 || m_fragmentShaderId == 0) {
            std::cerr << "Failed to create shader objects" << std::endl;
            return false;
        }
        
        // Compile shaders
        if (!compileShader(m_vertexShaderId, vertexSource)) {
            std::cerr << "Failed to compile vertex shader" << std::endl;
            return false;
        }
        
        if (!compileShader(m_fragmentShaderId, fragmentSource)) {
            std::cerr << "Failed to compile fragment shader" << std::endl;
            return false;
        }
        
        // Create and link program
        m_programId = glCreateProgram();
        if (m_programId == 0) {
            std::cerr << "Failed to create shader program" << std::endl;
            return false;
        }
        
        glAttachShader(m_programId, m_vertexShaderId);
        glAttachShader(m_programId, m_fragmentShaderId);
        
        if (!linkProgram()) {
            std::cerr << "Failed to link shader program" << std::endl;
            return false;
        }
        
        return true;
    }

    void ShaderProgram::use() {
        if (m_programId != 0) {
            glUseProgram(m_programId);
        }
    }

    void ShaderProgram::unuse() {
        glUseProgram(0);
    }

    void ShaderProgram::setUniform(const std::string& name, float value) {
        GLint location = getCachedUniformLocation(name);
        if (location != -1) {
            glUniform1f(location, value);
        }
    }

    void ShaderProgram::setUniform(const std::string& name, int value) {
        GLint location = getCachedUniformLocation(name);
        if (location != -1) {
            glUniform1i(location, value);
        }
    }

    void ShaderProgram::setUniform(const std::string& name, bool value) {
        setUniform(name, value ? 1 : 0);
    }

    void ShaderProgram::setUniform(const std::string& name, const Vec3& value) {
        GLint location = getCachedUniformLocation(name);
        if (location != -1) {
            glUniform3f(location, value.x, value.y, value.z);
        }
    }

    void ShaderProgram::setUniform(const std::string& name, const Mat4& value) {
        GLint location = getCachedUniformLocation(name);
        if (location != -1) {
            glUniformMatrix4fv(location, 1, GL_FALSE, value.m);
        }
    }

    void ShaderProgram::setUniform(const std::string& name, float x, float y, float z) {
        GLint location = getCachedUniformLocation(name);
        if (location != -1) {
            glUniform3f(location, x, y, z);
        }
    }

    void ShaderProgram::setUniform(const std::string& name, float x, float y, float z, float w) {
        GLint location = getCachedUniformLocation(name);
        if (location != -1) {
            glUniform4f(location, x, y, z, w);
        }
    }

    GLint ShaderProgram::getAttributeLocation(const std::string& name) {
        return getCachedAttributeLocation(name);
    }

    GLint ShaderProgram::getUniformLocation(const std::string& name) {
        return getCachedUniformLocation(name);
    }

    bool ShaderProgram::compileShader(GLuint shaderId, const std::string& source) {
        const char* sourcePtr = source.c_str();
        glShaderSource(shaderId, 1, &sourcePtr, nullptr);
        glCompileShader(shaderId);
        
        GLint success;
        glGetShaderiv(shaderId, GL_COMPILE_STATUS, &success);
        if (!success) {
            checkCompileErrors(shaderId, "SHADER");
            return false;
        }
        
        return true;
    }

    bool ShaderProgram::linkProgram() {
        glLinkProgram(m_programId);
        
        GLint success;
        glGetProgramiv(m_programId, GL_LINK_STATUS, &success);
        if (!success) {
            checkLinkErrors();
            return false;
        }
        
        return true;
    }

    std::string ShaderProgram::loadShaderFile(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Failed to open shader file: " << filepath << std::endl;
            return "";
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    void ShaderProgram::checkCompileErrors(GLuint shader, const std::string& type) {
        GLint maxLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);
        
        if (maxLength > 0) {
            std::vector<GLchar> infoLog(maxLength);
            glGetShaderInfoLog(shader, maxLength, &maxLength, &infoLog[0]);
            std::cerr << "Shader compilation error (" << type << "):\n" << &infoLog[0] << std::endl;
        }
    }

    void ShaderProgram::checkLinkErrors() {
        GLint maxLength = 0;
        glGetProgramiv(m_programId, GL_INFO_LOG_LENGTH, &maxLength);
        
        if (maxLength > 0) {
            std::vector<GLchar> infoLog(maxLength);
            glGetProgramInfoLog(m_programId, maxLength, &maxLength, &infoLog[0]);
            std::cerr << "Shader linking error:\n" << &infoLog[0] << std::endl;
        }
    }

    GLint ShaderProgram::getCachedUniformLocation(const std::string& name) {
        auto it = m_uniformLocations.find(name);
        if (it != m_uniformLocations.end()) {
            return it->second;
        }
        
        GLint location = glGetUniformLocation(m_programId, name.c_str());
        m_uniformLocations[name] = location;
        return location;
    }

    GLint ShaderProgram::getCachedAttributeLocation(const std::string& name) {
        auto it = m_attributeLocations.find(name);
        if (it != m_attributeLocations.end()) {
            return it->second;
        }
        
        GLint location = glGetAttribLocation(m_programId, name.c_str());
        m_attributeLocations[name] = location;
        return location;
    }

    // ShaderManager implementation
    ShaderManager::ShaderManager()
        : m_initialized(false)
    {
    }

    ShaderManager::~ShaderManager() {
        shutdown();
    }

    bool ShaderManager::initialize() {
        if (m_initialized) return true;
        
        // Load built-in shaders
        if (!loadBuiltinShaders()) {
            std::cerr << "Failed to load built-in shaders" << std::endl;
            return false;
        }
        
        m_initialized = true;
        return true;
    }

    void ShaderManager::shutdown() {
        clearShaders();
        m_initialized = false;
    }

    std::shared_ptr<ShaderProgram> ShaderManager::loadShader(const std::string& name, 
                                                             const std::string& vertexPath, 
                                                             const std::string& fragmentPath) {
        auto shader = std::make_shared<ShaderProgram>();
        if (!shader->loadFromFiles(vertexPath, fragmentPath)) {
            return nullptr;
        }
        
        m_shaders[name] = shader;
        return shader;
    }

    std::shared_ptr<ShaderProgram> ShaderManager::getShader(const std::string& name) {
        auto it = m_shaders.find(name);
        return (it != m_shaders.end()) ? it->second : nullptr;
    }

    bool ShaderManager::hasShader(const std::string& name) const {
        return m_shaders.find(name) != m_shaders.end();
    }

    void ShaderManager::removeShader(const std::string& name) {
        m_shaders.erase(name);
    }

    void ShaderManager::clearShaders() {
        m_shaders.clear();
    }

    bool ShaderManager::loadBuiltinShaders() {
        // Load mesh shader
        auto meshShader = std::make_shared<ShaderProgram>();
        if (!meshShader->loadFromStrings(getMeshVertexShaderSource(), getMeshFragmentShaderSource())) {
            std::cerr << "Failed to load built-in mesh shader" << std::endl;
            return false;
        }
        m_shaders["mesh"] = meshShader;
        
        // Load wireframe shader
        auto wireframeShader = std::make_shared<ShaderProgram>();
        if (!wireframeShader->loadFromStrings(getWireframeVertexShaderSource(), getWireframeFragmentShaderSource())) {
            std::cerr << "Failed to load built-in wireframe shader" << std::endl;
            return false;
        }
        m_shaders["wireframe"] = wireframeShader;
        
        return true;
    }

    std::string ShaderManager::getMeshVertexShaderSource() {
        return R"(
#version 120

attribute vec3 a_position;
attribute vec3 a_normal;
attribute vec3 a_color;

uniform mat4 u_modelViewProjectionMatrix;
uniform mat4 u_modelViewMatrix;
uniform mat4 u_normalMatrix;

varying vec3 v_position;
varying vec3 v_normal;
varying vec3 v_color;

void main() {
    v_position = (u_modelViewMatrix * vec4(a_position, 1.0)).xyz;
    v_normal = normalize((u_normalMatrix * vec4(a_normal, 0.0)).xyz);
    v_color = a_color;

    gl_Position = u_modelViewProjectionMatrix * vec4(a_position, 1.0);
}
)";
    }

    std::string ShaderManager::getMeshFragmentShaderSource() {
        return R"(
#version 120

varying vec3 v_position;
varying vec3 v_normal;
varying vec3 v_color;

uniform vec3 u_lightDirection;
uniform vec3 u_lightColor;
uniform vec3 u_ambientLight;
uniform bool u_enableLighting;

void main() {
    vec3 color = v_color;

    if (u_enableLighting) {
        // Simple directional lighting
        vec3 normal = normalize(v_normal);
        vec3 lightDir = normalize(-u_lightDirection);

        float diffuse = max(dot(normal, lightDir), 0.0);
        vec3 lighting = u_ambientLight + u_lightColor * diffuse;

        color = color * lighting;
    }

    gl_FragColor = vec4(color, 1.0);
}
)";
    }

    std::string ShaderManager::getWireframeVertexShaderSource() {
        return R"(
#version 120

attribute vec3 a_position;
attribute vec3 a_color;

uniform mat4 u_modelViewProjectionMatrix;

varying vec3 v_color;

void main() {
    v_color = a_color;
    gl_Position = u_modelViewProjectionMatrix * vec4(a_position, 1.0);
}
)";
    }

    std::string ShaderManager::getWireframeFragmentShaderSource() {
        return R"(
#version 120

varying vec3 v_color;

void main() {
    gl_FragColor = vec4(v_color, 1.0);
}
)";
    }

} // namespace alice2
