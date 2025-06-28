#pragma once

#ifndef ALICE2_SHADER_MANAGER_H
#define ALICE2_SHADER_MANAGER_H

#include "../utils/OpenGL.h"
#include "../utils/Math.h"
#include <string>
#include <unordered_map>
#include <memory>

namespace alice2 {

    // Shader program wrapper
    class ShaderProgram {
    public:
        ShaderProgram();
        ~ShaderProgram();

        // Shader compilation and linking
        bool loadFromFiles(const std::string& vertexPath, const std::string& fragmentPath);
        bool loadFromStrings(const std::string& vertexSource, const std::string& fragmentSource);
        
        // Program management
        void use();
        void unuse();
        bool isValid() const { return m_programId != 0; }
        GLuint getProgramId() const { return m_programId; }

        // Uniform setters
        void setUniform(const std::string& name, float value);
        void setUniform(const std::string& name, int value);
        void setUniform(const std::string& name, bool value);
        void setUniform(const std::string& name, const Vec3& value);
        void setUniform(const std::string& name, const Mat4& value);
        void setUniform(const std::string& name, float x, float y, float z);
        void setUniform(const std::string& name, float x, float y, float z, float w);

        // Attribute locations
        GLint getAttributeLocation(const std::string& name);
        GLint getUniformLocation(const std::string& name);

    private:
        GLuint m_programId;
        GLuint m_vertexShaderId;
        GLuint m_fragmentShaderId;
        
        // Cached uniform locations
        std::unordered_map<std::string, GLint> m_uniformLocations;
        std::unordered_map<std::string, GLint> m_attributeLocations;

        // Helper methods
        bool compileShader(GLuint shaderId, const std::string& source);
        bool linkProgram();
        std::string loadShaderFile(const std::string& filepath);
        void checkCompileErrors(GLuint shader, const std::string& type);
        void checkLinkErrors();
        GLint getCachedUniformLocation(const std::string& name);
        GLint getCachedAttributeLocation(const std::string& name);
    };

    // Shader manager for loading and caching shaders
    class ShaderManager {
    public:
        ShaderManager();
        ~ShaderManager();

        // Initialization
        bool initialize();
        void shutdown();

        // Shader loading and management
        std::shared_ptr<ShaderProgram> loadShader(const std::string& name, 
                                                  const std::string& vertexPath, 
                                                  const std::string& fragmentPath);
        
        std::shared_ptr<ShaderProgram> getShader(const std::string& name);
        bool hasShader(const std::string& name) const;
        void removeShader(const std::string& name);
        void clearShaders();

        // Built-in shaders
        bool loadBuiltinShaders();
        std::shared_ptr<ShaderProgram> getMeshShader() { return getShader("mesh"); }
        std::shared_ptr<ShaderProgram> getWireframeShader() { return getShader("wireframe"); }

        // Utility
        bool isInitialized() const { return m_initialized; }

    private:
        bool m_initialized;
        std::unordered_map<std::string, std::shared_ptr<ShaderProgram>> m_shaders;
        
        // Built-in shader sources
        std::string getMeshVertexShaderSource();
        std::string getMeshFragmentShaderSource();
        std::string getWireframeVertexShaderSource();
        std::string getWireframeFragmentShaderSource();
    };

} // namespace alice2

#endif // ALICE2_SHADER_MANAGER_H
