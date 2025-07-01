#pragma once

#ifndef ALICE2_SHADER_MANAGER_H
#define ALICE2_SHADER_MANAGER_H

#include <GL/glew.h>
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>
#include "utils/Vector.h"

namespace alice2 {

    /**
     * Shader types supported by the ShaderManager
     */
    enum class ShaderType {
        Vertex = GL_VERTEX_SHADER,
        Fragment = GL_FRAGMENT_SHADER,
        Compute = GL_COMPUTE_SHADER
    };

    /**
     * Represents a compiled shader program
     */
    class ShaderProgram {
    public:
        ShaderProgram(GLuint programId);
        ~ShaderProgram();

        // Non-copyable but movable
        ShaderProgram(const ShaderProgram&) = delete;
        ShaderProgram& operator=(const ShaderProgram&) = delete;
        ShaderProgram(ShaderProgram&& other) noexcept;
        ShaderProgram& operator=(ShaderProgram&& other) noexcept;

        GLuint getId() const { return m_programId; }
        bool isValid() const { return m_programId != 0; }

        // Uniform management
        void use() const;
        void setUniform(const std::string& name, int value) const;
        void setUniform(const std::string& name, float value) const;
        void setUniform(const std::string& name, const Vec3& value) const;
        void setUniform(const std::string& name, const float* values, int count) const;
        GLint getUniformLocation(const std::string& name) const;

        // Compute shader specific
        void dispatch(GLuint numGroupsX, GLuint numGroupsY = 1, GLuint numGroupsZ = 1) const;
        void memoryBarrier(GLbitfield barriers = GL_SHADER_STORAGE_BARRIER_BIT) const;

    private:
        GLuint m_programId;
        mutable std::unordered_map<std::string, GLint> m_uniformCache;
    };

    /**
     * Manages shader compilation, linking, and caching
     */
    class ShaderManager {
    public:
        ShaderManager();
        ~ShaderManager();

        // Initialization
        bool initialize();
        void shutdown();

        // Shader loading from files
        std::shared_ptr<ShaderProgram> loadComputeShader(const std::string& name, const std::string& filePath);
        std::shared_ptr<ShaderProgram> loadVertexFragmentShader(const std::string& name, 
                                                               const std::string& vertexPath, 
                                                               const std::string& fragmentPath);

        // Shader loading from source strings
        std::shared_ptr<ShaderProgram> createComputeShader(const std::string& name, const std::string& source);
        std::shared_ptr<ShaderProgram> createVertexFragmentShader(const std::string& name,
                                                                 const std::string& vertexSource,
                                                                 const std::string& fragmentSource);

        // Shader retrieval
        std::shared_ptr<ShaderProgram> getShader(const std::string& name) const;
        bool hasShader(const std::string& name) const;

        // Shader management
        void reloadShader(const std::string& name);
        void reloadAllShaders();
        void removeShader(const std::string& name);
        void clearCache();

        // Utility
        static bool isComputeShaderSupported();
        static std::string getOpenGLVersion();
        static std::string getGLSLVersion();

    private:
        bool m_initialized;
        std::unordered_map<std::string, std::shared_ptr<ShaderProgram>> m_shaderCache;
        std::unordered_map<std::string, std::string> m_shaderPaths; // For reloading

        // Shader compilation
        GLuint compileShader(const std::string& source, ShaderType type) const;
        GLuint linkProgram(const std::vector<GLuint>& shaders) const;
        std::string loadShaderFile(const std::string& filePath) const;
        
        // Error handling
        bool checkShaderCompileErrors(GLuint shader, const std::string& type) const;
        bool checkProgramLinkErrors(GLuint program) const;
        void logError(const std::string& message) const;
    };

} // namespace alice2

#endif // ALICE2_SHADER_MANAGER_H
