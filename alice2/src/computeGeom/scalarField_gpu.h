#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include "../include/alice2.h"
#include <GL/glew.h>

using namespace alice2;

// Forward declarations
namespace alice2 {
    class ShaderManager;
}

/**
 * Fully GPU-based Scalar Field class
 * Stores data directly on GPU memory - NO automatic CPU/GPU copying
 * All operations performed on GPU using compute shaders
 * CPU data only downloaded on explicit request for visualization/debugging
 */
class ScalarFieldGPU {
private:
    // Grid properties
    Vec3 m_min_bounds;
    Vec3 m_max_bounds;
    int m_res_x;
    int m_res_y;

    // GPU buffer management
    static std::shared_ptr<alice2::ShaderManager> s_shaderManager;
    static bool s_gpuEnabled;

    // GPU-only storage - no automatic CPU mirroring
    mutable GLuint m_valueBuffer;     // Scalar field values
    mutable GLuint m_positionBuffer;  // World positions (computed in shader)
    mutable bool m_buffersInitialized;

    // GPU rendering buffers
    mutable GLuint m_vertexBuffer;    // Vertex positions for rendering
    mutable GLuint m_colorBuffer;     // Vertex colors for rendering
    mutable bool m_renderBuffersInitialized;

    // CPU cache - only populated on explicit request
    mutable std::vector<float> m_cpuCache;
    mutable bool m_cpuCacheValid;

    // Helper methods
    inline int get_index(int x, int y) const {
        return y * m_res_x + x;
    }

    inline bool is_valid_coords(int x, int y) const {
        return x >= 0 && x < m_res_x && y >= 0 && y < m_res_y;
    }

    inline size_t get_buffer_size() const {
        return m_res_x * m_res_y * sizeof(float);
    }

    // GPU-only operations
    void initialize_gpu_buffers() const;
    void cleanup_gpu_buffers() const;
    void dispatch_clear_shader() const;
    void dispatch_circle_shader(const Vec3& center, float radius, float strength) const;
    void dispatch_boolean_union_shader(const ScalarFieldGPU& other) const;
    void invalidate_cpu_cache() const { m_cpuCacheValid = false; }

    // GPU rendering operations
    void initialize_render_buffers(int step) const;
    void cleanup_render_buffers() const;
    void update_render_buffers(int step) const;

public:
    // Constructor with RAII principles - matches existing API
    ScalarFieldGPU(const Vec3& min_bb = Vec3(-50, -50, 0),
                   const Vec3& max_bb = Vec3(50, 50, 0),
                   int res_x = 100,
                   int res_y = 100);
    
    // Destructor
    ~ScalarFieldGPU();
    
    // Copy constructor and assignment (deep copy)
    ScalarFieldGPU(const ScalarFieldGPU& other);
    ScalarFieldGPU& operator=(const ScalarFieldGPU& other);
    
    // Move constructor and assignment
    ScalarFieldGPU(ScalarFieldGPU&& other) noexcept;
    ScalarFieldGPU& operator=(ScalarFieldGPU&& other) noexcept;
    
    // Field generation methods - GPU-only operations (no CPU involvement)
    void clear_field();
    void apply_scalar_circle(const Vec3& center, float radius, float strength = 1.0f);

    // Boolean operations - pure GPU compute shaders
    void boolean_union(const ScalarFieldGPU& other);

    // CPU fallback methods (only used when GPU unavailable)
    void clear_field_fallback();
    void apply_scalar_circle_fallback(const Vec3& center, float radius, float strength = 1.0f);
    void boolean_union_fallback(const ScalarFieldGPU& other);

    // Visualization methods - GPU-direct rendering when possible
    void draw_points_gpu(Renderer& renderer, int step = 4) const;  // Direct GPU rendering
    void draw_points_cpu(Renderer& renderer, int step = 4) const;  // CPU fallback with download
    void draw_points(Renderer& renderer, int step = 4) const;      // Auto-select best method

    // GPU management
    static void initialize_gpu(std::shared_ptr<alice2::ShaderManager> shaderManager);
    static void shutdown_gpu();
    static bool is_gpu_enabled() { return s_gpuEnabled; }

    // Helper method for shader loading
    static std::shared_ptr<alice2::ShaderProgram> loadShaderWithPaths(
        const std::string& name,
        const std::string& filename,
        const std::vector<std::string>& basePaths);

    // Data access methods - explicit CPU download (use sparingly!)
    void download_to_cpu_cache() const;  // Explicit download from GPU
    float get_field_value(int x, int y) const;  // Downloads entire field if cache invalid
    Vec3 get_world_position(int x, int y) const;

    // Direct GPU buffer access for advanced users
    GLuint get_value_buffer() const { initialize_gpu_buffers(); return m_valueBuffer; }
    GLuint get_position_buffer() const { initialize_gpu_buffers(); return m_positionBuffer; }

    // Getters
    int get_resolution_x() const { return m_res_x; }
    int get_resolution_y() const { return m_res_y; }
    Vec3 get_min_bounds() const { return m_min_bounds; }
    Vec3 get_max_bounds() const { return m_max_bounds; }
};
