#include <computeGeom/scalarField_gpu.h>
#include <core/ShaderManager.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <algorithm>

// Static member definitions
std::shared_ptr<alice2::ShaderManager> ScalarFieldGPU::s_shaderManager = nullptr;
bool ScalarFieldGPU::s_gpuEnabled = false;

// Constructor - GPU-first approach
ScalarFieldGPU::ScalarFieldGPU(const Vec3& min_bb, const Vec3& max_bb, int res_x, int res_y)
    : m_min_bounds(min_bb), m_max_bounds(max_bb), m_res_x(res_x), m_res_y(res_y),
      m_valueBuffer(0), m_positionBuffer(0), m_buffersInitialized(false), m_cpuCacheValid(false) {

    if (res_x <= 0 || res_y <= 0) {
        throw std::invalid_argument("Resolution must be positive");
    }

    // Reserve CPU cache but don't populate it - GPU is primary storage
    m_cpuCache.reserve(m_res_x * m_res_y);

    // Initialize GPU buffers immediately if available
    if (s_gpuEnabled) {
        initialize_gpu_buffers();
        // Initialize with zeros using GPU compute shader
        dispatch_clear_shader();
    }
}

// Destructor
ScalarFieldGPU::~ScalarFieldGPU() {
    cleanup_gpu_buffers();
}

// Copy constructor - GPU-to-GPU copy when possible
ScalarFieldGPU::ScalarFieldGPU(const ScalarFieldGPU& other)
    : m_min_bounds(other.m_min_bounds), m_max_bounds(other.m_max_bounds),
      m_res_x(other.m_res_x), m_res_y(other.m_res_y),
      m_valueBuffer(0), m_positionBuffer(0), m_buffersInitialized(false), m_cpuCacheValid(false) {

    m_cpuCache.reserve(m_res_x * m_res_y);

    if (s_gpuEnabled && other.m_buffersInitialized) {
        // GPU-to-GPU copy (much faster)
        initialize_gpu_buffers();

        // Copy GPU buffer contents directly
        glBindBuffer(GL_COPY_READ_BUFFER, other.m_valueBuffer);
        glBindBuffer(GL_COPY_WRITE_BUFFER, m_valueBuffer);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, get_buffer_size());
        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    } else if (s_gpuEnabled) {
        // Initialize GPU and copy from other's CPU cache if available
        initialize_gpu_buffers();
        if (other.m_cpuCacheValid) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_valueBuffer);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, get_buffer_size(), other.m_cpuCache.data());
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        } else {
            dispatch_clear_shader();
        }
    }
}

// Copy assignment - GPU-first approach
ScalarFieldGPU& ScalarFieldGPU::operator=(const ScalarFieldGPU& other) {
    if (this != &other) {
        // Clean up existing GPU resources
        cleanup_gpu_buffers();

        // Copy properties
        m_min_bounds = other.m_min_bounds;
        m_max_bounds = other.m_max_bounds;
        m_res_x = other.m_res_x;
        m_res_y = other.m_res_y;

        m_cpuCache.clear();
        m_cpuCache.reserve(m_res_x * m_res_y);
        m_cpuCacheValid = false;

        if (s_gpuEnabled && other.m_buffersInitialized) {
            // GPU-to-GPU copy
            initialize_gpu_buffers();
            glBindBuffer(GL_COPY_READ_BUFFER, other.m_valueBuffer);
            glBindBuffer(GL_COPY_WRITE_BUFFER, m_valueBuffer);
            glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, get_buffer_size());
            glBindBuffer(GL_COPY_READ_BUFFER, 0);
            glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
        } else if (s_gpuEnabled) {
            initialize_gpu_buffers();
            dispatch_clear_shader();
        }
    }
    return *this;
}

// Move constructor - GPU-first approach
ScalarFieldGPU::ScalarFieldGPU(ScalarFieldGPU&& other) noexcept
    : m_min_bounds(other.m_min_bounds), m_max_bounds(other.m_max_bounds),
      m_res_x(other.m_res_x), m_res_y(other.m_res_y),
      m_valueBuffer(other.m_valueBuffer), m_positionBuffer(other.m_positionBuffer),
      m_buffersInitialized(other.m_buffersInitialized),
      m_cpuCache(std::move(other.m_cpuCache)), m_cpuCacheValid(other.m_cpuCacheValid) {

    // Reset other's resources
    other.m_valueBuffer = 0;
    other.m_positionBuffer = 0;
    other.m_buffersInitialized = false;
    other.m_cpuCacheValid = false;
}

// Move assignment - GPU-first approach
ScalarFieldGPU& ScalarFieldGPU::operator=(ScalarFieldGPU&& other) noexcept {
    if (this != &other) {
        // Clean up existing GPU resources
        cleanup_gpu_buffers();

        // Move properties
        m_min_bounds = other.m_min_bounds;
        m_max_bounds = other.m_max_bounds;
        m_res_x = other.m_res_x;
        m_res_y = other.m_res_y;
        m_valueBuffer = other.m_valueBuffer;
        m_positionBuffer = other.m_positionBuffer;
        m_buffersInitialized = other.m_buffersInitialized;
        m_cpuCache = std::move(other.m_cpuCache);
        m_cpuCacheValid = other.m_cpuCacheValid;

        // Reset other's resources
        other.m_valueBuffer = 0;
        other.m_positionBuffer = 0;
        other.m_buffersInitialized = false;
        other.m_cpuCacheValid = false;
    }
    return *this;
}

// Clear field - GPU-first approach
void ScalarFieldGPU::clear_field() {
    if (s_gpuEnabled && m_buffersInitialized) {
        // GPU path - no CPU involvement
        dispatch_clear_shader();
    } else {
        // CPU fallback
        clear_field_fallback();
    }
}

// Apply scalar circle - GPU-first approach
void ScalarFieldGPU::apply_scalar_circle(const Vec3& center, float radius, float strength) {
    if (s_gpuEnabled && m_buffersInitialized) {
        // GPU path - no CPU involvement
        dispatch_circle_shader(center, radius, strength);
    } else {
        // CPU fallback
        apply_scalar_circle_fallback(center, radius, strength);
    }
}

// Boolean union - pure GPU approach
void ScalarFieldGPU::boolean_union(const ScalarFieldGPU& other) {
    if (s_gpuEnabled && m_buffersInitialized && other.m_buffersInitialized) {
        // Pure GPU path - no CPU data copying
        dispatch_boolean_union_shader(other);
    } else {
        // CPU fallback
        boolean_union_fallback(other);
    }
}
// Old boolean union code removed - replaced with GPU-first approach

// Draw points - CPU fallback method (requires download)
void ScalarFieldGPU::draw_points_cpu(Renderer& renderer, int step) const {
    // Download data from GPU if needed (expensive!)
    if (!m_cpuCacheValid) {
        download_to_cpu_cache();
    }

    if (!m_cpuCacheValid) {
        std::cerr << "ScalarFieldGPU: Cannot draw points - no valid data" << std::endl;
        return;
    }

    const float dx = (m_max_bounds.x - m_min_bounds.x) / (m_res_x - 1);
    const float dy = (m_max_bounds.y - m_min_bounds.y) / (m_res_y - 1);

    for (int j = 0; j < m_res_y; j += step) {
        for (int i = 0; i < m_res_x; i += step) {
            const int idx = get_index(i, j);
            if (idx >= m_cpuCache.size()) continue;

            const float value = m_cpuCache[idx];

            // Calculate world position
            const float x = m_min_bounds.x + i * dx;
            const float y = m_min_bounds.y + j * dy;

            // Color based on field value
            Vec3 color;
            if (value < 0.0f) {
                // Inside - red to yellow gradient
                float t = std::min(-value / 10.0f, 1.0f);
                color = Vec3(1.0f, t, 0.0f);
            } else {
                // Outside - blue to cyan gradient
                float t = std::min(value / 10.0f, 1.0f);
                color = Vec3(0.0f, t, 1.0f);
            }

            renderer.setColor(color);
            renderer.drawPoint(Vec3(x, y, 0.0f));
        }
    }
}

// GPU-direct rendering (future implementation)
void ScalarFieldGPU::draw_points_gpu(Renderer& renderer, int step) const {
    // TODO: Implement direct GPU rendering using vertex buffer objects
    // For now, fallback to CPU method
    draw_points_cpu(renderer, step);
}

// Auto-select best drawing method
void ScalarFieldGPU::draw_points(Renderer& renderer, int step) const {
    if (s_gpuEnabled && m_buffersInitialized) {
        draw_points_gpu(renderer, step);
    } else {
        draw_points_cpu(renderer, step);
    }
}

// GPU management methods
void ScalarFieldGPU::initialize_gpu(std::shared_ptr<alice2::ShaderManager> shaderManager) {
    s_shaderManager = shaderManager;

    if (!s_shaderManager) {
        std::cerr << "ScalarFieldGPU: Invalid shader manager" << std::endl;
        return;
    }

    // Load all required compute shaders
    std::vector<std::string> possiblePaths = {
        "src/shaders/",
        "alice2/src/shaders/",
        "../src/shaders/",
        "./src/shaders/"
    };

    bool allShadersLoaded = true;

    // Load clear shader
    auto clearShader = loadShaderWithPaths("scalar_field_gpu_clear", "scalar_field_gpu_clear.comp", possiblePaths);
    if (!clearShader) {
        std::cerr << "ScalarFieldGPU: Failed to load clear shader" << std::endl;
        allShadersLoaded = false;
    }

    // Load circle shader
    auto circleShader = loadShaderWithPaths("scalar_field_gpu_circle", "scalar_field_gpu_circle.comp", possiblePaths);
    if (!circleShader) {
        std::cerr << "ScalarFieldGPU: Failed to load circle shader" << std::endl;
        allShadersLoaded = false;
    }

    // Load operations shader
    auto opsShader = loadShaderWithPaths("scalar_field_gpu_ops", "scalar_field_gpu_ops.comp", possiblePaths);
    if (!opsShader) {
        std::cerr << "ScalarFieldGPU: Failed to load operations shader" << std::endl;
        allShadersLoaded = false;
    }

    if (allShadersLoaded) {
        s_gpuEnabled = true;
        std::cout << "ScalarFieldGPU: GPU acceleration enabled with all shaders loaded" << std::endl;
    } else {
        std::cerr << "ScalarFieldGPU: GPU acceleration disabled due to missing shaders" << std::endl;
    }
}

void ScalarFieldGPU::shutdown_gpu() {
    s_gpuEnabled = false;
    s_shaderManager = nullptr;
}

// Helper method to load shaders with multiple path attempts
std::shared_ptr<alice2::ShaderProgram> ScalarFieldGPU::loadShaderWithPaths(
    const std::string& name,
    const std::string& filename,
    const std::vector<std::string>& basePaths) {

    for (const auto& basePath : basePaths) {
        std::string fullPath = basePath + filename;

        // Check if file exists
        std::ifstream file(fullPath);
        if (!file.good()) {
            continue;
        }
        file.close();

        std::cout << "ScalarFieldGPU: Attempting to load " << name << " from: " << fullPath << std::endl;

        auto shader = s_shaderManager->loadComputeShader(name, fullPath);
        if (shader) {
            std::cout << "ScalarFieldGPU: Successfully loaded " << name << " from: " << fullPath << std::endl;
            return shader;
        }
    }

    std::cerr << "ScalarFieldGPU: Failed to load " << name << " from any path" << std::endl;
    return nullptr;
}

// GPU buffer management - fully GPU-focused
void ScalarFieldGPU::initialize_gpu_buffers() const {
    if (m_buffersInitialized || !s_gpuEnabled) {
        return;
    }

    const size_t bufferSize = get_buffer_size();

    // Create value buffer (scalar field values)
    glGenBuffers(1, &m_valueBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_valueBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // Create position buffer (world coordinates - computed in shader)
    glGenBuffers(1, &m_positionBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_positionBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bufferSize * 3, nullptr, GL_DYNAMIC_DRAW); // Vec3 positions
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    m_buffersInitialized = true;
    //std::cout << "ScalarFieldGPU: GPU buffers initialized (" << bufferSize << " bytes each)" << std::endl;
}

void ScalarFieldGPU::cleanup_gpu_buffers() const {
    if (m_buffersInitialized) {
        if (m_valueBuffer != 0) {
            glDeleteBuffers(1, &m_valueBuffer);
            m_valueBuffer = 0;
        }
        if (m_positionBuffer != 0) {
            glDeleteBuffers(1, &m_positionBuffer);
            m_positionBuffer = 0;
        }
        m_buffersInitialized = false;
    }
}

// Old methods removed - GPU-first architecture doesn't need automatic CPU/GPU sync

// Access methods - explicit CPU download
float ScalarFieldGPU::get_field_value(int x, int y) const {
    if (!is_valid_coords(x, y)) {
        return 0.0f;
    }

    // Download entire field if cache is invalid (expensive operation!)
    if (!m_cpuCacheValid) {
        download_to_cpu_cache();
    }

    if (m_cpuCacheValid && get_index(x, y) < m_cpuCache.size()) {
        return m_cpuCache[get_index(x, y)];
    }

    return 0.0f;
}

Vec3 ScalarFieldGPU::get_world_position(int x, int y) const {
    const float dx = (m_max_bounds.x - m_min_bounds.x) / (m_res_x - 1);
    const float dy = (m_max_bounds.y - m_min_bounds.y) / (m_res_y - 1);

    const float world_x = m_min_bounds.x + x * dx;
    const float world_y = m_min_bounds.y + y * dy;

    return Vec3(world_x, world_y, 0.0f);
}

// GPU-only dispatch methods
void ScalarFieldGPU::dispatch_clear_shader() const {
    if (!s_gpuEnabled || !m_buffersInitialized || !s_shaderManager) {
        return;
    }

    auto shader = s_shaderManager->getShader("scalar_field_gpu_clear");
    if (!shader) {
        std::cerr << "ScalarFieldGPU: Clear shader not found" << std::endl;
        return;
    }

    // Bind value buffer
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_valueBuffer);

    // Set uniforms
    shader->use();
    shader->setUniform("gridWidth", m_res_x);
    shader->setUniform("gridHeight", m_res_y);

    // Dispatch compute shader
    const int groupSizeX = (m_res_x + 31) / 32;  // Round up to nearest 32
    const int groupSizeY = (m_res_y + 31) / 32;
    glDispatchCompute(groupSizeX, groupSizeY, 1);

    // Memory barrier to ensure writes are complete
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    invalidate_cpu_cache();
}

void ScalarFieldGPU::dispatch_circle_shader(const Vec3& center, float radius, float strength) const {
    if (!s_gpuEnabled || !m_buffersInitialized || !s_shaderManager) {
        return;
    }

    auto shader = s_shaderManager->getShader("scalar_field_gpu_circle");
    if (!shader) {
        std::cerr << "ScalarFieldGPU: Circle shader not found" << std::endl;
        return;
    }

    // Bind value buffer
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_valueBuffer);

    // Set uniforms
    shader->use();
    shader->setUniform("gridWidth", m_res_x);
    shader->setUniform("gridHeight", m_res_y);
    shader->setUniform("minBounds", m_min_bounds);
    shader->setUniform("maxBounds", m_max_bounds);
    shader->setUniform("circleCenter", center);
    shader->setUniform("circleRadius", radius);
    shader->setUniform("strength", strength);

    // Dispatch compute shader
    const int groupSizeX = (m_res_x + 31) / 32;
    const int groupSizeY = (m_res_y + 31) / 32;
    glDispatchCompute(groupSizeX, groupSizeY, 1);

    // Memory barrier
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    invalidate_cpu_cache();
}

void ScalarFieldGPU::dispatch_boolean_union_shader(const ScalarFieldGPU& other) const {
    if (!s_gpuEnabled || !m_buffersInitialized || !other.m_buffersInitialized || !s_shaderManager) {
        return;
    }

    auto shader = s_shaderManager->getShader("scalar_field_gpu_ops");
    if (!shader) {
        std::cerr << "ScalarFieldGPU: Boolean union shader not found" << std::endl;
        return;
    }

    // Bind buffers
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_valueBuffer);      // Input A and Output
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, other.m_valueBuffer); // Input B
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_valueBuffer);      // Result (same as A)

    // Set uniforms
    shader->use();
    shader->setUniform("gridWidth", m_res_x);
    shader->setUniform("gridHeight", m_res_y);
    shader->setUniform("operation", 0); // 0 = union

    // Dispatch compute shader
    const int groupSizeX = (m_res_x + 31) / 32;
    const int groupSizeY = (m_res_y + 31) / 32;
    glDispatchCompute(groupSizeX, groupSizeY, 1);

    // Memory barrier
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    invalidate_cpu_cache();
}

// CPU fallback methods
void ScalarFieldGPU::clear_field_fallback() {
    if (m_cpuCache.size() != m_res_x * m_res_y) {
        m_cpuCache.resize(m_res_x * m_res_y);
    }
    std::fill(m_cpuCache.begin(), m_cpuCache.end(), 0.0f);
    m_cpuCacheValid = true;
}

void ScalarFieldGPU::apply_scalar_circle_fallback(const Vec3& center, float radius, float strength) {
    if (m_cpuCache.size() != m_res_x * m_res_y) {
        m_cpuCache.resize(m_res_x * m_res_y);
    }

    const float dx = (m_max_bounds.x - m_min_bounds.x) / (m_res_x - 1);
    const float dy = (m_max_bounds.y - m_min_bounds.y) / (m_res_y - 1);

    for (int j = 0; j < m_res_y; ++j) {
        for (int i = 0; i < m_res_x; ++i) {
            const int idx = get_index(i, j);

            // Calculate world position
            const float x = m_min_bounds.x + i * dx;
            const float y = m_min_bounds.y + j * dy;

            // Calculate distance to center
            const float dist_x = x - center.x;
            const float dist_y = y - center.y;
            const float distance = std::sqrt(dist_x * dist_x + dist_y * dist_y);

            // SDF: negative inside, positive outside
            const float sdf = distance - radius;
            m_cpuCache[idx] = sdf * strength;
        }
    }

    m_cpuCacheValid = true;
}

// Data access methods - explicit CPU download
void ScalarFieldGPU::download_to_cpu_cache() const {
    if (!s_gpuEnabled || !m_buffersInitialized) {
        return;
    }

    if (m_cpuCache.size() != m_res_x * m_res_y) {
        m_cpuCache.resize(m_res_x * m_res_y);
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_valueBuffer);
    float* data = static_cast<float*>(glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY));

    if (data) {
        std::copy(data, data + m_cpuCache.size(), m_cpuCache.begin());
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        m_cpuCacheValid = true;
        //std::cout << "ScalarFieldGPU: Downloaded " << m_cpuCache.size() << " values from GPU" << std::endl;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void ScalarFieldGPU::boolean_union_fallback(const ScalarFieldGPU& other) {
    // Ensure both fields have CPU data
    if (!m_cpuCacheValid) {
        download_to_cpu_cache();
    }
    if (!other.m_cpuCacheValid) {
        other.download_to_cpu_cache();
    }

    // Perform union operation on CPU
    for (size_t i = 0; i < m_cpuCache.size() && i < other.m_cpuCache.size(); ++i) {
        m_cpuCache[i] = std::min(m_cpuCache[i], other.m_cpuCache[i]);
    }

    m_cpuCacheValid = true;
}
