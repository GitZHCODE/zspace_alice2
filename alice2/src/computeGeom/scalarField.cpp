#include <computeGeom/scalarField.h>
#include <core/ShaderManager.h>

// Static member definitions
std::shared_ptr<ShaderManager> ScalarField2D::s_shaderManager = nullptr;
bool ScalarField2D::s_gpuEnabled = false;

// Implementation of key methods
ScalarField2D::ScalarField2D(const Vec3& min_bb, const Vec3& max_bb, int res_x, int res_y)
    : m_min_bounds(min_bb), m_max_bounds(max_bb), m_res_x(res_x), m_res_y(res_y),
      m_gpuBuffer(0), m_gpuBufferInitialized(false), m_dataOnGPU(false) {
    if (res_x <= 0 || res_y <= 0) {
        throw std::invalid_argument("Resolution must be positive");
    }

    const int total_points = m_res_x * m_res_y;
    m_grid_points.reserve(total_points);
    m_field_values.resize(total_points, 0.0f);
    m_normalized_values.resize(total_points, 0.0f);
    m_gradient_field.resize(total_points, Vec3(0, 0, 0));

    initialize_grid();
}

ScalarField2D::ScalarField2D(const ScalarField2D& other)
    : m_min_bounds(other.m_min_bounds), m_max_bounds(other.m_max_bounds)
    , m_res_x(other.m_res_x), m_res_y(other.m_res_y)
    , m_grid_points(other.m_grid_points), m_field_values(other.m_field_values)
    , m_normalized_values(other.m_normalized_values), m_gradient_field(other.m_gradient_field)
    , m_gpuBuffer(0), m_gpuBufferInitialized(false), m_dataOnGPU(false) {
}

ScalarField2D& ScalarField2D::operator=(const ScalarField2D& other) {
    if (this != &other) {
        // Clean up existing GPU buffers
        cleanupGPUBuffers();

        m_min_bounds = other.m_min_bounds;
        m_max_bounds = other.m_max_bounds;
        m_res_x = other.m_res_x;
        m_res_y = other.m_res_y;
        m_grid_points = other.m_grid_points;
        m_field_values = other.m_field_values;
        m_normalized_values = other.m_normalized_values;
        m_gradient_field = other.m_gradient_field;

        // Reset GPU buffer state
        m_gpuBuffer = 0;
        m_gpuBufferInitialized = false;
        m_dataOnGPU = false;
    }
    return *this;
}

ScalarField2D::ScalarField2D(ScalarField2D&& other) noexcept
    : m_min_bounds(std::move(other.m_min_bounds)), m_max_bounds(std::move(other.m_max_bounds))
    , m_res_x(other.m_res_x), m_res_y(other.m_res_y)
    , m_grid_points(std::move(other.m_grid_points)), m_field_values(std::move(other.m_field_values))
    , m_normalized_values(std::move(other.m_normalized_values)), m_gradient_field(std::move(other.m_gradient_field))
    , m_gpuBuffer(other.m_gpuBuffer), m_gpuBufferInitialized(other.m_gpuBufferInitialized)
    , m_dataOnGPU(other.m_dataOnGPU) {
    other.m_res_x = other.m_res_y = 0;
    other.m_gpuBuffer = 0;
    other.m_gpuBufferInitialized = false;
    other.m_dataOnGPU = false;
}

ScalarField2D& ScalarField2D::operator=(ScalarField2D&& other) noexcept {
    if (this != &other) {
        // Clean up existing GPU buffers
        cleanupGPUBuffers();

        m_min_bounds = std::move(other.m_min_bounds);
        m_max_bounds = std::move(other.m_max_bounds);
        m_res_x = other.m_res_x;
        m_res_y = other.m_res_y;
        m_grid_points = std::move(other.m_grid_points);
        m_field_values = std::move(other.m_field_values);
        m_normalized_values = std::move(other.m_normalized_values);
        m_gradient_field = std::move(other.m_gradient_field);

        // Move GPU buffer state
        m_gpuBuffer = other.m_gpuBuffer;
        m_gpuBufferInitialized = other.m_gpuBufferInitialized;
        m_dataOnGPU = other.m_dataOnGPU;

        other.m_res_x = other.m_res_y = 0;
        other.m_gpuBuffer = 0;
        other.m_gpuBufferInitialized = false;
        other.m_dataOnGPU = false;
    }
    return *this;
}

// Helper method implementations
void ScalarField2D::initialize_grid() {
    m_grid_points.clear();
    m_grid_points.reserve(m_res_x * m_res_y);

    const Vec3 span = m_max_bounds - m_min_bounds;
    const float step_x = span.x / (m_res_x - 1);
    const float step_y = span.y / (m_res_y - 1);

    for (int j = 0; j < m_res_y; ++j) {
        for (int i = 0; i < m_res_x; ++i) {
            const float x = m_min_bounds.x + i * step_x;
            const float y = m_min_bounds.y + j * step_y;
            m_grid_points.emplace_back(x, y, 0.0f);
        }
    }
}

void ScalarField2D::normalize_field() {
    if (m_field_values.empty()) return;

    // find extrema
    auto [min_it, max_it] = std::minmax_element(m_field_values.begin(),
                                                 m_field_values.end());
    const float min_val = *min_it;
    const float max_val = *max_it;
    const float range   = std::max(max_val - min_val, 1e-6f);

    // do we need a [-1,1] stretch?
    const bool use_bipolar = (min_val < 0.0f);

    for (size_t i = 0, n = m_field_values.size(); i < n; ++i) {
        // first normalize into [0,1]
        float t = (m_field_values[i] - min_val) / range;
        // if any value was negative, remap [0,1] -> [-1,1]
        m_normalized_values[i] = use_bipolar ? (t * 2.0f - 1.0f)
                                              :  t;
    }
}

void ScalarField2D::clear_field() {
    std::fill(m_field_values.begin(), m_field_values.end(), 0.0f);
    std::fill(m_normalized_values.begin(), m_normalized_values.end(), 0.0f);
}

// Scalar function implementations

void ScalarField2D::apply_scalar_circle(const Vec3& center, float radius) {
    for (int j = 0; j < m_res_y; ++j) {
        for (int i = 0; i < m_res_x; ++i) {
            const int idx = get_index(i, j);
            const Vec3& pt = m_grid_points[idx];
            const float d = ScalarFieldUtils::distance_to(pt, center);
            const float sdf = d - radius; // SDF: negative inside, positive outside
            m_field_values[idx] = sdf;
        }
    }
}

void ScalarField2D::apply_scalar_rect(const Vec3& center, const Vec3& half_size, float angle_radians) {
    const float cos_angle = std::cos(angle_radians);
    const float sin_angle = std::sin(angle_radians);

    for (int j = 0; j < m_res_y; ++j) {
        for (int i = 0; i < m_res_x; ++i) {
            const int idx = get_index(i, j);
            const Vec3 p = m_grid_points[idx] - center;

            // Rotate point into box's local frame
            const Vec3 pr(
                cos_angle * p.x + sin_angle * p.y,
                -sin_angle * p.x + cos_angle * p.y,
                0.0f
            );

            const Vec3 d = ScalarFieldUtils::vec_max(Vec3(std::abs(pr.x), std::abs(pr.y), 0.0f) - half_size, Vec3(0, 0, 0));
            const float outside_dist = d.length();
            const float inside_dist = std::min(std::max(std::abs(pr.x) - half_size.x, std::abs(pr.y) - half_size.y), 0.0f);

            // SDF: negative inside, positive outside
            const float sdf = (outside_dist > 0.0f) ? outside_dist : inside_dist;
            m_field_values[idx] = sdf;
        }
    }
}

void ScalarField2D::apply_scalar_voronoi(const std::vector<Vec3>& sites) {
    for (int j = 0; j < m_res_y; ++j) {
        for (int i = 0; i < m_res_x; ++i) {
            const int idx = get_index(i, j);
            const Vec3& pt = m_grid_points[idx];

            float min_dist = std::numeric_limits<float>::max();
            float second_min_dist = std::numeric_limits<float>::max();

            for (const auto& site : sites) {
                const float d = ScalarFieldUtils::distance_to(pt, site);
                if (d < min_dist) {
                    second_min_dist = min_dist;
                    min_dist = d;
                } else if (d < second_min_dist) {
                    second_min_dist = d;
                }
            }

            // Voronoi edge distance (distance to second closest minus closest)
            m_field_values[idx] = second_min_dist - min_dist;
        }
    }
}

// Boolean operations with GPU acceleration
void ScalarField2D::boolean_union(const ScalarField2D& other) {
    if (m_field_values.size() != other.m_field_values.size()) {
        throw std::invalid_argument("Field dimensions must match for boolean operations");
    }

    // Try GPU acceleration first, fall back to CPU if it fails
    if (s_gpuEnabled && performGPUBooleanOperation(other, 0)) {
        // GPU operation succeeded - data is now on GPU
        // Only download if we need CPU access immediately
        return;
    }

    // CPU fallback - ensure we have CPU data
    if (m_dataOnGPU) {
        downloadFromGPU();
    }
    boolean_union_fallback(other);
}

void ScalarField2D::boolean_intersect(const ScalarField2D& other) {
    if (m_field_values.size() != other.m_field_values.size()) {
        throw std::invalid_argument("Field dimensions must match for boolean operations");
    }

    // Try GPU acceleration first, fall back to CPU if it fails
    if (s_gpuEnabled && performGPUBooleanOperation(other, 1)) {
        return; // GPU operation succeeded
    }

    // CPU fallback
    boolean_intersect_fallback(other);
}

void ScalarField2D::boolean_subtract(const ScalarField2D& other) {
    if (m_field_values.size() != other.m_field_values.size()) {
        throw std::invalid_argument("Field dimensions must match for boolean operations");
    }

    // Try GPU acceleration first, fall back to CPU if it fails
    if (s_gpuEnabled && performGPUBooleanOperation(other, 2)) {
        return; // GPU operation succeeded
    }

    // CPU fallback
    boolean_subtract_fallback(other);
}

void ScalarField2D::boolean_smin(const ScalarField2D& other, float smoothing) {
    if (m_field_values.size() != other.m_field_values.size()) {
        throw std::invalid_argument("Field dimensions must match for boolean operations");
    }

    // Try GPU acceleration first, fall back to CPU if it fails
    if (s_gpuEnabled && performGPUBooleanOperation(other, 3, smoothing)) {
        return; // GPU operation succeeded
    }

    // CPU fallback
    boolean_smin_fallback(other, smoothing);
}

void ScalarField2D::boolean_smin_weighted(const ScalarField2D& other, float smoothing, float wt) {
    if (m_field_values.size() != other.m_field_values.size()) {
        throw std::invalid_argument("Field dimensions must match for boolean operations");
    }

    // Try GPU acceleration first, fall back to CPU if it fails
    if (s_gpuEnabled && performGPUBooleanOperation(other, 4, smoothing, wt)) {
        return; // GPU operation succeeded
    }

    // CPU fallback
    boolean_smin_weighted_fallback(other, smoothing, wt);
}

void ScalarField2D::interpolate(const ScalarField2D& other, float t) {
    if (m_field_values.size() != other.m_field_values.size()) {
        throw std::invalid_argument("Field dimensions must match for interpolation");
    }

    for (size_t i = 0; i < m_field_values.size(); ++i) {
        m_field_values[i] = (1.0f - t) * m_field_values[i] + t * other.m_field_values[i];
    }
}

// Rendering methods
void ScalarField2D::draw_points(Renderer& renderer, int step) const {
    // We need to cast away const to normalize - this is a design compromise
    const_cast<ScalarField2D*>(this)->normalize_field();

    for (int j = 0; j < m_res_y; j += step) {
        for (int i = 0; i < m_res_x; i += step) {
            const int idx = get_index(i, j);
            const float f = m_normalized_values[idx];

            float r, g, b;
            ScalarFieldUtils::get_jet_color(f * 2.0f - 1.0f, r, g, b);
            const Vec3 color(r, g, b);

            renderer.drawPoint(m_grid_points[idx], color, 3.0f);
        }
    }
}

void ScalarField2D::draw_values(Renderer& renderer, int step) const {
    for (int j = 0; j < m_res_y; j += step) {
        for (int i = 0; i < m_res_x; i += step) {
            const int idx = get_index(i, j);
            const float value = m_field_values[idx];

            // Draw 3D text showing the scalar value
            const std::string text = std::to_string(value).substr(0, 5); // Limit to 5 characters
            renderer.drawText(text, m_grid_points[idx] + Vec3(0, 0, 0), 0.8f);
        }
    }
}

// Legacy compatibility method for contour drawing
void ScalarField2D::drawIsocontours(Renderer& renderer, float threshold) const {
    const ContourData contours = get_contours(threshold);

    for (const auto& segment : contours.line_segments) {
        renderer.drawLine(segment.first, segment.second, renderer.getCurrentColor(), 2.0f);
    }
}

// Analysis methods - simplified implementations
ContourData ScalarField2D::get_contours(float threshold) const {
    ContourData result(threshold);

    // Simple marching squares implementation for contour extraction
    for (int j = 0; j < m_res_y - 1; ++j) {
        for (int i = 0; i < m_res_x - 1; ++i) {
            const int idx00 = get_index(i, j);
            const int idx10 = get_index(i + 1, j);
            const int idx01 = get_index(i, j + 1);
            const int idx11 = get_index(i + 1, j + 1);

            const float v00 = m_field_values[idx00];
            const float v10 = m_field_values[idx10];
            const float v01 = m_field_values[idx01];
            const float v11 = m_field_values[idx11];

            // Check for threshold crossings and create line segments
            std::vector<Vec3> crossings;

            // Check each edge of the quad
            if ((v00 < threshold && v10 >= threshold) || (v10 < threshold && v00 >= threshold)) {
                const float t = (threshold - v00) / (v10 - v00);
                crossings.push_back(Vec3::lerp(m_grid_points[idx00], m_grid_points[idx10], t));
            }
            if ((v10 < threshold && v11 >= threshold) || (v11 < threshold && v10 >= threshold)) {
                const float t = (threshold - v10) / (v11 - v10);
                crossings.push_back(Vec3::lerp(m_grid_points[idx10], m_grid_points[idx11], t));
            }
            if ((v11 < threshold && v01 >= threshold) || (v01 < threshold && v11 >= threshold)) {
                const float t = (threshold - v11) / (v01 - v11);
                crossings.push_back(Vec3::lerp(m_grid_points[idx11], m_grid_points[idx01], t));
            }
            if ((v01 < threshold && v00 >= threshold) || (v00 < threshold && v01 >= threshold)) {
                const float t = (threshold - v01) / (v00 - v01);
                crossings.push_back(Vec3::lerp(m_grid_points[idx01], m_grid_points[idx00], t));
            }

            if (crossings.size() == 2) {
                result.line_segments.emplace_back(crossings[0], crossings[1]);
            }
        }
    }

    return result;
}

std::vector<Vec3> ScalarField2D::get_gradient() const {
    std::vector<Vec3> gradient(m_field_values.size(), Vec3(0, 0, 0));

    for (int j = 1; j < m_res_y - 1; ++j) {
        for (int i = 1; i < m_res_x - 1; ++i) {
            const int idx = get_index(i, j);
            const int idx_left = get_index(i - 1, j);
            const int idx_right = get_index(i + 1, j);
            const int idx_down = get_index(i, j - 1);
            const int idx_up = get_index(i, j + 1);

            const float dx = (m_field_values[idx_right] - m_field_values[idx_left]) * 0.5f;
            const float dy = (m_field_values[idx_up] - m_field_values[idx_down]) * 0.5f;

            gradient[idx] = Vec3(dx, dy, 0.0f);
        }
    }

    return gradient;
}

// Additional helper methods for missing functionality
void ScalarField2D::apply_scalar_line(const Vec3& start, const Vec3& end, float thickness) {
    // Simple line SDF implementation
    for (int j = 0; j < m_res_y; ++j) {
        for (int i = 0; i < m_res_x; ++i) {
            const int idx = get_index(i, j);
            const Vec3& pt = m_grid_points[idx];

            const Vec3 pa = pt - start;
            const Vec3 ba = end - start;
            const float h = clamp(pa.dot(ba) / ba.dot(ba), 0.0f, 1.0f);
            const float dist = (pa - ba * h).length();

            m_field_values[idx] = dist - thickness; // SDF: negative inside, positive outside
        }
    }
}

void ScalarField2D::apply_scalar_polygon(const std::vector<Vec3>& vertices) {
    // Simple polygon SDF - placeholder implementation
    if (vertices.empty()) return;

    for (int j = 0; j < m_res_y; ++j) {
        for (int i = 0; i < m_res_x; ++i) {
            const int idx = get_index(i, j);
            const Vec3& pt = m_grid_points[idx];

            // Find distance to closest vertex (simplified)
            float min_dist = std::numeric_limits<float>::max();
            for (const auto& vertex : vertices) {
                const float dist = ScalarFieldUtils::distance_to(pt, vertex);
                min_dist = std::min(min_dist, dist);
            }

            m_field_values[idx] = min_dist;
        }
    }
}

void ScalarField2D::set_values(const std::vector<float>& values) {
    if (values.size() != m_field_values.size()) {
        throw std::invalid_argument("Value array size must match field resolution");
    }
    m_field_values = values;
}

void ScalarField2D::boolean_difference(const ScalarField2D& other) {
    // Difference is A - B = A ∩ ¬B = max(A, -B)
    boolean_subtract(other);
}

// GPU acceleration methods
void ScalarField2D::initializeGPU(std::shared_ptr<alice2::ShaderManager> shaderManager) {
    s_shaderManager = shaderManager;
    s_gpuEnabled = false;

    if (!s_shaderManager) {
        std::cout << "ScalarField2D: No shader manager provided" << std::endl;
        return;
    }

    if (!s_shaderManager->isComputeShaderSupported()) {
        std::cout << "ScalarField2D: Compute shaders not supported on this system" << std::endl;
        return;
    }

    std::cout << "ScalarField2D: Attempting to load compute shader..." << std::endl;

    // Load the compute shader (try multiple possible paths)
    std::vector<std::string> possiblePaths = {
        "src/shaders/scalar_field_ops.comp",
        "alice2/src/shaders/scalar_field_ops.comp",
        "../src/shaders/scalar_field_ops.comp",
        "./src/shaders/scalar_field_ops.comp"
    };

    std::shared_ptr<alice2::ShaderProgram> shader = nullptr;
    for (const auto& path : possiblePaths) {
        std::cout << "ScalarField2D: Trying to load shader from: " << path << std::endl;

        // Check if file exists
        std::ifstream file(path);
        if (!file.good()) {
            std::cout << "ScalarField2D: File not found: " << path << std::endl;
            continue;
        }
        file.close();

        shader = s_shaderManager->loadComputeShader("scalar_field_ops", path);
        if (shader) {
            std::cout << "ScalarField2D: Successfully loaded compute shader from: " << path << std::endl;
            break;
        } else {
            std::cout << "ScalarField2D: Failed to compile shader from: " << path << std::endl;
        }
    }
    if (!shader) {
        std::cout << "ScalarField2D: Failed to load compute shader" << std::endl;
        return;
    }

    s_gpuEnabled = true;
    std::cout << "ScalarField2D: GPU acceleration enabled" << std::endl;
}

void ScalarField2D::shutdownGPU() {
    s_gpuEnabled = false;
    s_shaderManager = nullptr;
}

void ScalarField2D::initializeGPUBuffers() const {
    if (m_gpuBufferInitialized || !s_gpuEnabled) {
        return;
    }

    const size_t bufferSize = m_field_values.size() * sizeof(float);

    // Create single persistent GPU buffer
    glGenBuffers(1, &m_gpuBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_gpuBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bufferSize, m_field_values.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    m_gpuBufferInitialized = true;
    m_dataOnGPU = true;
}

void ScalarField2D::cleanupGPUBuffers() const {
    if (!m_gpuBufferInitialized) {
        return;
    }

    if (m_gpuBuffer != 0) {
        glDeleteBuffers(1, &m_gpuBuffer);
        m_gpuBuffer = 0;
    }

    m_gpuBufferInitialized = false;
    m_dataOnGPU = false;
}

bool ScalarField2D::performGPUBooleanOperation(const ScalarField2D& other, int operation,
                                              float smoothing, float weight) const {
    if (!s_gpuEnabled || !s_shaderManager) {
        return false;
    }

    auto shader = s_shaderManager->getShader("scalar_field_ops");
    if (!shader) {
        return false;
    }

    // Ensure both fields have GPU buffers
    initializeGPUBuffers();
    const_cast<ScalarField2D&>(other).initializeGPUBuffers();

    // Upload data to GPU only if not already there
    if (!m_dataOnGPU) {
        uploadToGPU();
    }
    if (!other.m_dataOnGPU) {
        const_cast<ScalarField2D&>(other).uploadToGPU();
    }

    // Create temporary buffer for the other field's data
    GLuint tempBuffer;
    glGenBuffers(1, &tempBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, tempBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, m_field_values.size() * sizeof(float), nullptr, GL_DYNAMIC_COPY);

    // Copy other field's data to temp buffer
    glBindBuffer(GL_COPY_READ_BUFFER, other.m_gpuBuffer);
    glBindBuffer(GL_COPY_WRITE_BUFFER, tempBuffer);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, m_field_values.size() * sizeof(float));

    // Bind buffers to shader (this = input A, other = input B, this = output)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_gpuBuffer);      // Input A
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, tempBuffer);       // Input B
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_gpuBuffer);      // Output (overwrite A)

    // Set uniforms
    shader->use();
    shader->setUniform("gridWidth", m_res_x);
    shader->setUniform("gridHeight", m_res_y);
    shader->setUniform("operation", operation);
    shader->setUniform("smoothing", smoothing);
    shader->setUniform("weight", weight);

    // Dispatch compute shader
    const int groupSizeX = 32;
    const int groupSizeY = 32;
    const int numGroupsX = (m_res_x + groupSizeX - 1) / groupSizeX;
    const int numGroupsY = (m_res_y + groupSizeY - 1) / groupSizeY;

    shader->dispatch(numGroupsX, numGroupsY, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Clean up temp buffer
    glDeleteBuffers(1, &tempBuffer);

    // Data is now on GPU, mark as such
    m_dataOnGPU = true;

    return true;
}

// CPU fallback implementations
void ScalarField2D::boolean_union_fallback(const ScalarField2D& other) {
    for (size_t i = 0; i < m_field_values.size(); ++i) {
        m_field_values[i] = std::min(m_field_values[i], other.m_field_values[i]);
    }
}

void ScalarField2D::boolean_intersect_fallback(const ScalarField2D& other) {
    for (size_t i = 0; i < m_field_values.size(); ++i) {
        m_field_values[i] = std::max(m_field_values[i], other.m_field_values[i]);
    }
}

void ScalarField2D::boolean_subtract_fallback(const ScalarField2D& other) {
    for (size_t i = 0; i < m_field_values.size(); ++i) {
        m_field_values[i] = std::max(m_field_values[i], -other.m_field_values[i]);
    }
}

void ScalarField2D::boolean_smin_fallback(const ScalarField2D& other, float smoothing) {
    for (size_t i = 0; i < m_field_values.size(); ++i) {
        m_field_values[i] = ScalarFieldUtils::smooth_min(m_field_values[i], other.m_field_values[i], smoothing);
    }
}

void ScalarField2D::boolean_smin_weighted_fallback(const ScalarField2D& other, float smoothing, float wt) {
    for (size_t i = 0; i < m_field_values.size(); ++i) {
        m_field_values[i] = ScalarFieldUtils::smooth_min_weighted(m_field_values[i], other.m_field_values[i], smoothing, wt);
    }
}

// GPU optimization methods
void ScalarField2D::uploadToGPU() const {
    if (!s_gpuEnabled || !m_gpuBufferInitialized) {
        return;
    }

    const size_t bufferSize = m_field_values.size() * sizeof(float);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_gpuBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bufferSize, m_field_values.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    m_dataOnGPU = true;
}

void ScalarField2D::downloadFromGPU() const {
    if (!s_gpuEnabled || !m_gpuBufferInitialized || !m_dataOnGPU) {
        return;
    }

    const size_t bufferSize = m_field_values.size() * sizeof(float);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_gpuBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bufferSize, const_cast<float*>(m_field_values.data()));
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void ScalarField2D::ensureCPUData() const {
    if (m_dataOnGPU) {
        downloadFromGPU();
    }
}
