// #define __MAIN__
#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <igl/barycenter.h>
#include <igl/cotmatrix.h>
#include <igl/cotmatrix_entries.h>
#include <igl/massmatrix.h>
#include <igl/per_face_normals.h>
#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <stdexcept>

using namespace alice2;

// Compact port of the upstream Gauss-image thinning loop, using the supplied hollow-box input.
class GaussThinningSketch : public ISketch {
public:
    std::string getName() const override { return "Gauss Thinning Box Hollow"; }
    std::string getDescription() const override { return "libigl Gauss-image thinning; box_hollow.obj input"; }
    std::string getAuthor() const override { return "alice2 User"; }

    void setup() override {
        scene().setBackgroundColor(Color(1.0f, 1.0f, 1.0f));
        scene().setShowGrid(false);
        scene().setShowAxes(true);
        scene().setAxesLength(0.8f);

        loadBoxHollow();
        normalize(m_vertices);
        m_originalVertices = m_vertices;
        m_mesh = std::make_shared<MeshObject>("gauss_thinning_box_hollow");
        scene().addObject(m_mesh);
        prepareSolver();
        updateMesh();
    }

    void update(float deltaTime) override {
        if (!m_running || m_iteration >= m_maxIterations) return;
        m_elapsed += deltaTime;
        while (m_elapsed >= 0.05f && m_iteration < m_maxIterations) {
            m_elapsed -= 0.05f;
            step();
        }
    }

    void draw(Renderer& renderer, Camera&) override {
        renderer.setColor(Color(0.1f, 0.1f, 0.1f));
        if (m_showOriginalWireframe) drawOriginalWireframe(renderer);
        drawMinConeEdges(renderer);
        drawFittingNormals(renderer);
        renderer.drawString("Box hollow: " + std::to_string(m_iteration) + "/" + std::to_string(m_maxIterations) +
                            " iterations | u step | p run/pause | r reset | e export OBJ | n normals | w wireframe | c curvature", 10, 30);
        if (!m_exportStatus.empty()) renderer.drawString(m_exportStatus, 10, 50);
    }

    bool onKeyPress(unsigned char key, int, int) override {
        if (key == 'u' && m_iteration < m_maxIterations) {
            step();
            return true;
        }
        if (key == 'p') {
            m_running = !m_running;
            return true;
        }
        if (key == 'e') {
            exportOptimizedMesh();
            return true;
        }
        if (key == 'n') {
            m_showFittingNormals = !m_showFittingNormals;
            return true;
        }
        if (key == 'w') {
            m_showOriginalWireframe = !m_showOriginalWireframe;
            return true;
        }
        if (key == 'c') {
            m_showCurvature = !m_showCurvature;
            updateMesh();
            return true;
        }
        if (key == 'r') {
            m_running = true;
            m_iteration = 0;
            m_elapsed = 0.0f;
            loadBoxHollow();
            normalize(m_vertices);
        m_originalVertices = m_vertices;
            prepareSolver();
            updateMesh();
            return true;
        }
        return false;
    }

private:
    static constexpr double kPi = 3.14159265358979323846;

    static std::filesystem::path dataPath(const char* filename) {
        const std::filesystem::path workingDirectoryPath = std::filesystem::path("data") / filename;
        if (std::filesystem::exists(workingDirectoryPath)) return workingDirectoryPath;

        // __FILE__ is userSrc/geometry/sketch_gauss_thinning.cpp, so this resolves
        // the repository's alice2/data directory even when launched from build/bin.
        return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / "data" / filename;
    }

    void loadBoxHollow() {
        const std::filesystem::path meshPath = dataPath("box_hollow.obj");
        if (!igl::read_triangle_mesh(meshPath.string(), m_vertices, m_faces)) {
            throw std::runtime_error("Failed to load Gauss-thinning box_hollow.obj: " + meshPath.string());
        }
    }

    void exportOptimizedMesh() {
        const std::filesystem::path exportPath = dataPath("box_hollow_optimized.obj");
        Eigen::MatrixXd exportVertices = m_vertices;
        exportVertices /= m_normalizationScale;
        exportVertices.rowwise() += m_normalizationCenter;
        m_exportStatus = igl::write_triangle_mesh(exportPath.string(), exportVertices, m_faces)
            ? "Exported box_hollow_optimized.obj"
            : "Failed to export box_hollow_optimized.obj";
    }
    void normalize(Eigen::MatrixXd& vertices) {
        m_normalizationCenter = vertices.colwise().mean();
        vertices.rowwise() -= m_normalizationCenter;
        m_normalizationScale = 1.0 / (2.0 * vertices.rowwise().norm().maxCoeff());
        vertices *= m_normalizationScale;
    }

    static std::vector<std::vector<int>> triangleAdjacency(const Eigen::MatrixXi& faces, int vertexCount) {
        std::vector<std::vector<int>> incident(vertexCount);
        for (int face = 0; face < faces.rows(); ++face) {
            for (int corner = 0; corner < 3; ++corner) incident[faces(face, corner)].push_back(face);
        }

        std::vector<std::vector<int>> result(faces.rows());
        std::vector<int> visited(faces.rows(), -1);
        for (int face = 0; face < faces.rows(); ++face) {
            for (int corner = 0; corner < 3; ++corner) {
                for (int neighbour : incident[faces(face, corner)]) {
                    if (neighbour != face && visited[neighbour] != face) {
                        result[face].push_back(neighbour);
                        visited[neighbour] = face;
                    }
                }
            }
        }
        return result;
    }

    static std::uint64_t edgeKey(int a, int b) {
        const auto low = static_cast<std::uint32_t>(std::min(a, b));
        const auto high = static_cast<std::uint32_t>(std::max(a, b));
        return (static_cast<std::uint64_t>(low) << 32) | high;
    }

    static std::vector<int> boundaryVertices(const Eigen::MatrixXi& faces, int vertexCount) {
        std::unordered_map<std::uint64_t, int> edgeUse;
        for (int face = 0; face < faces.rows(); ++face) {
            for (int corner = 0; corner < 3; ++corner) {
                ++edgeUse[edgeKey(faces(face, corner), faces(face, (corner + 1) % 3))];
            }
        }

        std::vector<bool> isBoundary(vertexCount, false);
        for (const auto& [key, useCount] : edgeUse) {
            if (useCount != 1) continue;
            isBoundary[static_cast<std::uint32_t>(key >> 32)] = true;
            isBoundary[static_cast<std::uint32_t>(key)] = true;
        }

        std::vector<int> result;
        for (int vertex = 0; vertex < vertexCount; ++vertex) {
            if (isBoundary[vertex]) result.push_back(vertex);
        }
        return result;
    }
    static std::vector<std::vector<int>> collectNeighbours(
        const std::vector<std::vector<int>>& adjacency,
        const Eigen::MatrixXd& centres,
        const Eigen::MatrixXd& normals,
        double radius,
        double coneAngleDegrees) {
        const double normalThreshold = std::cos(coneAngleDegrees * kPi / 180.0);
        std::vector<std::vector<int>> result(centres.rows());
        std::vector<int> marks(centres.rows(), -1);
        std::vector<int> stack;

        for (int seed = 0; seed < centres.rows(); ++seed) {
            stack.push_back(seed);
            marks[seed] = seed;
            while (!stack.empty()) {
                const int current = stack.back();
                stack.pop_back();
                result[seed].push_back(current);
                for (int neighbour : adjacency[current]) {
                    if (marks[neighbour] != seed &&
                        (centres.row(seed) - centres.row(neighbour)).norm() < radius &&
                        normals.row(seed).dot(normals.row(neighbour)) > normalThreshold) {
                        marks[neighbour] = seed;
                        stack.push_back(neighbour);
                    }
                }
            }
        }
        return result;
    }
    static void fitNormals(const std::vector<std::vector<int>>& neighbours,
                           const Eigen::MatrixXd& normals,
                           double coneAngleDegrees,
                           Eigen::MatrixXd& fittedNormals) {
        fittedNormals.resizeLike(normals);
        const double threshold = coneAngleDegrees * kPi / 180.0;
        for (int face = 0; face < normals.rows(); ++face) {
            const auto& neighbourhood = neighbours[face];
            Eigen::MatrixXd samples(neighbourhood.size(), 3);
            for (int i = 0; i < static_cast<int>(neighbourhood.size()); ++i) {
                samples.row(i) = normals.row(neighbourhood[i]);
            }

            Eigen::VectorXd weights(neighbourhood.size());
            for (int i = 0; i < weights.size(); ++i) {
                const double dot = std::clamp(samples.row(0).dot(samples.row(i)), -1.0, 1.0);
                weights(i) = dot <= 0.0 ? 0.0 : std::exp(-std::pow(std::acos(dot) / threshold / 2.0, 2.0));
            }

            const Eigen::Matrix3d covariance = samples.transpose() * weights.asDiagonal() * samples;
            const Eigen::Matrix3d frame = Eigen::JacobiSVD<Eigen::Matrix3d>(covariance, Eigen::ComputeFullV).matrixV();
            fittedNormals.row(face) = (frame.leftCols(2) * frame.leftCols(2).transpose() * normals.row(face).transpose()).normalized();
        }
    }

    static void findRotations(const Eigen::MatrixXd& before,
                              const Eigen::MatrixXd& after,
                              std::vector<Eigen::Matrix3d>& rotations) {
        rotations.resize(before.rows());
        for (int face = 0; face < before.rows(); ++face) {
            const Eigen::Vector3d n0 = before.row(face);
            const Eigen::Vector3d n1 = after.row(face);
            const Eigen::Vector3d axis = n0.cross(n1);
            const double cosine = n0.dot(n1);
            if (cosine > -1.0 + 1e-8) {
                Eigen::Matrix3d cross;
                cross << 0.0, -axis.z(), axis.y(), axis.z(), 0.0, -axis.x(), -axis.y(), axis.x(), 0.0;
                rotations[face] = Eigen::Matrix3d::Identity() + cross + cross * cross / (1.0 + cosine);
            } else {
                rotations[face] = -Eigen::Matrix3d::Identity();
            }
        }
    }

    void prepareSolver() {
        igl::cotmatrix_entries(m_vertices, m_faces, m_cotangents);
        igl::cotmatrix(m_vertices, m_faces, m_laplacian);
        igl::massmatrix(m_vertices, m_faces, igl::MASSMATRIX_TYPE_BARYCENTRIC, m_mass);
        m_adjacency = triangleAdjacency(m_faces, m_vertices.rows());
        m_boundaryVertices = boundaryVertices(m_faces, m_vertices.rows());

        std::vector<Eigen::Triplet<double>> anchorTriplets;
        anchorTriplets.reserve(m_boundaryVertices.size());
        for (const int vertex : m_boundaryVertices) {
            anchorTriplets.emplace_back(vertex, vertex, m_boundaryWeight);
        }
        m_boundaryAnchors.resize(m_vertices.rows(), m_vertices.rows());
        m_boundaryAnchors.setFromTriplets(anchorTriplets.begin(), anchorTriplets.end());

        m_solver.compute(-m_laplacian + m_smooth * m_laplacian.transpose() * m_laplacian +
                         m_epsilon * m_mass + m_boundaryAnchors);
    }

    void step() {
        Eigen::MatrixXd normals, centres, fitted, rhs(m_vertices.rows(), 3);
        igl::per_face_normals(m_vertices, m_faces, normals);
        igl::barycenter(m_vertices, m_faces, centres);
        m_fittingNormals = normals;
        m_fittingCentres = centres;
        const double coneAngle = std::max(m_minConeAngle, m_startConeAngle * std::pow(0.95, m_iteration));
        fitNormals(collectNeighbours(m_adjacency, centres, normals, m_radius, coneAngle), normals, coneAngle, fitted);

        std::vector<Eigen::Matrix3d> rotations;
        findRotations(normals, fitted, rotations);
        rhs.setZero();
        for (int face = 0; face < m_faces.rows(); ++face) {
            for (int corner = 0; corner < 3; ++corner) {
                const int v0 = m_faces(face, (corner + 1) % 3);
                const int v1 = m_faces(face, (corner + 2) % 3);
                const Eigen::Vector3d contribution = m_cotangents(face, corner) * rotations[face] *
                    (m_vertices.row(v0) - m_vertices.row(v1)).transpose();
                rhs.row(v0) -= contribution.transpose();
                rhs.row(v1) += contribution.transpose();
            }
        }

        m_vertices = m_solver.solve(m_epsilon * m_mass * m_vertices - rhs + m_boundaryAnchors * m_originalVertices);
        ++m_iteration;
        updateMesh();
    }

    void drawMinConeEdges(Renderer& renderer) const {
        if (m_fittingNormals.rows() != m_faces.rows()) return;

        std::unordered_map<std::uint64_t, std::vector<int>> edgeFaces;
        for (int face = 0; face < m_faces.rows(); ++face) {
            for (int corner = 0; corner < 3; ++corner) {
                edgeFaces[edgeKey(m_faces(face, corner), m_faces(face, (corner + 1) % 3))].push_back(face);
            }
        }

        const Color edgeColor(0.95f, 0.05f, 0.1f, 1.0f);
        for (const auto& [key, faces] : edgeFaces) {
            if (faces.size() != 2) continue;
            const double cosine = std::clamp(m_fittingNormals.row(faces[0]).dot(m_fittingNormals.row(faces[1])), -1.0, 1.0);
            const double angleDegrees = std::acos(cosine) * 180.0 / kPi;
            if (angleDegrees <= m_minConeAngle) continue;

            const int a = static_cast<int>(key >> 32);
            const int b = static_cast<int>(key);
            renderer.drawLine(Vec3(static_cast<float>(m_vertices(a, 0)), static_cast<float>(m_vertices(a, 1)), static_cast<float>(m_vertices(a, 2))),
                              Vec3(static_cast<float>(m_vertices(b, 0)), static_cast<float>(m_vertices(b, 1)), static_cast<float>(m_vertices(b, 2))),
                              edgeColor, 3.0f);
        }
    }
    void drawFittingNormals(Renderer& renderer) const {
        if (!m_showFittingNormals || m_fittingCentres.rows() != m_fittingNormals.rows()) return;
        const Color normalColor(0.0f, 0.75f, 0.95f, 1.0f);
        constexpr double normalLength = 0.06;
        for (int face = 0; face < m_fittingCentres.rows(); ++face) {
            const Eigen::Vector3d start = m_fittingCentres.row(face);
            const Eigen::Vector3d end = start + normalLength * m_fittingNormals.row(face).transpose();
            renderer.drawLine(Vec3(static_cast<float>(start.x()), static_cast<float>(start.y()), static_cast<float>(start.z())),
                              Vec3(static_cast<float>(end.x()), static_cast<float>(end.y()), static_cast<float>(end.z())),
                              normalColor, 1.0f);
        }
    }
    void drawOriginalWireframe(Renderer& renderer) const {
        if (m_originalVertices.rows() == 0) return;
        const Color referenceColor(0.55f, 0.55f, 0.55f, 1.0f);
        std::unordered_set<std::uint64_t> drawnEdges;
        for (int face = 0; face < m_faces.rows(); ++face) {
            for (int corner = 0; corner < 3; ++corner) {
                const int a = m_faces(face, corner);
                const int b = m_faces(face, (corner + 1) % 3);
                if (!drawnEdges.insert(edgeKey(a, b)).second) continue;
                const Vec3 start(static_cast<float>(m_originalVertices(a, 0)), static_cast<float>(m_originalVertices(a, 1)), static_cast<float>(m_originalVertices(a, 2)));
                const Vec3 end(static_cast<float>(m_originalVertices(b, 0)), static_cast<float>(m_originalVertices(b, 1)), static_cast<float>(m_originalVertices(b, 2)));
                renderer.drawLine(start, end, referenceColor, 1.0f);
            }
        }
    }
    void updateMesh() {
        std::vector<Vec3> positions;
        std::vector<std::vector<int>> faces;
        positions.reserve(m_vertices.rows());
        faces.reserve(m_faces.rows());
        for (int vertex = 0; vertex < m_vertices.rows(); ++vertex) {
            positions.emplace_back(static_cast<float>(m_vertices(vertex, 0)),
                                   static_cast<float>(m_vertices(vertex, 1)),
                                   static_cast<float>(m_vertices(vertex, 2)));
        }
        for (int face = 0; face < m_faces.rows(); ++face) {
            faces.push_back({m_faces(face, 0), m_faces(face, 1), m_faces(face, 2)});
        }
        m_mesh->createFromVerticesAndFaces(positions, faces);
        if (m_showCurvature) m_mesh->meanCurvature(true);
        m_mesh->setShowEdges(true);
    }

    std::shared_ptr<MeshObject> m_mesh;
    Eigen::MatrixXd m_vertices;
    Eigen::RowVector3d m_normalizationCenter{0.0, 0.0, 0.0};
    double m_normalizationScale{1.0};
    Eigen::MatrixXd m_originalVertices;
    Eigen::MatrixXd m_fittingCentres;
    Eigen::MatrixXd m_fittingNormals;
    Eigen::MatrixXi m_faces;
    Eigen::MatrixXd m_cotangents;
    Eigen::SparseMatrix<double> m_laplacian;
    Eigen::SparseMatrix<double> m_mass;
    Eigen::SparseMatrix<double> m_boundaryAnchors;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> m_solver;
    std::vector<std::vector<int>> m_adjacency;
    std::vector<int> m_boundaryVertices;
    std::string m_exportStatus;
    int m_iteration{0};
    int m_maxIterations{1000};
    bool m_running{false};
    bool m_showFittingNormals{true};
    bool m_showOriginalWireframe{true};
    bool m_showCurvature{false};
    float m_elapsed{0.0f};
    double m_minConeAngle{13.0};
    // double m_minConeAngle{9.0};
    double m_startConeAngle{25.0};
    double m_radius{0.07};
    double m_smooth{1e-4};
    double m_epsilon{0.05};
    double m_boundaryWeight{100.0};

   /*
   
   */
};

ALICE2_REGISTER_SKETCH_AUTO(GaussThinningSketch)

#endif
