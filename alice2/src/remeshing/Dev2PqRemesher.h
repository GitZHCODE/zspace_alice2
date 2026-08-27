#pragma once

#ifndef ALICE2_DEV2PQ_REMESHER_H
#define ALICE2_DEV2PQ_REMESHER_H

#include "../objects/MeshObject.h"

#include <memory>
#include <string>
#include <vector>

namespace alice2 {

    enum class Dev2PqCurvatureEstimator {
        MeshObject,
        Libigl
    };

    // A compact Dev2PQ-style prototype. It targets near-developable triangle
    // meshes: a curvature-derived ruling field is smoothed, integrated into
    // strip coordinates, and sampled into a planar quad-dominant mesh.
    struct Dev2PqOptions {
        float stripSpacing{0.08f};       // relative to the input bounding-box diagonal
        Dev2PqCurvatureEstimator curvatureEstimator{Dev2PqCurvatureEstimator::Libigl};
        int fieldIterations{40};
        float fieldSmoothing{0.62f};
        float confidenceThreshold{0.16f};
        int planarizationIterations{40};
        float planarityTolerance{1e-4f};
    };

    struct Dev2PqResult {
        bool success{false};
        std::string diagnostic;
        // Direct curvature-derived ruling candidates, before coherence filtering
        // and strip-coordinate integration alter the field.
        std::vector<Vec3> rawFaceRulings;
        std::vector<float> rawFaceConfidence;
        std::vector<Vec3> faceRulings;
        std::vector<float> faceConfidence;
        std::vector<float> scalarU;
        std::vector<float> scalarV;
        std::vector<std::vector<Vec3>> rulingIsolines;
        std::shared_ptr<MeshData> mesh;
        int quadCount{0};
        int planarFaceCount{0};
        float maxQuadNonPlanarity{0.0f};
    };

    class Dev2PqRemesher {
    public:
        Dev2PqResult remesh(const MeshData& mesh, const Dev2PqOptions& options = {}) const;
    };

} // namespace alice2

#endif // ALICE2_DEV2PQ_REMESHER_H
