#pragma once

#ifndef ALICE2_DEV2PQ_REMESHER_H
#define ALICE2_DEV2PQ_REMESHER_H

#include "../objects/MeshObject.h"

#include <string>
#include <vector>

namespace alice2 {

    struct Dev2PqOptions {
        float stripSpacing{0.08f};
        float alignmentWeight{3.0f};
        bool useDirectionalCurlProjection{true};
    };

    struct Dev2PqResult {
        bool success{false};
        std::string diagnostic;
        std::vector<Vec3> faceCentres;
        std::vector<Vec3> rawRulings;
        std::vector<Vec3> optimizedRulings;
        std::vector<float> confidence;
        std::vector<float> scalarU;
        std::vector<std::vector<Vec3>> isolines;
        int singularityCount{0};
        float maxCurlBefore{0.0f};
        float maxCurlAfter{0.0f};
    };

    // Directional-backed field stage for Dev2PQ. It accepts polygonal meshes,
    // triangulates them internally, and uses a sign-symmetric power-2 field.
    class Dev2PqRemesher {
    public:
        Dev2PqResult remesh(const MeshData& mesh, const Dev2PqOptions& options = {}) const;
    };

} // namespace alice2

#endif // ALICE2_DEV2PQ_REMESHER_H
