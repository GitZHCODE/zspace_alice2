#pragma once

#ifndef ALICE2_MIQ_FIELD_REMESHER_H
#define ALICE2_MIQ_FIELD_REMESHER_H

#include "../computeGeom/TensorField.h"
#include "../objects/MeshObject.h"

#include <array>
#include <string>
#include <vector>

namespace alice2 {

    struct MiqRemeshOptions {
        float targetSpacing{0.1f};
        unsigned int stiffnessIterations{5};
        unsigned int localIterations{5};
    };

    struct MiqGridLines {
        std::vector<std::vector<Vec3>> u;
        std::vector<std::vector<Vec3>> v;
    };

    struct MiqRemeshResult {
        bool success{false};
        std::string diagnostic;
        std::vector<Vec2> uv;
        std::vector<std::array<int, 3>> uvFaces;
        MiqGridLines gridLines;
        int seamVertexCount{0};
    };

    // Generic adapter for a coherent per-face tangent cross field. The first
    // experiment exposes MIQ's integer-grid isolines, not an all-quad mesh.
    class MiqFieldRemesher {
    public:
        MiqRemeshResult parameterize(const MeshData& mesh,
                                     const TensorField& field,
                                     const MiqRemeshOptions& options = {}) const;
    };

} // namespace alice2

#endif // ALICE2_MIQ_FIELD_REMESHER_H
