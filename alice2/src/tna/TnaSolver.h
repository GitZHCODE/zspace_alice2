#pragma once

#ifndef ALICE2_TNA_SOLVER_H
#define ALICE2_TNA_SOLVER_H

#include "../objects/MeshObject.h"

#include <string>
#include <vector>

namespace alice2 {

// Stage 1 of TNA: the planar input mesh with one exterior polygon appended
// for every support-delimited boundary chain. The recorded chains will later
// provide the exterior faces needed by the dual force diagram.
struct TnaFormDiagram {
    bool success{false};
    std::string diagnostic;

    MeshData mesh;
    // Source vertex IDs accepted as supports. If the caller supplies none,
    // all topological boundary vertices are used.
    std::vector<int> supportVertices;
    // One chain per support-to-support boundary walk, inclusive of both
    // support endpoints. A boundary loop without supports is one group.
    std::vector<std::vector<int>> boundaryGroups;
    int appendedBoundaryFaces{0};
    int skippedDegenerateGroups{0};
};

class TnaSolver {
public:
    // Builds the topological form diagram only. Horizontal and vertical
    // equilibrium will be added as separate stages after topology is tested.
    TnaFormDiagram makeFormDiagram(const MeshData& input,
                                   const std::vector<int>& supportVertices = {}) const;
};

} // namespace alice2

#endif // ALICE2_TNA_SOLVER_H
