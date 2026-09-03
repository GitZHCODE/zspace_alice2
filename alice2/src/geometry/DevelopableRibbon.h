#pragma once

#ifndef ALICE2_DEVELOPABLE_RIBBON_H
#define ALICE2_DEVELOPABLE_RIBBON_H

#include "../objects/MeshObject.h"

#include <array>
#include <string>
#include <vector>

namespace alice2 {

// A single open, one-quad-wide strip.  Faces are stored in traversal order as
// [P[i], P[i + 1], Q[i + 1], Q[i]], independent of the input OBJ winding.
struct QuadRibbon {
    std::vector<Vec3> vertices;
    std::vector<std::array<int, 4>> faces;
    // Corresponding MeshData face index for every ordered ribbon face.
    std::vector<int> sourceFaceIndices;
    std::vector<int> railP;
    std::vector<int> railQ;
};

struct RibbonPlanarizationResult {
    int iterations = 0;
    float maxPlanarityError = 0.0f;
    bool converged = false;
    bool usedSequentialFallback = false;
};

struct RibbonSignature {
    int startFace = 0;
    int faceCount = 0;
    // Normalized arclength location of each descriptor sample. Empty means
    // uniformly spaced samples, retained for backwards-compatible callers.
    std::vector<double> station;
    std::vector<double> bend;
    std::vector<double> rulingAngle;
    // Physical length of the ruling at each sample. Empty means that width is
    // unavailable and should not contribute to a compatibility cost.
    std::vector<double> rulingLength;
};

struct RibbonMatch {
    int stripA = -1;
    int stripB = -1;
    double distance = 0.0;
    bool reversed = false;
};

// Validates the mesh as one connected, open quad chain and reconstructs its
// longitudinal rails. Returns false and fills diagnostic if that is not true.
bool orderRibbon(const MeshData& mesh, QuadRibbon& ribbon, std::string* diagnostic = nullptr);

// Largest scale-independent scalar-triple-product residual over all quads.
float maxRibbonPlanarityError(const QuadRibbon& ribbon);

// Shape-Up-style local/global planarisation. originalWeight keeps the result
// close to the positions present when this method is called. If its iteration
// budget cannot meet tolerance, an exact one-rail-fixed sequential projection
// is used as the documented drift-prone fallback.
RibbonPlanarizationResult planarizeRibbon(QuadRibbon& ribbon,
                                          int maxIterations = 50,
                                          float tolerance = 1e-5f,
                                          float originalWeight = 0.05f);

// Returns a parallel ribbon whose vertices are displaced by averaged incident
// face normals. Use a negative offset for the lower skin of a thick ribbon.
QuadRibbon offsetRibbonAlongVertexNormals(const QuadRibbon& ribbon, float offset);

// Builds signatures for every complete sliding window. A window of N faces
// has N-1 interior bend/ruling-angle samples, so N must be at least two.
std::vector<RibbonSignature> buildRibbonSignatures(const QuadRibbon& ribbon,
                                                    int facesPerStrip,
                                                    int stride = 0);

// Compares equal-length signatures. reverseB applies the convention induced
// by reversing strip traversal: bend changes sign and beta becomes pi-beta.
double ribbonSignatureDistance(const RibbonSignature& a,
                               const RibbonSignature& b,
                               double bendWeight = 1.0,
                               double rulingWeight = 1.0,
                               bool reverseB = false);

// Globally standardises bend and ruling-angle channels before exhaustive
// pairwise comparison, then returns the nearest topK distinct window pairs.
std::vector<RibbonMatch> findSimilarRibbonStrips(const std::vector<RibbonSignature>& signatures,
                                                  int topK = 10,
                                                  double bendWeight = 1.0,
                                                  double rulingWeight = 1.0);

} // namespace alice2

#endif // ALICE2_DEVELOPABLE_RIBBON_H
