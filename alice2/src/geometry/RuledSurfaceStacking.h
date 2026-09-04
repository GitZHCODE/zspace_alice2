#pragma once

#ifndef ALICE2_RULED_SURFACE_STACKING_H
#define ALICE2_RULED_SURFACE_STACKING_H

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace alice2 {

// A plane expressed as n.dot(x) = d.  Face normals are consistently wound
// toward the positive stack direction for the supplied diagnostic surfaces.
struct RuledSurfacePlane {
    Eigen::Vector3d n = Eigen::Vector3d::UnitZ();
    double d = 0.0;
};

struct RuledSurfaceFace {
    std::array<Eigen::Vector3d, 4> vertices;
    RuledSurfacePlane plane;
};

struct RuledSurfaceRuling {
    Eigen::Vector3d left;
    Eigen::Vector3d right;
};

// A zero-thickness, piecewise-planar quad strip.  Rulings are retained for
// drawing/debugging; all computations operate on the planar faces.
struct RuledSurface {
    std::vector<RuledSurfaceRuling> rulings;
    std::vector<RuledSurfaceFace> faces;
};

struct RuledSurfaceBounds2D {
    Eigen::Vector2d min = Eigen::Vector2d::Zero();
    Eigen::Vector2d max = Eigen::Vector2d::Zero();
};

struct RuledSurfaceStackSolution {
    std::vector<int> order;
    // Translation along the stack direction, indexed by layer in order.
    std::vector<double> stackZ;
    double totalHeight = 0.0;
    // Selected physical orientation, indexed by original surface index.
    std::vector<bool> flippedBySurface;
};

// Controls a deterministic family of similar, finite wire sweeps.  Each
// generated strip has faceCount exactly planar quad faces and is single-valued
// along +Z, so it is appropriate for the first vertical-stacking prototype.
struct RuledSurfaceProceduralSettings {
    int surfaceCount = 6;
    int faceCount = 6;
    double length = 6.0;
    double width = 1.0;
    // 0 creates coincident baseline strips; 1 allows the full profile/tilt
    // variation used by the sketch.
    double randomness = 0.50;
    std::uint32_t seed = 12345u;
    // 0 keeps the nominal length. At 1, individual strips range from 15% to
    // 100% of it, allowing short sweeps in the same generated family.
    double lengthRandomness = 0.0;
    // Random turn between consecutive rulings, in the current face plane.
    // A value of 1 maps to a capped 20-degree per-face turn.
    double rulingRotation = 0.0;
};

// Construct a strip from consecutive left/right rulings.  Faces are wound so
// an ordinary XY strip has +Z normals.  Invalid/degenerate rulings are kept
// as faces with a zero normal, which makes direction validation fail clearly.
RuledSurface makeRuledSurface(const std::vector<RuledSurfaceRuling>& rulings);
// A physical 180-degree flip about the strip's local longitudinal axis. A
// result is returned only if the flipped geometry remains a +Z height field.
std::optional<RuledSurface> flipRuledSurfaceForStack(const RuledSurface& surface,
                                                      double epsilon = 1e-6);

RuledSurface makeFlatStraightRuledSurface();
RuledSurface makeBentRuledSurface();
RuledSurface makeTwistedRuledSurface();
// Additional controls and close variants used to exercise directional gaps
// and ordering decisions in the diagnostic sketch.
RuledSurface makeElevatedFlatRuledSurface();
RuledSurface makeBentVariantRuledSurface();
RuledSurface makeTwistedVariantRuledSurface();
std::vector<RuledSurface> makeDiagnosticRuledSurfaces();
std::vector<RuledSurface> makeProceduralRuledSurfaces(
    const RuledSurfaceProceduralSettings& settings = {});

bool isValidForStackDirection(const RuledSurface& surface,
                              const Eigen::Vector3d& stackDirection = Eigen::Vector3d::UnitZ(),
                              double epsilon = 1e-6);

double ruledSurfacePlaneHeight(const RuledSurfacePlane& plane, const Eigen::Vector2d& xy);
RuledSurfaceBounds2D ruledSurfaceBoundsXY(const RuledSurface& surface);
RuledSurfaceBounds2D ruledSurfaceGroupBoundsXY(const std::vector<RuledSurface>& surfaces);
int findRuledSurfaceFaceAtXY(const RuledSurface& surface,
                             const Eigen::Vector2d& xy,
                             double epsilon = 1e-9);
std::optional<double> ruledSurfaceHeightAtXY(const RuledSurface& surface,
                                             const Eigen::Vector2d& xy,
                                             double epsilon = 1e-9);

// G(lower, upper): the Z translation required to put upper above lower.
// This first prototype intentionally uses uniform samples rather than exact
// projected-quad clipping.
double sampledRuledSurfaceNestingGap(const RuledSurface& below,
                                     const RuledSurface& above,
                                     double clearance = 0.0,
                                     int resolution = 100);
Eigen::MatrixXd buildRuledSurfaceGapMatrix(const std::vector<RuledSurface>& surfaces,
                                           double clearance = 0.0,
                                           int resolution = 100);

// Exact hot-wire gap model. Every bounding ruling is extended to
// foamFootprint and the finite swept patch between the extended ruling pair
// is compared with every original finite face of the other surface. The pair
// cost enforces both possible cutting sweeps.
Eigen::MatrixXd buildExtendedSweepGapMatrix(const std::vector<RuledSurface>& surfaces,
                                            const RuledSurfaceBounds2D& foamFootprint,
                                            double clearance = 0.0);

// stackZ is indexed by the layer position in order, not by surface index.
std::vector<double> solveRuledSurfaceStackHeights(const std::vector<int>& order,
                                                  const Eigen::MatrixXd& gapMatrix);
double computeRuledSurfaceStackHeight(const std::vector<RuledSurface>& surfaces,
                                      const std::vector<int>& order,
                                      const std::vector<double>& stackZ);
RuledSurfaceStackSolution findBestRuledSurfaceStackBruteForce(
    const std::vector<RuledSurface>& surfaces,
    const Eigen::MatrixXd& gapMatrix);

// Uses exhaustive ordering through bruteForceLimit surfaces. Above that it
// uses a deterministic multi-start greedy ordering, then always resolves all
// lower-to-upper sampled-gap constraints for the selected order.
RuledSurfaceStackSolution findRuledSurfaceStack(
    const std::vector<RuledSurface>& surfaces,
    const Eigen::MatrixXd& gapMatrix,
    int bruteForceLimit = 8);

// Optimises stack order together with a binary physical flip state per
// surface. It starts unflipped, tests each individual flip against the full
// re-solved stack, and retains only height-improving flip states.
RuledSurfaceStackSolution findExtendedSweepStackWithFlips(
    const std::vector<RuledSurface>& surfaces,
    const RuledSurfaceBounds2D& foamFootprint,
    double clearance = 0.0,
    int bruteForceLimit = 8);

// Flip-aware variant with a selectable collision model. When
// useExtendedSweeps is false, finite strip gaps use the sampled baseline
// model; when true, box-extended ruling sweeps are used.
RuledSurfaceStackSolution findRuledSurfaceStackWithFlips(
    const std::vector<RuledSurface>& surfaces,
    const RuledSurfaceBounds2D& foamFootprint,
    double clearance,
    bool useExtendedSweeps,
    int sampledResolution = 100,
    int bruteForceLimit = 8);

} // namespace alice2

#endif // ALICE2_RULED_SURFACE_STACKING_H
