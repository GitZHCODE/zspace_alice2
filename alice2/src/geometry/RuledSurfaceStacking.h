#pragma once

#ifndef ALICE2_RULED_SURFACE_STACKING_H
#define ALICE2_RULED_SURFACE_STACKING_H

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
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
};

// Construct a strip from consecutive left/right rulings.  Faces are wound so
// an ordinary XY strip has +Z normals.  Invalid/degenerate rulings are kept
// as faces with a zero normal, which makes direction validation fail clearly.
RuledSurface makeRuledSurface(const std::vector<RuledSurfaceRuling>& rulings);

RuledSurface makeFlatStraightRuledSurface();
RuledSurface makeBentRuledSurface();
RuledSurface makeTwistedRuledSurface();
// Additional controls and close variants used to exercise directional gaps
// and ordering decisions in the diagnostic sketch.
RuledSurface makeElevatedFlatRuledSurface();
RuledSurface makeBentVariantRuledSurface();
RuledSurface makeTwistedVariantRuledSurface();
std::vector<RuledSurface> makeDiagnosticRuledSurfaces();

bool isValidForStackDirection(const RuledSurface& surface,
                              const Eigen::Vector3d& stackDirection = Eigen::Vector3d::UnitZ(),
                              double epsilon = 1e-6);

double ruledSurfacePlaneHeight(const RuledSurfacePlane& plane, const Eigen::Vector2d& xy);
RuledSurfaceBounds2D ruledSurfaceBoundsXY(const RuledSurface& surface);
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

// stackZ is indexed by the layer position in order, not by surface index.
std::vector<double> solveRuledSurfaceStackHeights(const std::vector<int>& order,
                                                  const Eigen::MatrixXd& gapMatrix);
double computeRuledSurfaceStackHeight(const std::vector<RuledSurface>& surfaces,
                                      const std::vector<int>& order,
                                      const std::vector<double>& stackZ);
RuledSurfaceStackSolution findBestRuledSurfaceStackBruteForce(
    const std::vector<RuledSurface>& surfaces,
    const Eigen::MatrixXd& gapMatrix);

} // namespace alice2

#endif // ALICE2_RULED_SURFACE_STACKING_H
