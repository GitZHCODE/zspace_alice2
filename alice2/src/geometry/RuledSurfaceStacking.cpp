#include "RuledSurfaceStacking.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

namespace alice2 {
namespace {

constexpr double kDegenerateEpsilon = 1e-12;

Eigen::Vector2d projectXY(const Eigen::Vector3d& point) {
    return {point.x(), point.y()};
}

double cross2D(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    return a.x() * b.y() - a.y() * b.x();
}

void centreRulingsInXY(std::vector<RuledSurfaceRuling>& rulings) {
    if (rulings.empty()) return;
    Eigen::Vector2d minimum = Eigen::Vector2d::Constant(std::numeric_limits<double>::infinity());
    Eigen::Vector2d maximum = Eigen::Vector2d::Constant(-std::numeric_limits<double>::infinity());
    for (const RuledSurfaceRuling& ruling : rulings) {
        minimum = minimum.cwiseMin(projectXY(ruling.left));
        minimum = minimum.cwiseMin(projectXY(ruling.right));
        maximum = maximum.cwiseMax(projectXY(ruling.left));
        maximum = maximum.cwiseMax(projectXY(ruling.right));
    }
    const Eigen::Vector2d centre = 0.5 * (minimum + maximum);
    for (RuledSurfaceRuling& ruling : rulings) {
        ruling.left.x() -= centre.x();
        ruling.left.y() -= centre.y();
        ruling.right.x() -= centre.x();
        ruling.right.y() -= centre.y();
    }
}

bool pointInTriangle(const Eigen::Vector2d& point,
                     const Eigen::Vector2d& a,
                     const Eigen::Vector2d& b,
                     const Eigen::Vector2d& c,
                     double epsilon) {
    const double area = cross2D(b - a, c - a);
    if (std::abs(area) <= epsilon) return false;

    const double u = cross2D(b - a, point - a);
    const double v = cross2D(c - b, point - b);
    const double w = cross2D(a - c, point - c);
    return area > 0.0 ? (u >= -epsilon && v >= -epsilon && w >= -epsilon)
                      : (u <= epsilon && v <= epsilon && w <= epsilon);
}

std::pair<double, double> localZRange(const RuledSurface& surface) {
    double minZ = std::numeric_limits<double>::infinity();
    double maxZ = -std::numeric_limits<double>::infinity();
    for (const RuledSurfaceFace& face : surface.faces) {
        for (const Eigen::Vector3d& vertex : face.vertices) {
            minZ = std::min(minZ, vertex.z());
            maxZ = std::max(maxZ, vertex.z());
        }
    }
    return std::isfinite(minZ) ? std::pair<double, double>{minZ, maxZ}
                               : std::pair<double, double>{0.0, 0.0};
}

} // namespace

RuledSurface makeRuledSurface(const std::vector<RuledSurfaceRuling>& rulings) {
    RuledSurface result;
    result.rulings = rulings;
    if (rulings.size() < 2) return result;

    result.faces.reserve(rulings.size() - 1);
    for (size_t i = 0; i + 1 < rulings.size(); ++i) {
        // The order L_i, L_(i+1), R_(i+1), R_i gives an upward normal for
        // the conventional strip whose left edge has lower Y coordinates.
        RuledSurfaceFace face;
        face.vertices = {rulings[i].left, rulings[i + 1].left,
                         rulings[i + 1].right, rulings[i].right};
        const Eigen::Vector3d e0 = face.vertices[1] - face.vertices[0];
        const Eigen::Vector3d e1 = face.vertices[3] - face.vertices[0];
        const Eigen::Vector3d unnormalized = e0.cross(e1);
        const double length = unnormalized.norm();
        if (length > kDegenerateEpsilon) {
            face.plane.n = unnormalized / length;
        } else {
            face.plane.n = Eigen::Vector3d::Zero();
        }
        face.plane.d = face.plane.n.dot(face.vertices[0]);
        result.faces.push_back(face);
    }
    return result;
}

RuledSurface makeFlatStraightRuledSurface() {
    return makeRuledSurface({
        {{-3.0, -0.5, 0.0}, {-3.0, 0.5, 0.0}},
        {{-1.0, -0.5, 0.0}, {-1.0, 0.5, 0.0}},
        {{ 1.0, -0.5, 0.0}, { 1.0, 0.5, 0.0}},
        {{ 3.0, -0.5, 0.0}, { 3.0, 0.5, 0.0}}
    });
}

RuledSurface makeBentRuledSurface() {
    const std::array<double, 4> heights{0.0, 0.3, 0.6, 0.3};
    const std::array<double, 4> xs{-3.0, -1.0, 1.0, 3.0};
    std::vector<RuledSurfaceRuling> rulings;
    rulings.reserve(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) {
        rulings.push_back({{xs[i], -0.5, heights[i]}, {xs[i], 0.5, heights[i]}});
    }
    return makeRuledSurface(rulings);
}

RuledSurface makeTwistedRuledSurface() {
    // Parallel but sloped rulings keep each quad exactly planar.  Their
    // pronounced, alternating longitudinal slope creates strong normal
    // variation while preserving the single-valued Z assumption.
    const std::array<double, 5> xs{-3.0, -1.5, 0.0, 1.5, 3.0};
    const std::array<double, 5> baseZ{0.0, 0.45, -0.15, 0.55, 0.10};
    std::vector<RuledSurfaceRuling> rulings;
    rulings.reserve(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) {
        rulings.push_back({{xs[i], -0.5, baseZ[i]}, {xs[i], 0.5, baseZ[i] + 0.35}});
    }
    return makeRuledSurface(rulings);
}

RuledSurface makeElevatedFlatRuledSurface() {
    // Exact control: it is Surface 0 translated upward by 0.5.  With a 0.05
    // clearance, G(flat, elevatedFlat) is zero whereas the reverse is 0.55.
    return makeRuledSurface({
        {{-3.0, -0.5, 0.5}, {-3.0, 0.5, 0.5}},
        {{-1.0, -0.5, 0.5}, {-1.0, 0.5, 0.5}},
        {{ 1.0, -0.5, 0.5}, { 1.0, 0.5, 0.5}},
        {{ 3.0, -0.5, 0.5}, { 3.0, 0.5, 0.5}}
    });
}

RuledSurface makeBentVariantRuledSurface() {
    // A close counterpart of Surface 1: same footprint and profile, lifted
    // by 0.15.  This is a second directional control with non-flat faces.
    const std::array<double, 4> heights{0.15, 0.45, 0.75, 0.45};
    const std::array<double, 4> xs{-3.0, -1.0, 1.0, 3.0};
    std::vector<RuledSurfaceRuling> rulings;
    rulings.reserve(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) {
        rulings.push_back({{xs[i], -0.5, heights[i]}, {xs[i], 0.5, heights[i]}});
    }
    return makeRuledSurface(rulings);
}

RuledSurface makeTwistedVariantRuledSurface() {
    // Similar to Surface 3, but with a changed longitudinal profile and
    // ruling slope. Each face remains exactly planar because the difference
    // between the two boundary heights is constant along the strip.
    const std::array<double, 5> xs{-3.0, -1.5, 0.0, 1.5, 3.0};
    const std::array<double, 5> baseZ{0.10, 0.40, -0.10, 0.65, 0.20};
    std::vector<RuledSurfaceRuling> rulings;
    rulings.reserve(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) {
        rulings.push_back({{xs[i], -0.5, baseZ[i]}, {xs[i], 0.5, baseZ[i] + 0.25}});
    }
    return makeRuledSurface(rulings);
}

std::vector<RuledSurface> makeDiagnosticRuledSurfaces() {
    return {makeFlatStraightRuledSurface(), makeBentRuledSurface(), makeTwistedRuledSurface(),
            makeElevatedFlatRuledSurface(), makeBentVariantRuledSurface(), makeTwistedVariantRuledSurface()};
}

std::vector<RuledSurface> makeProceduralRuledSurfaces(
    const RuledSurfaceProceduralSettings& settings) {
    const int surfaceCount = std::clamp(settings.surfaceCount, 1, 50);
    const int faceCount = std::clamp(settings.faceCount, 2, 64);
    const double length = std::max(settings.length, 1e-6);
    const double nominalWidth = std::max(settings.width, 1e-6);
    const double randomness = std::clamp(settings.randomness, 0.0, 1.0);
    const double lengthRandomness = std::clamp(settings.lengthRandomness, 0.0, 1.0);
    const double rulingRotation = std::clamp(settings.rulingRotation, 0.0, 1.0);

    std::mt19937 random(settings.seed);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    constexpr double twoPi = 6.28318530717958647692;
    std::vector<RuledSurface> result;
    result.reserve(surfaceCount);

    for (int surface = 0; surface < surfaceCount; ++surface) {
        // Every surface starts in a comparable local Z frame.  Random values
        // change profile, width, and ruling slope, not a global Z placement;
        // the stack solver is solely responsible for final placement.
        const double phase = randomness * unit(random) * 0.5 * twoPi;
        const double secondaryPhase = randomness * unit(random) * 0.5 * twoPi;
        const double amplitude = 0.12 + randomness * (0.12 + 0.14 * (0.5 + 0.5 * unit(random)));
        const double rulingRise = randomness * 0.45 * unit(random);
        const double width = nominalWidth * (1.0 + randomness * 0.18 * unit(random));
        const double surfaceLength = length * (1.0 - 0.85 * lengthRandomness *
                                                std::uniform_real_distribution<double>(0.0, 1.0)(random));

        std::vector<double> heights(faceCount + 1, 0.0);
        for (int i = 0; i <= faceCount; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(faceCount);
            const double profile = 0.70 * std::sin(twoPi * t + phase) +
                                   0.30 * std::sin(2.0 * twoPi * t + secondaryPhase);
            const double localNoise = randomness * 0.06 * unit(random);
            heights[i] = amplitude * profile + localNoise;
        }
        // Remove arbitrary random vertical drift. This makes generated strips
        // comparable before the solver determines their stack translations.
        const double meanHeight = std::accumulate(heights.begin(), heights.end(), 0.0) /
                                  static_cast<double>(heights.size());
        for (double& height : heights) height -= meanHeight;

        std::vector<RuledSurfaceRuling> rulings;
        rulings.reserve(faceCount + 1);
        if (rulingRotation <= 1e-12) {
            for (int i = 0; i <= faceCount; ++i) {
                const double t = static_cast<double>(i) / static_cast<double>(faceCount);
                const double x = -0.5 * surfaceLength + t * surfaceLength;
                // The cross-strip Z difference is constant within this surface.
                // Together with linear interpolation between consecutive rulings,
                // that makes each four-vertex face exactly planar.
                rulings.push_back({{x, -0.5 * width, heights[i] - 0.5 * rulingRise},
                                    {x,  0.5 * width, heights[i] + 0.5 * rulingRise}});
            }
        } else {
            // Build one face at a time.  The following ruling is rotated in
            // the current face plane, so it is non-parallel to the preceding
            // ruling while the shared quad remains exactly planar.
            const Eigen::Vector3d initialRuling(0.0, width, rulingRise);
            const double rulingLength = initialRuling.norm();
            Eigen::Vector3d rulingDirection = initialRuling / rulingLength;
            Eigen::Vector3d centre(-0.5 * surfaceLength, 0.0, heights.front());
            rulings.push_back({centre - 0.5 * rulingLength * rulingDirection,
                                centre + 0.5 * rulingLength * rulingDirection});

            constexpr double kMaxTurnRadians = 0.3490658503988659; // 20 degrees
            const double stepLength = surfaceLength / static_cast<double>(faceCount);
            for (int i = 0; i < faceCount; ++i) {
                const double targetSlope = (heights[i + 1] - heights[i]) / stepLength;
                Eigen::Vector3d normalCandidate(-targetSlope, -rulingRise / width, 1.0);
                Eigen::Vector3d faceNormal = normalCandidate -
                    rulingDirection * normalCandidate.dot(rulingDirection);
                if (faceNormal.norm() <= kDegenerateEpsilon || faceNormal.z() <= 1e-4) {
                    faceNormal = Eigen::Vector3d::UnitZ() -
                        rulingDirection * rulingDirection.z();
                }
                faceNormal.normalize();
                if (faceNormal.z() < 0.0) faceNormal = -faceNormal;

                // Keep this orientation tied to rulingDirection x faceNormal:
                // reversing it later in the walk would reverse the generated
                // quad winding even though faceNormal itself points upward.
                Eigen::Vector3d advance = rulingDirection.cross(faceNormal).normalized();
                const double requestedTurn = kMaxTurnRadians * rulingRotation * unit(random);
                double acceptedTurn = requestedTurn;
                Eigen::Vector3d nextDirection = rulingDirection;
                // Avoid near-vertical wire directions, which would make the
                // next upward-facing supporting plane poorly conditioned.
                for (int attempt = 0; attempt < 5; ++attempt) {
                    nextDirection = Eigen::AngleAxisd(acceptedTurn, faceNormal) * rulingDirection;
                    if (std::abs(nextDirection.z()) <= 0.60) break;
                    acceptedTurn *= 0.5;
                }
                // A negative turn moves the next left endpoint back along
                // the advance direction. Move the ruling centre farther when
                // needed so the planar quad cannot fold over and reverse its
                // normal, especially on a short, high-density strip.
                const double noFoldAdvance = 0.5 * rulingLength *
                    std::max(0.0, -std::sin(acceptedTurn)) + 0.05 * stepLength;
                centre += advance * std::max(stepLength, noFoldAdvance);
                rulings.push_back({centre - 0.5 * rulingLength * nextDirection,
                                    centre + 0.5 * rulingLength * nextDirection});
                rulingDirection = nextDirection;
            }
        }
        // Generated sweeps all begin at a construction-side ruling. Recenter
        // their XY footprints before stacking so shorter or turned strips do
        // not artificially enlarge the group bounding box in one direction.
        centreRulingsInXY(rulings);
        result.push_back(makeRuledSurface(rulings));
    }
    return result;
}

bool isValidForStackDirection(const RuledSurface& surface,
                              const Eigen::Vector3d& stackDirection,
                              double epsilon) {
    const double length = stackDirection.norm();
    if (surface.faces.empty() || length <= kDegenerateEpsilon) return false;
    const Eigen::Vector3d direction = stackDirection / length;
    for (const RuledSurfaceFace& face : surface.faces) {
        if (face.plane.n.dot(direction) <= epsilon) return false;
    }
    return true;
}

double ruledSurfacePlaneHeight(const RuledSurfacePlane& plane, const Eigen::Vector2d& xy) {
    return (plane.d - plane.n.x() * xy.x() - plane.n.y() * xy.y()) / plane.n.z();
}

RuledSurfaceBounds2D ruledSurfaceBoundsXY(const RuledSurface& surface) {
    RuledSurfaceBounds2D result;
    if (surface.faces.empty()) return result;
    result.min = Eigen::Vector2d::Constant(std::numeric_limits<double>::infinity());
    result.max = Eigen::Vector2d::Constant(-std::numeric_limits<double>::infinity());
    for (const RuledSurfaceFace& face : surface.faces) {
        for (const Eigen::Vector3d& vertex : face.vertices) {
            result.min = result.min.cwiseMin(projectXY(vertex));
            result.max = result.max.cwiseMax(projectXY(vertex));
        }
    }
    return result;
}

int findRuledSurfaceFaceAtXY(const RuledSurface& surface,
                             const Eigen::Vector2d& xy,
                             double epsilon) {
    for (size_t index = 0; index < surface.faces.size(); ++index) {
        const auto& v = surface.faces[index].vertices;
        const Eigen::Vector2d a = projectXY(v[0]);
        const Eigen::Vector2d b = projectXY(v[1]);
        const Eigen::Vector2d c = projectXY(v[2]);
        const Eigen::Vector2d d = projectXY(v[3]);
        if (pointInTriangle(xy, a, b, c, epsilon) || pointInTriangle(xy, a, c, d, epsilon)) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

std::optional<double> ruledSurfaceHeightAtXY(const RuledSurface& surface,
                                             const Eigen::Vector2d& xy,
                                             double epsilon) {
    const int face = findRuledSurfaceFaceAtXY(surface, xy, epsilon);
    if (face < 0 || std::abs(surface.faces[face].plane.n.z()) <= epsilon) return std::nullopt;
    return ruledSurfacePlaneHeight(surface.faces[face].plane, xy);
}

double sampledRuledSurfaceNestingGap(const RuledSurface& below,
                                     const RuledSurface& above,
                                     double clearance,
                                     int resolution) {
    if (below.faces.empty() || above.faces.empty() || resolution < 1) return 0.0;
    const RuledSurfaceBounds2D lowerBounds = ruledSurfaceBoundsXY(below);
    const RuledSurfaceBounds2D upperBounds = ruledSurfaceBoundsXY(above);
    const Eigen::Vector2d minimum = lowerBounds.min.cwiseMax(upperBounds.min);
    const Eigen::Vector2d maximum = lowerBounds.max.cwiseMin(upperBounds.max);
    if (minimum.x() > maximum.x() || minimum.y() > maximum.y()) return 0.0;

    bool foundOverlap = false;
    double maxDifference = -std::numeric_limits<double>::infinity();
    for (int y = 0; y < resolution; ++y) {
        const double ty = resolution == 1 ? 0.5 : static_cast<double>(y) / (resolution - 1);
        for (int x = 0; x < resolution; ++x) {
            const double tx = resolution == 1 ? 0.5 : static_cast<double>(x) / (resolution - 1);
            const Eigen::Vector2d point = minimum + Eigen::Vector2d(tx, ty).cwiseProduct(maximum - minimum);
            const auto lowerHeight = ruledSurfaceHeightAtXY(below, point);
            const auto upperHeight = ruledSurfaceHeightAtXY(above, point);
            if (!lowerHeight || !upperHeight) continue;
            foundOverlap = true;
            maxDifference = std::max(maxDifference, *lowerHeight - *upperHeight);
        }
    }
    return foundOverlap ? std::max(0.0, maxDifference + std::max(0.0, clearance)) : 0.0;
}

Eigen::MatrixXd buildRuledSurfaceGapMatrix(const std::vector<RuledSurface>& surfaces,
                                           double clearance,
                                           int resolution) {
    const Eigen::Index count = static_cast<Eigen::Index>(surfaces.size());
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(count, count);
    for (Eigen::Index lower = 0; lower < count; ++lower) {
        for (Eigen::Index upper = 0; upper < count; ++upper) {
            if (lower != upper) {
                result(lower, upper) = sampledRuledSurfaceNestingGap(
                    surfaces[lower], surfaces[upper], clearance, resolution);
            }
        }
    }
    return result;
}

std::vector<double> solveRuledSurfaceStackHeights(const std::vector<int>& order,
                                                  const Eigen::MatrixXd& gapMatrix) {
    std::vector<double> result(order.size(), 0.0);
    for (size_t upperLayer = 1; upperLayer < order.size(); ++upperLayer) {
        const int upper = order[upperLayer];
        if (upper < 0 || upper >= gapMatrix.cols()) return {};
        for (size_t lowerLayer = 0; lowerLayer < upperLayer; ++lowerLayer) {
            const int lower = order[lowerLayer];
            if (lower < 0 || lower >= gapMatrix.rows()) return {};
            result[upperLayer] = std::max(result[upperLayer], result[lowerLayer] + gapMatrix(lower, upper));
        }
    }
    return result;
}

double computeRuledSurfaceStackHeight(const std::vector<RuledSurface>& surfaces,
                                      const std::vector<int>& order,
                                      const std::vector<double>& stackZ) {
    if (order.empty() || order.size() != stackZ.size()) return 0.0;
    double bottom = std::numeric_limits<double>::infinity();
    double top = -std::numeric_limits<double>::infinity();
    for (size_t layer = 0; layer < order.size(); ++layer) {
        const int surface = order[layer];
        if (surface < 0 || surface >= static_cast<int>(surfaces.size())) return std::numeric_limits<double>::infinity();
        const auto [minZ, maxZ] = localZRange(surfaces[surface]);
        bottom = std::min(bottom, stackZ[layer] + minZ);
        top = std::max(top, stackZ[layer] + maxZ);
    }
    return top - bottom;
}

RuledSurfaceStackSolution findBestRuledSurfaceStackBruteForce(
    const std::vector<RuledSurface>& surfaces,
    const Eigen::MatrixXd& gapMatrix) {
    RuledSurfaceStackSolution best;
    if (surfaces.empty() || gapMatrix.rows() != static_cast<Eigen::Index>(surfaces.size()) ||
        gapMatrix.cols() != static_cast<Eigen::Index>(surfaces.size())) return best;

    std::vector<int> order(surfaces.size());
    std::iota(order.begin(), order.end(), 0);
    best.totalHeight = std::numeric_limits<double>::infinity();
    do {
        const std::vector<double> stackZ = solveRuledSurfaceStackHeights(order, gapMatrix);
        const double height = computeRuledSurfaceStackHeight(surfaces, order, stackZ);
        if (height + 1e-12 < best.totalHeight) {
            best.order = order;
            best.stackZ = stackZ;
            best.totalHeight = height;
        }
    } while (std::next_permutation(order.begin(), order.end()));
    return best;
}

RuledSurfaceStackSolution findRuledSurfaceStack(
    const std::vector<RuledSurface>& surfaces,
    const Eigen::MatrixXd& gapMatrix,
    int bruteForceLimit) {
    const int count = static_cast<int>(surfaces.size());
    if (count == 0 || gapMatrix.rows() != count || gapMatrix.cols() != count) return {};
    if (count <= std::max(1, bruteForceLimit)) {
        return findBestRuledSurfaceStackBruteForce(surfaces, gapMatrix);
    }

    // A different starting surface can produce a different directed path, so
    // retain the best of all starts. At every step, select the next candidate
    // that gives the lowest actual stack height, not merely the lowest gap to
    // the immediately preceding layer.
    RuledSurfaceStackSolution best;
    best.totalHeight = std::numeric_limits<double>::infinity();
    for (int start = 0; start < count; ++start) {
        std::vector<int> order{start};
        std::vector<bool> used(count, false);
        used[start] = true;

        while (static_cast<int>(order.size()) < count) {
            int next = -1;
            double nextHeight = std::numeric_limits<double>::infinity();
            for (int candidate = 0; candidate < count; ++candidate) {
                if (used[candidate]) continue;
                std::vector<int> trial = order;
                trial.push_back(candidate);
                const std::vector<double> trialZ = solveRuledSurfaceStackHeights(trial, gapMatrix);
                const double trialHeight = computeRuledSurfaceStackHeight(surfaces, trial, trialZ);
                if (trialHeight + 1e-12 < nextHeight ||
                    (std::abs(trialHeight - nextHeight) <= 1e-12 && candidate < next)) {
                    next = candidate;
                    nextHeight = trialHeight;
                }
            }
            if (next < 0) return {};
            order.push_back(next);
            used[next] = true;
        }

        const std::vector<double> stackZ = solveRuledSurfaceStackHeights(order, gapMatrix);
        const double height = computeRuledSurfaceStackHeight(surfaces, order, stackZ);
        if (height + 1e-12 < best.totalHeight) {
            best.order = std::move(order);
            best.stackZ = stackZ;
            best.totalHeight = height;
        }
    }
    return best;
}

} // namespace alice2
