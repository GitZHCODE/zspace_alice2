#include "RibbonStacking.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace alice2 {
namespace {

constexpr double kPi = 3.14159265358979323846;

double wrapAngle(double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

double frobeniusSquared(const RibbonCurvatureTensor& tensor) {
    return tensor.xx * tensor.xx + 2.0 * tensor.xy * tensor.xy + tensor.yy * tensor.yy;
}

RibbonCurvatureTensor subtract(const RibbonCurvatureTensor& a, const RibbonCurvatureTensor& b) {
    return {a.xx - b.xx, a.xy - b.xy, a.yy - b.yy};
}

RibbonCurvatureTensor add(const RibbonCurvatureTensor& a, const RibbonCurvatureTensor& b) {
    return {a.xx + b.xx, a.xy + b.xy, a.yy + b.yy};
}

bool validPair(const RibbonSignature& a, const RibbonSignature& b) {
    return !a.bend.empty() && a.bend.size() == a.rulingAngle.size() &&
           a.bend.size() == b.bend.size() && b.bend.size() == b.rulingAngle.size();
}

RibbonSignature reversedSignature(const RibbonSignature& signature) {
    RibbonSignature result = signature;
    std::reverse(result.bend.begin(), result.bend.end());
    std::reverse(result.rulingAngle.begin(), result.rulingAngle.end());
    for (double& bend : result.bend) bend = -bend;
    for (double& angle : result.rulingAngle) angle = wrapAngle(kPi - angle);
    return result;
}

struct OrientedStackOrder {
    std::vector<int> order;
    std::vector<bool> reversed;
    double cost = std::numeric_limits<double>::infinity();
};

} // namespace

RibbonCurvatureTensor ribbonCurvatureTensor(double bend, double rulingAngle) {
    const double sine = std::sin(rulingAngle);
    const double cosine = std::cos(rulingAngle);
    return {bend * sine * sine, -bend * sine * cosine, bend * cosine * cosine};
}

RibbonPairCompatibility ribbonStackingCost(const RibbonSignature& a,
                                           const RibbonSignature& b,
                                           const RibbonStackingSettings& settings,
                                           bool reverseB) {
    RibbonPairCompatibility result;
    result.reversed = reverseB;
    if (!validPair(a, b)) {
        result.localCost = result.accumulatedCost = result.totalCost = std::numeric_limits<double>::infinity();
        return result;
    }

    RibbonCurvatureTensor accumulated;
    double localSum = 0.0;
    double accumulatedSum = 0.0;
    for (size_t i = 0; i < a.bend.size(); ++i) {
        const size_t bIndex = reverseB ? a.bend.size() - 1 - i : i;
        const double bendB = reverseB ? -b.bend[bIndex] : b.bend[bIndex];
        const double betaB = reverseB ? wrapAngle(kPi - b.rulingAngle[bIndex]) : b.rulingAngle[bIndex];
        const RibbonCurvatureTensor delta = subtract(ribbonCurvatureTensor(a.bend[i], a.rulingAngle[i]),
                                                      ribbonCurvatureTensor(bendB, betaB));
        localSum += frobeniusSquared(delta);
        accumulated = add(accumulated, delta); // Unit station spacing as specified for the first pass.
        accumulatedSum += frobeniusSquared(accumulated);
    }
    const double sampleCount = static_cast<double>(a.bend.size());
    result.localCost = localSum / sampleCount;
    result.accumulatedCost = accumulatedSum / sampleCount;
    result.totalCost = std::max(0.0, settings.localWeight) * result.localCost +
                       std::max(0.0, settings.accumulatedWeight) * result.accumulatedCost;
    return result;
}

RibbonPairCompatibility ribbonStackingCostOriented(const RibbonSignature& a,
                                                   const RibbonSignature& b,
                                                   const RibbonStackingSettings& settings,
                                                   bool reverseA,
                                                   bool reverseB) {
    const RibbonSignature orientedA = reverseA ? reversedSignature(a) : a;
    const RibbonSignature orientedB = reverseB ? reversedSignature(b) : b;
    RibbonPairCompatibility result = ribbonStackingCost(orientedA, orientedB, settings, false);
    result.reversed = reverseB;
    return result;
}

RibbonStackingCostMatrix buildRibbonStackingCostMatrix(const std::vector<RibbonSignature>& signatures,
                                                       const RibbonStackingSettings& settings) {
    RibbonStackingCostMatrix costs(signatures.size(), std::vector<RibbonPairCompatibility>(signatures.size()));
    for (int a = 0; a < static_cast<int>(signatures.size()); ++a) {
        for (int b = a + 1; b < static_cast<int>(signatures.size()); ++b) {
            RibbonPairCompatibility best = ribbonStackingCost(signatures[a], signatures[b], settings, false);
            if (settings.allowPairwiseReversal) {
                const RibbonPairCompatibility reversed = ribbonStackingCost(signatures[a], signatures[b], settings, true);
                if (reversed.totalCost < best.totalCost) best = reversed;
            }
            costs[a][b] = costs[b][a] = best;
        }
    }
    return costs;
}

std::vector<int> nearestNeighborStackOrder(const RibbonStackingCostMatrix& costs, int start) {
    const int count = static_cast<int>(costs.size());
    if (start < 0 || start >= count) return {};
    std::vector<bool> used(count, false);
    std::vector<int> order;
    order.reserve(count);
    order.push_back(start);
    used[start] = true;
    while (static_cast<int>(order.size()) < count) {
        const int current = order.back();
        int next = -1;
        double bestCost = std::numeric_limits<double>::infinity();
        for (int candidate = 0; candidate < count; ++candidate) {
            if (!used[candidate] && costs[current][candidate].totalCost < bestCost) {
                next = candidate;
                bestCost = costs[current][candidate].totalCost;
            }
        }
        if (next < 0) return {};
        used[next] = true;
        order.push_back(next);
    }
    return order;
}

double ribbonStackOrderCost(const std::vector<int>& order, const RibbonStackingCostMatrix& costs) {
    double result = 0.0;
    for (size_t i = 1; i < order.size(); ++i) {
        const int a = order[i - 1];
        const int b = order[i];
        if (a < 0 || b < 0 || a >= static_cast<int>(costs.size()) || b >= static_cast<int>(costs.size())) {
            return std::numeric_limits<double>::infinity();
        }
        result += costs[a][b].totalCost;
    }
    return result;
}

void improveRibbonStackOrderTwoOpt(std::vector<int>& order, const RibbonStackingCostMatrix& costs) {
    if (order.size() < 4) return;
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t begin = 1; begin + 2 < order.size() && !changed; ++begin) {
            for (size_t end = begin + 1; end + 1 < order.size(); ++end) {
                std::vector<int> candidate = order;
                std::reverse(candidate.begin() + static_cast<std::ptrdiff_t>(begin),
                             candidate.begin() + static_cast<std::ptrdiff_t>(end + 1));
                if (ribbonStackOrderCost(candidate, costs) + 1e-12 < ribbonStackOrderCost(order, costs)) {
                    order = std::move(candidate);
                    changed = true;
                    break;
                }
            }
        }
    }
}

RibbonStackResult findBestRibbonStack(const std::vector<RibbonSignature>& signatures,
                                      const RibbonStackingSettings& settings) {
    return findBestRibbonStack(signatures, signatures, settings);
}

RibbonStackResult findBestRibbonStack(const std::vector<RibbonSignature>& topSignatures,
                                      const std::vector<RibbonSignature>& bottomSignatures,
                                      const RibbonStackingSettings& settings) {
    RibbonStackResult result;
    if (topSignatures.size() != bottomSignatures.size()) return result;
    const int count = static_cast<int>(topSignatures.size());
    result.pairCosts.assign(count, std::vector<RibbonPairCompatibility>(count));
    for (int lower = 0; lower < count; ++lower) {
        for (int upper = 0; upper < count; ++upper) {
            if (lower == upper) continue;
            RibbonPairCompatibility best = ribbonStackingCost(topSignatures[lower], bottomSignatures[upper], settings, false);
            if (settings.allowPairwiseReversal) {
                const RibbonPairCompatibility reversed = ribbonStackingCost(topSignatures[lower], bottomSignatures[upper], settings, true);
                if (reversed.totalCost < best.totalCost) best = reversed;
            }
            result.pairCosts[lower][upper] = best;
        }
    }
    if (topSignatures.empty()) return result;

    // A pairwise best reversal is not a physically valid stack: one strip
    // cannot be forward for one neighbour and reversed for the other.  Search
    // a small two-state (forward/reverse) path instead.  This remains the
    // requested nearest-neighbour + directed 2-opt heuristic, not an exact
    // Hamiltonian-path solver.
    const auto interfaceCost = [&](int lower, bool reverseLower, int upper, bool reverseUpper) {
        return ribbonStackingCostOriented(topSignatures[lower], bottomSignatures[upper], settings,
                                          reverseLower, reverseUpper).totalCost;
    };
    const auto evaluate = [&](OrientedStackOrder& candidate) {
        candidate.cost = 0.0;
        for (size_t layer = 1; layer < candidate.order.size(); ++layer) {
            candidate.cost += interfaceCost(candidate.order[layer - 1], candidate.reversed[layer - 1],
                                             candidate.order[layer], candidate.reversed[layer]);
        }
    };
    const auto improve = [&](OrientedStackOrder& candidate) {
        evaluate(candidate);
        bool changed = true;
        while (changed) {
            changed = false;

            // Coordinate descent selects one physical orientation per layer
            // against both of its neighbours.
            if (settings.allowPairwiseReversal) {
                for (size_t layer = 0; layer < candidate.order.size(); ++layer) {
                    OrientedStackOrder toggled = candidate;
                    toggled.reversed[layer] = !toggled.reversed[layer];
                    evaluate(toggled);
                    if (toggled.cost + 1e-12 < candidate.cost) {
                        candidate = std::move(toggled);
                        changed = true;
                    }
                }
            }

            // For a directed path, evaluate each full reversal rather than
            // using the invalid symmetric four-edge shortcut.
            for (size_t begin = 1; begin + 2 < candidate.order.size() && !changed; ++begin) {
                for (size_t end = begin + 1; end + 1 < candidate.order.size(); ++end) {
                    OrientedStackOrder swapped = candidate;
                    std::reverse(swapped.order.begin() + static_cast<std::ptrdiff_t>(begin),
                                 swapped.order.begin() + static_cast<std::ptrdiff_t>(end + 1));
                    std::reverse(swapped.reversed.begin() + static_cast<std::ptrdiff_t>(begin),
                                 swapped.reversed.begin() + static_cast<std::ptrdiff_t>(end + 1));
                    evaluate(swapped);
                    if (swapped.cost + 1e-12 < candidate.cost) {
                        candidate = std::move(swapped);
                        changed = true;
                        break;
                    }
                }
            }
        }
    };

    OrientedStackOrder best;
    for (int start = 0; start < count; ++start) {
        for (int startState = 0; startState < (settings.allowPairwiseReversal ? 2 : 1); ++startState) {
            OrientedStackOrder candidate;
            candidate.order.push_back(start);
            candidate.reversed.push_back(startState != 0);
            std::vector<bool> used(count, false);
            used[start] = true;
            while (static_cast<int>(candidate.order.size()) < count) {
                const int lower = candidate.order.back();
                const bool reverseLower = candidate.reversed.back();
                int next = -1;
                bool reverseNext = false;
                double nextCost = std::numeric_limits<double>::infinity();
                for (int upper = 0; upper < count; ++upper) {
                    if (used[upper]) continue;
                    for (int state = 0; state < (settings.allowPairwiseReversal ? 2 : 1); ++state) {
                        const double cost = interfaceCost(lower, reverseLower, upper, state != 0);
                        if (cost < nextCost) {
                            next = upper;
                            reverseNext = state != 0;
                            nextCost = cost;
                        }
                    }
                }
                if (next < 0) break;
                used[next] = true;
                candidate.order.push_back(next);
                candidate.reversed.push_back(reverseNext);
            }
            if (static_cast<int>(candidate.order.size()) != count) continue;
            improve(candidate);
            if (candidate.cost < best.cost) best = std::move(candidate);
        }
    }

    result.order = std::move(best.order);
    result.reversedInOrder = std::move(best.reversed);
    result.totalCost = best.cost;
    for (size_t layer = 1; layer < result.order.size(); ++layer) {
        result.interfaceCosts.push_back(ribbonStackingCostOriented(
            topSignatures[result.order[layer - 1]], bottomSignatures[result.order[layer]], settings,
            result.reversedInOrder[layer - 1], result.reversedInOrder[layer]));
    }
    return result;
}

} // namespace alice2
