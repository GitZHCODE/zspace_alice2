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

RibbonCurvatureTensor scale(const RibbonCurvatureTensor& tensor, double value) {
    return {tensor.xx * value, tensor.xy * value, tensor.yy * value};
}

bool validSignature(const RibbonSignature& signature) {
    const size_t count = signature.bend.size();
    return count > 0 && count == signature.rulingAngle.size() &&
           (signature.station.empty() || signature.station.size() == count) &&
           (signature.rulingLength.empty() || signature.rulingLength.size() == count);
}

bool validPair(const RibbonSignature& a, const RibbonSignature& b) {
    return validSignature(a) && validSignature(b);
}

double stationAt(const RibbonSignature& signature, size_t index) {
    if (signature.station.size() == signature.bend.size()) return std::clamp(signature.station[index], 0.0, 1.0);
    if (signature.bend.size() == 1) return 0.5;
    return static_cast<double>(index) / static_cast<double>(signature.bend.size() - 1);
}

struct SignatureSample {
    RibbonCurvatureTensor tensor;
    double width = 0.0;
    bool hasWidth = false;
};

SignatureSample sampleSignature(const RibbonSignature& signature, double station) {
    const size_t count = signature.bend.size();
    size_t left = 0;
    size_t right = 0;
    if (station <= stationAt(signature, 0)) {
        right = 0;
    } else if (station >= stationAt(signature, count - 1)) {
        left = right = count - 1;
    } else {
        right = 1;
        while (right < count && stationAt(signature, right) < station) ++right;
        left = right - 1;
    }
    const double leftStation = stationAt(signature, left);
    const double rightStation = stationAt(signature, right);
    const double blend = rightStation > leftStation ?
        std::clamp((station - leftStation) / (rightStation - leftStation), 0.0, 1.0) : 0.0;
    const RibbonCurvatureTensor a = ribbonCurvatureTensor(signature.bend[left], signature.rulingAngle[left]);
    const RibbonCurvatureTensor b = ribbonCurvatureTensor(signature.bend[right], signature.rulingAngle[right]);
    SignatureSample result;
    result.tensor = add(scale(a, 1.0 - blend), scale(b, blend));
    if (signature.rulingLength.size() == count) {
        result.width = signature.rulingLength[left] +
                       (signature.rulingLength[right] - signature.rulingLength[left]) * blend;
        result.hasWidth = true;
    }
    return result;
}

RibbonSignature reversedSignature(const RibbonSignature& signature) {
    RibbonSignature result = signature;
    std::reverse(result.station.begin(), result.station.end());
    std::reverse(result.bend.begin(), result.bend.end());
    std::reverse(result.rulingAngle.begin(), result.rulingAngle.end());
    std::reverse(result.rulingLength.begin(), result.rulingLength.end());
    for (double& station : result.station) station = 1.0 - station;
    for (double& bend : result.bend) bend = -bend;
    for (double& angle : result.rulingAngle) angle = wrapAngle(kPi - angle);
    return result;
}

struct OrientedStackOrder {
    std::vector<int> order;
    std::vector<bool> reversed;
    double maxInterfaceCost = std::numeric_limits<double>::infinity();
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

    const RibbonSignature reversed = reverseB ? reversedSignature(b) : RibbonSignature{};
    const RibbonSignature& comparison = reverseB ? reversed : b;
    const size_t sampleCount = std::max(a.bend.size(), comparison.bend.size());
    const double deltaS = sampleCount > 1 ? 1.0 / static_cast<double>(sampleCount - 1) : 1.0;
    RibbonCurvatureTensor accumulated;
    for (size_t i = 0; i < sampleCount; ++i) {
        const double station = sampleCount > 1 ? static_cast<double>(i) * deltaS : 0.5;
        const double integrationWeight = sampleCount > 1 && (i == 0 || i + 1 == sampleCount) ?
            0.5 * deltaS : deltaS;
        const SignatureSample sampleA = sampleSignature(a, station);
        const SignatureSample sampleB = sampleSignature(comparison, station);
        const RibbonCurvatureTensor delta = subtract(sampleA.tensor, sampleB.tensor);
        result.localCost += frobeniusSquared(delta) * integrationWeight;
        accumulated = add(accumulated, scale(delta, deltaS));
        result.accumulatedCost += frobeniusSquared(accumulated) * integrationWeight;
        if (sampleA.hasWidth && sampleB.hasWidth) {
            const double meanWidth = std::max(1e-8, 0.5 * (std::abs(sampleA.width) + std::abs(sampleB.width)));
            const double relativeDifference = (sampleA.width - sampleB.width) / meanWidth;
            result.widthCost += relativeDifference * relativeDifference * integrationWeight;
        }
    }
    result.totalCost = std::max(0.0, settings.localWeight) * result.localCost +
                       std::max(0.0, settings.accumulatedWeight) * result.accumulatedCost +
                       std::max(0.0, settings.widthWeight) * result.widthCost;
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
        candidate.maxInterfaceCost = 0.0;
        for (size_t layer = 1; layer < candidate.order.size(); ++layer) {
            const double cost = interfaceCost(candidate.order[layer - 1], candidate.reversed[layer - 1],
                                              candidate.order[layer], candidate.reversed[layer]);
            candidate.cost += cost;
            candidate.maxInterfaceCost = std::max(candidate.maxInterfaceCost, cost);
        }
    };
    const auto better = [](const OrientedStackOrder& candidate, const OrientedStackOrder& current) {
        constexpr double epsilon = 1e-12;
        if (candidate.maxInterfaceCost + epsilon < current.maxInterfaceCost) return true;
        if (current.maxInterfaceCost + epsilon < candidate.maxInterfaceCost) return false;
        return candidate.cost + epsilon < current.cost;
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
                    if (better(toggled, candidate)) {
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
                    if (better(swapped, candidate)) {
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
            if (better(candidate, best)) best = std::move(candidate);
        }
    }

    result.order = std::move(best.order);
    result.reversedInOrder = std::move(best.reversed);
    result.maxInterfaceCost = best.maxInterfaceCost;
    result.totalCost = best.cost;
    for (size_t layer = 1; layer < result.order.size(); ++layer) {
        result.interfaceCosts.push_back(ribbonStackingCostOriented(
            topSignatures[result.order[layer - 1]], bottomSignatures[result.order[layer]], settings,
            result.reversedInOrder[layer - 1], result.reversedInOrder[layer]));
    }
    return result;
}

} // namespace alice2
