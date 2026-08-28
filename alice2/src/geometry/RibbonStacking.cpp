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
                const int a = order[begin - 1];
                const int b = order[begin];
                const int c = order[end];
                const int d = order[end + 1];
                const double previous = costs[a][b].totalCost + costs[c][d].totalCost;
                const double replacement = costs[a][c].totalCost + costs[b][d].totalCost;
                if (replacement + 1e-12 < previous) {
                    std::reverse(order.begin() + static_cast<std::ptrdiff_t>(begin),
                                 order.begin() + static_cast<std::ptrdiff_t>(end + 1));
                    changed = true;
                    break;
                }
            }
        }
    }
}

RibbonStackResult findBestRibbonStack(const std::vector<RibbonSignature>& signatures,
                                      const RibbonStackingSettings& settings) {
    RibbonStackResult result;
    result.pairCosts = buildRibbonStackingCostMatrix(signatures, settings);
    if (signatures.empty()) return result;

    result.totalCost = std::numeric_limits<double>::infinity();
    for (int start = 0; start < static_cast<int>(signatures.size()); ++start) {
        std::vector<int> candidate = nearestNeighborStackOrder(result.pairCosts, start);
        improveRibbonStackOrderTwoOpt(candidate, result.pairCosts);
        const double cost = ribbonStackOrderCost(candidate, result.pairCosts);
        if (cost < result.totalCost) {
            result.order = std::move(candidate);
            result.totalCost = cost;
        }
    }
    return result;
}

} // namespace alice2
