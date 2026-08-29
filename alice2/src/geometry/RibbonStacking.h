#pragma once

#ifndef ALICE2_RIBBON_STACKING_H
#define ALICE2_RIBBON_STACKING_H

#include "DevelopableRibbon.h"

#include <vector>

namespace alice2 {

// Symmetric 2x2 tensor [xx xy; xy yy] in the strip's local tangent frame.
struct RibbonCurvatureTensor {
    double xx = 0.0;
    double xy = 0.0;
    double yy = 0.0;
};

struct RibbonPairCompatibility {
    double localCost = 0.0;
    double accumulatedCost = 0.0;
    double totalCost = 0.0;
    // True when the lower pairwise cost used B traversed in reverse. This is
    // pair-relative diagnostic data; the first ordering pass does not flip
    // individual strips globally.
    bool reversed = false;
};

struct RibbonStackingSettings {
    double localWeight = 1.0;
    double accumulatedWeight = 1.0;
    bool allowPairwiseReversal = true;
};

using RibbonStackingCostMatrix = std::vector<std::vector<RibbonPairCompatibility>>;

struct RibbonStackResult {
    std::vector<int> order;
    double totalCost = 0.0;
    RibbonStackingCostMatrix pairCosts;
};

RibbonCurvatureTensor ribbonCurvatureTensor(double bend, double rulingAngle);

// Computes tensor local and persistent/accumulated mismatch. reverseB uses
// the same reversal convention as ribbonSignatureDistance.
RibbonPairCompatibility ribbonStackingCost(const RibbonSignature& a,
                                           const RibbonSignature& b,
                                           const RibbonStackingSettings& settings = {},
                                           bool reverseB = false);

RibbonStackingCostMatrix buildRibbonStackingCostMatrix(const std::vector<RibbonSignature>& signatures,
                                                       const RibbonStackingSettings& settings = {});

std::vector<int> nearestNeighborStackOrder(const RibbonStackingCostMatrix& costs, int start);
double ribbonStackOrderCost(const std::vector<int>& order, const RibbonStackingCostMatrix& costs);
void improveRibbonStackOrderTwoOpt(std::vector<int>& order, const RibbonStackingCostMatrix& costs);
RibbonStackResult findBestRibbonStack(const std::vector<RibbonSignature>& signatures,
                                      const RibbonStackingSettings& settings = {});

// Directional solid-stack variant: C(lower, upper) compares the top side of
// lower with the normal-offset bottom side of upper.
RibbonStackResult findBestRibbonStack(const std::vector<RibbonSignature>& topSignatures,
                                      const std::vector<RibbonSignature>& bottomSignatures,
                                      const RibbonStackingSettings& settings = {});

} // namespace alice2

#endif // ALICE2_RIBBON_STACKING_H
