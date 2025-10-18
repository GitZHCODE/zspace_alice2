#pragma once
#include <alice2.h>
#include <fstream>
#include <utility>
#include <algorithm>
#include <limits>
#include <memory>
#include <random>
#include <cmath>

// Include nlohmann/json for JSON parsing
#include <nlohmann/json.hpp>

namespace DeepSDF {

// ----------------- Utility -----------------
inline float clampSDF(float d, float beta = 0.1f) {
    float v = d / beta;
    if (v < -1.f) v = -1.f;
    if (v >  1.f) v =  1.f;
    return v;
}

// ---------- Analytic SDFs ----------
inline float sdCircle(float x, float y, float r = 0.6f) {
    return std::sqrt(x*x + y*y) - r;
}
inline float sdBox(float x, float y, float hx = 0.55f, float hy = 0.55f) {
    float ax = std::fabs(x) - hx;
    float ay = std::fabs(y) - hy;
    float ox = std::max(ax, 0.0f);
    float oy = std::max(ay, 0.0f);
    float outside = std::sqrt(ox*ox + oy*oy);
    float inside  = std::max(ax, ay);
    return (inside <= 0.0f) ? inside : outside;
}
inline float sdTriangleUp(float x, float y, float s = 1.1f) {
    const float k = std::sqrt(3.0f);
    x = std::fabs(x);
    float d1 = (k*x + y) - s;
    float d2 = (k*x - y) - s;
    float d3 = -y - s * 0.3f;
    float outside = std::max(std::max(d1, d2), d3);
    if (outside > 0.0f) return outside;
    float de1 = (k*x + y) - s;
    float de2 = (k*x - y) - s;
    float de3 = -y - s * 0.3f;
    float m = std::max(std::max(de1, de2), de3);
    return m;
}
inline float labelFromSDF(float d, float eps = 0.02f) {
    if (d < -eps) return -1.0f;
    if (d >  eps) return +1.0f;
    return 0.0f;
}

// ---------- Sampler ----------
struct Sampler {
    Sampler(unsigned seed = 999) : rng(seed) {}

    struct GridDesc {
        std::vector<float> values;   // flattened (row-major) length = resX*resY
        int   resX = 0, resY = 0;
        float xMin = -1.f, xMax = 1.f;
        float yMin = -1.f, yMax = 1.f;
    };

    void clear() { custom_.clear(); }
    int  numShapes() const { return static_cast<int>(custom_.size()); }

    void addShapeGrid(const std::vector<float>& sdfValues,
                      int resX, int resY,
                      float xMin, float xMax,
                      float yMin, float yMax)
    {
        GridDesc g;
        g.values = sdfValues;
        g.resX = resX; g.resY = resY;
        g.xMin = xMin; g.xMax = xMax;
        g.yMin = yMin; g.yMax = yMax;
        custom_.push_back(std::move(g));
    }

    std::tuple<float,float,float> sampleForShape(int shapeIdx) {
        if (custom_.empty()) {
            return {0.f, 0.f, 0.f};
        }
        const int clampedIdx = std::clamp(shapeIdx, 0, numShapes() - 1);
        const GridDesc& g = custom_[clampedIdx];
        if (g.values.empty() || g.resX <= 0 || g.resY <= 0) {
            return {0.f, 0.f, 0.f};
        }

        std::uniform_real_distribution<float> Ux(g.xMin, g.xMax);
        std::uniform_real_distribution<float> Uy(g.yMin, g.yMax);
        const float x = Ux(rng);
        const float y = Uy(rng);
        const float d = sampleGridBilinear(g, x, y);
        return {x, y, clampSDF(d)};
    }

    // Legacy analytic helpers remain available for other places that still rely on them.
    static float sdf_analytic(int shapeIdx, float x, float y) {
        switch (shapeIdx) {
            case 0: return sdCircle(x, y, 0.6f);
            case 1: return sdBox   (x, y, 0.55f, 0.55f);
            case 2: return sdTriangleUp(x, y, 1.1f);
            default: return 1e9f;
        }
    }

    static float sdf(int shapeIdx, float x, float y) {
        return sdf_analytic(shapeIdx, x, y);
    }

private:
    static float sampleGridBilinear(const GridDesc& g, float x, float y) {
        if (g.values.empty() || g.resX <= 0 || g.resY <= 0) {
            return 0.f;
        }

        const float u = std::clamp((x - g.xMin) / (g.xMax - g.xMin), 0.f, 1.f);
        const float v = std::clamp((y - g.yMin) / (g.yMax - g.yMin), 0.f, 1.f);

        const float fx = u * float(g.resX - 1);
        const float fy = v * float(g.resY - 1);
        const int x0 = static_cast<int>(std::floor(fx));
        const int y0 = static_cast<int>(std::floor(fy));
        const int x1 = std::min(x0 + 1, g.resX - 1);
        const int y1 = std::min(y0 + 1, g.resY - 1);
        const float tx = fx - float(x0);
        const float ty = fy - float(y0);

        auto idx = [&](int ix, int iy) {
            return static_cast<size_t>(iy) * static_cast<size_t>(g.resX) + static_cast<size_t>(ix);
        };

        const float v00 = g.values[idx(x0, y0)];
        const float v10 = g.values[idx(x1, y0)];
        const float v01 = g.values[idx(x0, y1)];
        const float v11 = g.values[idx(x1, y1)];

        const float vx0 = v00 * (1.f - tx) + v10 * tx;
        const float vx1 = v01 * (1.f - tx) + v11 * tx;
        return vx0 * (1.f - ty) + vx1 * ty;
    }

    std::mt19937 rng;
    std::vector<GridDesc> custom_;
};

// ---------- TrainingDataSet ----------
// struct TrainingDataset {
//     std::vector<int>   shapeIdx;
//     std::vector<float> x, y, target, pred;

//     void reserve(size_t n){
//         shapeIdx.reserve(n); x.reserve(n); y.reserve(n); target.reserve(n); pred.reserve(n);
//     }
//     void add(int s, float xx, float yy, float t, float p){
//         shapeIdx.push_back(s); x.push_back(xx); y.push_back(yy); target.push_back(t); pred.push_back(p);
//     }
//     void clear(){
//         shapeIdx.clear(); x.clear(); y.clear(); target.clear(); pred.clear();
//     }
//     size_t size() const { return x.size(); }

//     // ---- NEW: persistence helpers ----
//     // Save all samples to a JSON file (newline-free, compact)
//     bool saveJSON(const std::string& path) const {
//         using nlohmann::json;
//         try {
//             json j;
//             j["version"] = 1;
//             j["count"]   = size();
//             j["shapeIdx"] = shapeIdx;
//             j["x"]        = x;
//             j["y"]        = y;
//             j["target"]   = target;
//             j["pred"]     = pred;
//             std::ofstream ofs(path, std::ios::binary);
//             if (!ofs) return false;
//             ofs << j.dump(); // compact
//             return true;
//         } catch (...) { return false; }
//     }

//     // Load samples from a JSON file (replaces current content)
//     bool loadJSON(const std::string& path) {
//         using nlohmann::json;
//         try {
//             std::ifstream ifs(path, std::ios::binary);
//             if (!ifs) return false;
//             json j; ifs >> j;

//             // basic validation
//             if (!j.contains("shapeIdx") || !j.contains("x") || !j.contains("y") ||
//                 !j.contains("target") || !j.contains("pred")) return false;

//             std::vector<int>   s  = j["shapeIdx"].get<std::vector<int>>();
//             std::vector<float> vx = j["x"].get<std::vector<float>>();
//             std::vector<float> vy = j["y"].get<std::vector<float>>();
//             std::vector<float> vt = j["target"].get<std::vector<float>>();
//             std::vector<float> vp = j["pred"].get<std::vector<float>>();

//             const size_t n = s.size();
//             if (vx.size()!=n || vy.size()!=n || vt.size()!=n || vp.size()!=n) return false;

//             shapeIdx = std::move(s);
//             x = std::move(vx);
//             y = std::move(vy);
//             target = std::move(vt);
//             pred = std::move(vp);
//             return true;
//         } catch (...) { return false; }
//     }

//     // Generate a default dataset (current behaviour): empty (fresh) or a tiny seed.
//     void generateDefault(bool withTinySeed=false) {
//         clear();
//         if (!withTinySeed) return;
//         // Minimal seed: one sample per shape at origin
//         for (int s=0; s<3; ++s) {
//             float d = clampSDF(Sampler::sdf(s, 0.f, 0.f));
//             add(s, 0.f, 0.f, d, d);
//         }
//     }
// };

}
