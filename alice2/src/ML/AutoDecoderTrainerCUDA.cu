
// AutoDecoderTrainerCUDA.cu
#include "AutoDecoderTrainerCUDA.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cstdio>

#ifndef CUDA_CHECK
#define CUDA_CHECK(expr) do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "[CUDA] %s failed at %s:%d : %s\n", #expr, __FILE__, __LINE__, cudaGetErrorString(_err)); \
        abort(); \
    } \
} while(0)
#endif

// ---------------------- Device helpers ----------------------
__device__ inline float dtanh(float y) { // derivative using output y = tanh(x)
    return 1.f - y*y;
}

// weights/biases are flattened per layer in row-major (out x in)
// We pack all layers back-to-back; layerIn[k], layerOut[k] give sizes.
// Offsets are computed by scanning.
__device__ float* layerW(int layerIdx, float* Wflat, const int* layerIn, const int* layerOut) {
    // compute offset
    int offset = 0;
    for (int l=0; l<layerIdx; ++l) offset += layerOut[l]*layerIn[l];
    return Wflat + offset;
}
__device__ float* layerB(int layerIdx, float* Bflat, const int* layerOut) {
    int offset = 0;
    for (int l=0; l<layerIdx; ++l) offset += layerOut[l];
    return Bflat + offset;
}
__device__ int weightOffset(int upto, const int* layerIn, const int* layerOut) {
    int off=0; for(int l=0;l<upto;++l) off += layerOut[l]*layerIn[l]; return off;
}
__device__ int biasOffset(int upto, const int* layerOut) {
    int off=0; for(int l=0;l<upto;++l) off += layerOut[l]; return off;
}

// Mat-vec: y = W*x + b ; W[rows x cols] row-major
__device__ void matvec(const float* W, const float* x, const float* b,
                       float* y, int rows, int cols) {
    for (int r=0; r<rows; ++r) {
        float acc = b ? b[r] : 0.f;
        const float* wrow = W + r*cols;
        for (int c=0; c<cols; ++c) acc += wrow[c]*x[c];
        y[r] = acc;
    }
}

// In-place tanh on vector
__device__ void vec_tanh(float* a, int n) {
    for (int i=0;i<n;++i) a[i] = tanhf(a[i]);
}

// y  := y + alpha*x
__device__ void axpy(float* y, const float* x, float alpha, int n) {
    for (int i=0;i<n;++i) y[i] += alpha * x[i];
}

// Outer product update: W += alpha * (u ⊗ v) where W is [rows x cols] row-major
__device__ void sgd_rank1(float* W, const float* u, const float* v, float alpha, int rows, int cols) {
    for (int r=0; r<rows; ++r) {
        float ar = alpha * u[r];
        float* wrow = W + r*cols;
        for (int c=0; c<cols; ++c) wrow[c] += ar * v[c];
    }
}

// ---------------- Persistent kernel (single block OK) ----------------
extern "C" __global__
void adt_persistent_epoch(
    // network
    float* Wflat, float* Bflat,
    const int* layerIn, const int* layerOut, int numLayers,
    // data
    const float* Z, int numShapes, int latentDim,
    const float* coords, const float* targets, const int32_t* shapeIdx,
    const int32_t* order, int N, int coordDim,
    // hparams
    float lrW, float lrZ, float lambda,
    // scratch (shared or global)
    float* scratch,
    // stats
    float* lossOut, float* lastOut
) {
    // We'll use thread 0 to run the serial loop; other threads idle.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float* act0 = scratch;                          // max width = max(layerIn[0], layerOut[..])
    float* act1 = act0 + 1024;                      // same size scratch
    float* delta = act1 + 1024;                     // for backprop
    float* dInput = delta + 1024;                   // gradient wrt input of layer0
    // Safety cap: assume max layer width <= 1024; adapt as needed.
    // For compactness we allocate a fixed-size scratch; for production use, size dynamically.

    float totalLoss = 0.f;
    float last = 0.f;

    for (int it=0; it<N; ++it) {
        int idx = order[it];
        int s   = shapeIdx[idx];
        const float* coord = coords + idx*coordDim;
        float target = targets[idx];

        // Build input = [Z[s,*], coord[*]]
        int in0 = layerIn[0];
        // sanity
        // load latent
        for (int i=0;i<latentDim;++i) act0[i] = Z[s*latentDim + i];
        // load coord
        for (int i=0;i<coordDim;++i) act0[latentDim + i] = coord[i];

        // --- forward ---
        float* cur = act0;
        float* nxt = act1;
        for (int l=0; l<numLayers; ++l) {
            int rows = layerOut[l];
            int cols = layerIn[l];
            float* W = layerW(l, Wflat, layerIn, layerOut);
            float* B = layerB(l, Bflat, layerOut);
            matvec(W, cur, B, nxt, rows, cols);
            if (l < numLayers-1) {
                vec_tanh(nxt, rows);
            }
            // swap
            float* tmp = cur; cur = nxt; nxt = tmp;
        }
        float pred = cur[0];
        float diff = pred - target;
        float sampleLoss = diff*diff + lambda * 0.0f; // latent penalty added later for stats only
        last = sampleLoss;

        // --- backward --- (compute gradOut = 2*diff)
        // We'll backprop layer by layer, updating W/B immediately (SGD)
        float gradOut = 2.f*diff;

        // We need activations per layer for backprop; for brevity, do a second forward storing them.
        // (Still cheap because nets are small; and keeps code simple.)
        // a[l] = activation after layer l (post-activation), with a[-1] = input
        // We'll reuse scratch: act0/act1 toggling and keep a small array of pointers to copies.
        // Here we store them in-place in a fixed buffer.
        // Layout: we store only references by recomputing again (cheap & deterministic).

        // Recompute with storage
        // store input in act0_copy
        float* act_in0 = dInput; // reuse region
        for (int i=0;i<in0;++i) act_in0[i] = act0[i];
        // act per layer will be written into contiguous areas after dInput; but to keep it simple
        // we recompute on the fly during backward as well (two-pass technique).

        // Backward pass requires deltas per layer. We do classic backprop with immediate SGD.
        // We'll do layer L-1..0. First compute delta for output layer: size 1
        delta[0] = gradOut; // dL/dy for linear output

        // We also need previous activation vector to form rank-1 update
        // So we forward again to get per-layer pre-activation vectors.
        // Forward to store activations in a small array (max 16 layers assumed).
        const int MAXL=16;
        int inA[MAXL]; int outA[MAXL];
        float* aStore[MAXL+1]; // a[0]=input, a[l+1]=post-act of layer l
        aStore[0] = act_in0;
        int curIn = in0;
        cur = act_in0; nxt = act1;
        for (int l=0;l<numLayers;++l){
            int rows = layerOut[l], cols = layerIn[l];
            float* W = layerW(l, Wflat, layerIn, layerOut);
            float* B = layerB(l, Bflat, layerOut);
            matvec(W, cur, B, nxt, rows, cols);
            if (l < numLayers-1) vec_tanh(nxt, rows);
            aStore[l+1] = nxt; // store pointer to current buffer (overwritten next iter but we copy below)
            // copy to act0 region to preserve later
            for (int i=0;i<rows;++i) act0[i] = nxt[i];
            // swap: store saved to act0; move act0->nxt for next layer
            for (int i=0;i<rows;++i) nxt[i]=act0[i];
            cur = nxt; nxt = act1;
            inA[l] = cols; outA[l]=rows;
        }

        // Now do backward + SGD updates
        // Work arrays: curDelta size = outA[l]
        for (int l=numLayers-1; l>=0; --l) {
            int rows = outA[l];
            int cols = inA[l];
            float* W = layerW(l, Wflat, layerIn, layerOut);
            float* B = layerB(l, Bflat, layerOut);

            // Get a(l-1) (input to this layer) into act0; recompute if needed
            // We'll recompute activations up to layer l-1 quickly
            // For simplicity and determinism, recompute from input each time (still cheap for small nets).
            // Build input again:
            for (int i=0;i<latentDim;++i) act1[i] = Z[s*latentDim + i];
            for (int i=0;i<coordDim;++i) act1[latentDim + i] = coord[i];
            float* aPrev = act1;
            float* aNext = act0;
            for (int j=0;j<l;++j){
                float* Wj = layerW(j, Wflat, layerIn, layerOut);
                float* Bj = layerB(j, Bflat, layerOut);
                matvec(Wj, aPrev, Bj, aNext, outA[j], inA[j]);
                if (j < numLayers-1) vec_tanh(aNext, outA[j]);
                float* t=aPrev; aPrev=aNext; aNext=t;
            }
            // aPrev now holds a(l-1)
            // Update W and B: W += -lrW * (delta ⊗ aPrev^T), B += -lrW * delta
            sgd_rank1(W, delta, aPrev, -lrW, rows, cols);
            axpy(B, delta, -lrW, rows);

            if (l == 0) {
                // dInput = W^T * delta  (and tanh' applied to earlier layers handled in prev iterations)
                // For l==0 we also need the gradient wrt input vector to update latent part
                // Compute dInput (cols)
                for (int c=0;c<cols;++c){
                    float acc=0.f;
                    for (int r=0;r<rows;++r) acc += W[r*cols + c] * delta[r];
                    dInput[c] = acc;
                }
            }

            // Prepare delta for previous layer if any
            if (l>0){
                // d(aPrev) = W^T * delta  hadamard dtanh(aPrev)
                for (int c=0;c<cols;++c){
                    float acc=0.f;
                    for (int r=0;r<rows;++r) acc += W[r*cols + c] * delta[r];
                    // aPrev is post-activation for layer l-1 (tanh)
                    float ap = aPrev[c];
                    act1[c] = acc * dtanh(ap);
                }
                // move act1 -> delta
                for (int c=0;c<cols;++c) delta[c] = act1[c];
            }
        }

        // --- update latent ---
        // latent grad is first latentDim entries of dInput, add 2*lambda*z
        for (int j=0;j<latentDim;++j){
            float g = dInput[j] + 2.f*lambda * Z[s*latentDim + j];
            // SGD
            // Note: we cannot write Z (const) here because Z is const pointer;
            // in this kernel design we keep Z immutable and only report loss. For strict update,
            // we would pass Z as non-const and allow in-place update. Keep compatibility:
        }
        // We *do* need to update Z; so we cast away const-ness safely because we know it's our buffer.
        float* Zmut = const_cast<float*>(Z);
        for (int j=0;j<latentDim;++j){
            float g = dInput[j] + 2.f*lambda * Zmut[s*latentDim + j];
            Zmut[s*latentDim + j] -= lrZ * g;
            // accumulate latent penalty to the loss for stats
            sampleLoss += lambda * (Zmut[s*latentDim + j]*Zmut[s*latentDim + j]);
        }

        totalLoss += sampleLoss;
    }

    *lossOut = totalLoss / (float)N;
    *lastOut = last;
}

// ---------------------- Host side impl ----------------------

AutoDecoderTrainerCUDA::AutoDecoderTrainerCUDA() {}
AutoDecoderTrainerCUDA::~AutoDecoderTrainerCUDA(){ freeDevice_(); }

void AutoDecoderTrainerCUDA::setNetwork(const SimpleMLP& mlp, int numShapes, int latentDim, int coordDim){
    hostMLP_ = mlp;
    numShapes_ = numShapes;
    latentDim_ = latentDim;
    coordDim_ = coordDim;
    // init latents
    std::mt19937 rng(42);
    std::normal_distribution<float> N01(0.f, 0.01f);
    h_Z_.assign(numShapes_*latentDim_, 0.f);
    for (auto& v : h_Z_) v = N01(rng);
}

void AutoDecoderTrainerCUDA::setSamples(const std::vector<ADTSample>& samples){
    dataset_ = samples;
    order_.resize((int)samples.size());
    std::iota(order_.begin(), order_.end(), 0);
}

void AutoDecoderTrainerCUDA::allocDevice_(){
    freeDevice_();
    // flatten weights & biases
    numLayers_ = (int)hostMLP_.hidden.size() + 1;
    std::vector<int> layerIn, layerOut;
    int inDim = hostMLP_.inputDim;
    for (size_t i=0;i<hostMLP_.hidden.size();++i){
        layerIn.push_back(inDim);
        layerOut.push_back(hostMLP_.hidden[i]);
        inDim = hostMLP_.hidden[i];
    }
    layerIn.push_back(inDim);
    layerOut.push_back(hostMLP_.outputDim);

    CUDA_CHECK(cudaMalloc(&d_layerIn_,  layerIn.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_layerOut_, layerOut.size()*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_layerIn_,  layerIn.data(),  layerIn.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_layerOut_, layerOut.data(), layerOut.size()*sizeof(int), cudaMemcpyHostToDevice));

    int wCount=0, bCount=0;
    for (int l=0;l<numLayers_;++l){ wCount += layerOut[l]*layerIn[l]; bCount += layerOut[l]; }
    CUDA_CHECK(cudaMalloc(&d_W_, wCount*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_, bCount*sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_Z_, numShapes_*latentDim_*sizeof(float)));
    int N = (int)dataset_.size();
    CUDA_CHECK(cudaMalloc(&d_coords_,  N*coordDim_*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets_, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_shapeIdx_, N*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_order_,   N*sizeof(int32_t)));
}

void AutoDecoderTrainerCUDA::freeDevice_()
{
    // explicit helper to avoid lambda capture issues with NVCC
    auto freeIf = [](auto*& p) {
        if (p) {
            cudaFree(p);
            p = nullptr;
        }
    };

    freeIf(d_W_);
    freeIf(d_B_);
    freeIf(d_layerIn_);
    freeIf(d_layerOut_);
    freeIf(d_Z_);
    freeIf(d_coords_);
    freeIf(d_targets_);
    freeIf(d_shapeIdx_);
    freeIf(d_order_);
}

void AutoDecoderTrainerCUDA::uploadModel_(){
    // Flatten and upload
    int inDim = hostMLP_.inputDim;
    int offsetW=0, offsetB=0;
    for (size_t l=0;l<hostMLP_.hidden.size();++l){
        int rows = hostMLP_.hidden[l];
        int cols = inDim;
        CUDA_CHECK(cudaMemcpy((float*)d_W_ + offsetW, hostMLP_.weights[l].data(), rows*cols*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((float*)d_B_ + offsetB, hostMLP_.biases[l].data(), rows*sizeof(float), cudaMemcpyHostToDevice));
        offsetW += rows*cols; offsetB += rows; inDim = rows;
    }
    // output layer
    {
        int rows = hostMLP_.outputDim;
        int cols = inDim;
        CUDA_CHECK(cudaMemcpy((float*)d_W_ + offsetW, hostMLP_.weights.back().data(), rows*cols*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((float*)d_B_ + offsetB, hostMLP_.biases.back().data(), rows*sizeof(float), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_Z_, h_Z_.data(), h_Z_.size()*sizeof(float), cudaMemcpyHostToDevice));
}

void AutoDecoderTrainerCUDA::downloadModel_(){
    // read back weights and Z
    int inDim = hostMLP_.inputDim;
    int offsetW=0, offsetB=0;
    for (size_t l=0;l<hostMLP_.hidden.size();++l){
        int rows = hostMLP_.hidden[l];
        int cols = inDim;
        CUDA_CHECK(cudaMemcpy(hostMLP_.weights[l].data(), (float*)d_W_ + offsetW, rows*cols*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostMLP_.biases[l].data(),  (float*)d_B_ + offsetB, rows*sizeof(float), cudaMemcpyDeviceToHost));
        offsetW += rows*cols; offsetB += rows; inDim = rows;
    }
    {
        int rows = hostMLP_.outputDim;
        int cols = inDim;
        CUDA_CHECK(cudaMemcpy(hostMLP_.weights.back().data(), (float*)d_W_ + offsetW, rows*cols*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostMLP_.biases.back().data(),  (float*)d_B_ + offsetB, rows*sizeof(float), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaMemcpy(h_Z_.data(), d_Z_, h_Z_.size()*sizeof(float), cudaMemcpyDeviceToHost));
}

void AutoDecoderTrainerCUDA::uploadData_(){
    int N = (int)dataset_.size();
    std::vector<float> hcoords(N*coordDim_);
    std::vector<float> htargets(N);
    std::vector<int32_t> hshape(N);
    for (int i=0;i<N;++i){
        std::memcpy(&hcoords[i*coordDim_], dataset_[i].coord.data(), coordDim_*sizeof(float));
        htargets[i] = dataset_[i].target;
        hshape[i]   = dataset_[i].shapeIndex;
    }
    CUDA_CHECK(cudaMemcpy(d_coords_,  hcoords.data(),  hcoords.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets_, htargets.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_shapeIdx_,hshape.data(),   N*sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_order_,   order_.data(),   N*sizeof(int32_t), cudaMemcpyHostToDevice));
}

ADTStats AutoDecoderTrainerCUDA::train(const ADTConfig& cfg){
    if (dataset_.empty()) throw std::runtime_error("No dataset set.");
    if (latentDim_ + coordDim_ != hostMLP_.inputDim)
        throw std::runtime_error("MLP inputDim must equal latentDim + coordDim");

    allocDevice_();
    uploadModel_();
    uploadData_();

    int N = (int)dataset_.size();
    // make an order per epoch
    std::mt19937_64 rng(cfg.shuffleSeed);

    // Scratch buffer (4 * 1024 floats)
    float* d_scratch=nullptr;
    CUDA_CHECK(cudaMalloc(&d_scratch, 4*1024*sizeof(float)));
    float *d_loss, *d_last;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_last, sizeof(float)));

    ADTStats stats{};

    for (int e=0;e<cfg.epochs;++e){
        std::shuffle(order_.begin(), order_.end(), rng);
        CUDA_CHECK(cudaMemcpy(d_order_, order_.data(), N*sizeof(int32_t), cudaMemcpyHostToDevice));
        adt_persistent_epoch<<<1, 1>>>(
            (float*)d_W_, (float*)d_B_,
            d_layerIn_, d_layerOut_, numLayers_,
            d_Z_, numShapes_, latentDim_,
            d_coords_, d_targets_, d_shapeIdx_,
            d_order_, N, coordDim_,
            cfg.lrW, cfg.lrZ, cfg.lambda,
            d_scratch,
            d_loss, d_last
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        float h_loss=0.f, h_last=0.f;
        CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_last, d_last, sizeof(float), cudaMemcpyDeviceToHost));
        stats.avgLoss = h_loss;
        stats.lastLoss = h_last;
        stats.epochsRun++;
    }

    CUDA_CHECK(cudaFree(d_scratch));
    CUDA_CHECK(cudaFree(d_loss));
    CUDA_CHECK(cudaFree(d_last));

    downloadModel_();
    freeDevice_();
    return stats;
}
