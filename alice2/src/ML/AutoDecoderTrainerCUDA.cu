// AutoDecoderTrainerCUDA.cu  (v3: shared-memory scratch, no global aStore, no WT cache)
#include "AutoDecoderTrainerCUDA.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>
#include <random>
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

// ---------------- Device kernel ----------------
__device__ inline float dtanhf_out(float y) { return 1.f - y*y; }

// matvec (rows x cols) * x (+b) -> y ; one thread per row
template<bool WithBias>
__device__ inline void matvec_rows(const float* __restrict__ W,
                                   const float* __restrict__ x,
                                   const float* __restrict__ b,
                                   float* __restrict__ y,
                                   int rows, int cols,
                                   int tid, int T)
{
    for (int r = tid; r < rows; r += T) {
        float acc = WithBias ? b[r] : 0.f;
        const float* __restrict__ wrow = W + (size_t)r * (size_t)cols;
        #pragma unroll 4
        for (int c=0;c<cols;++c) acc = fmaf(wrow[c], x[c], acc);
        y[r] = acc;
    }
    __syncthreads();
}

// y = W^T * u ; W is (rows x cols) row-major; one thread per col
__device__ inline void WT_times_vec(const float* __restrict__ W,
                                    const float* __restrict__ u,
                                    float* __restrict__ y,
                                    int rows, int cols,
                                    int tid, int T)
{
    for (int c=tid; c<cols; c+=T) {
        float acc = 0.f;
        #pragma unroll 4
        for (int r=0;r<rows;++r) acc = fmaf(W[(size_t)r*(size_t)cols + (size_t)c], u[r], acc);
        y[c] = acc;
    }
    __syncthreads();
}

// rank-1: W += alpha * u v^T ; B += alpha * u  (parallel over columns)
__device__ inline void sgd_rank1(float* __restrict__ W,
                                 float* __restrict__ B,
                                 const float* __restrict__ u,
                                 const float* __restrict__ v,
                                 float alpha, int rows, int cols,
                                 int tid, int T)
{
    for (int r=0;r<rows;++r) {
        const float ar = alpha * u[r];
        for (int c=tid; c<cols; c+=T) {
            W[(size_t)r*(size_t)cols + (size_t)c] += ar * v[c];
        }
    }
    for (int r=tid; r<rows; r+=T) B[r] += alpha * u[r];
    __syncthreads();
}

extern "C" __global__
void adt_persistent_epoch(
    float* __restrict__ Wflat, float* __restrict__ Bflat,
    const int* __restrict__ layerIn, const int* __restrict__ layerOut, int numLayers,
    const int* __restrict__ wOffsets, const int* __restrict__ bOffsets,
    const float* __restrict__ Z_in, int numShapes, int latentDim,
    const float* __restrict__ coords, const float* __restrict__ targets, const int32_t* __restrict__ shapeIdx,
    const int32_t* __restrict__ order, int N, int coordDim,
    float lrW, float lrZ, float lambda, int maxWidth,
    float* __restrict__ lossOut, float* __restrict__ lastOut)
{
    extern __shared__ float smem[];
    // scratch layout in shared memory
    float* act0   = smem;                 // maxWidth
    float* act1   = act0   + maxWidth;    // maxWidth
    float* delta  = act1   + maxWidth;    // maxWidth
    float* dInput = delta  + maxWidth;    // layerIn[0]
    float* aStore = dInput + layerIn[0];  // (numLayers+1)*maxWidth
    auto A = [&](int l)->float* { return aStore + (size_t)l * (size_t)maxWidth; };

    const int tid = threadIdx.x;
    const int T   = blockDim.x;
    float* Z = const_cast<float*>(Z_in);

    float totalLoss = 0.f;
    float last = 0.f;

    for (int it=0; it<N; ++it) {
        const int idx = order[it];
        const int s   = shapeIdx[idx];
        const float* coord = coords + (size_t)idx * (size_t)coordDim;
        const float target = targets[idx];

        // input activation A(0) = [Z[s,*], coord]
        if (tid == 0) {
            for (int i=0;i<latentDim;++i) A(0)[i] = Z[(size_t)s*(size_t)latentDim + (size_t)i];
            for (int i=0;i<coordDim; ++i) A(0)[latentDim + i] = coord[i];
        }
        __syncthreads();

        // forward store in shared memory
        float* cur = A(0);
        for (int l=0; l<numLayers; ++l) {
            const int rows = layerOut[l];
            const int cols = layerIn[l];
            float* W  = Wflat + wOffsets[l];
            float* B  = Bflat + bOffsets[l];

            matvec_rows<true>(W, cur, B, act1, rows, cols, tid, T);
            if (l < numLayers-1) {
                for (int i=tid;i<rows;i+=T) act1[i] = tanhf(act1[i]);
                __syncthreads();
            }

            for (int i=tid;i<rows;i+=T) A(l+1)[i] = act1[i];
            __syncthreads();
            cur = A(l+1);
        }

        const float pred = A(numLayers)[0];
        const float diff = pred - target;
        float sampleLoss = diff*diff;
        if (tid == 0) delta[0] = 2.f*diff;
        __syncthreads();

        // backward + immediate SGD
        for (int l=numLayers-1; l>=0; --l) {
            const int rows = layerOut[l];
            const int cols = layerIn[l];
            float* W  = Wflat + wOffsets[l];
            float* B  = Bflat + bOffsets[l];
            float* aPrev = A(l);

            // W,B update
            sgd_rank1(W, B, delta, aPrev, -lrW, rows, cols, tid, T);

            // dInput for latent update (first layer) or prev delta
            WT_times_vec(W, delta, act0, rows, cols, tid, T);

            if (l == 0) {
                for (int i=tid;i<cols;i+=T) dInput[i] = act0[i];
                __syncthreads();
            }

            if (l > 0) {
                // apply dtanh on aPrev
                for (int i=tid;i<cols;i+=T) act0[i] *= (1.f - aPrev[i]*aPrev[i]);
                __syncthreads();
                for (int i=tid;i<cols;i+=T) delta[i] = act0[i];
                __syncthreads();
            }
        }

        // update latent
        for (int j=tid; j<latentDim; j+=T) {
            float zj = Z[(size_t)s*(size_t)latentDim + (size_t)j];
            float g  = dInput[j] + 2.f*lambda*zj;
            zj      -= lrZ * g;
            Z[(size_t)s*(size_t)latentDim + (size_t)j] = zj;
        }
        __syncthreads();

        if (tid == 0) {
            float lp=0.f;
            for (int j=0;j<latentDim;++j) { float zj = Z[(size_t)s*(size_t)latentDim + (size_t)j]; lp += lambda*(zj*zj); }
            totalLoss += sampleLoss + lp;
            last = sampleLoss + lp;
        }
        __syncthreads();
    }

    if (tid == 0) {
        *lossOut = totalLoss / (float)N;
        *lastOut = last;
    }
}

// ---------------- Host implementation ----------------
AutoDecoderTrainerCUDA::AutoDecoderTrainerCUDA() {}
AutoDecoderTrainerCUDA::~AutoDecoderTrainerCUDA(){ freeDevice_(); }

void AutoDecoderTrainerCUDA::setNetwork(const SimpleMLP& mlp, int numShapes, int latentDim, int coordDim){
    hostMLP_ = mlp;
    numShapes_ = numShapes;
    latentDim_ = latentDim;
    coordDim_  = coordDim;

    // init host latents
    std::mt19937 rng(42);
    std::normal_distribution<float> N01(0.f, 0.01f);
    h_Z_.assign((size_t)numShapes_ * (size_t)latentDim_, 0.f);
    for (auto& v : h_Z_) v = N01(rng);
}

void AutoDecoderTrainerCUDA::setSamples(const std::vector<ADTSample>& samples){
    dataset_ = samples;
    order_.resize((int)samples.size());
    std::iota(order_.begin(), order_.end(), 0);
}

void AutoDecoderTrainerCUDA::allocDevice_(){
    freeDevice_();

    // layer dims
    numLayers_ = (int)hostMLP_.hidden.size() + 1;
    h_layerIn_.clear(); h_layerOut_.clear();
    int inDim = hostMLP_.inputDim;
    for (size_t i=0;i<hostMLP_.hidden.size();++i){
        h_layerIn_.push_back(inDim);
        h_layerOut_.push_back(hostMLP_.hidden[i]);
        inDim = hostMLP_.hidden[i];
    }
    h_layerIn_.push_back(inDim);
    h_layerOut_.push_back(hostMLP_.outputDim);

    // offsets
    h_wOffsets_.assign(numLayers_+1, 0);
    h_bOffsets_.assign(numLayers_+1, 0);
    for (int l=0; l<numLayers_; ++l){
        h_wOffsets_[l+1] = h_wOffsets_[l] + h_layerOut_[l]*h_layerIn_[l];
        h_bOffsets_[l+1] = h_bOffsets_[l] + h_layerOut_[l];
    }

    // max width
    maxWidth_ = 0;
    for (int l=0; l<numLayers_; ++l) maxWidth_ = std::max(maxWidth_, h_layerOut_[l]);
    maxWidth_ = std::max(maxWidth_, h_layerIn_[0]);

    // device allocs
    CUDA_CHECK(cudaMalloc(&d_layerIn_,  sizeof(int)*numLayers_));
    CUDA_CHECK(cudaMalloc(&d_layerOut_, sizeof(int)*numLayers_));
    CUDA_CHECK(cudaMemcpy(d_layerIn_,  h_layerIn_.data(),  sizeof(int)*numLayers_, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_layerOut_, h_layerOut_.data(), sizeof(int)*numLayers_, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_wOffsets_, sizeof(int)*(numLayers_+1)));
    CUDA_CHECK(cudaMalloc(&d_bOffsets_, sizeof(int)*(numLayers_+1)));
    CUDA_CHECK(cudaMemcpy(d_wOffsets_, h_wOffsets_.data(), sizeof(int)*(numLayers_+1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bOffsets_, h_bOffsets_.data(), sizeof(int)*(numLayers_+1), cudaMemcpyHostToDevice));

    int wCount = h_wOffsets_.back();
    int bCount = h_bOffsets_.back();
    CUDA_CHECK(cudaMalloc(&d_W_,  sizeof(float)*wCount));
    CUDA_CHECK(cudaMalloc(&d_B_,  sizeof(float)*bCount));

    int N = (int)dataset_.size();
    CUDA_CHECK(cudaMalloc(&d_Z_,        sizeof(float)*(size_t)numShapes_*(size_t)latentDim_));
    CUDA_CHECK(cudaMalloc(&d_coords_,   sizeof(float)*(size_t)N*(size_t)coordDim_));
    CUDA_CHECK(cudaMalloc(&d_targets_,  sizeof(float)*(size_t)N));
    CUDA_CHECK(cudaMalloc(&d_shapeIdx_, sizeof(int32_t)*(size_t)N));
    CUDA_CHECK(cudaMalloc(&d_order_,    sizeof(int32_t)*(size_t)N));
}

void AutoDecoderTrainerCUDA::freeDevice_(){
    auto freeIf = [](auto*& p){ if (p) { cudaFree(p); p=nullptr; } };
    freeIf(d_W_);
    freeIf(d_B_);
    freeIf(d_layerIn_);
    freeIf(d_layerOut_);
    freeIf(d_wOffsets_);
    freeIf(d_bOffsets_);
    freeIf(d_Z_);
    freeIf(d_coords_);
    freeIf(d_targets_);
    freeIf(d_shapeIdx_);
    freeIf(d_order_);
}

void AutoDecoderTrainerCUDA::uploadModel_(){
    // flatten upload
    int inDim = hostMLP_.inputDim;
    int oW=0, oB=0;
    for (size_t l=0; l<hostMLP_.hidden.size(); ++l){
        const int rows = hostMLP_.hidden[l];
        const int cols = inDim;
        CUDA_CHECK(cudaMemcpy(d_W_ + oW, hostMLP_.weights[l].data(), sizeof(float)*rows*cols, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B_ + oB, hostMLP_.biases[l].data(),  sizeof(float)*rows,     cudaMemcpyHostToDevice));
        oW += rows*cols; oB += rows; inDim = rows;
    }
    // output
    {
        const int rows = hostMLP_.outputDim;
        const int cols = inDim;
        CUDA_CHECK(cudaMemcpy(d_W_ + oW, hostMLP_.weights.back().data(), sizeof(float)*rows*cols, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B_ + oB, hostMLP_.biases.back().data(),  sizeof(float)*rows,     cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_Z_, h_Z_.data(), sizeof(float)*h_Z_.size(), cudaMemcpyHostToDevice));
}

void AutoDecoderTrainerCUDA::downloadModel_(){
    int inDim = hostMLP_.inputDim;
    int oW=0, oB=0;
    for (size_t l=0; l<hostMLP_.hidden.size(); ++l){
        const int rows = hostMLP_.hidden[l];
        const int cols = inDim;
        CUDA_CHECK(cudaMemcpy(hostMLP_.weights[l].data(), d_W_ + oW, sizeof(float)*rows*cols, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostMLP_.biases[l].data(),  d_B_ + oB, sizeof(float)*rows,     cudaMemcpyDeviceToHost));
        oW += rows*cols; oB += rows; inDim = rows;
    }
    {
        const int rows = hostMLP_.outputDim;
        const int cols = inDim;
        CUDA_CHECK(cudaMemcpy(hostMLP_.weights.back().data(), d_W_ + oW, sizeof(float)*rows*cols, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostMLP_.biases.back().data(),  d_B_ + oB, sizeof(float)*rows,     cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaMemcpy(h_Z_.data(), d_Z_, sizeof(float)*h_Z_.size(), cudaMemcpyDeviceToHost));
}

void AutoDecoderTrainerCUDA::uploadData_(){
    const int N = (int)dataset_.size();
    std::vector<float> hcoords((size_t)N*(size_t)coordDim_);
    std::vector<float> htargets((size_t)N);
    std::vector<int32_t> hshape((size_t)N);
    for (int i=0;i<N;++i){
        std::memcpy(&hcoords[(size_t)i*(size_t)coordDim_], dataset_[i].coord.data(), sizeof(float)*(size_t)coordDim_);
        htargets[(size_t)i] = dataset_[i].target;
        hshape[(size_t)i]   = dataset_[i].shapeIndex;
    }
    CUDA_CHECK(cudaMemcpy(d_coords_,  hcoords.data(),  sizeof(float)*hcoords.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets_, htargets.data(), sizeof(float)*N,             cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_shapeIdx_,hshape.data(),   sizeof(int32_t)*N,           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_order_,   order_.data(),   sizeof(int32_t)*N,           cudaMemcpyHostToDevice));
}

ADTStats AutoDecoderTrainerCUDA::train(const ADTConfig& cfg){
    if (dataset_.empty()) throw std::runtime_error("No dataset set.");
    if (latentDim_ + coordDim_ != hostMLP_.inputDim)
        throw std::runtime_error("MLP inputDim must equal latentDim + coordDim");

    allocDevice_();
    uploadModel_();
    uploadData_();

    const int N = (int)dataset_.size();
    std::mt19937_64 rng(cfg.shuffleSeed);

    ADTStats stats{};

    // choose threads and shared mem size
    const int threads = 256;
    size_t shmemFloats = (size_t)(3*maxWidth_ + latentDim_ + (numLayers_+1)*maxWidth_);
    size_t shmemBytes  = shmemFloats * sizeof(float);

    float *d_loss=nullptr, *d_last=nullptr;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_last, sizeof(float)));

    for (int e=0; e<cfg.epochs; ++e){
        std::shuffle(order_.begin(), order_.end(), rng);
        CUDA_CHECK(cudaMemcpy(d_order_, order_.data(), sizeof(int32_t)*N, cudaMemcpyHostToDevice));

        adt_persistent_epoch<<<1, threads, shmemBytes>>>(
            d_W_, d_B_,
            d_layerIn_, d_layerOut_, numLayers_,
            d_wOffsets_, d_bOffsets_,
            d_Z_, numShapes_, latentDim_,
            d_coords_, d_targets_, d_shapeIdx_,
            d_order_, N, coordDim_,
            cfg.lrW, cfg.lrZ, cfg.lambda, maxWidth_,
            d_loss, d_last
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&stats.avgLoss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&stats.lastLoss, d_last, sizeof(float), cudaMemcpyDeviceToHost));
        stats.epochsRun++;
    }

    CUDA_CHECK(cudaFree(d_loss));
    CUDA_CHECK(cudaFree(d_last));

    downloadModel_();
    freeDevice_();
    return stats;
}
