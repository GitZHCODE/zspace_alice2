
#include "CUDAUtils.h"
#include "AutoDecoderTrainerCUDA.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath>

#include <nlohmann/json.hpp>

// Include CPU MLP to read/write weights
// genericMLP is included via AutoDecoderTrainerCUDA.h

using std::size_t;

namespace alice2 {

using json = nlohmann::json;

// --- training state parity with CPU ---

void AutoDecoderTrainerCUDA::setLatentCodesHost(const std::vector<std::vector<float>>& Z)
{
    if ((int)Z.size() != numShapes_ || numShapes_ == 0)
        throw std::runtime_error("setLatentCodesHost: numShapes mismatch or trainer not initialized");
    if ((int)Z[0].size() != latentDim_)
        throw std::runtime_error("setLatentCodesHost: latentDim mismatch");

    h_latents_ = Z; // copy
    // flatten and upload
    std::vector<float> flat;
    flat.reserve((size_t)numShapes_ * latentDim_);
    for (const auto& row : h_latents_) flat.insert(flat.end(), row.begin(), row.end());
    CUDA_CHECK(cudaMemcpy(d_Z_, flat.data(), flat.size()*sizeof(float), cudaMemcpyHostToDevice));
    latentsInit_ = true; // skip random init
}

// ---------------- device kernels (B=1) ----------------

__global__ void kConcatInput(const float* __restrict__ z, const float* __restrict__ coord,
                             int latentDim, int coordDim, float* __restrict__ out)
{
    // out = [z | coord]
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int total = latentDim + coordDim;
    if (i < total) {
        out[i] = (i < latentDim) ? z[i] : coord[i - latentDim];
    }
}

__global__ void kAffineRowMajor(const float* __restrict__ W, const float* __restrict__ b,
                                const float* __restrict__ a_in,
                                int outDim, int inDim,
                                float* __restrict__ a_out)
{
    // z = W * a_in + b ; W row-major [out x in]
    int o = threadIdx.x + blockIdx.x * blockDim.x;
    if (o < outDim) {
        float acc = b[o];
        const float* wrow = W + o * inDim;
        for (int j = 0; j < inDim; ++j) acc += wrow[j] * a_in[j];
        a_out[o] = acc;
    }
}

__global__ void kTanhInplace(float* __restrict__ v, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) v[i] = tanhf(v[i]);
}

__global__ void kTanhDerivInplace(float* __restrict__ v, const float* __restrict__ activ, int n)
{
    // v *= (1 - activ^2) ; where activ is the pre-existing activation A_l
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        float a = activ[i];
        v[i] *= (1.0f - a * a);
    }
}

__global__ void kMSEGradAndLoss(float pred, float target, float* __restrict__ gradOut, float* __restrict__ lossOut)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float diff = pred - target;
        *gradOut = 2.0f * diff;
        *lossOut = diff * diff;
    }
}

__global__ void kLatentUpdate(float* __restrict__ zRow, const float* __restrict__ gradLatent,
                              int latentDim, float lrZ, float lambda)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < latentDim) {
        float g = gradLatent[i] + 2.0f * lambda * zRow[i];
        zRow[i] -= lrZ * g;
    }
}

__global__ void kSGDParamUpdate(float* __restrict__ W, float* __restrict__ b,
                                const float* __restrict__ delta, const float* __restrict__ prevAct,
                                int outDim, int inDim, float lr)
{
    int o = blockIdx.x; // one block per output neuron
    int j = threadIdx.x; // threads per input feature
    // update weights
    for (int idx = j; idx < inDim; idx += blockDim.x) {
        float grad = delta[o] * prevAct[idx];
        W[o * inDim + idx] -= lr * grad;
    }
    // update bias (thread 0)
    if (j == 0) b[o] -= lr * delta[o];
}

// delta_{l-1} = W^T * delta_l
__global__ void kBackpropDelta(const float* __restrict__ W, const float* __restrict__ deltaCur,
                               int outDim, int inDim, float* __restrict__ deltaPrev)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x; // index in previous layer
    if (i < inDim) {
        float acc = 0.0f;
        for (int o = 0; o < outDim; ++o) {
            acc += W[o * inDim + i] * deltaCur[o];
        }
        deltaPrev[i] = acc;
    }
}

// ---------------- host impl ----------------

AutoDecoderTrainerCUDA::AutoDecoderTrainerCUDA(MLP& cpuDecoder)
: model_(cpuDecoder), rng_(1337u) {}

AutoDecoderTrainerCUDA::~AutoDecoderTrainerCUDA() {
    freeDevice_();
}

void AutoDecoderTrainerCUDA::initialize(int numShapes, int latentDim, int coordinateDim)
{
    if (numShapes <= 0 || latentDim <= 0 || coordinateDim <= 0)
        throw std::runtime_error("AutoDecoderTrainerCUDA::initialize invalid dims");

    numShapes_ = numShapes;
    latentDim_ = latentDim;
    coordDim_  = coordinateDim;

    inputDim_ = latentDim_ + coordDim_;
    outputDim_ = 1;

    // Read CPU model structure
    // genericMLP.h uses row-major W[l][out][in]
    std::vector<int> layerDims;
    layerDims.push_back(model_.inputDim);
    for (int h : model_.hiddenDims) layerDims.push_back(h);
    layerDims.push_back(model_.outputDim);

    numLayers_ = (int)layerDims.size() - 1;
    layerIn_.resize(numLayers_);
    layerOut_.resize(numLayers_);
    for (int l = 0; l < numLayers_; ++l) {
        layerIn_[l]  = layerDims[l];
        layerOut_[l] = layerDims[l+1];
    }

    // Host latents
    h_latents_.assign((size_t)numShapes_, std::vector<float>((size_t)latentDim_, 0.0f));
    latentsInit_ = false;

    syncWeightsFromCPU_();
    allocDevice_();
}

void AutoDecoderTrainerCUDA::setSamples(const std::vector<AutoDecoderSample>& samples)
{
    if (inputDim_ != model_.inputDim)
        throw std::runtime_error("CUDA trainer: model input dim mismatch");

    dataset_ = samples;
    for (auto& s : dataset_) {
        if ((int)s.coordinate.size() != coordDim_) throw std::runtime_error("sample coordDim mismatch");
        if (s.shapeIndex < 0 || s.shapeIndex >= numShapes_) throw std::runtime_error("shape index OOB");
    }
}

void AutoDecoderTrainerCUDA::ensureLatentInit_(float stddev)
{
    if (latentsInit_) return;
    if (stddev <= 0.0f) stddev = 0.01f;
    std::normal_distribution<float> dist(0.0f, stddev);
    for (auto& row : h_latents_)
        for (auto& v : row) v = dist(rng_);
    // upload
    std::vector<float> flat; flat.reserve((size_t)numShapes_ * latentDim_);
    for (auto& row : h_latents_) flat.insert(flat.end(), row.begin(), row.end());
    CUDA_CHECK(cudaMemcpy(d_Z_, flat.data(), flat.size()*sizeof(float), cudaMemcpyHostToDevice));
    latentsInit_ = true;
}

void AutoDecoderTrainerCUDA::syncWeightsFromCPU_()
{
    // allocate raw device buffers for each layer & copy data
    freeDevice_(); // in case called again
    dW_raw_.resize(numLayers_, nullptr);
    db_raw_.resize(numLayers_, nullptr);
    dAct_raw_.resize(numLayers_ + 1, nullptr);   // A_0..A_L
    dDelta_raw_.resize(numLayers_ + 1, nullptr); // δ_0..δ_L

    for (int l = 0; l < numLayers_; ++l) {
        int out = layerOut_[l], in = layerIn_[l];
        CUDA_CHECK(cudaMalloc(&dW_raw_[l], out * in * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&db_raw_[l], out * sizeof(float)));

        // pack CPU weights row-major [out x in]
        std::vector<float> hostW(out * in);
        for (int o = 0; o < out; ++o)
            for (int j = 0; j < in; ++j)
                hostW[o*in + j] = model_.W[l][o][j];
        CUDA_CHECK(cudaMemcpy(dW_raw_[l], hostW.data(),
                              hostW.size()*sizeof(float), cudaMemcpyHostToDevice));

        std::vector<float> hostB(out);
        for (int o = 0; o < out; ++o) hostB[o] = model_.b[l][o];
        CUDA_CHECK(cudaMemcpy(db_raw_[l], hostB.data(),
                              hostB.size()*sizeof(float), cudaMemcpyHostToDevice));

        // allocate δ_l (note: index aligned with activations A_l)
        CUDA_CHECK(cudaMalloc(&dDelta_raw_[l+1], out * sizeof(float)));
    }

    // input activation A_0 and output activations A_l
    CUDA_CHECK(cudaMalloc(&dAct_raw_[0], layerIn_[0]*sizeof(float)));
    for (int l = 0; l < numLayers_; ++l) {
        CUDA_CHECK(cudaMalloc(&dAct_raw_[l+1], layerOut_[l]*sizeof(float)));
    }

    CUDA_CHECK(cudaMalloc(&dDelta_raw_[0], layerIn_[0] * sizeof(float)));
}

void AutoDecoderTrainerCUDA::syncWeightsToCPU_()
{
    for (int l = 0; l < numLayers_; ++l) {
        int out = layerOut_[l], in = layerIn_[l];
        std::vector<float> hostW(out * in), hostB(out);
        CUDA_CHECK(cudaMemcpy(hostW.data(), dW_raw_[l], hostW.size()*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostB.data(), db_raw_[l], hostB.size()*sizeof(float), cudaMemcpyDeviceToHost));
        for (int o = 0; o < out; ++o) {
            for (int j = 0; j < in; ++j) {
                model_.W[l][o][j] = hostW[o*in + j];
            }
            model_.b[l][o] = hostB[o];
        }
    }
}

void AutoDecoderTrainerCUDA::allocDevice_()
{
    // pointer arrays
    CUDA_CHECK(cudaMalloc(&d_W_, numLayers_ * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_b_, numLayers_ * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_act_, (numLayers_ + 1) * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_delta_, (numLayers_ + 1) * sizeof(float*)));

    CUDA_CHECK(cudaMemcpy(d_W_, dW_raw_.data(), numLayers_ * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_, db_raw_.data(), numLayers_ * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_act_, dAct_raw_.data(), (numLayers_ + 1) * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_delta_, dDelta_raw_.data(), (numLayers_ + 1) * sizeof(float*), cudaMemcpyHostToDevice));

    // input concat, target, loss
    CUDA_CHECK(cudaMalloc(&d_inputConcat_, (latentDim_ + coordDim_) * sizeof(float)));

    // latent table
    CUDA_CHECK(cudaMalloc(&d_Z_, (size_t)numShapes_ * latentDim_ * sizeof(float)));
}

void AutoDecoderTrainerCUDA::freeDevice_()
{
    auto freeVec = [](std::vector<float*>& v){
        for (auto* p : v) if (p) cudaFree(p);
        v.clear();
    };
    freeVec(dW_raw_);
    freeVec(db_raw_);
    freeVec(dAct_raw_);
    freeVec(dDelta_raw_);

    if (d_W_) cudaFree(d_W_), d_W_ = nullptr;
    if (d_b_) cudaFree(d_b_), d_b_ = nullptr;
    if (d_act_) cudaFree(d_act_), d_act_ = nullptr;
    if (d_delta_) cudaFree(d_delta_), d_delta_ = nullptr;
    if (d_inputConcat_) cudaFree(d_inputConcat_), d_inputConcat_ = nullptr;
    if (d_Z_) cudaFree(d_Z_), d_Z_ = nullptr;
}

void AutoDecoderTrainerCUDA::stepSample_(int shapeIndex, const float* h_coord, float target,
                                         float lrW, float lrZ, float lambda, float& outLoss)
{
    // Gather latent row
    float* zRow = d_Z_ + (size_t)shapeIndex * latentDim_;

    // Build A_0 = [z | coord]
    CUDA_CHECK(cudaMemcpy(dAct_raw_[0] + 0,           zRow,    latentDim_ * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dAct_raw_[0] + latentDim_,  h_coord,  coordDim_ * sizeof(float), cudaMemcpyHostToDevice));

    // Forward pass
    for (int l = 0; l < numLayers_; ++l) {
        const int in  = layerIn_[l];
        const int out = layerOut_[l];

        int block = 128;
        int grid  = (out + block - 1) / block;
        kAffineRowMajor<<<grid, block>>>(dW_raw_[l], db_raw_[l], dAct_raw_[l], out, in, dAct_raw_[l+1]);
        CUDA_CHECK(cudaGetLastError());

        // tanh on hidden layers only
        if (l < numLayers_ - 1) {
            int n   = out;
            int g2  = (n + block - 1) / block;
            kTanhInplace<<<g2, block>>>(dAct_raw_[l+1], n);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // Read prediction (outputDim == 1)
    float h_pred = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_pred, dAct_raw_[numLayers_], sizeof(float), cudaMemcpyDeviceToHost));

    // Compute loss and output grad on host
    float diff    = h_pred - target;
    float gradOut = 2.0f * diff;     // d(MSE)/d(pred) with no 1/2 factor
    outLoss       = diff * diff;

    // Set delta_L on device
    CUDA_CHECK(cudaMemcpy(dDelta_raw_[numLayers_], &gradOut, sizeof(float), cudaMemcpyHostToDevice));

    // Backward pass that mirrors CPU order exactly:
    // For l = L-1..0:
    //   1) deltaPrev = W[l]^T * deltaCur   (pre-update weights)
    //   2) if (l > 0) deltaPrev *= tanh'(activations[l])
    //   3) SGD update on W[l], b[l] using deltaCur and activations[l]
    for (int l = numLayers_ - 1; l >= 0; --l) {
        const int out = layerOut_[l];
        const int in  = layerIn_[l];

        // (1) Propagate to previous layer with current weights
        if (l > 0) {
            int block = 128, grid = (in + block - 1) / block;
            kBackpropDelta<<<grid, block>>>(dW_raw_[l], dDelta_raw_[l+1], out, in, dDelta_raw_[l]);
            CUDA_CHECK(cudaGetLastError());

            // (2) Apply tanh' on deltaPrev using activations[l] (post-activation at layer l)
            grid = (in + block - 1) / block;
            kTanhDerivInplace<<<grid, block>>>(dDelta_raw_[l], dAct_raw_[l], in);
            CUDA_CHECK(cudaGetLastError());
        }

        // (3) SGD update W,b using deltaCur and activations[l] (prev activations)
        int threads = std::min(256, ((in + 31) / 32) * 32);
        kSGDParamUpdate<<<out, threads>>>(dW_raw_[l], db_raw_[l],
                                        dDelta_raw_[l+1],  // deltaCur at layer l
                                        dAct_raw_[l],      // activations[l]
                                        out, in, lrW);
        CUDA_CHECK(cudaGetLastError());
    }
}

AutoDecoderTrainingStats AutoDecoderTrainerCUDA::train(const AutoDecoderTrainingConfig& cfg)
{
    AutoDecoderTrainingStats stats{};

    if (dataset_.empty()) return stats;
    if (model_.outputDim != 1) throw std::runtime_error("CUDA trainer expects outputDim=1");
    if (model_.inputDim != inputDim_) throw std::runtime_error("CUDA trainer inputDim mismatch");

    // --- Seed once like CPU ---
    if (epochsRun_ == 0) {
        rng_.seed(cfg.shuffleSeed);
    }

    // Ensure latents exist
    if (!latentsInit_) ensureLatentInit_(cfg.latentInitStd);

    // Order vector
    std::vector<size_t> order(dataset_.size());
    std::iota(order.begin(), order.end(), 0);

    // Host scratch buffers for parity (latent + grad copies are tiny)
    std::vector<float> z_before(latentDim_);
    std::vector<float> gradA0(inputDim_, 0.0f); // δ_0 size = inputDim (latentDim + coordDim)

    for (int ep = 0; ep < cfg.epochs; ++ep) {
        std::shuffle(order.begin(), order.end(), rng_);

        double epochLoss = 0.0;

        for (size_t idx : order) {
            const auto& s = dataset_[idx];
            const int shapeIndex = s.shapeIndex;

            // --- (a) read z BEFORE update for penalty parity ---
            CUDA_CHECK(cudaMemcpy(z_before.data(),
                                  d_Z_ + (size_t)shapeIndex * latentDim_,
                                  latentDim_ * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            double latentPenalty = 0.0;
            if (cfg.latentRegularization > 0.0f) {
                for (float v : z_before) latentPenalty += double(v) * double(v);
                latentPenalty *= double(cfg.latentRegularization);
            }

            // --- (b) one SGD step up to δ_0 on device, WITHOUT applying device-side latent update ---
            float sampleMSE = 0.0f;

            // stepSample_ currently does the device-side latent update.
            // For strict parity, we reroute latent update to host:
            //  -> duplicate stepSample_ logic here: forward, loss grad, backward to δ_0, W/b updates
            //  -> then perform latent update on host
            // To minimize surgery, we’ll keep stepSample_ but disable its latent update part.
            // If your current stepSample_ always updates latents, move that small kernel into a helper
            // and guard it behind a flag. For now, we emulate by:
            //   1) run stepSample_ as-is
            //   2) immediately overwrite z on device with host-updated z (this wins deterministically)

            stepSample_(shapeIndex, s.coordinate.data(), s.sdf,
                        cfg.learningRateWeights, /* lrZ unused here */ 0.0f,
                        cfg.latentRegularization, sampleMSE);

            // Fetch δ_0 (device) to host
            CUDA_CHECK(cudaMemcpy(gradA0.data(),
                                  dDelta_raw_[0],
                                  inputDim_ * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            // --- (c) host-side latent update for strict parity ---
            // Use grad wrt A_0's first latentDim entries + 2λz
            // z_host := z_before (we’ll update this and write back)
            for (int k = 0; k < latentDim_; ++k) {
                float g = gradA0[k] + 2.0f * cfg.latentRegularization * z_before[k];
                z_before[k] -= cfg.learningRateLatent * g;
            }

            // Write updated z back to device and to host mirror table
            CUDA_CHECK(cudaMemcpy(d_Z_ + (size_t)shapeIndex * latentDim_,
                                  z_before.data(),
                                  latentDim_ * sizeof(float),
                                  cudaMemcpyHostToDevice));
            h_latents_[shapeIndex] = z_before; // keep host mirror in sync

            // --- (d) accumulate loss (MSE + λ‖z_pre‖²) ---
            const double total = double(sampleMSE) + latentPenalty;
            epochLoss += total;

            lastBatchLoss = float(total);
            stats.lastLoss = lastBatchLoss;
            stats.totalSamples++;
        }

        runningAverageLoss = float(epochLoss / double(dataset_.size())); // last epoch avg
        epochsRun = ep + 1;
        stats.averageLoss     = runningAverageLoss;
        stats.epochsCompleted = epochsRun;
    }

    // Sync weights back to CPU model
    syncWeightsToCPU_();

    // Latents are already mirrored into h_latents_ each step; no need to read back now,
    // but keep this for safety in case of out-of-band changes:
    std::vector<float> flat((size_t)numShapes_ * latentDim_);
    CUDA_CHECK(cudaMemcpy(flat.data(), d_Z_, flat.size()*sizeof(float), cudaMemcpyDeviceToHost));
    for (int s = 0; s < numShapes_; ++s) {
        auto& row = h_latents_[(size_t)s];
        std::copy(flat.begin() + s*latentDim_, flat.begin() + (s+1)*latentDim_, row.begin());
    }

    epochsRun_ += cfg.epochs;
    return stats;
}

bool AutoDecoderTrainerCUDA::saveToJson(const std::string& filePath) const
{
    if (h_latents_.empty())
        return false;

    json root;

    json decoder;
    decoder["input_dim"] = model_.inputDim;
    decoder["output_dim"] = model_.outputDim;
    decoder["hidden_dims"] = model_.hiddenDims;
    decoder["weights"] = model_.W;
    decoder["biases"] = model_.b;
    root["decoder"] = decoder;

    root["latent_codes"] = h_latents_;

    json training;
    training["epochs_completed"] = epochsRun_;
    training["last_loss"] = lastBatchLoss;
    training["average_loss"] = runningAverageLoss;
    training["latent_regularization"] = lastLatentRegularization;
    training["samples_per_epoch"] = dataset_.size();
    root["training"] = training;

    json metadata;
    metadata["num_shapes"] = h_latents_.size();
    metadata["latent_dim"] = latentDim_;
    metadata["coordinate_dim"] = coordDim_;
    metadata["model_input_dim"] = model_.inputDim;
    metadata["model_output_dim"] = model_.outputDim;
    root["metadata"] = metadata;

    std::ofstream file(filePath);
    if (!file.is_open())
        return false;

    file << root.dump(4);
    return true;
}

} // namespace alice2
