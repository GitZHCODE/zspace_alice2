
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(expr) do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "[CUDA] %s failed at %s:%d with %s\n", #expr, __FILE__, __LINE__, cudaGetErrorString(_err)); \
        abort(); \
    } \
} while(0)
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(expr) do { \
    cublasStatus_t _st = (expr); \
    if (_st != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "[cuBLAS] %s failed at %s:%d (status=%d)\n", #expr, __FILE__, __LINE__, (int)_st); \
        abort(); \
    } \
} while(0)
#endif
