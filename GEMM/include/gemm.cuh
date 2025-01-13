#include <cuda_fp16.h>
#include <cuda_fp8.h>
#ifndef GEMM_H
#define GEMM_H

typedef __nv_fp8_e5m2 e5m2;
typedef __nv_fp8_e4m3 e4m3;

namespace gemm{

// kernel1
cudaError_t GEMM(int m, int k, int n, 
    const e4m3* __restrict__ a_matrix,
    const e4m3* __restrict__ b_matrix,
    float* __restrict__ output_matrix);

// kernel2
cudaError_t GEMM4(int m, int k, int n, 
    const e4m3* __restrict__ a_matrix,
    const e4m3* __restrict__ b_matrix,
    float* __restrict__ output_matrix);

// kernel3
cudaError_t GEMM(int m, int k, int n, 
    const e5m2* __restrict__ a_matrix,
    const e5m2* __restrict__ b_matrix,
    float* __restrict__ output_matrix);

// kernel4
cudaError_t GEMM4(int m, int k, int n, 
    const e5m2* __restrict__ a_matrix,
    const e5m2* __restrict__ b_matrix,
    float* __restrict__ output_matrix);

// kernel5
cudaError_t GEMM(int m, int k, int n, 
    const e4m3* __restrict__ a_matrix,
    const e5m2* __restrict__ b_matrix,
    float* __restrict__ output_matrix);
// kernel6
cudaError_t GEMM4(int m, int k, int n, 
    const e4m3* __restrict__ a_matrix,
    const e5m2* __restrict__ b_matrix,
    float* __restrict__ output_matrix);

// kernel7
cudaError_t GEMM(int m, int k, int n, 
    const e5m2* __restrict__ a_matrix,
    const e4m3* __restrict__ b_matrix,
    float* __restrict__ output_matrix);
// kernel8
cudaError_t GEMM4(int m, int k, int n, 
    const e5m2* __restrict__ a_matrix,
    const e4m3* __restrict__ b_matrix,
    float* __restrict__ output_matrix);
} // namespace gemm

#endif