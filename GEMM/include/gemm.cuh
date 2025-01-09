#include <cuda_fp16.h>
#include <cuda_fp8.h>
#ifndef GEMM_H
#define GEMM_H

typedef __nv_fp8_e5m2 e5m2;
typedef __nv_fp8_e4m3 e4m3;

namespace gemm{
// original version
cudaError_t GEMM(int m, int k, int n, 
    const e5m2* __restrict__ a_matrix,
    const e5m2* __restrict__ b_matrix,
    half* __restrict__ output_matrix);

cudaError_t GEMM(int m, int k, int n, 
    const e5m2* __restrict__ a_matrix,
    const e5m2* __restrict__ b_matrix,
    float* __restrict__ output_matrix);

cudaError_t GEMM(int m, int k, int n, 
    const e4m3* __restrict__ a_matrix,
    const e4m3* __restrict__ b_matrix,
    half* __restrict__ output_matrix);

cudaError_t GEMM(int m, int k, int n, 
    const e4m3* __restrict__ a_matrix,
    const e4m3* __restrict__ b_matrix,
    float* __restrict__ output_matrix);

cudaError_t GEMM(int m, int k, int n, 
    const e4m3* __restrict__ a_matrix,
    const e5m2* __restrict__ b_matrix,
    half* __restrict__ output_matrix);

cudaError_t GEMM(int m, int k, int n, 
    const e4m3* __restrict__ a_matrix,
    const e5m2* __restrict__ b_matrix,
    float* __restrict__ output_matrix);

cudaError_t GEMM(int m, int k, int n, 
    const e5m2* __restrict__ a_matrix,
    const e4m3* __restrict__ b_matrix,
    half* __restrict__ output_matrix);

cudaError_t GEMM(int m, int k, int n, 
    const e5m2* __restrict__ a_matrix,
    const e4m3* __restrict__ b_matrix,
    float* __restrict__ output_matrix);
    
} // namespace gemm

#endif