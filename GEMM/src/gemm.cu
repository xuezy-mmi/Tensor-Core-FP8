#include <cassert>
#include <chrono>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <time.h>
#include <type_traits>
#include <vector>
#include <stdio.h>
#include <mma.h>
#include <float.h>

using namespace nvcuda;
typedef __nv_fp8_e5m2 e5m2;
typedef __nv_fp8_e4m3 e4m3;
// static constexpr int BLOCKM = 128;
// static constexpr int BLOCKN = 128;
// static constexpr int BLOCKK = 32;

// #define HOST_DEVICE __forceinline__ __host__ __device__
// #define DEVICE __forceinline__ __device__

namespace gemm{

//kernel1////////////////e4m3 * e4m3 = f32//////////////////
__global__ void GEMM_e4m3_e4m3_o32_stage2_row_col(
    int M, int K, int N,
    const e4m3 * __restrict__ A_Value,
    const e4m3 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
    // thread num = 128
    constexpr int Block_M = 128;
    constexpr int Block_K = 64;
    constexpr int Block_N = 128;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int wid = tid / 32;// 128 / 32 = 4

    if(bx >= M / Block_M || by >= N / Block_N){
        return;
    }
    const int APAD = 0;
    const int BPAD = 0;

    // extern __shared__ float4 smem[];
    const int smem_a_offset = Block_M * (Block_K + APAD);
    const int smem_b_offset = Block_N * (Block_K + BPAD);
    // const int smem_size_float4 = 2 * (smem_a_offset + smem_b_offset) / 16;
    __shared__ e4m3 smem_a[smem_a_offset * 2];
    __shared__ e4m3 smem_b[smem_b_offset * 2];
    // extern __shared__ float4 smem[];
    // e4m3 * smem_a = reinterpret_cast<e4m3 *>(smem);
    // e4m3 * smem_b = reinterpret_cast<e4m3 *>(smem_a + 2 * smem_a_offset);

    float4 matrix_a_fragment[8];// 8 float4 = 32 reg
    float4 matrix_b_fragment[8];// 8 float4 = 32 reg
    float output_fragment[128];// 4(m) * 8(n) * 4(one tile) = 128 float

    int smem_a_m = (tid / 4) * 4;
    int smem_a_k = (tid % 4) * 16;
    int smem_b_k = (tid % 4) * 16;
    int smem_b_n = (tid / 4) * 4;
    int smem_a_base_addr = __cvta_generic_to_shared(smem_a);
    int smem_b_base_addr = __cvta_generic_to_shared(smem_b);
    // block_size = 128 * 64 
    // thred num = 128 
    // each thread load = 128 * 64 / 128 = 64 fp8 = 4 float4
    int smem_a_addr0 = smem_a_base_addr + (smem_a_m * (Block_K + APAD) + smem_a_k) * sizeof(char);
    int smem_a_addr1 = smem_a_addr0 + 1 * (Block_K + APAD) * sizeof(char);
    int smem_a_addr2 = smem_a_addr0 + 2 * (Block_K + APAD) * sizeof(char);
    int smem_a_addr3 = smem_a_addr0 + 3 * (Block_K + APAD) * sizeof(char);
    
    int smem_b_addr0 = smem_b_base_addr + (smem_b_n * (Block_K + BPAD) + smem_b_k) * sizeof(char);
    int smem_b_addr1 = smem_b_addr0 + 1 * (Block_K + BPAD) * sizeof(char);
    int smem_b_addr2 = smem_b_addr0 + 2 * (Block_K + BPAD) * sizeof(char);
    int smem_b_addr3 = smem_b_addr0 + 3 * (Block_K + BPAD) * sizeof(char);

    int gmem_a_m = bx * Block_M + smem_a_m;
    int gmem_a_k = smem_a_k;
    int gmem_b_k = smem_b_k;
    int gmem_b_n = by * Block_N + smem_b_n;

    int gmem_a_addr = gmem_a_m * K + gmem_a_k;
    int gmem_b_addr = gmem_b_n * K + gmem_b_k;

    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr0), "l"(&A_Value[gmem_a_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr1), "l"(&A_Value[gmem_a_addr + 1 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr2), "l"(&A_Value[gmem_a_addr + 2 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr3), "l"(&A_Value[gmem_a_addr + 3 * K]));

    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr0), "l"(&B_Value[gmem_b_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr1), "l"(&B_Value[gmem_b_addr + 1 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr2), "l"(&B_Value[gmem_b_addr + 2 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr3), "l"(&B_Value[gmem_b_addr + 3 * K]));

    asm ("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    int warp_x = wid % 2;//0 1 0 1
    int warp_y = wid / 2;//0 0 1 1
    #pragma unroll 32
    for(int bk = 1; bk < K/Block_K; bk++){
        int sel_com = (bk % 2) ^ 1; // 0 1 0 1
        int sel_mem = bk % 2; // 1 0 1 0
        gmem_a_addr += Block_K;
        gmem_b_addr += Block_K;
        
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_a_addr0 + sel_mem * smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_a_addr1 + sel_mem * smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr + 1 * K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_a_addr2 + sel_mem * smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr + 2 * K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_a_addr3 + sel_mem * smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr + 3 * K]));

        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_b_addr0 + sel_mem * smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_b_addr1 + sel_mem * smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr + 1 * K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_b_addr2 + sel_mem * smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr + 2 * K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_b_addr3 + sel_mem * smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr + 3 * K]));
        
        float4 * smem_a_sel = reinterpret_cast<float4 *>(smem_a + sel_com * smem_a_offset + warp_x * 64 * (Block_K+APAD));
        float4 * smem_b_sel = reinterpret_cast<float4 *>(smem_b + sel_com * smem_b_offset + warp_y * 64 * (Block_K+BPAD));
        matrix_a_fragment[0] = *(smem_a_sel + (lane_id % 4) + (int)(lane_id / 4 + 0 ) * (4+0));
        matrix_a_fragment[1] = *(smem_a_sel + (lane_id % 4) + (int)(lane_id / 4 + 8 ) * (4+0));
        matrix_a_fragment[2] = *(smem_a_sel + (lane_id % 4) + (int)(lane_id / 4 + 16) * (4+0));
        matrix_a_fragment[3] = *(smem_a_sel + (lane_id % 4) + (int)(lane_id / 4 + 24) * (4+0));
        matrix_a_fragment[4] = *(smem_a_sel + (lane_id % 4) + (int)(lane_id / 4 + 32) * (4+0));
        matrix_a_fragment[5] = *(smem_a_sel + (lane_id % 4) + (int)(lane_id / 4 + 40) * (4+0));
        matrix_a_fragment[6] = *(smem_a_sel + (lane_id % 4) + (int)(lane_id / 4 + 48) * (4+0));
        matrix_a_fragment[7] = *(smem_a_sel + (lane_id % 4) + (int)(lane_id / 4 + 56) * (4+0));
        matrix_b_fragment[0] = *(smem_b_sel + (lane_id % 4) + (int)(lane_id / 4 + 0 ) * (4+0));
        matrix_b_fragment[1] = *(smem_b_sel + (lane_id % 4) + (int)(lane_id / 4 + 8 ) * (4+0));
        matrix_b_fragment[2] = *(smem_b_sel + (lane_id % 4) + (int)(lane_id / 4 + 16) * (4+0));
        matrix_b_fragment[3] = *(smem_b_sel + (lane_id % 4) + (int)(lane_id / 4 + 24) * (4+0));
        matrix_b_fragment[4] = *(smem_b_sel + (lane_id % 4) + (int)(lane_id / 4 + 32) * (4+0));
        matrix_b_fragment[5] = *(smem_b_sel + (lane_id % 4) + (int)(lane_id / 4 + 40) * (4+0));
        matrix_b_fragment[6] = *(smem_b_sel + (lane_id % 4) + (int)(lane_id / 4 + 48) * (4+0));
        matrix_b_fragment[7] = *(smem_b_sel + (lane_id % 4) + (int)(lane_id / 4 + 56) * (4+0));

        int * a_fragment_int = reinterpret_cast<int *>(matrix_a_fragment);// 8 float4 --> 32 reg
        int * b_fragment_int = reinterpret_cast<int *>(matrix_b_fragment);// 8 float4 --> 32 reg

        #pragma unroll
        for(int i = 0; i < 4; i++){
            #pragma unroll
            for(int j = 0; j < 8; j++){
                asm ("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                    "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                    "r"(a_fragment_int[0 + 8 * i]), "r"(a_fragment_int[4 + 8 * i]),
                    "r"(a_fragment_int[1 + 8 * i]), "r"(a_fragment_int[5 + 8 * i]),
                    "r"(b_fragment_int[0 + 4 * j]), "r"(b_fragment_int[1 + 4 * j])
                );
                __syncthreads();
                asm ("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                    "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                    "r"(a_fragment_int[2 + 8 * i]), "r"(a_fragment_int[6 + 8 * i]),
                    "r"(a_fragment_int[3 + 8 * i]), "r"(a_fragment_int[7 + 8 * i]),
                    "r"(b_fragment_int[2 + 4 * j]), "r"(b_fragment_int[3 + 4 * j])
                );
            }
        }// end mma compute
        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);
        __syncthreads();

    }// end main loop
    int sel_com = ((K / Block_K) & 1) ^ 1;
    float4 * smem_a_sel = reinterpret_cast<float4 *>(smem_a + sel_com * smem_a_offset + warp_x * (smem_a_offset >> 1));
    float4 * smem_b_sel = reinterpret_cast<float4 *>(smem_b + sel_com * smem_b_offset + warp_y * (smem_b_offset >> 1));
    matrix_a_fragment[0] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 0 ) * (4+0));
    matrix_a_fragment[1] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 8 ) * (4+0));
    matrix_a_fragment[2] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 16) * (4+0));
    matrix_a_fragment[3] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 24) * (4+0));
    matrix_a_fragment[4] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 32) * (4+0));
    matrix_a_fragment[5] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 40) * (4+0));
    matrix_a_fragment[6] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 48) * (4+0));
    matrix_a_fragment[7] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 56) * (4+0));
    matrix_b_fragment[0] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 0 ) * (4+0));
    matrix_b_fragment[1] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 8 ) * (4+0));
    matrix_b_fragment[2] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 16) * (4+0));
    matrix_b_fragment[3] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 24) * (4+0));
    matrix_b_fragment[4] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 32) * (4+0));
    matrix_b_fragment[5] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 40) * (4+0));
    matrix_b_fragment[6] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 48) * (4+0));
    matrix_b_fragment[7] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 56) * (4+0));

    int * a_fragment_int = reinterpret_cast<int *>(matrix_a_fragment);// 8 float4 --> 32 reg
    int * b_fragment_int = reinterpret_cast<int *>(matrix_b_fragment);// 8 float4 --> 32 reg

    #pragma unroll
    for(int i = 0; i < 4; i++){
        #pragma unroll
        for(int j = 0; j < 8; j++){
            asm ("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 \t"
                "{%0, %1, %2, %3}, \t"
                "{%4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%0, %1, %2, %3}; ":
                "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                "r"(a_fragment_int[0 + 8 * i]), "r"(a_fragment_int[4 + 8 * i]),
                "r"(a_fragment_int[1 + 8 * i]), "r"(a_fragment_int[5 + 8 * i]),
                "r"(b_fragment_int[0 + 4 * j]), "r"(b_fragment_int[1 + 4 * j])
            );
            __syncthreads();
            asm ("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 \t"
                "{%0, %1, %2, %3}, \t"
                "{%4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%0, %1, %2, %3}; ":
                "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                "r"(a_fragment_int[2 + 8 * i]), "r"(a_fragment_int[6 + 8 * i]),
                "r"(a_fragment_int[3 + 8 * i]), "r"(a_fragment_int[7 + 8 * i]),
                "r"(b_fragment_int[2 + 4 * j]), "r"(b_fragment_int[3 + 4 * j])
            );
        }
    }// end mma compute
    // __syncthreads();
    int store_gmem_m = bx * Block_M + warp_x * 64;
    int store_gmem_n = by * Block_N + warp_y * 64;
    float2 * output_ = reinterpret_cast<float2 *>(Output_Value + (store_gmem_m + (int)(lane_id / 4)) * N + store_gmem_n + ((lane_id % 4) * 2));
    float2 * output_fragment_ = reinterpret_cast<float2 *>(output_fragment);// 128 reg --> 64 float2
    #pragma unroll
    for(int i = 0; i < 4; i++){
        #pragma unroll
        for(int j = 0; j < 8; j++){
            *(output_ + (int)(( (i*16  ) * N + j * 8 )/2)) = *(output_fragment_ + 2 * (i*8+j)    );
            *(output_ + (int)(( (i*16+8) * N + j * 8 )/2)) = *(output_fragment_ + 2 * (i*8+j) + 1);
        }
    }
}
cudaError_t GEMMex(
    int M, int K, int N,
    const e4m3 * __restrict__ A_Value,
    const e4m3 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
    const int Block_M = 128, Block_N = 128, Block_K = 64;
    dim3 block_dim(128,1,1);
	dim3 grid_dim(ceil(static_cast<float>(M) / Block_M), ceil(static_cast<float>(N) / Block_N), 1);

    GEMM_e4m3_e4m3_o32_stage2_row_col<<<grid_dim, block_dim>>>(
        M, K, N, A_Value, B_Value, Output_Value);

	return cudaGetLastError();
}
cudaError_t GEMM(
    int M, int K, int N,
    const e4m3 * __restrict__ A_Value,
    const e4m3 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
	return GEMMex(M, K, N, A_Value, B_Value, Output_Value);
}
//kernel2////////////////////e4m3 * e4m3 = f32//////////////////
__global__ void GEMM_e4m3_e4m3_o32_stage4_row_col(
    int M, int K, int N,
    const e4m3 * __restrict__ A_Value,
    const e4m3 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
    // thread num = 128
    constexpr int Block_M = 128;
    constexpr int Block_K = 64;
    constexpr int Block_N = 128;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int wid = tid / 32;// 128 / 32 = 4

    if(bx >= M / Block_M || by >= N / Block_N){
        return;
    }
    constexpr int APAD = 0;
    constexpr int BPAD = 0;


    constexpr int smem_a_offset = Block_M * (Block_K + APAD);
    constexpr int smem_b_offset = Block_N * (Block_K + BPAD);
    // const int smem_size_float4 = 4 * (smem_a_offset + smem_b_offset) / 16;
    // __shared__ float4 smem[smem_size_float4];
    __shared__ e4m3 smem_a[4 * smem_a_offset];
    __shared__ e4m3 smem_b[4 * smem_b_offset];
    // e4m3 * smem_a = smem;
    // e4m3 * smem_b = smem_a + 4 * smem_a_offset;

    // __align__(32) float4 matrix_a_fragment[8];// 8 float4 = 32 reg
    // __align__(32) float4 matrix_b_fragment[8];// 8 float4 = 32 reg
    // __align__(32) float output_fragment[128];// 4(m) * 8(n) * 4(one tile) = 128 float
    float4 matrix_a_fragment[8];// 8 float4 = 32 reg
    float4 matrix_b_fragment[8];// 8 float4 = 32 reg
    float output_fragment[128];// 4(m) * 8(n) * 4(one tile) = 128 float

    int smem_a_base_addr = __cvta_generic_to_shared(smem_a);
    int smem_b_base_addr = __cvta_generic_to_shared(smem_b);
    int smem_a_m = (tid / 4) * 4;
    int smem_a_k = (tid % 4) * 16;
    int smem_b_k = (tid % 4) * 16;
    int smem_b_n = (tid / 4) * 4;
    // block_size = 128 * 64 
    // thred num = 128 
    // each thread load = 128 * 64 / 128 = 64 fp8 = 4 float4
    int smem_a_addr0 = smem_a_base_addr + (smem_a_m * (Block_K + APAD) + smem_a_k) * sizeof(char);
    int smem_a_addr1 = smem_a_addr0 + 1 * (Block_K + APAD) * sizeof(char);
    int smem_a_addr2 = smem_a_addr0 + 2 * (Block_K + APAD) * sizeof(char);
    int smem_a_addr3 = smem_a_addr0 + 3 * (Block_K + APAD) * sizeof(char);
    
    int smem_b_addr0 = smem_b_base_addr + (smem_b_n * (Block_K + BPAD) + smem_b_k) * sizeof(char);
    int smem_b_addr1 = smem_b_addr0 + 1 * (Block_K + BPAD) * sizeof(char);
    int smem_b_addr2 = smem_b_addr0 + 2 * (Block_K + BPAD) * sizeof(char);
    int smem_b_addr3 = smem_b_addr0 + 3 * (Block_K + BPAD) * sizeof(char);

    int gmem_a_m = bx * Block_M + smem_a_m;
    int gmem_a_k = smem_a_k;
    int gmem_b_k = smem_b_k;
    int gmem_b_n = by * Block_N + smem_b_n;

    int gmem_a_addr = gmem_a_m * K + gmem_a_k;
    int gmem_b_addr = gmem_b_n * K + gmem_b_k;
    // stage 1
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr0), "l"(&A_Value[gmem_a_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr1), "l"(&A_Value[gmem_a_addr + 1 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr2), "l"(&A_Value[gmem_a_addr + 2 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr3), "l"(&A_Value[gmem_a_addr + 3 * K]));

    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr0), "l"(&B_Value[gmem_b_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr1), "l"(&B_Value[gmem_b_addr + 1 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr2), "l"(&B_Value[gmem_b_addr + 2 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr3), "l"(&B_Value[gmem_b_addr + 3 * K]));
    // stage 2
    gmem_a_addr += Block_K;
    gmem_b_addr += Block_K;
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr0+smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr1+smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr + 1 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr2+smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr + 2 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr3+smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr + 3 * K]));

    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr0+smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr1+smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr + 1 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr2+smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr + 2 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr3+smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr + 3 * K]));

    // stage 3
    gmem_a_addr += Block_K;
    gmem_b_addr += Block_K;
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr0+2*smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr1+2*smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr + 1 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr2+2*smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr + 2 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr3+2*smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr + 3 * K]));

    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr0+2*smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr1+2*smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr + 1 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr2+2*smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr + 2 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr3+2*smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr + 3 * K]));

    asm ("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    int warp_x = wid % 2;//0 1 0 1
    int warp_y = wid / 2;//0 0 1 1
    #pragma unroll 32
    for(int bk = 3; bk < K/Block_K; bk++){
        int sel_mem = bk % 4; // 3 0 1 2 3 0 1
        int sel_com = (bk - 3) % 4; // 0 1 2 3 0 1 2 3
        gmem_a_addr += Block_K;
        gmem_b_addr += Block_K;
        
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_a_addr0 + sel_mem * smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_a_addr1 + sel_mem * smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr + 1 * K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_a_addr2 + sel_mem * smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr + 2 * K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_a_addr3 + sel_mem * smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr + 3 * K]));

        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_b_addr0 + sel_mem * smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_b_addr1 + sel_mem * smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr + 1 * K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_b_addr2 + sel_mem * smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr + 2 * K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_b_addr3 + sel_mem * smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr + 3 * K]));
        
        float4 * smem_a_sel = reinterpret_cast<float4 *>(smem_a + sel_com * smem_a_offset + warp_x * 64 * (Block_K+APAD));
        float4 * smem_b_sel = reinterpret_cast<float4 *>(smem_b + sel_com * smem_b_offset + warp_y * 64 * (Block_K+BPAD));
        matrix_a_fragment[0] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 0 ) * 4);
        matrix_a_fragment[1] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 8 ) * 4);
        matrix_a_fragment[2] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 16) * 4);
        matrix_a_fragment[3] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 24) * 4);
        matrix_a_fragment[4] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 32) * 4);
        matrix_a_fragment[5] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 40) * 4);
        matrix_a_fragment[6] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 48) * 4);
        matrix_a_fragment[7] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 56) * 4);
        matrix_b_fragment[0] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 0 ) * 4);
        matrix_b_fragment[1] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 8 ) * 4);
        matrix_b_fragment[2] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 16) * 4);
        matrix_b_fragment[3] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 24) * 4);
        matrix_b_fragment[4] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 32) * 4);
        matrix_b_fragment[5] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 40) * 4);
        matrix_b_fragment[6] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 48) * 4);
        matrix_b_fragment[7] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 56) * 4);

        int * a_fragment_int = reinterpret_cast<int *>(matrix_a_fragment);// 8 float4 --> 32 reg
        int * b_fragment_int = reinterpret_cast<int *>(matrix_b_fragment);// 8 float4 --> 32 reg

        #pragma unroll
        for(int i = 0; i < 4; i++){
            #pragma unroll
            for(int j = 0; j < 8; j++){
                asm ("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                    "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                    "r"(a_fragment_int[0 + 8 * i]), "r"(a_fragment_int[4 + 8 * i]),
                    "r"(a_fragment_int[1 + 8 * i]), "r"(a_fragment_int[5 + 8 * i]),
                    "r"(b_fragment_int[0 + 4 * j]), "r"(b_fragment_int[1 + 4 * j])
                );
                // __syncthreads();
                asm ("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                    "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                    "r"(a_fragment_int[2 + 8 * i]), "r"(a_fragment_int[6 + 8 * i]),
                    "r"(a_fragment_int[3 + 8 * i]), "r"(a_fragment_int[7 + 8 * i]),
                    "r"(b_fragment_int[2 + 4 * j]), "r"(b_fragment_int[3 + 4 * j])
                );
            }
        }// end mma compute
        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);
        __syncthreads();

    }// end main loop
    int sel_com = ((int)(K / 64) - 3) % 4;
    float4 * smem_a_sel = reinterpret_cast<float4 *>(smem_a + sel_com * smem_a_offset + warp_x * 64 * (Block_K+APAD));
    float4 * smem_b_sel = reinterpret_cast<float4 *>(smem_b + sel_com * smem_b_offset + warp_y * 64 * (Block_K+BPAD));
    matrix_a_fragment[0] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 0 ) * 4);
    matrix_a_fragment[1] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 8 ) * 4);
    matrix_a_fragment[2] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 16) * 4);
    matrix_a_fragment[3] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 24) * 4);
    matrix_a_fragment[4] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 32) * 4);
    matrix_a_fragment[5] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 40) * 4);
    matrix_a_fragment[6] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 48) * 4);
    matrix_a_fragment[7] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 56) * 4);
    matrix_b_fragment[0] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 0 ) * 4);
    matrix_b_fragment[1] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 8 ) * 4);
    matrix_b_fragment[2] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 16) * 4);
    matrix_b_fragment[3] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 24) * 4);
    matrix_b_fragment[4] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 32) * 4);
    matrix_b_fragment[5] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 40) * 4);
    matrix_b_fragment[6] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 48) * 4);
    matrix_b_fragment[7] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 56) * 4);

    int * a_fragment_int = reinterpret_cast<int *>(matrix_a_fragment);// 8 float4 --> 32 reg
    int * b_fragment_int = reinterpret_cast<int *>(matrix_b_fragment);// 8 float4 --> 32 reg

    #pragma unroll
    for(int i = 0; i < 4; i++){
        #pragma unroll
        for(int j = 0; j < 8; j++){
            asm ("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 \t"
                "{%0, %1, %2, %3}, \t"
                "{%4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%0, %1, %2, %3}; ":
                "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                "r"(a_fragment_int[0 + 8 * i]), "r"(a_fragment_int[4 + 8 * i]),
                "r"(a_fragment_int[1 + 8 * i]), "r"(a_fragment_int[5 + 8 * i]),
                "r"(b_fragment_int[0 + 4 * j]), "r"(b_fragment_int[1 + 4 * j])
            );
            // __syncthreads();
            asm ("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 \t"
                "{%0, %1, %2, %3}, \t"
                "{%4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%0, %1, %2, %3}; ":
                "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                "r"(a_fragment_int[2 + 8 * i]), "r"(a_fragment_int[6 + 8 * i]),
                "r"(a_fragment_int[3 + 8 * i]), "r"(a_fragment_int[7 + 8 * i]),
                "r"(b_fragment_int[2 + 4 * j]), "r"(b_fragment_int[3 + 4 * j])
            );
        }
    }// end mma compute
    __syncthreads();
    sel_com = ((int)(K / 64)-2) % 4;
    smem_a_sel = reinterpret_cast<float4 *>(smem_a + sel_com * smem_a_offset + warp_x * 64 * (Block_K+APAD));
    smem_b_sel = reinterpret_cast<float4 *>(smem_b + sel_com * smem_b_offset + warp_y * 64 * (Block_K+BPAD));
    matrix_a_fragment[0] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 0 ) * 4);
    matrix_a_fragment[1] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 8 ) * 4);
    matrix_a_fragment[2] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 16) * 4);
    matrix_a_fragment[3] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 24) * 4);
    matrix_a_fragment[4] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 32) * 4);
    matrix_a_fragment[5] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 40) * 4);
    matrix_a_fragment[6] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 48) * 4);
    matrix_a_fragment[7] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 56) * 4);
    matrix_b_fragment[0] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 0 ) * 4);
    matrix_b_fragment[1] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 8 ) * 4);
    matrix_b_fragment[2] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 16) * 4);
    matrix_b_fragment[3] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 24) * 4);
    matrix_b_fragment[4] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 32) * 4);
    matrix_b_fragment[5] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 40) * 4);
    matrix_b_fragment[6] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 48) * 4);
    matrix_b_fragment[7] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 56) * 4);

    a_fragment_int = reinterpret_cast<int *>(matrix_a_fragment);// 8 float4 --> 32 reg
    b_fragment_int = reinterpret_cast<int *>(matrix_b_fragment);// 8 float4 --> 32 reg

    #pragma unroll
    for(int i = 0; i < 4; i++){
        #pragma unroll
        for(int j = 0; j < 8; j++){
            asm ("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 \t"
                "{%0, %1, %2, %3}, \t"
                "{%4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%0, %1, %2, %3}; ":
                "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                "r"(a_fragment_int[0 + 8 * i]), "r"(a_fragment_int[4 + 8 * i]),
                "r"(a_fragment_int[1 + 8 * i]), "r"(a_fragment_int[5 + 8 * i]),
                "r"(b_fragment_int[0 + 4 * j]), "r"(b_fragment_int[1 + 4 * j])
            );
            // __syncthreads();
            asm ("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 \t"
                "{%0, %1, %2, %3}, \t"
                "{%4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%0, %1, %2, %3}; ":
                "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                "r"(a_fragment_int[2 + 8 * i]), "r"(a_fragment_int[6 + 8 * i]),
                "r"(a_fragment_int[3 + 8 * i]), "r"(a_fragment_int[7 + 8 * i]),
                "r"(b_fragment_int[2 + 4 * j]), "r"(b_fragment_int[3 + 4 * j])
            );
        }
    }// end mma compute
    __syncthreads();
    sel_com = ((int)(K / 64)-1) % 4;
    smem_a_sel = reinterpret_cast<float4 *>(smem_a + sel_com * smem_a_offset + warp_x * 64 * (Block_K+APAD));
    smem_b_sel = reinterpret_cast<float4 *>(smem_b + sel_com * smem_b_offset + warp_y * 64 * (Block_K+BPAD));
    matrix_a_fragment[0] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 0 ) * 4);
    matrix_a_fragment[1] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 8 ) * 4);
    matrix_a_fragment[2] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 16) * 4);
    matrix_a_fragment[3] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 24) * 4);
    matrix_a_fragment[4] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 32) * 4);
    matrix_a_fragment[5] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 40) * 4);
    matrix_a_fragment[6] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 48) * 4);
    matrix_a_fragment[7] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 56) * 4);
    matrix_b_fragment[0] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 0 ) * 4);
    matrix_b_fragment[1] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 8 ) * 4);
    matrix_b_fragment[2] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 16) * 4);
    matrix_b_fragment[3] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 24) * 4);
    matrix_b_fragment[4] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 32) * 4);
    matrix_b_fragment[5] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 40) * 4);
    matrix_b_fragment[6] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 48) * 4);
    matrix_b_fragment[7] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 56) * 4);

    a_fragment_int = reinterpret_cast<int *>(matrix_a_fragment);// 8 float4 --> 32 reg
    b_fragment_int = reinterpret_cast<int *>(matrix_b_fragment);// 8 float4 --> 32 reg

    #pragma unroll
    for(int i = 0; i < 4; i++){
        #pragma unroll
        for(int j = 0; j < 8; j++){
            asm ("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 \t"
                "{%0, %1, %2, %3}, \t"
                "{%4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%0, %1, %2, %3}; ":
                "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                "r"(a_fragment_int[0 + 8 * i]), "r"(a_fragment_int[4 + 8 * i]),
                "r"(a_fragment_int[1 + 8 * i]), "r"(a_fragment_int[5 + 8 * i]),
                "r"(b_fragment_int[0 + 4 * j]), "r"(b_fragment_int[1 + 4 * j])
            );
            // __syncthreads();
            asm ("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 \t"
                "{%0, %1, %2, %3}, \t"
                "{%4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%0, %1, %2, %3}; ":
                "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                "r"(a_fragment_int[2 + 8 * i]), "r"(a_fragment_int[6 + 8 * i]),
                "r"(a_fragment_int[3 + 8 * i]), "r"(a_fragment_int[7 + 8 * i]),
                "r"(b_fragment_int[2 + 4 * j]), "r"(b_fragment_int[3 + 4 * j])
            );
        }
    }// end mma compute
    __syncthreads();

    float2 * output_ = reinterpret_cast<float2 *>(Output_Value + (bx * Block_M + warp_x * 64 + (int)(lane_id / 4)) * N + by * Block_N + warp_y * 64 + ((lane_id % 4) * 2));
    float2 * output_fragment_ = reinterpret_cast<float2 *>(output_fragment);// 128 reg --> 64 float2
    #pragma unroll
    for(int i = 0; i < 4; i++){
        #pragma unroll
        for(int j = 0; j < 8; j++){
            *(output_ + (int)(( (i*16  ) * N + j * 8 )/2)) = *(output_fragment_ + 2 * (i*8+j)    );
            *(output_ + (int)(( (i*16+8) * N + j * 8 )/2)) = *(output_fragment_ + 2 * (i*8+j) + 1);
        }
    }
}
cudaError_t GEMMex4(
    int M, int K, int N,
    const e4m3 * __restrict__ A_Value,
    const e4m3 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
    const int Block_M = 128, Block_N = 128, Block_K = 64;
    dim3 block_dim(128,1,1);
	dim3 grid_dim(ceil(static_cast<float>(M) / Block_M), ceil(static_cast<float>(N) / Block_N), 1);
    // unsigned int dsmem = 4 * (Block_M * (Block_K + 0) + Block_N * (Block_K + 0)) * sizeof(char);
    // cudaFuncSetAttribute(GEMM_e4m3_e4m3_o32_stage4_row_col,
        // cudaFuncAttributeMaxDynamicSharedMemorySize, 233472);

    GEMM_e4m3_e4m3_o32_stage4_row_col<<<grid_dim, block_dim>>>(
        M, K, N, A_Value, B_Value, Output_Value);
	return cudaGetLastError();
}
cudaError_t GEMM4(
    int M, int K, int N,
    const e4m3 * __restrict__ A_Value,
    const e4m3 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
	return GEMMex(M, K, N, A_Value, B_Value, Output_Value);
}
//kernel3////////////////////e5m2 * e5m2 = f32//////////////////
__global__ void GEMM_e5m2_e5m2_o32_stage2_row_col(
    int M, int K, int N,
    const e5m2 * __restrict__ A_Value,
    const e5m2 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
    // thread num = 128
    constexpr int Block_M = 128;
    constexpr int Block_K = 64;
    constexpr int Block_N = 128;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int wid = tid / 32;// 128 / 32 = 4

    if(bx >= M / Block_M || by >= N / Block_N){
        return;
    }
    const int APAD = 0;
    const int BPAD = 0;

    // extern __shared__ float4 smem[];
    const int smem_a_offset = Block_M * (Block_K + APAD);
    const int smem_b_offset = Block_N * (Block_K + BPAD);
    // const int smem_size_float4 = 2 * (smem_a_offset + smem_b_offset) / 16;
    __shared__ e5m2 smem_a[smem_a_offset * 2];
    __shared__ e5m2 smem_b[smem_b_offset * 2];
    // e5m2 * smem_a = reinterpret_cast<e5m2 *>(smem);
    // e5m2 * smem_b = smem_a + 2 * smem_a_offset;
    float4 matrix_a_fragment[8];// 8 float4 = 32 reg
    float4 matrix_b_fragment[8];// 8 float4 = 32 reg
    float output_fragment[128];// 4(m) * 8(n) * 4(one tile) = 128 float

    int smem_a_m = (int)(tid / 4) * 4;
    int smem_a_k = (tid % 4) * 16;
    int smem_b_k = (tid % 4) * 16;
    int smem_b_n = (int)(tid / 4) * 4;
    int smem_a_base_addr = __cvta_generic_to_shared(smem_a);
    int smem_b_base_addr = __cvta_generic_to_shared(smem_b);

    int smem_a_addr0 = smem_a_base_addr + (smem_a_m * (Block_K + APAD) + smem_a_k) * sizeof(char);
    int smem_a_addr1 = smem_a_addr0 + 1 * (Block_K + APAD) * sizeof(char);
    int smem_a_addr2 = smem_a_addr0 + 2 * (Block_K + APAD) * sizeof(char);
    int smem_a_addr3 = smem_a_addr0 + 3 * (Block_K + APAD) * sizeof(char);
    
    int smem_b_addr0 = smem_b_base_addr + (smem_b_n * (Block_K + BPAD) + smem_b_k) * sizeof(char);
    int smem_b_addr1 = smem_b_addr0 + 1 * (Block_K + BPAD) * sizeof(char);
    int smem_b_addr2 = smem_b_addr0 + 2 * (Block_K + BPAD) * sizeof(char);
    int smem_b_addr3 = smem_b_addr0 + 3 * (Block_K + BPAD) * sizeof(char);

    int gmem_a_m = bx * Block_M + smem_a_m;
    int gmem_a_k = smem_a_k;
    int gmem_b_k = smem_b_k;
    int gmem_b_n = by * Block_N + smem_b_n;

    int gmem_a_addr = gmem_a_m * K + gmem_a_k;
    int gmem_b_addr = gmem_b_n * K + gmem_b_k;

    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr0), "l"(&A_Value[gmem_a_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr1), "l"(&A_Value[gmem_a_addr + 1 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr2), "l"(&A_Value[gmem_a_addr + 2 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr3), "l"(&A_Value[gmem_a_addr + 3 * K]));

    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr0), "l"(&B_Value[gmem_b_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr1), "l"(&B_Value[gmem_b_addr + 1 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr2), "l"(&B_Value[gmem_b_addr + 2 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr3), "l"(&B_Value[gmem_b_addr + 3 * K]));

    asm ("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    int warp_x = wid % 2;//0 1 0 1
    int warp_y = wid / 2;//0 0 1 1
    #pragma unroll 16
    for(int bk = 1; bk < K/Block_K; bk++){
        int sel = (bk & 1) ^ 1; // 0 1 0 1
        int sel_mem = ((bk - 1) & 1) ^ 1; // 1 0 1 0
        gmem_a_addr += Block_K;
        gmem_b_addr += Block_K;
        
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_a_addr0 + sel_mem * smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_a_addr1 + sel_mem * smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr + 1 * K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_a_addr2 + sel_mem * smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr + 2 * K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_a_addr3 + sel_mem * smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr + 3 * K]));

        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_b_addr0 + sel_mem * smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_b_addr1 + sel_mem * smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr + 1 * K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_b_addr2 + sel_mem * smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr + 2 * K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_b_addr3 + sel_mem * smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr + 3 * K]));
        
        float4 * smem_a_sel = reinterpret_cast<float4 *>(smem_a + sel * smem_a_offset + warp_x * (int)(smem_a_offset/2));
        float4 * smem_b_sel = reinterpret_cast<float4 *>(smem_b + sel * smem_b_offset + warp_y * (int)(smem_b_offset/2));
        matrix_a_fragment[0] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 0 ) * 4);
        matrix_a_fragment[1] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 8 ) * 4);
        matrix_a_fragment[2] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 16) * 4);
        matrix_a_fragment[3] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 24) * 4);
        matrix_a_fragment[4] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 32) * 4);
        matrix_a_fragment[5] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 40) * 4);
        matrix_a_fragment[6] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 48) * 4);
        matrix_a_fragment[7] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 56) * 4);
        matrix_b_fragment[0] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 0 ) * 4);
        matrix_b_fragment[1] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 8 ) * 4);
        matrix_b_fragment[2] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 16) * 4);
        matrix_b_fragment[3] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 24) * 4);
        matrix_b_fragment[4] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 32) * 4);
        matrix_b_fragment[5] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 40) * 4);
        matrix_b_fragment[6] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 48) * 4);
        matrix_b_fragment[7] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 56) * 4);

        int * a_fragment_int = reinterpret_cast<int *>(matrix_a_fragment);// 8 float4 --> 32 reg
        int * b_fragment_int = reinterpret_cast<int *>(matrix_b_fragment);// 8 float4 --> 32 reg

        #pragma unroll
        for(int i = 0; i < 4; i++){
            #pragma unroll
            for(int j = 0; j < 8; j++){
                asm ("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                    "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                    "r"(a_fragment_int[0 + 8 * i]), "r"(a_fragment_int[4 + 8 * i]),
                    "r"(a_fragment_int[1 + 8 * i]), "r"(a_fragment_int[5 + 8 * i]),
                    "r"(b_fragment_int[0 + 4 * j]), "r"(b_fragment_int[1 + 4 * j])
                );
                asm ("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                    "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                    "r"(a_fragment_int[2 + 8 * i]), "r"(a_fragment_int[6 + 8 * i]),
                    "r"(a_fragment_int[3 + 8 * i]), "r"(a_fragment_int[7 + 8 * i]),
                    "r"(b_fragment_int[2 + 4 * j]), "r"(b_fragment_int[3 + 4 * j])
                );
            }
        }// end mma compute
        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);
        __syncthreads();

    }// end main loop
    int sel = ((K / Block_K) & 1) ^ 1;
    float4 * smem_a_sel = reinterpret_cast<float4 *>(smem_a + sel * smem_a_offset + warp_x * (int)(smem_a_offset/2));
    float4 * smem_b_sel = reinterpret_cast<float4 *>(smem_b + sel * smem_b_offset + warp_y * (int)(smem_b_offset/2));
    matrix_a_fragment[0] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 0 ) * 4);
    matrix_a_fragment[1] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 8 ) * 4);
    matrix_a_fragment[2] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 16) * 4);
    matrix_a_fragment[3] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 24) * 4);
    matrix_a_fragment[4] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 32) * 4);
    matrix_a_fragment[5] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 40) * 4);
    matrix_a_fragment[6] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 48) * 4);
    matrix_a_fragment[7] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 56) * 4);
    matrix_b_fragment[0] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 0 ) * 4);
    matrix_b_fragment[1] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 8 ) * 4);
    matrix_b_fragment[2] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 16) * 4);
    matrix_b_fragment[3] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 24) * 4);
    matrix_b_fragment[4] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 32) * 4);
    matrix_b_fragment[5] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 40) * 4);
    matrix_b_fragment[6] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 48) * 4);
    matrix_b_fragment[7] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 56) * 4);

    int * a_fragment_int = reinterpret_cast<int *>(matrix_a_fragment);// 8 float4 --> 32 reg
    int * b_fragment_int = reinterpret_cast<int *>(matrix_b_fragment);// 8 float4 --> 32 reg

    #pragma unroll
    for(int i = 0; i < 4; i++){
        #pragma unroll
        for(int j = 0; j < 8; j++){
            asm ("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 \t"
                "{%0, %1, %2, %3}, \t"
                "{%4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%0, %1, %2, %3}; ":
                "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                "r"(a_fragment_int[0 + 8 * i]), "r"(a_fragment_int[4 + 8 * i]),
                "r"(a_fragment_int[1 + 8 * i]), "r"(a_fragment_int[5 + 8 * i]),
                "r"(b_fragment_int[0 + 4 * j]), "r"(b_fragment_int[1 + 4 * j])
            );
            asm ("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 \t"
                "{%0, %1, %2, %3}, \t"
                "{%4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%0, %1, %2, %3}; ":
                "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                "r"(a_fragment_int[2 + 8 * i]), "r"(a_fragment_int[6 + 8 * i]),
                "r"(a_fragment_int[3 + 8 * i]), "r"(a_fragment_int[7 + 8 * i]),
                "r"(b_fragment_int[2 + 4 * j]), "r"(b_fragment_int[3 + 4 * j])
            );
        }
    }// end mma compute
    __syncthreads();
    float2 * output_ = reinterpret_cast<float2 *>(Output_Value + (bx * Block_M + warp_x * 64 + (int)(lane_id / 4)) * N + by * Block_N + warp_y * 64 + (int)((lane_id % 4) * 2));
    float2 * output_fragment_ = reinterpret_cast<float2 *>(output_fragment);// 128 reg --> 64 float2
    #pragma unroll
    for(int i = 0; i < 4; i++){
        #pragma unroll
        for(int j = 0; j < 8; j++){
            *(output_ + (i*8  ) * N + j * 4) = *(output_fragment_ + 2 * (i*8+j)    );
            *(output_ + (i*8+4) * N + j * 4) = *(output_fragment_ + 2 * (i*8+j) + 1);
        }
    }
}

cudaError_t GEMMex(
    int M, int K, int N,
    const e5m2 * __restrict__ A_Value,
    const e5m2 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
    const int Block_M = 128, Block_N = 128, Block_K = 64;
    dim3 block_dim(128,1,1);
	dim3 grid_dim(ceil(static_cast<float>(M) / Block_M), ceil(static_cast<float>(N) / Block_N), 1);

    GEMM_e5m2_e5m2_o32_stage2_row_col<<<grid_dim, block_dim>>>(
        M, K, N, A_Value, B_Value, Output_Value);
	return cudaGetLastError();
}
cudaError_t GEMM(
    int M, int K, int N,
    const e5m2 * __restrict__ A_Value,
    const e5m2 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
	return GEMMex(M, K, N, A_Value, B_Value, Output_Value);
}
//kernel5////////////////////e4m3 * e5m2 = f32//////////////////
__global__ void GEMM_e4m3_e5m2_o32_stage2_row_col(
    int M, int K, int N,
    const e4m3 * __restrict__ A_Value,
    const e5m2 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
    // thread num = 128
    constexpr int Block_M = 128;
    constexpr int Block_K = 64;
    constexpr int Block_N = 128;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int wid = tid >> 5;// 128 / 32 = 4

    if(bx >= M / Block_M || by >= N / Block_N){
        return;
    }
    constexpr int APAD = 0;
    constexpr int BPAD = 0;

    // extern __shared__ float4 smem[];
    constexpr int smem_a_offset = Block_M * (Block_K + APAD);
    constexpr int smem_b_offset = Block_N * (Block_K + BPAD);
    const int smem_size_float4 = 2 * (smem_a_offset + smem_b_offset) / 16;
    __shared__ float4 smem[smem_size_float4];
    e4m3 * smem_a = reinterpret_cast<e4m3 *>(smem);
    e5m2 * smem_b = reinterpret_cast<e5m2 *>(smem) + 2 * smem_a_offset;

    // __align__(32) float4 matrix_a_fragment[8];// 8 float4 = 32 reg
    // __align__(32) float4 matrix_b_fragment[8];// 8 float4 = 32 reg
    // __align__(32) float output_fragment[128];// 4(m) * 8(n) * 4(one tile) = 128 float
    float4 matrix_a_fragment[8];// 8 float4 = 32 reg
    float4 matrix_b_fragment[8];// 8 float4 = 32 reg
    float output_fragment[128];// 4(m) * 8(n) * 4(one tile) = 128 float

    int smem_a_base_addr = __cvta_generic_to_shared(smem_a);
    int smem_b_base_addr = __cvta_generic_to_shared(smem_b);
    int smem_a_m = (tid / 4) << 2;
    int smem_a_k = (tid &  3) << 4;
    int smem_b_k = (tid &  3) << 4;
    int smem_b_n = (tid / 4) << 2;
    // block_size = 128 * 64 
    // thred num = 128 
    // each thread load = 128 * 64 / 128 = 64 fp8 = 4 float4
    int smem_a_addr0 = smem_a_base_addr + (smem_a_m * (Block_K + APAD) + smem_a_k) * sizeof(char);
    int smem_a_addr1 = smem_a_addr0 + 1 * (Block_K + APAD) * sizeof(char);
    int smem_a_addr2 = smem_a_addr0 + 2 * (Block_K + APAD) * sizeof(char);
    int smem_a_addr3 = smem_a_addr0 + 3 * (Block_K + APAD) * sizeof(char);
    
    int smem_b_addr0 = smem_b_base_addr + (smem_b_n * (Block_K + BPAD) + smem_b_k) * sizeof(char);
    int smem_b_addr1 = smem_b_addr0 + 1 * (Block_K + BPAD) * sizeof(char);
    int smem_b_addr2 = smem_b_addr0 + 2 * (Block_K + BPAD) * sizeof(char);
    int smem_b_addr3 = smem_b_addr0 + 3 * (Block_K + BPAD) * sizeof(char);

    int gmem_a_m = bx * Block_M + smem_a_m;
    int gmem_a_k = smem_a_k;
    int gmem_b_k = smem_b_k;
    int gmem_b_n = by * Block_N + smem_b_n;

    int gmem_a_addr = gmem_a_m * K + gmem_a_k;
    int gmem_b_addr = gmem_b_n * K + gmem_b_k;

    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr0), "l"(&A_Value[gmem_a_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr1), "l"(&A_Value[gmem_a_addr + 1 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr2), "l"(&A_Value[gmem_a_addr + 2 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr3), "l"(&A_Value[gmem_a_addr + 3 * K]));

    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr0), "l"(&B_Value[gmem_b_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr1), "l"(&B_Value[gmem_b_addr + 1 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr2), "l"(&B_Value[gmem_b_addr + 2 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr3), "l"(&B_Value[gmem_b_addr + 3 * K]));

    asm ("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    int warp_x = wid &  1;//0 1 0 1
    int warp_y = wid >> 1;//0 0 1 1
    #pragma unroll 16
    for(int bk = 1; bk < K/Block_K; bk++){
        int sel = (bk & 1) ^ 1; // 0 1 0 1
        int sel_mem = ((bk - 1) & 1) ^ 1; // 1 0 1 0
        gmem_a_addr += Block_K;
        gmem_b_addr += Block_K;
        
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_a_addr0 + sel_mem * smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_a_addr1 + sel_mem * smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr + 1 * K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_a_addr2 + sel_mem * smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr + 2 * K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_a_addr3 + sel_mem * smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr + 3 * K]));

        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_b_addr0 + sel_mem * smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_b_addr1 + sel_mem * smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr + 1 * K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_b_addr2 + sel_mem * smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr + 2 * K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_b_addr3 + sel_mem * smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr + 3 * K]));
        
        float4 * smem_a_sel = reinterpret_cast<float4 *>(smem_a + sel * smem_a_offset + warp_x * (smem_a_offset >> 1));
        float4 * smem_b_sel = reinterpret_cast<float4 *>(smem_b + sel * smem_b_offset + warp_y * (smem_b_offset >> 1));
        matrix_a_fragment[0] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 0 ) * 4);
        matrix_a_fragment[1] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 8 ) * 4);
        matrix_a_fragment[2] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 16) * 4);
        matrix_a_fragment[3] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 24) * 4);
        matrix_a_fragment[4] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 32) * 4);
        matrix_a_fragment[5] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 40) * 4);
        matrix_a_fragment[6] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 48) * 4);
        matrix_a_fragment[7] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 56) * 4);
        matrix_b_fragment[0] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 0 ) * 4);
        matrix_b_fragment[1] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 8 ) * 4);
        matrix_b_fragment[2] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 16) * 4);
        matrix_b_fragment[3] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 24) * 4);
        matrix_b_fragment[4] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 32) * 4);
        matrix_b_fragment[5] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 40) * 4);
        matrix_b_fragment[6] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 48) * 4);
        matrix_b_fragment[7] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 56) * 4);

        int * a_fragment_int = reinterpret_cast<int *>(matrix_a_fragment);// 8 float4 --> 32 reg
        int * b_fragment_int = reinterpret_cast<int *>(matrix_b_fragment);// 8 float4 --> 32 reg

        #pragma unroll
        for(int i = 0; i < 4; i++){
            #pragma unroll
            for(int j = 0; j < 8; j++){
                asm ("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                    "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                    "r"(a_fragment_int[0 + 8 * i]), "r"(a_fragment_int[1 + 8 * i]),
                    "r"(a_fragment_int[4 + 8 * i]), "r"(a_fragment_int[5 + 8 * i]),
                    "r"(b_fragment_int[0 + 4 * j]), "r"(b_fragment_int[1 + 4 * j])
                );
                asm ("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                    "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                    "r"(a_fragment_int[2 + 8 * i]), "r"(a_fragment_int[3 + 8 * i]),
                    "r"(a_fragment_int[6 + 8 * i]), "r"(a_fragment_int[7 + 8 * i]),
                    "r"(b_fragment_int[2 + 4 * j]), "r"(b_fragment_int[3 + 4 * j])
                );
            }
        }// end mma compute
        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);
        __syncthreads();

    }// end main loop
    int sel = ((K / Block_K) & 1) ^ 1;
    float4 * smem_a_sel = reinterpret_cast<float4 *>(smem_a + sel * smem_a_offset + warp_x * (smem_a_offset >> 1));
    float4 * smem_b_sel = reinterpret_cast<float4 *>(smem_b + sel * smem_b_offset + warp_y * (smem_b_offset >> 1));
    matrix_a_fragment[0] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 0 ) * 4);
    matrix_a_fragment[1] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 8 ) * 4);
    matrix_a_fragment[2] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 16) * 4);
    matrix_a_fragment[3] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 24) * 4);
    matrix_a_fragment[4] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 32) * 4);
    matrix_a_fragment[5] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 40) * 4);
    matrix_a_fragment[6] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 48) * 4);
    matrix_a_fragment[7] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 56) * 4);
    matrix_b_fragment[0] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 0 ) * 4);
    matrix_b_fragment[1] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 8 ) * 4);
    matrix_b_fragment[2] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 16) * 4);
    matrix_b_fragment[3] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 24) * 4);
    matrix_b_fragment[4] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 32) * 4);
    matrix_b_fragment[5] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 40) * 4);
    matrix_b_fragment[6] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 48) * 4);
    matrix_b_fragment[7] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 56) * 4);

    int * a_fragment_int = reinterpret_cast<int *>(matrix_a_fragment);// 8 float4 --> 32 reg
    int * b_fragment_int = reinterpret_cast<int *>(matrix_b_fragment);// 8 float4 --> 32 reg

    #pragma unroll
    for(int i = 0; i < 4; i++){
        #pragma unroll
        for(int j = 0; j < 8; j++){
            asm ("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 \t"
                "{%0, %1, %2, %3}, \t"
                "{%4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%0, %1, %2, %3}; ":
                "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                "r"(a_fragment_int[0 + 8 * i]), "r"(a_fragment_int[1 + 8 * i]),
                "r"(a_fragment_int[4 + 8 * i]), "r"(a_fragment_int[5 + 8 * i]),
                "r"(b_fragment_int[0 + 4 * j]), "r"(b_fragment_int[1 + 4 * j])
            );
            asm ("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 \t"
                "{%0, %1, %2, %3}, \t"
                "{%4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%0, %1, %2, %3}; ":
                "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                "r"(a_fragment_int[2 + 8 * i]), "r"(a_fragment_int[3 + 8 * i]),
                "r"(a_fragment_int[6 + 8 * i]), "r"(a_fragment_int[7 + 8 * i]),
                "r"(b_fragment_int[2 + 4 * j]), "r"(b_fragment_int[3 + 4 * j])
            );
        }
    }// end mma compute
    __syncthreads();
    float2 * output_ = reinterpret_cast<float2 *>(Output_Value + (bx * Block_M + warp_x * 64 + (int)(lane_id / 4)) * N + by * Block_N + warp_y * 64 + (int)((lane_id % 4) << 1));
    float2 * output_fragment_ = reinterpret_cast<float2 *>(output_fragment);// 128 reg --> 64 float2
    #pragma unroll
    for(int i = 0; i < 4; i++){
        #pragma unroll
        for(int j = 0; j < 8; j++){
            *(output_ + (i*8  ) * N + j * 4) = *(output_fragment_ + 2 * (i*8+j)    );
            *(output_ + (i*8+4) * N + j * 4) = *(output_fragment_ + 2 * (i*8+j) + 1);
        }
    }
}

cudaError_t GEMMex(
    int M, int K, int N,
    const e4m3 * __restrict__ A_Value,
    const e5m2 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
    const int Block_M = 128, Block_N = 128, Block_K = 64;
    dim3 block_dim(128,1,1);
	dim3 grid_dim(ceil(static_cast<float>(M) / Block_M), ceil(static_cast<float>(N) / Block_N), 1);

    GEMM_e4m3_e5m2_o32_stage2_row_col<<<grid_dim, block_dim>>>(
        M, K, N, A_Value, B_Value, Output_Value);
	return cudaGetLastError();
}
cudaError_t GEMM(
    int M, int K, int N,
    const e4m3 * __restrict__ A_Value,
    const e5m2 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
	return GEMMex(M, K, N, A_Value, B_Value, Output_Value);
}
//kernel7////////////////////e5m2 * e4m3 = f32//////////////////
__global__ void GEMM_e5m2_e4m3_o32_stage2_row_col(
    int M, int K, int N,
    const e5m2 * __restrict__ A_Value,
    const e4m3 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
    // thread num = 128
    constexpr int Block_M = 128;
    constexpr int Block_K = 64;
    constexpr int Block_N = 128;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int wid = tid >> 5;// 128 / 32 = 4

    if(bx >= M / Block_M || by >= N / Block_N){
        return;
    }
    constexpr int APAD = 0;
    constexpr int BPAD = 0;

    // extern __shared__ float4 smem[];
    constexpr int smem_a_offset = Block_M * (Block_K + APAD);
    constexpr int smem_b_offset = Block_N * (Block_K + BPAD);
    const int smem_size_float4 = 2 * (smem_a_offset + smem_b_offset) / 16;
    __shared__ float4 smem[smem_size_float4];
    e5m2 * smem_a = reinterpret_cast<e5m2 *>(smem);
    e4m3 * smem_b = reinterpret_cast<e4m3 *>(smem) + 2 * smem_a_offset;

    // __align__(32) float4 matrix_a_fragment[8];// 8 float4 = 32 reg
    // __align__(32) float4 matrix_b_fragment[8];// 8 float4 = 32 reg
    // __align__(32) float output_fragment[128];// 4(m) * 8(n) * 4(one tile) = 128 float
    float4 matrix_a_fragment[8];// 8 float4 = 32 reg
    float4 matrix_b_fragment[8];// 8 float4 = 32 reg
    float output_fragment[128];// 4(m) * 8(n) * 4(one tile) = 128 float

    int smem_a_base_addr = __cvta_generic_to_shared(smem_a);
    int smem_b_base_addr = __cvta_generic_to_shared(smem_b);
    int smem_a_m = (tid / 4) << 2;
    int smem_a_k = (tid &  3) << 4;
    int smem_b_k = (tid &  3) << 4;
    int smem_b_n = (tid / 4) << 2;
    // block_size = 128 * 64 
    // thred num = 128 
    // each thread load = 128 * 64 / 128 = 64 fp8 = 4 float4
    int smem_a_addr0 = smem_a_base_addr + (smem_a_m * (Block_K + APAD) + smem_a_k) * sizeof(char);
    int smem_a_addr1 = smem_a_addr0 + 1 * (Block_K + APAD) * sizeof(char);
    int smem_a_addr2 = smem_a_addr0 + 2 * (Block_K + APAD) * sizeof(char);
    int smem_a_addr3 = smem_a_addr0 + 3 * (Block_K + APAD) * sizeof(char);
    
    int smem_b_addr0 = smem_b_base_addr + (smem_b_n * (Block_K + BPAD) + smem_b_k) * sizeof(char);
    int smem_b_addr1 = smem_b_addr0 + 1 * (Block_K + BPAD) * sizeof(char);
    int smem_b_addr2 = smem_b_addr0 + 2 * (Block_K + BPAD) * sizeof(char);
    int smem_b_addr3 = smem_b_addr0 + 3 * (Block_K + BPAD) * sizeof(char);

    int gmem_a_m = bx * Block_M + smem_a_m;
    int gmem_a_k = smem_a_k;
    int gmem_b_k = smem_b_k;
    int gmem_b_n = by * Block_N + smem_b_n;

    int gmem_a_addr = gmem_a_m * K + gmem_a_k;
    int gmem_b_addr = gmem_b_n * K + gmem_b_k;

    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr0), "l"(&A_Value[gmem_a_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr1), "l"(&A_Value[gmem_a_addr + 1 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr2), "l"(&A_Value[gmem_a_addr + 2 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_a_addr3), "l"(&A_Value[gmem_a_addr + 3 * K]));

    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr0), "l"(&B_Value[gmem_b_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr1), "l"(&B_Value[gmem_b_addr + 1 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr2), "l"(&B_Value[gmem_b_addr + 2 * K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(smem_b_addr3), "l"(&B_Value[gmem_b_addr + 3 * K]));

    asm ("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    int warp_x = wid &  1;//0 1 0 1
    int warp_y = wid >> 1;//0 0 1 1
    #pragma unroll 16
    for(int bk = 1; bk < K/Block_K; bk++){
        int sel = (bk & 1) ^ 1; // 0 1 0 1
        int sel_mem = ((bk - 1) & 1) ^ 1; // 1 0 1 0
        gmem_a_addr += Block_K;
        gmem_b_addr += Block_K;
        
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_a_addr0 + sel_mem * smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_a_addr1 + sel_mem * smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr + 1 * K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_a_addr2 + sel_mem * smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr + 2 * K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_a_addr3 + sel_mem * smem_a_offset * (int)sizeof(char)), "l"(&A_Value[gmem_a_addr + 3 * K]));

        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_b_addr0 + sel_mem * smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_b_addr1 + sel_mem * smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr + 1 * K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_b_addr2 + sel_mem * smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr + 2 * K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(smem_b_addr3 + sel_mem * smem_b_offset * (int)sizeof(char)), "l"(&B_Value[gmem_b_addr + 3 * K]));
        
        float4 * smem_a_sel = reinterpret_cast<float4 *>(smem_a + sel * smem_a_offset + warp_x * (smem_a_offset >> 1));
        float4 * smem_b_sel = reinterpret_cast<float4 *>(smem_b + sel * smem_b_offset + warp_y * (smem_b_offset >> 1));
        matrix_a_fragment[0] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 0 ) * 4);
        matrix_a_fragment[1] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 8 ) * 4);
        matrix_a_fragment[2] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 16) * 4);
        matrix_a_fragment[3] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 24) * 4);
        matrix_a_fragment[4] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 32) * 4);
        matrix_a_fragment[5] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 40) * 4);
        matrix_a_fragment[6] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 48) * 4);
        matrix_a_fragment[7] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 56) * 4);
        matrix_b_fragment[0] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 0 ) * 4);
        matrix_b_fragment[1] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 8 ) * 4);
        matrix_b_fragment[2] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 16) * 4);
        matrix_b_fragment[3] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 24) * 4);
        matrix_b_fragment[4] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 32) * 4);
        matrix_b_fragment[5] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 40) * 4);
        matrix_b_fragment[6] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 48) * 4);
        matrix_b_fragment[7] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 56) * 4);

        int * a_fragment_int = reinterpret_cast<int *>(matrix_a_fragment);// 8 float4 --> 32 reg
        int * b_fragment_int = reinterpret_cast<int *>(matrix_b_fragment);// 8 float4 --> 32 reg

        #pragma unroll
        for(int i = 0; i < 4; i++){
            #pragma unroll
            for(int j = 0; j < 8; j++){
                asm ("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                    "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                    "r"(a_fragment_int[0 + 8 * i]), "r"(a_fragment_int[1 + 8 * i]),
                    "r"(a_fragment_int[4 + 8 * i]), "r"(a_fragment_int[5 + 8 * i]),
                    "r"(b_fragment_int[0 + 4 * j]), "r"(b_fragment_int[1 + 4 * j])
                );
                asm ("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                    "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                    "r"(a_fragment_int[2 + 8 * i]), "r"(a_fragment_int[3 + 8 * i]),
                    "r"(a_fragment_int[6 + 8 * i]), "r"(a_fragment_int[7 + 8 * i]),
                    "r"(b_fragment_int[2 + 4 * j]), "r"(b_fragment_int[3 + 4 * j])
                );
            }
        }// end mma compute
        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);
        __syncthreads();

    }// end main loop
    int sel = ((K / Block_K) & 1) ^ 1;
    float4 * smem_a_sel = reinterpret_cast<float4 *>(smem_a + sel * smem_a_offset + warp_x * (smem_a_offset >> 1));
    float4 * smem_b_sel = reinterpret_cast<float4 *>(smem_b + sel * smem_b_offset + warp_y * (smem_b_offset >> 1));
    matrix_a_fragment[0] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 0 ) * 4);
    matrix_a_fragment[1] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 8 ) * 4);
    matrix_a_fragment[2] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 16) * 4);
    matrix_a_fragment[3] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 24) * 4);
    matrix_a_fragment[4] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 32) * 4);
    matrix_a_fragment[5] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 40) * 4);
    matrix_a_fragment[6] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 48) * 4);
    matrix_a_fragment[7] = *(smem_a_sel + (lane_id % 4) + (lane_id / 4 + 56) * 4);
    matrix_b_fragment[0] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 0 ) * 4);
    matrix_b_fragment[1] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 8 ) * 4);
    matrix_b_fragment[2] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 16) * 4);
    matrix_b_fragment[3] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 24) * 4);
    matrix_b_fragment[4] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 32) * 4);
    matrix_b_fragment[5] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 40) * 4);
    matrix_b_fragment[6] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 48) * 4);
    matrix_b_fragment[7] = *(smem_b_sel + (lane_id % 4) + (lane_id / 4 + 56) * 4);

    int * a_fragment_int = reinterpret_cast<int *>(matrix_a_fragment);// 8 float4 --> 32 reg
    int * b_fragment_int = reinterpret_cast<int *>(matrix_b_fragment);// 8 float4 --> 32 reg

    #pragma unroll
    for(int i = 0; i < 4; i++){
        #pragma unroll
        for(int j = 0; j < 8; j++){
            asm ("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32 \t"
                "{%0, %1, %2, %3}, \t"
                "{%4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%0, %1, %2, %3}; ":
                "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                "r"(a_fragment_int[0 + 8 * i]), "r"(a_fragment_int[1 + 8 * i]),
                "r"(a_fragment_int[4 + 8 * i]), "r"(a_fragment_int[5 + 8 * i]),
                "r"(b_fragment_int[0 + 4 * j]), "r"(b_fragment_int[1 + 4 * j])
            );
            asm ("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32 \t"
                "{%0, %1, %2, %3}, \t"
                "{%4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%0, %1, %2, %3}; ":
                "+f"(output_fragment[0 + 4 * (8*i+j)]), "+f"(output_fragment[1 + 4 * (8*i+j)]),
                "+f"(output_fragment[2 + 4 * (8*i+j)]), "+f"(output_fragment[3 + 4 * (8*i+j)]):
                "r"(a_fragment_int[2 + 8 * i]), "r"(a_fragment_int[3 + 8 * i]),
                "r"(a_fragment_int[6 + 8 * i]), "r"(a_fragment_int[7 + 8 * i]),
                "r"(b_fragment_int[2 + 4 * j]), "r"(b_fragment_int[3 + 4 * j])
            );
        }
    }// end mma compute
    __syncthreads();
    float2 * output_ = reinterpret_cast<float2 *>(Output_Value + (bx * Block_M + warp_x * 64 + (int)(lane_id / 4)) * N + by * Block_N + warp_y * 64 + (int)((lane_id % 4) << 1));
    float2 * output_fragment_ = reinterpret_cast<float2 *>(output_fragment);// 128 reg --> 64 float2
    #pragma unroll
    for(int i = 0; i < 4; i++){
        #pragma unroll
        for(int j = 0; j < 8; j++){
            *(output_ + (i*8  ) * N + j * 4) = *(output_fragment_ + 2 * (i*8+j)    );
            *(output_ + (i*8+4) * N + j * 4) = *(output_fragment_ + 2 * (i*8+j) + 1);
        }
    }
}

cudaError_t GEMMex(
    int M, int K, int N,
    const e5m2 * __restrict__ A_Value,
    const e4m3 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
    const int Block_M = 128, Block_N = 128, Block_K = 64;
    dim3 block_dim(128,1,1);
	dim3 grid_dim(ceil(static_cast<float>(M) / Block_M), ceil(static_cast<float>(N) / Block_N), 1);

    GEMM_e5m2_e4m3_o32_stage2_row_col<<<grid_dim, block_dim>>>(
        M, K, N, A_Value, B_Value, Output_Value);
	return cudaGetLastError();
}
cudaError_t GEMM(
    int M, int K, int N,
    const e5m2 * __restrict__ A_Value,
    const e4m3 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
	return GEMMex(M, K, N, A_Value, B_Value, Output_Value);
}

//kernel4////////////////////e5m2*e5m2 = f32//////////////////
__global__ void GEMM_e5m2_e5m2_o32_stage4_row_col(
    int M, int K, int N,
    const e5m2 * __restrict__ A_Value,
    const e5m2 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
}
cudaError_t GEMMex4(
    int M, int K, int N,
    const e5m2 * __restrict__ A_Value,
    const e5m2 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
    const int Block_M = 128, Block_N = 128, Block_K = 64;
    dim3 block_dim(128,1,1);
	dim3 grid_dim(ceil(static_cast<float>(M) / Block_M), ceil(static_cast<float>(N) / Block_N), 1);
    // unsigned int dsmem = 4 * (Block_M * (Block_K + 0) + Block_N * (Block_K + 0)) * sizeof(char);

    GEMM_e5m2_e5m2_o32_stage4_row_col<<<grid_dim, block_dim>>>(
        M, K, N, A_Value, B_Value, Output_Value);
	return cudaGetLastError();
}
cudaError_t GEMM4(
    int M, int K, int N,
    const e5m2 * __restrict__ A_Value,
    const e5m2 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
	return GEMMex(M, K, N, A_Value, B_Value, Output_Value);
}
//kernel6////////////////////e4m3*e5m2 = f32//////////////////
__global__ void GEMM_e4m3_e5m2_o32_stage4_row_col(
    int M, int K, int N,
    const e4m3 * __restrict__ A_Value,
    const e5m2 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
}
cudaError_t GEMMex4(
    int M, int K, int N,
    const e4m3 * __restrict__ A_Value,
    const e5m2 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
    const int Block_M = 128, Block_N = 128, Block_K = 64;
    dim3 block_dim(128,1,1);
	dim3 grid_dim(ceil(static_cast<float>(M) / Block_M), ceil(static_cast<float>(N) / Block_N), 1);
    // unsigned int dsmem = 4 * (Block_M * (Block_K + 0) + Block_N * (Block_K + 0)) * sizeof(char);

    GEMM_e4m3_e5m2_o32_stage4_row_col<<<grid_dim, block_dim>>>(
        M, K, N, A_Value, B_Value, Output_Value);
	return cudaGetLastError();
}
cudaError_t GEMM4(
    int M, int K, int N,
    const e4m3 * __restrict__ A_Value,
    const e5m2 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
	return GEMMex(M, K, N, A_Value, B_Value, Output_Value);
}
//kernel8////////////////////e5m2*e4m3 = f32//////////////////
__global__ void GEMM_e5m2_e4m3_o32_stage4_row_col(
    int M, int K, int N,
    const e5m2 * __restrict__ A_Value,
    const e4m3 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
}
cudaError_t GEMMex4(
    int M, int K, int N,
    const e5m2 * __restrict__ A_Value,
    const e4m3 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
    const int Block_M = 128, Block_N = 128, Block_K = 64;
    dim3 block_dim(128,1,1);
	dim3 grid_dim(ceil(static_cast<float>(M) / Block_M), ceil(static_cast<float>(N) / Block_N), 1);
    // unsigned int dsmem = 4 * (Block_M * (Block_K + 0) + Block_N * (Block_K + 0)) * sizeof(char);

    GEMM_e5m2_e4m3_o32_stage4_row_col<<<grid_dim, block_dim>>>(
        M, K, N, A_Value, B_Value, Output_Value);
	return cudaGetLastError();
}
cudaError_t GEMM4(
    int M, int K, int N,
    const e5m2 * __restrict__ A_Value,
    const e4m3 * __restrict__ B_Value,
    float * __restrict__ Output_Value)
{
	return GEMMex(M, K, N, A_Value, B_Value, Output_Value);
}
}