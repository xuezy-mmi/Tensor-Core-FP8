#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>
#include <float.h>
#include <random>
#include <assert.h>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
// #include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mma.h>

#include "./include/bm_test_utils.h"
#include "./include/gemm.cuh"

using namespace std;
#define repeat 200
#define warmup 20

template <typename AType, typename BType, typename CType>
void BmFN(int M, int K, int N, int kernel, bool func, bool record){

    // declare Matrix A and Matrix B and Matrix Output in cpu
    std::default_random_engine generator;
    AType * h_A_Value = new AType[M * K];// row major
    BType * h_B_Value = new BType[N * K];// col major
    MakeDenseMatrix<AType>(M, K, h_A_Value, generator);
    MakeDenseMatrix<BType>(N, K, h_B_Value, generator);
    float * h_Output_Value = new float[M * N];// row major

    if(func == 1){// compute output in cpu
        // init output in cpu
        for(int i = 0; i < M*N; i++){
            h_Output_Value[i] = 0.0f;
        }
        // gemm in cpu
        for(int m = 0; m <M; m++){
            for(int n = 0; n < N; n++){
                for(int k = 0; k < K; k++){
                    h_Output_Value[m*N+n] += (float)h_A_Value[m*K+k] * (float)h_B_Value[n*K+k];
                }
                
            }
        }
    }// end compute output in cpu

    // init Values in gpu
    AType * d_A_Value;
    BType * d_B_Value;
    CType * d_Output_Value;
    checkCuda(cudaMalloc(&d_A_Value, (M*K) * sizeof(AType)));
    checkCuda(cudaMalloc(&d_B_Value, (N*K) * sizeof(BType)));
    checkCuda(cudaMalloc(&d_Output_Value, (M*N) * sizeof(CType)));
    // copy value into gpu
    checkCuda(cudaMemcpy(d_A_Value, h_A_Value, (M*K) * sizeof(AType), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B_Value, h_B_Value, (N*K) * sizeof(BType), cudaMemcpyHostToDevice));
    
    if(kernel % 2 == 1){// execute gemm
        float total_msec = 0.0;
        double Gperf = 0;
        for(int i = 0; i < repeat+warmup; i++){
            float msec = 0.0;
            cudaEvent_t start;
            cudaEvent_t end;
            checkCuda(cudaEventCreate(&start));
            checkCuda(cudaEventCreate(&end));
            checkCuda(cudaEventRecord(start));
            ////////////////////////////////////////////
            gemm::GEMM(M, K, N, d_A_Value, d_B_Value, d_Output_Value);
            ////////////////////////////////////////////
            checkCuda(cudaEventRecord(end));
            checkCuda(cudaEventSynchronize(end));
            checkCuda(cudaEventElapsedTime(&msec, start, end));
            if(i >= warmup) {total_msec = total_msec + msec;}
        }
        Gperf = ((double)M) * K * N * 2 / 1000/ 1000 / total_msec * repeat;
        printf("GEMM-stage2:  M = %d, K = %d, N = %d\n", M, K, N);
        printf("\033[33mFP8-TC Avager Performance = \033[35m%lf\033[0m \033[33mGFlops\033[0m\n", Gperf);
        if(record){// record performance output into csv file
            std::ofstream outFile;
            //-------------edit here to change the output file-----------------------------
            std::string output_dir = "./data/";
            std::string line;
            if(kernel == 1){
                line = "fp8_gemm_v1.csv";
            }
            else if(kernel == 3){
                line = "fp8_gemm_v3.csv";
            }
            else if(kernel == 5){
                line = "fp8_gemm_v5.csv";
            }
            else if(kernel == 7){
                line = "fp8_gemm_v7.csv";
            }
            output_dir = output_dir + line;

            outFile.open(output_dir, std::ios::app);
            //-----------------------------------------------------------------------------
            outFile << M << ',' << K << ',' << N << ',' << Gperf << std::endl;
            outFile.close();
        }
    }
    else if(kernel % 2 == 0){// execute gemm
        float total_msec = 0.0;
        double Gperf = 0;
        for(int i = 0; i < repeat+warmup; i++){
            float msec = 0.0;
            cudaEvent_t start;
            cudaEvent_t end;
            checkCuda(cudaEventCreate(&start));
            checkCuda(cudaEventCreate(&end));
            checkCuda(cudaEventRecord(start));
            ////////////////////////////////////////////
            gemm::GEMM4(M, K, N, d_A_Value, d_B_Value, d_Output_Value);
            ////////////////////////////////////////////
            checkCuda(cudaEventRecord(end));
            checkCuda(cudaEventSynchronize(end));
            checkCuda(cudaEventElapsedTime(&msec, start, end));
            if(i >= warmup) {total_msec = total_msec + msec;}
        }
        Gperf = ((double)M) * K * N * 2 / 1000/ 1000 / total_msec * repeat;
        printf("GEMM-stage4:  M = %d, K = %d, N = %d\n", M, K, N);
        printf("\033[33mFP8-TC Avager Performance = \033[35m%lf\033[0m \033[33mGFlops\033[0m\n", Gperf);
        if(record){// record performance output into csv file
            std::ofstream outFile;
            //-------------edit here to change the output file-----------------------------
            std::string output_dir = "./data/";
            std::string line;
            if(kernel == 2){
                line = "fp8_gemm_v2.csv";
            }
            else if(kernel == 4){
                line = "fp8_gemm_v4.csv";
            }
            else if(kernel == 6){
                line = "fp8_gemm_v6.csv";
            }
            else if(kernel == 8){
                line = "fp8_gemm_v8.csv";
            }
            output_dir = output_dir + line;

            outFile.open(output_dir, std::ios::app);
            //-----------------------------------------------------------------------------
            outFile << M << ',' << K << ',' << N << ',' << Gperf << std::endl;
            outFile.close();
        }
    }

    unsigned int num_errors = 0;
    if(func){//verify
        CType * Output_Value_base = new CType[M*N];
        checkCuda(cudaMemcpy(Output_Value_base, d_Output_Value, M*N * sizeof(CType), cudaMemcpyDeviceToHost));
        // verify result
        float max_error = abs((Output_Value_base[0] - h_Output_Value[0]));
        for(int i = 0; i < M*N; i++){
            float temp_error = abs((Output_Value_base[i] - h_Output_Value[i]));
            // std::cout << Output_Value_base[i] << " " << h_Output_Value[i] << " difference: " << temp_error << std::endl;
            if(temp_error > 0.25){
                num_errors = num_errors + 1;
            }
            if(temp_error > max_error){
                max_error = temp_error;
            }
        }
        if(num_errors = 0){
            printf("Result Verified!\n");
        }
        else{
            // printf("There are %d errors.\n", num_errors);
            std::cout << "There are " << num_errors << " errors." << std::endl;
        }
        printf("MAx Error is %f\n", max_error);
        delete Output_Value_base;
    }
    cudaFree(d_A_Value);
    cudaFree(d_B_Value);
    cudaFree(d_Output_Value);

    delete h_A_Value;
    delete h_B_Value;
    delete h_Output_Value;
    
    printf("end BmFN\n");
}

void usage(void){
    printf("!!!!!!!\n");
    printf("Input Help!\n");
    printf("Input Format is  ./gemm_main [dimM] [dimK] [dimN] [kernel] [verify] [reocrd] [testbench]\n");
    printf("dim: M-K-N in GEMM\n");
    printf("kernel: {0 , 1 }\n");
    printf("verify: {0 not verify result with cpu, 1: verify result}\n");
    printf("record: {record in csv files}\n");
    printf("!!!!!!!\n");
    exit(1);
}
int main(int argc, char **argv){

    if(argc != 8 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0){
        usage();
    }
    // // Run the benchmark
    // else{
    int dimM = std::atoi(argv[1]);//A_rows_num
    int dimK = std::atoi(argv[2]);//A_cols_num // B_rows_num
    int dimN = std::atoi(argv[3]);//B_cols_num
    const int kernel = std::atoi(argv[4]);//0:
    const int verify = std::atoi(argv[5]);//0:not verify  1:verify
    const int record = std::atoi(argv[6]);//
    const int testbench = std::atoi(argv[7]);//
    if(testbench == 0){
        if(kernel == 1){
            printf("e4m3 * e4m3 = fp16\n");
            BmFN<__nv_fp8_e4m3, __nv_fp8_e4m3, float>(dimM, dimK, dimN, kernel, verify, record);
        }
        else if(kernel == 2){
            printf("e4m3 * e4m3 = fp32\n");
            BmFN<__nv_fp8_e4m3, __nv_fp8_e4m3, float>(dimM, dimK, dimN, kernel, verify, record);
        }
        else if(kernel == 3){
            printf("e5m2 * e5m2 = fp16\n");
            BmFN<__nv_fp8_e5m2, __nv_fp8_e5m2, float>(dimM, dimK, dimN, kernel, verify, record);
        }
        else if(kernel == 4){
            printf("e5m2 * e5m2 = fp32\n");
            // BmFN<__nv_fp8_e5m2, __nv_fp8_e5m2, float>(dimM, dimK, dimN, kernel, verify, record);
        }
        else if(kernel == 5){
            printf("e4m3 * e5m2 = fp16\n");
            BmFN<__nv_fp8_e4m3, __nv_fp8_e5m2, float>(dimM, dimK, dimN, kernel, verify, record);
        }
        else if(kernel == 6){
            printf("e4m3 * e5m2 = fp32\n");
            // BmFN<__nv_fp8_e4m3, __nv_fp8_e5m2, flo   at>(dimM, dimK, dimN, kernel, verify, record);
        }
        else if(kernel == 7){
            printf("e5m2 * e4m3 = fp16\n");
            BmFN<__nv_fp8_e5m2, __nv_fp8_e4m3, float>(dimM, dimK, dimN, kernel, verify, record);
        }
        else if(kernel == 8){
            printf("e5m2 * e4m3 = fp32\n");
            // BmFN<__nv_fp8_e5m2, __nv_fp8_e4m3, float>(dimM, dimK, dimN, kernel, verify, record);
        }
        else{
            printf("unsupported kernel\n");
        }
    }
    else{
        const int test_num = 64;
        // int M_t[test_num];
        // int K_t[test_num];
        // int N_t[test_num];
        for(int i = 0; i < test_num; i++){
            // M_t[i] = 256*(i+1);
            // K_t[i] = 256*(i+1);
            // N_t[i] = 256*(i+1);
            dimM = 256*(i+1);
            dimK = 256*(i+1);
            dimN = 256*(i+1);

            if(kernel == 1){
                printf("e4m3 * e4m3 = fp16\n");
                BmFN<__nv_fp8_e4m3, __nv_fp8_e4m3, float>(dimM, dimK, dimN, kernel, verify, record);
            }
            else if(kernel == 2){
                printf("e4m3 * e4m3 = fp32\n");
                BmFN<__nv_fp8_e4m3, __nv_fp8_e4m3, float>(dimM, dimK, dimN, kernel, verify, record);
            }
            else if(kernel == 3){
                printf("e5m2 * e5m2 = fp16\n");
                BmFN<__nv_fp8_e5m2, __nv_fp8_e5m2, float>(dimM, dimK, dimN, kernel, verify, record);
            }
            else if(kernel == 4){
                printf("e5m2 * e5m2 = fp32\n");
                // BmFN<__nv_fp8_e5m2, __nv_fp8_e5m2, float>(dimM, dimK, dimN, kernel, verify, record);
            }
            else if(kernel == 5){
                printf("e4m3 * e5m2 = fp16\n");
                BmFN<__nv_fp8_e4m3, __nv_fp8_e5m2, float>(dimM, dimK, dimN, kernel, verify, record);
            }
            else if(kernel == 6){
                printf("e4m3 * e5m2 = fp32\n");
                // BmFN<__nv_fp8_e4m3, __nv_fp8_e5m2, float>(dimM, dimK, dimN, kernel, verify, record);
            }
            else if(kernel == 7){
                printf("e5m2 * e4m3 = fp16\n");
                BmFN<__nv_fp8_e5m2, __nv_fp8_e4m3, float>(dimM, dimK, dimN, kernel, verify, record);
            }
            else if(kernel == 8){
                printf("e5m2 * e4m3 = fp32\n");
                // BmFN<__nv_fp8_e5m2, __nv_fp8_e4m3, float>(dimM, dimK, dimN, kernel, verify, record);
            }
            else{
                printf("unsupported kernel\n");
            }
        }
    }
    return 0;
}
