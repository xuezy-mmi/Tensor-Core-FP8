# Tensor-Core-FP8-Kernel

NVIDIA released MMA instructions for FP8 (e4m3 and e5m2) in PTX 8.4.

However, there are few open-source kernels that support FP8, except for TransformerEngine and cuBLAS.

I will gradually implement GEMM, GEMV, SpMM, SpMV and other CUDA kernels based on FP8-MMA Instruction in this project.


|                  |     4090     |  H100 PCIe  |
|:----------------:|:------------:|:-----------:|
| Memory Bandwidth |   1008 GB/s  |  2039 GB/s  |
|  FP8 Peak Perf.  | 660.6 TFLOPS | 1513 TFLOPS |
| Shared Memory/SM |    128 KB    |    228 KB   |
|        SMs       |      128     |     114     |

|kernel|Block Size| layout |Precision| Pipeline Stage |
|------|--------|------|------|------|
|GEMM  |128x128x64|row-col|e4m3*e4m3=fp32|2-stage|
|GEMM  |128x128x64|row-col|e5m2*e5m2=fp32|2-stage|
|GEMM  |128x128x64|row-col|e4m3*e5m2=fp32|2-stage|
|GEMM  |128x128x64|row-col|e5m2*e4m3=fp32|2-stage|
|GEMM  |128x128x64|row-col|e4m3*e4m3=fp32|4-stage|
|GEMM  |128x128x64|row-col|e5m2*e5m2=fp32|4-stage|
|GEMM  |128x128x64|row-col|e4m3*e5m2=fp32|4-stage|
|GEMM  |128x128x64|row-col|e5m2*e4m3=fp32|4-stage|

