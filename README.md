# Tensor-Core-FP8-Kernel

NVIDIA released MMA instructions for FP8 (e4m3 and e5m2) in PTX 8.4.

However, there are few open-source kernels that support FP8, except for TransformerEngine and cuBLAS.

I will gradually implement GEMM, GEMV, SpMM, SpMV and other CUDA kernels based on FP8-MMA Instruction in this project.


|kernel|Block Size| layout |Precision|
|------|--------|------|------|
|GEMM  |128x128x64|row-col|e4m3*e4m3=fp32|

