# GEMM-Tensor-Core-FP8

NVIDIA released MMA instructions for fp8 (e4m3 and e5m2) in PTX 8.4.

However, there are no kernels that support FP8, except for TransformerEngine and cuBLAS.

I will gradually implement GEMM, GEMV, SpMM, SpMV and other CUDA kernels based on FP8-MMA Instruction in this project.
