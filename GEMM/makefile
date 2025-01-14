NVCC = /usr/local/cuda-12.4/bin/nvcc
NVCC_FLAGS = -std=c++11 -arch=sm_90 -lineinfo --ptxas-options=-v

##################################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file directory
INC_DIR = include

# Library directory
LIB_DIR = /usr/local/cuda/lib64

#output directory
OUT_DIR = out
#csv output directory
RECORD_DIR = data
##################################################################

## Compile ##

# Compile main file to object file
$(OBJ_DIR)/%.o : ./%.cpp
	@$(NVCC) $(NVCC_FLAGS) -x c++ -c $< -o $@

# Compile CUDA source files to object files
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	@$(NVCC) $(NVCC_FLAGS) -x cu -c $< -o $@

gemm: $(OBJ_DIR)/gemm_main.o $(OBJ_DIR)/gemm.o
	@$(NVCC) $(NVCC_FLAGS) $^  -o $(OUT_DIR)/$@

cublas:
	@$(NVCC) -lcublasLt $(SRC_DIR)/cublas.cu -o $(OUT_DIR)/cublas
#  -Xcompiler -Wall
cublas_run:
	$(OUT_DIR)/cublas 100

baseline:
	@$(NVCC) $(SRC_DIR)/matmul-fp8-v0.cu -o $(OUT_DIR)/baseline -arch=compute_90a -code=sm_90a

gemm_test:
	$(OUT_DIR)/gemm 128 128 128 2 1 0 0
gemm_run:
	$(OUT_DIR)/gemm 4096 4096 4096 1 0 0 0

baseline_run:
	$(OUT_DIR)/baseline
clean:
	@rm -f $(OBJ_DIR)/*
	@rm -f $(OUT_DIR)/*
	@rm -f $(RECORD_DIR)/*

obj_clean:
	@rm -f $(OBJ_DIR)/*
out_clean:
	@rm -f $(OUT_DIR)/*
record_clean:
	@rm -f $(RECORD_DIR)/*
