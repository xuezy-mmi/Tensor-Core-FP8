NVCC = /usr/local/cuda-12.4/bin/nvcc
NVCC_FLAGS = -std=c++11 -arch=sm_90 -lineinfo -lcublas --ptxas-options=-v

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

baseline:
	@$(NVCC) matmul-fp8-v0.cu -o baseline -arch=compute_90a -code=sm_90a --ptxas-options=-v

gemm_test:
	$(OUT_DIR)/gemm 128 128 128 2 1 0 0
gemm_run:
	$(OUT_DIR)/gemm 4096 4096 4096 1 0 0 0


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
