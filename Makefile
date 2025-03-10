# Compiler and flags
CC = nvcc
CXX = g++
CFLAGS = -arch=sm_89 -O3 --use_fast_math -std=c++17 \
         -Xcompiler -Wall,-Wextra -Iinclude

# Directories
SRC_DIR = attn/native_sparse_attn/src
OBJ_DIR = build
INC_DIR = attn/native_sparse_attn/include

# Target and objects
EXEC = cuda_program
SRCS = $(wildcard $(SRC_DIR)/*.cu) $(SRC_DIR)/main.cpp
OBJS = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(filter %.cu, $(SRCS))) \
       $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(filter %.cpp, $(SRCS)))

# Ensure build directory exists
$(shell mkdir -p $(OBJ_DIR))

# Default target
all: $(EXEC)

# Link executable
$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@

# Compile CUDA files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(CC) $(CFLAGS) -c $< -o $@ -MD -MP

# Compile C++ files with proper dependencies
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) -std=c++17 -Wall -Wextra -I$(INC_DIR) -c $< -o $@ -MD -MP

# Include dependency files
-include $(OBJS:.o=.d)

# Clean with directory support
clean:
	rm -rf $(OBJ_DIR)/*.o $(OBJ_DIR)/*.d $(EXEC)

.PHONY: all clean

