#ifndef NATIVE_SPARSE_ATTENTION_H
#define NATIVE_SPARSE_ATTENTION_H

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

/*
 * Cooperative tile loading for shared memory optimization
 * Loads matrix tiles from global memory to shared memory
 */
template <int TILE_SIZE>
__device__ __inline__ void
load_shared_tile(const __nv_bfloat16 *global_ptr, __nv_bfloat16 *shared_ptr,
                 int global_stride, int shared_stride, int row_offset,
                 int col_offset);

/*
 * Warp-level matrix multiply-accumulate using Tensor Cores
 * Performs D = A * B + C where dimensions are matrix_a[M][K], matrix_b[K][N]
 *
 * Template parameters:
 * - M: Rows in matrix A and matrix D
 * - N: Columns in matrix B and matrix D
 * - K: Columns in matrix A / Rows in matrix B
 *
 * All matrices must be aligned to 16-element boundaries
 */
template <int M, int N, int K>
__device__ __inline__ void
bf16_warp_mma(const __nv_bfloat16 *matrix_a, // [M][K] row-major
              const __nv_bfloat16 *matrix_b, // [K][N] column-major
              float *accumulators,           // [M][N] row-major
              int warp_row,                  // Warp's row position in block
              int warp_col                   // Warp's column position in block
);

/*
 * Multi-head attention kernel using bfloat16 precision
 * Implements Native Sparse Attention
 *
 * Expects input tensors in [batch_size, seq_len, num_heads, head_dim] format
 *
 * @param query         Input query tensor (device pointer)
 * @param key           Input key tensor (device pointer)
 * @param value         Input value tensor (device pointer)
 * @param output        Output tensor (device pointer)
 * @param batch_size    Batch size
 * @param seq_len       Sequence length
 * @param num_heads     Number of attention heads
 * @param head_dim      Dimension of each attention head
 * @param block_indices K,V blocks for each query
 * @param block_counts  Number of blocks per query
 * @param scale_factor  Scaling factor for attention scores (1/sqrt(head_dim))
 * @param stream        CUDA stream for kernel execution
 */
void launch_mha_kernel(const __nv_bfloat16 *query, const __nv_bfloat16 *key,
                       const __nv_bfloat16 *value, __nv_bfloat16 *output,
                       int batch_size, int seq_len, int num_heads, int head_dim,
                       long **block_indices, long *block_counts,
                       float scale_factor, cudaStream_t stream = 0);

#endif // NATIVE_SPARSE_ATTENTION_H
