#ifndef UTILS_H
#define UTILS_H

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

/**
 * Cooperative tile loading for shared memory optimization
 * Loads matrix tiles from global memory to shared memory
 */
template <typename T, int TILE_SIZE>
__device__ __inline__ void load_shared_tile(const T *global_ptr, T *shared_ptr, int global_stride, int shared_stride,
                                            int row_offset, int col_offset);

/**
 * Warp-level matrix multiply accumulate using Tensor Cores
 * Performs C = A * B where dimensions are matrix_a[M][K], matrix_b[K][N]
 *
 * Template parameters:
 * - M: Rows in matrix A and matrix D
 * - N: Columns in matrix B and matrix D
 * - K: Columns in matrix A / Rows in matrix B
 * --A: Accumulate
 *
 * All matrices must be aligned to 16-element boundaries
 */
template <int M, int N, int K, bool A>
__device__ __inline__ void bf16_warp_mma(const __nv_bfloat16 *matrix_a, // [M][K] column-major
                                         const __nv_bfloat16 *matrix_b, // [K][N] row-major
                                         float *matrix_c                // [M][N] row-major
);

/**
 * Performs a sum reduction within a warp using CUDA's warp-level primitives.
 * This function assumes that threads in the same warp are participating.
 *
 * @param val The value to sum from the current thread
 * @return The sum of values from all threads in the warp (returned to all threads)
 */
__device__ inline float warpReduceSum(float val);

/**
 * A complete block reduction sum that uses the warp reduction internally.
 * Handles cases where block size > 32 (one warp) by first reducing within warps,
 * then combining warp results.
 *
 * @param val The value to sum from the current thread
 * @param shared The shared memory array pointer
 * @return The sum of values from all threads in the block (returned to thread 0)
 */
__device__ float blockReduceSum(float val, float *shared);

/**
 * Find the maximum value within a warp using warp shuffle operations.
 *
 * @param val The input value from each thread
 * @return The maximum value across all threads in the block (returned to all threads)
 */
__device__ float warpReduceMax(float val);

/**
 * Find the maximum across the rows in the block.
 *
 * @param val The value for each thread
 * @param shared The shared memory to hold warp max
 * @return The maximum value across block row
 */
__device__ float blockReduceMax(float val, float *shared);
#endif // UTILS_H
