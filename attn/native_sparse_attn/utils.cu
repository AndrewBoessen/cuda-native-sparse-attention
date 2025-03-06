#include "utils.h"
#include <assert.h>
#include <mma.h>

using namespace nvcuda;

__device__ inline float warpReduceSum(float val) {
  // Use warp shuffle down operations to perform reduction
  // Each step halves the number of active threads but combines values
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }

  // At this point, lane 0 has the sum of all values in the warp
  return val;
}

__device__ float blockReduceSum(float val) {
  // Shared memory for storing warp sums
  static __shared__ float shared[32]; // One element per warp

  int lane = threadIdx.x % 32;   // Lane index within warp
  int warpId = threadIdx.x / 32; // Warp index within block

  // Perform warp reduction on each warp
  val = warpReduceSum(val);

  // Store the warp sum in shared memory (only lane 0 of each warp)
  if (lane == 0) {
    shared[warpId] = val;
  }

  // Make sure all warp sums are visible to all threads
  __syncthreads();

  // Read warp sums from shared memory (first warp only)
  val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;

  // Final reduction of warp sums (first warp only)
  if (warpId == 0) {
    val = warpReduceSum(val);
  }

  return val; // Result is stored in thread 0
}

template <int TILE_SIZE>
__device__ __inline__ void load_shared_tile(const __nv_bfloat16 *global_ptr, __nv_bfloat16 *shared_ptr,
                                            int global_stride, int shared_stride, int row_offset, int col_offset) {
#pragma unroll
  for (int i = threadIdx.y; i < TILE_SIZE; i += blockDim.y) {
#pragma unroll
    for (int j = threadIdx.x; j < TILE_SIZE; j += blockDim.x) {
      shared_ptr[i * shared_stride + j] = global_ptr[(row_offset + i) * global_stride + (col_offset + j)];
    }
  }
  __syncthreads();
}

template <int M, int N, int K>
__device__ __inline__ void bf16_warp_mm(const __nv_bfloat16 *matrix_a, // [M][K] column-major
                                        const __nv_bfloat16 *matrix_b, // [K][N] row-major
                                        float *matrix_c                // [M][N] row-major
) {
  // check if matrix fits in block
  assert(blockDim.x >= N);
  assert(blockDim.y >= M);

  // define warps tile
  int row_id = threadIdx.y / WMMA_M;
  int col_id = threadIdx.x / WMMA_N;

  // strides
  const int a_stride = M; // stride between cols
  const int b_stride = N; // stride between rows
  const int c_stride = N; // stride between rows

  // initial offsets
  int c_offset = row_id * (c_stride * WMMA_M) + col_id * WMMA_N;
  int a_offset = row_id * WMMA_M;
  int b_offset = col_id + WMMA_N;

  // declare wmma fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, nv_bfloat16, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  // fill output tile with 0
  wmma::fill_fragment(c_frag, 0.0f);

  // one thread per element in output matrix
  if (threadIdx.x < N && threadIdx.y < M) {
    // loop over inner dimension
#pragma unroll
    for (int k = 0; k < K; k += WMMA_K) {
      // load fragments
      wmma::load_matrix_sync(a_frag, matrix_a + a_offset + k * a_stride, a_stride);
      wmma::load_matrix_sync(b_frag, matrix_b + b_offset + k * b_stride, b_stride);

      // matmul accumulate on current tiles, C = AB + C
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // store tile result to output matrix
    wmma::store_matrix_sync(matrix_c + c_offset, c_frag, c_stride, wmma::mem_row_major);
  }
}
