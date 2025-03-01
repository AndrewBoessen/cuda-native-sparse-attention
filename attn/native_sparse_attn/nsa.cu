#include "nsa.h"

template <int TILE_SIZE>
__device__ __inline__ void
load_shared_tile(const __nv_bfloat16 *global_ptr, __nv_bfloat16 *shared_ptr,
                 int global_stride, int shared_stride, int row_offset,
                 int col_offset) {
#pragma unroll
  for (int i = threadIdx.y; i < TILE_SIZE; i += blockDim.y) {
#pragma unroll
    for (int j = threadIdx.x; j < TILE_SIZE; j += blockDim.x) {
      shared_ptr[i * shared_stride + j] =
          global_ptr[(row_offset + i) * global_stride + (col_offset + j)];
    }
  }
  __syncthreads();
}
