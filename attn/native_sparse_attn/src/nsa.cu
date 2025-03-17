#include "../include/nsa.h"
#include "../include/utils.h"
#include <assert.h>
#include <cmath>

// Helper function to convert float to bfloat16
void convertFloatToBFloat16(const float *src, __nv_bfloat16 *dst, size_t size) {
  for (size_t i = 0; i < size; i++) {
    dst[i] = __float2bfloat16(src[i]);
  }
}

__global__ void mqa_kernel(const __nv_bfloat16 *query, const __nv_bfloat16 *key, const __nv_bfloat16 *value,
                           float *output, int seq_len, int num_heads, int head_dim, long **block_indices,
                           long *block_counts, int block_size, float scale_factor) {
  extern __shared__ __nv_bfloat16 smem[];

  // Offsets into shared memory
  __nv_bfloat16 *p_q = smem;
  __nv_bfloat16 *p_k = p_q + num_heads * head_dim;
  __nv_bfloat16 *p_v = p_k + block_size * head_dim;
  float *s = (float *)(p_v + block_size * head_dim);
  __nv_bfloat16 *p = (__nv_bfloat16 *)(s + num_heads * block_size);

  // Outer Loop (Q)
  int grid_row = blockIdx.x;
  float *o_bos = output + (grid_row * num_heads * head_dim);

  // Load query heads to smem
  // Loops over heads
  for (int head = 0; head < num_heads; head++) {
    load_shared_tile<__nv_bfloat16, THREADS_IN_BLOCK>(query, p_q + head, 1, num_heads,
                                                      (grid_row * num_heads * head_dim) + (head * head_dim), 0);
  }

  // Array to hold M for each head
  float m_p[NUM_HEADS] = {-INFINITY};
  // Array to hold accumulator
  float acc_p[NUM_HEADS] = {0};

  // Inner Loop (KV)
  long num_blocks = block_counts[grid_row];
  for (int i = 0; i < num_blocks; i++) {
    long block_id = block_indices[grid_row][i];

    // Load KV blocks to shared memory
    for (int t = 0; t < block_size; t++) {
      load_shared_tile<__nv_bfloat16, THREADS_IN_BLOCK>(key, p_k + t, 1, block_size,
                                                        (block_id * block_size + t) * head_dim, 0);
      load_shared_tile<__nv_bfloat16, THREADS_IN_BLOCK>(value, p_v + (t * head_dim), 1, 1,
                                                        (block_id * block_size + t) * head_dim, 0);
    }
    // Compute QK^T
    bf16_warp_mma<NUM_HEADS, BLOCK_SIZE, HEAD_DIM, false>(p_q, p_k, s);
    // Intermediate row operations
    for (int row = threadIdx.x / block_size; row < num_heads; row += blockDim.x) {
      int col = threadIdx.x % block_size;
      // rowmax of S
      float max = warpReduceMax(s[row * block_size + col]);
      // max(m^-1, m)
      float m = max > m_p[row] ? max : m_p[row];
      // P = exp(m - S)
      s[row * block_size + col] = expf(s[row * block_size + col] - max);
      // R = exp(m^-1 - m)
      float r = expf(m_p[row] - max);

      // rowsum of P
      float sum = warpReduceSum(s[row * block_size + col]);
      // broadcast result to warp
      sum = __shfl_sync(0xffffffff, sum, 0);

      // l = exp(m^-1 - m) * l^-1 + rowsum(P)
      float acc = r * acc_p[row] + sum;

      // O = O * diag(R)
      for (int off = col; off < head_dim; off += block_size) {
        o_bos[row * head_dim + off] *= r;
      }
      // Update intermediate values
      m_p[row] = m;
      acc_p[row] = acc;
      // cast to bfloat16
      p[row * block_size + col] = __float2bfloat16(s[row * block_size + col]);
    }
    // O = O + PV
    bf16_warp_mma<NUM_HEADS, HEAD_DIM, BLOCK_SIZE, true>(p, p_v, o_bos);
  }
  // O = O / diag(l)
  for (int row = 0; row < num_heads; row++) {
    o_bos[row * head_dim + threadIdx.x] /= acc_p[row];

    // update M
    m_p[row] += log(acc_p[row]);
  }
}

void launch_mqa_kernel(const __nv_bfloat16 *query, const __nv_bfloat16 *key, const __nv_bfloat16 *value, float *output,
                       int seq_len, int num_heads, int head_dim, long **block_indices, long *block_counts,
                       int block_size, float scale_factor, cudaStream_t stream) {
  // size must be compatible with wmma tiles
  assert(num_heads % 16 == 0);
  assert(head_dim % 16 == 0);
  assert(block_size % 16 == 0);

  // the dimension connot be split between blocks
  assert(head_dim <= THREADS_IN_BLOCK);

  // Number of bytes in shared memory
  size_t qkv_mem_size = (num_heads * head_dim + 2 * (block_size * head_dim)) * sizeof(__nv_bfloat16);
  size_t s_mem_size = num_heads * block_size * sizeof(float);
  size_t p_mem_size = num_heads * block_size * sizeof(__nv_bfloat16);

  size_t sharedMem = qkv_mem_size + s_mem_size + p_mem_size;

  dim3 blockDim(THREADS_IN_BLOCK);
  dim3 gridDim(seq_len);

  mqa_kernel<<<gridDim, blockDim, sharedMem, stream>>>(query, key, value, output, seq_len, num_heads, head_dim,
                                                       block_indices, block_counts, block_size, scale_factor);
}

void native_sparse_attention(const float *query, const float *key, const float *value, float *output, int batch_size,
                             int seq_len, int num_heads, int head_dim, long **block_indices, long *block_counts,
                             int block_size, float scale_factor) {
  // Number of elements in array
  size_t q_size = batch_size * seq_len * num_heads * head_dim;
  size_t kv_size = batch_size * seq_len * head_dim;
  size_t o_size = q_size;

  // Allocate memory on device
  __nv_bfloat16 *d_q, *d_k, *d_v;
  float *d_o;
  cudaMalloc(&d_q, q_size * sizeof(__nv_bfloat16));
  cudaMalloc(&d_k, kv_size * sizeof(__nv_bfloat16));
  cudaMalloc(&d_v, kv_size * sizeof(__nv_bfloat16));
  cudaMalloc(&d_o, o_size * sizeof(float));

  // Allocate host bfloat16 arrays
  __nv_bfloat16 *bf16_query = (__nv_bfloat16 *)malloc(q_size * sizeof(__nv_bfloat16));
  __nv_bfloat16 *bf16_key = (__nv_bfloat16 *)malloc(kv_size * sizeof(__nv_bfloat16));
  __nv_bfloat16 *bf16_value = (__nv_bfloat16 *)malloc(kv_size * sizeof(__nv_bfloat16));

  // Cast host float arrays to bfloat16
  convertFloatToBFloat16(query, bf16_query, q_size);
  convertFloatToBFloat16(key, bf16_key, kv_size);
  convertFloatToBFloat16(value, bf16_value, kv_size);

  // Split batch over number of streams (i.e. one kernel per batch)
  int num_streams = batch_size;
  cudaStream_t *streams = new cudaStream_t[num_streams];

  // Initialize streams
  for (int i = 0; i < num_streams; i++) {
    cudaStreamCreate(&streams[i]);
  }

  // Multi-Query Attention (K,V pair shared across all heads in Q)
  size_t kv_batch_stride = seq_len * head_dim;
  size_t q_batch_stride = kv_batch_stride * num_heads;
  // Output shape [B, T, H, D]
  size_t o_batch_stride = q_batch_stride;

  // Launch kernel grids
  for (int i = 0; i < num_streams; i++) {
    // Offsets into input host arrays
    int kv_offset = kv_batch_stride * i;
    int q_offset = q_batch_stride * i;
    int o_offset = o_batch_stride * i;

    // Copy memory to device for current stream
    cudaMemcpyAsync(d_q + q_offset, bf16_query + q_offset, q_offset * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice,
                    streams[i]);
    cudaMemcpyAsync(d_k + kv_offset, bf16_key + kv_offset, kv_offset * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice,
                    streams[i]);
    cudaMemcpyAsync(d_v + kv_offset, bf16_value + kv_offset, kv_offset * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice,
                    streams[i]);

    // Launch kernel for current stream
    launch_mqa_kernel(d_q, d_k, d_v, d_o, seq_len, num_heads, head_dim, block_indices, block_counts, block_size,
                      scale_factor, streams[i]);

    // Copy result back to host
    cudaMemcpyAsync(output + o_offset, d_o + o_offset, o_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
  }

  // Wait for all streams to finish
  for (int i = 0; i < num_streams; ++i) {
    cudaStreamSynchronize(streams[i]);
  }

  // Free resources
  for (int i = 0; i < num_streams; ++i) {
    cudaStreamDestroy(streams[i]);
  }

  // Free device memory
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_o);
}
