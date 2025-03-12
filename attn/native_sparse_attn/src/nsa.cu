#include "../include/nsa.h"
#include "../include/utils.h"

// Helper function to convert float to bfloat16
void convertFloatToBFloat16(const float *src, __nv_bfloat16 *dst, size_t size) {
  for (size_t i = 0; i < size; i++) {
    dst[i] = __float2bfloat16(src[i]);
  }
}

__global__ void mqa_kernel(const __nv_bfloat16 *query, const __nv_bfloat16 *key, const __nv_bfloat16 *value,
                           float *output, int seq_len, int num_heads, int head_dim, long **block_indices,
                           long *block_counts, int block_size, float scale_factor);

void launch_mqa_kernel(const __nv_bfloat16 *query, const __nv_bfloat16 *key, const __nv_bfloat16 *value, float *output,
                       int seq_len, int num_heads, int head_dim, long **block_indices, long *block_counts,
                       int block_size, float scale_factor, cudaStream_t stream) {
  // Number of bytes in shared memory
  size_t qkv_mem_size = (num_heads * head_dim + 2 * (block_size * head_dim)) * sizeof(__nv_bfloat16);
  size_t output_tile_size = (num_heads * block_size) * sizeof(float);
  size_t warp_reduce_scratch_size = (block_size / warpSize * num_heads) * sizeof(float);

  size_t sharedMem = qkv_mem_size + output_tile_size + warp_reduce_scratch_size;

  dim3 blockDim(block_size, num_heads);
  dim3 gridDim(seq_len, seq_len);

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
