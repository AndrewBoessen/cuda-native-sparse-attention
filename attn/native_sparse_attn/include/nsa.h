#ifndef NATIVE_SPARSE_ATTENTION_H
#define NATIVE_SPARSE_ATTENTION_H

#include <assert.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#define THREADS_IN_BLOCK 128

#define NUM_HEADS 16
#define BLOCK_SIZE 32
#define HEAD_DIM 128

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
/**
 * Multi-query self attention using Native Sparse Attention
 *
 * Template parameters:
 * - T: Sequence length
 * - H: Heads in Q. MQA so K,V have 1 head
 * - D: Dimension per head
 *
 * @param query         Query array [B, T, H, D]
 * @param key           Key array [B, T, 1, D]
 * @param value         Value array [B, T, 1, D]
 * @param output        Output matrix [B, T, 1, D]
 * @param seq_len       Sequence length
 * @param num_heads     Heads in Q. MQA so K,V have 1 head
 * @param head_dim      Dimension per head
 * @param block-indices Indices of K,V pairs per row [T, T]
 * @param block_counts  Number of K,V blocks per row [T]
 * @param block_size    Number of tokens per block
 * @param scale_factor  Scale factor for QK^T default 1/sqrt(D)
 */
__global__ void mqa_kernel(const __nv_bfloat16 *query, const __nv_bfloat16 *key, const __nv_bfloat16 *value,
                           float *output, int seq_len, int num_heads, int head_dim, long **block_indices,
                           long *block_counts, int block_size, float scale_factor);

/**
 * Multi-query attention kernel using bfloat16 precision
 * Implements Native Sparse Attention
 *
 * Expects input tensors in [seq_len, num_heads, head_dim] format
 *
 * @param query         Input query tensor (device pointer)
 * @param key           Input key tensor (device pointer)
 * @param value         Input value tensor (device pointer)
 * @param output        Output tensor (device pointer)
 * @param seq_len       Sequence length
 * @param num_heads     Number of attention heads
 * @param head_dim      Dimension of each attention head
 * @param block_indices K,V blocks for each query
 * @param block_counts  Number of blocks per query
 * @param block_size    Number of token per block
 * @param scale_factor  Scaling factor for attention scores (1/sqrt(head_dim))
 * @param stream        CUDA stream for kernel execution
 */
void launch_mqa_kernel(const __nv_bfloat16 *query, const __nv_bfloat16 *key, const __nv_bfloat16 *value, float *output,
                       int seq_len, int num_heads, int head_dim, long **block_indices, long *block_counts,
                       int block_size, float scale_factor, cudaStream_t stream = 0);

/**
 * Native Sparse Attention
 * Implemented with bfloat16, Tensor Core MMA, and stream optimized scheduling.
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
 * @param block_size    Number of token per block
 * @param scale_factor  Scaling factor for attention scores (1/sqrt(head_dim))
 */
void native_sparse_attention(const float *query, const float *key, const float *value, float *output, int batch_size,
                             int seq_len, int num_heads, int head_dim, long **block_indices, long *block_counts,
                             int block_size, float scale_factor);

#endif // NATIVE_SPARSE_ATTENTION_H
