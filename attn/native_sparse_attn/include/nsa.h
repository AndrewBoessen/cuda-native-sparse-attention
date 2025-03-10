#ifndef NATIVE_SPARSE_ATTENTION_H
#define NATIVE_SPARSE_ATTENTION_H

#include <cuda_bf16.h>
#include <cuda_runtime.h>

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
void launch_mha_kernel(const __nv_bfloat16 *query, const __nv_bfloat16 *key, const __nv_bfloat16 *value,
                       __nv_bfloat16 *output, int batch_size, int seq_len, int num_heads, int head_dim,
                       long **block_indices, long *block_counts, float scale_factor, cudaStream_t stream = 0);

#endif // NATIVE_SPARSE_ATTENTION_H
