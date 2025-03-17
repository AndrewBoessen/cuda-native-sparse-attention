#include "../include/nsa.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Helper function to fill array with random float values
void fill_random_float(float *array, size_t size) {
  for (size_t i = 0; i < size; i++) {
    array[i] = (float)rand() / RAND_MAX;
  }
}

// Helper function to fill array with random indices
void fill_random_block_indices(long **block_indices, long *block_counts, int seq_len, int block_size) {
  for (int i = 0; i < seq_len; i++) {
    // Random number of blocks for this query (between 1 and seq_len/block_size)
    int max_blocks = (seq_len + block_size - 1) / block_size;
    block_counts[i] = 1 + rand() % max_blocks;

    // Allocate memory for this row's block indices
    cudaMallocHost((void **)&block_indices[i], block_counts[i] * sizeof(long));

    // Fill with random block indices
    for (int j = 0; j < block_counts[i]; j++) {
      block_indices[i][j] = rand() % (seq_len / block_size);
    }
  }
}

int main() {
  // Set random seed
  srand(time(NULL));

  // Parameters
  int batch_size = 1;
  int seq_len = 1024;
  int num_heads = NUM_HEADS;
  int head_dim = HEAD_DIM;
  int block_size = BLOCK_SIZE;
  float scale_factor = 1.0f / sqrtf(head_dim);

  // Calculate sizes
  size_t query_size = batch_size * seq_len * num_heads * head_dim * sizeof(float);
  size_t key_size = batch_size * seq_len * 1 * head_dim * sizeof(float);   // MQA: 1 head for K
  size_t value_size = batch_size * seq_len * 1 * head_dim * sizeof(float); // MQA: 1 head for V
  size_t output_size = batch_size * seq_len * num_heads * head_dim * sizeof(float);

  // Allocate pinned memory for inputs and output
  float *query, *key, *value, *output;
  cudaError_t status;

  status = cudaHostAlloc((void **)&query, query_size, cudaHostAllocDefault);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaHostAlloc failed for query: %s\n", cudaGetErrorString(status));
    return 1;
  }

  status = cudaHostAlloc((void **)&key, key_size, cudaHostAllocDefault);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaHostAlloc failed for key: %s\n", cudaGetErrorString(status));
    return 1;
  }

  status = cudaHostAlloc((void **)&value, value_size, cudaHostAllocDefault);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaHostAlloc failed for value: %s\n", cudaGetErrorString(status));
    return 1;
  }

  status = cudaHostAlloc((void **)&output, output_size, cudaHostAllocDefault);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaHostAlloc failed for output: %s\n", cudaGetErrorString(status));
    return 1;
  }

  // Allocate block indices and counts
  long **block_indices;
  long *block_counts;

  status = cudaHostAlloc((void **)&block_indices, seq_len * sizeof(long *), cudaHostAllocDefault);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaHostAlloc failed for block_indices: %s\n", cudaGetErrorString(status));
    return 1;
  }

  status = cudaHostAlloc((void **)&block_counts, seq_len * sizeof(long), cudaHostAllocDefault);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaHostAlloc failed for block_counts: %s\n", cudaGetErrorString(status));
    return 1;
  }

  // Fill arrays with random float data
  fill_random_float(query, batch_size * seq_len * num_heads * head_dim);
  fill_random_float(key, batch_size * seq_len * 1 * head_dim);
  fill_random_float(value, batch_size * seq_len * 1 * head_dim);

  // Fill output array with zeros
  memset(output, 0, output_size);

  // Fill block indices
  fill_random_block_indices(block_indices, block_counts, seq_len, block_size);

  // Call the native sparse attention function
  printf("Calling native_sparse_attention with sequence length %d...\n", seq_len);
  native_sparse_attention(query, key, value, output, batch_size, seq_len, num_heads, head_dim, block_indices,
                          block_counts, block_size, scale_factor);

  // Check for CUDA errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
  } else {
    printf("Native sparse attention completed successfully.\n");
  }

  // Free allocated memory
  cudaFreeHost(query);
  cudaFreeHost(key);
  cudaFreeHost(value);
  cudaFreeHost(output);

  // Free block indices
  for (int i = 0; i < seq_len; i++) {
    cudaFreeHost(block_indices[i]);
  }
  cudaFreeHost(block_indices);
  cudaFreeHost(block_counts);

  return 0;
}
