// Kernel code for bst-sum example, used in video 6.7.

#include <stdio.h>

#include "bst-sum-kernels.cuh"

#define kernelCheckError(code)                                  \
  {                                                             \
    if ((code) != cudaSuccess) {                                \
      printf("Cuda failure %s:%d: '%s' \n", __FILE__, __LINE__, \
             cudaGetErrorString(code));                         \
      return;                                                   \
    }                                                           \
  }

__device__ void build_subtree(Tree *root, const int *source, int left,
                              int right)
{
  int middle = (left + right) / 2;
  root->value = source[middle];

  if (left >= right) {
    return;
  }

  if (middle == left) {
    root->left = nullptr;
  } else {
    root->left = new Tree();
    build_subtree(root->left, source, left, middle - 1);
  }

  if (middle == right) {
    root->right = nullptr;
  } else {
    root->right = new Tree();
    build_subtree(root->right, source, middle + 1, right);
  }
}

__global__ void build_tree(const int *source, unsigned int length, Tree *root)
{
  build_subtree(root, source, 0, length - 1);
}

__device__ void destroy_subtree(Tree *root)
{
  if (root->left) {
    destroy_subtree(root->left);
  }
  if (root->right) {
    destroy_subtree(root->right);
  }
  delete root;
}

__global__ void destroy_tree(Tree *root)
{
  if (root->left) {
    destroy_subtree(root->left);
  }
  if (root->right) {
    destroy_subtree(root->right);
  }
  // Do not destroy root! It was allocated with cudaMalloc and must be freed
  // from host code.
}

__global__ void sum_tree(const Tree *root, int *result)
{
  // Allocate temporary global memory for storing subtree results
  int *left_sum = new int;
  int *right_sum = new int;

  // Create independent streams to sum each subtree
  cudaStream_t left_stream, right_stream;
  kernelCheckError(
      cudaStreamCreateWithFlags(&left_stream, cudaStreamNonBlocking));
  kernelCheckError(
      cudaStreamCreateWithFlags(&right_stream, cudaStreamNonBlocking));

  if (root->left) {
    sum_tree<<<1, 1, 0, left_stream>>>(root->left, left_sum);
  } else {
    *left_sum = 0;
  }

  if (root->right) {
    sum_tree<<<1, 1, 0, right_stream>>>(root->right, right_sum);
  } else {
    *right_sum = 0;
  }

  // Wait for both streams to finish
  kernelCheckError(cudaDeviceSynchronize());

  *result = root->value + *left_sum + *right_sum;

  kernelCheckError(cudaStreamDestroy(left_stream));
  kernelCheckError(cudaStreamDestroy(right_stream));

  delete left_sum;
  delete right_sum;
}
