// Build and print a binary search tree on the device, using dynamic global
// memory allocation.
// Example for video 6.6.

#include <thrust/device_vector.h>
#include <memory>

// Standard CUDA API functions
#include <cuda_runtime_api.h>

#include "../utils.h"

struct Tree {
  int value;
  Tree *left;
  Tree *right;
};

// Helper function to construct a binary search tree from a sorted array.
__device__ void build_subtree(Tree *root, const int *source, int left,
                              int right)
{
  int middle = (left + right) / 2;
  root->value = source[middle];

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

// Construct a binary search tree from a sorted array. This kernel should be
// run with a single thread.
__global__ void build_tree(const int *source, unsigned int length, Tree *root)
{
  build_subtree(root, source, 0, length - 1);
}

// Print the nodes of a tree, in order.
__device__ void print_subtree(const Tree *root)
{
  if (root->left) {
    print_subtree(root->left);
  }
  printf("%d\n", root->value);
  if (root->right) {
    print_subtree(root->right);
  }
}
__global__ void print_tree(const Tree *root) { print_subtree(root); }

// Free a device-allocated tree.
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

int main(int argc, char **argv)
{
  const unsigned int COUNT = 128;

  // Create device vector with sequential integers
  thrust::device_vector<int> source(COUNT);
  thrust::sequence(source.begin(), source.end());

  // Allocate a root for the tree
  Tree *root;
  cudaCheckError(cudaMalloc(&root, sizeof(Tree)));

  // Build the tree from a sorted array
  build_tree<<<1, 1>>>(thrust::raw_pointer_cast(&source[0]), source.size(),
                       root);

  // Print the tree values, in order
  print_tree<<<1, 1>>>(root);

  // Destroy all the subtrees which were allocated with new in device
  // code.
  destroy_tree<<<1, 1>>>(root);

  // Destroy the root which was allocated with cudaMalloc.
  cudaCheckError(cudaFree(root));

  cudaCheckError(cudaDeviceSynchronize());
}
