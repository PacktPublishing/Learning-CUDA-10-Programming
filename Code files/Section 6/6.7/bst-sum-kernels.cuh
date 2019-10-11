// Header for bst-sum example, using in video 6.7.

struct Tree {
  int value;
  Tree *left;
  Tree *right;
};

__global__ void build_tree(const int *source, unsigned int length, Tree *root);
__global__ void destroy_tree(Tree *root);
__global__ void sum_tree(const Tree *root, int *result);
