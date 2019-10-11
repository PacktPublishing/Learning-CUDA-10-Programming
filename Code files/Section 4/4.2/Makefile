CUDAFLAGS ?= -g

ALL = reduce

all: $(ALL)

../utils.o: ../utils.cu ../utils.h
	nvcc -std=c++11 $(CUDAFLAGS) -o $@ -c $<

%: %.cu ../utils.o
	nvcc -std=c++11 $(CUDAFLAGS) -o $@ $^

# Dynamic parallelism requires separate compilation of kernels and
# host code
bst-sum-kernels.o: bst-sum-kernels.cu bst-sum-kernels.cuh
	nvcc -std=c++11 $(CUDAFLAGS) -arch compute_35 -dc bst-sum-kernels.cu

bst-sum.o: ../utils.h bst-sum.cu
	nvcc -std=c++11 $(CUDAFLAGS) -arch compute_35 -c bst-sum.cu

bst-sum: bst-sum.o bst-sum-kernels.o
	nvcc -std=c++11 $(CUDAFLAGS) -arch compute_35 -o bst-sum bst-sum.o bst-sum-kernels.o

clean:
	rm -f ../utils.o bst-sum.o bst-sum-kernels.o manylights1 $(ALL)
