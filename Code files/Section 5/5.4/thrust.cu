// Demonstration of basic thrust functionality.
// Example for video 5.4.

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <iostream>

int main(void)
{
  // Allocate two device_vectors with 10 elements
  thrust::device_vector<int> vec1(10);
  thrust::device_vector<int> vec2(10);

  // Initialize vec1 to 0,1,2,3, ....
  thrust::sequence(vec1.begin(), vec1.end());

  // vec2 = -vec1
  thrust::transform(vec1.begin(), vec1.end(), vec2.begin(),
                    thrust::negate<int>());

  // print vec2
  thrust::copy(vec2.begin(), vec2.end(),
               std::ostream_iterator<int>(std::cout, "\n"));

  return 0;
}
