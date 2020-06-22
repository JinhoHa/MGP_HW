#include <stdio.h>
#include <stdlib.h>

__global__ void print_from_gpu(void) {
  printf(
      "Hello World from device!\n\
    threadIdx.x: %d\n\
    threadIdx.y: %d\n\
    blockIdx.x: %d\n\
    blockIdx.y: %d\n\
    blockDim.x: %d\n\
    blockDim.y: %d\n",
      threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y);
}

int main(void) {
  printf("Hello World from host!\n");
  dim3 DimGrid(2, 2);
  dim3 DimBlock(3, 2);
  print_from_gpu<<<DimGrid, DimBlock>>>();
  cudaDeviceSynchronize();
  printf("Can this be ahead of kernel?\n");

  printf("Dim change\n");
  DimGrid.x = 2; DimGrid.y = 1;
  DimBlock.x = 2; DimBlock.y = 3;
  print_from_gpu<<<DimGrid, DimBlock>>>();
  cudaDeviceSynchronize();
  return 0;
}
