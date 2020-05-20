#include "matmul.h"

void matmul_ref(const int* const matrixA, const int* const matrixB,
                int* const matrixC, const int n) {
  // You can assume matrixC is initialized with zero
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        matrixC[i * n + j] += matrixA[i * n + k] * matrixB[k * n + j];
}

void matmul_optimized(const int* const matrixA, const int* const matrixB,
                      int* const matrixC, const int n) {
  // TODO: Implement your code
  for(int i=0; i<n; i++) {
    for(int k=0; k<n; k++) {
      int a_ik = matrixA[i*n + k];
      for(int j=0; j<n; j++) {
        matrixC[i*n + j] += a_ik * matrixB[k*n + j];
      }
    }
  }
}
