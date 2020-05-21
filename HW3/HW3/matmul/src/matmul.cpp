#include "matmul.h"

void matmul_ref(const int* const matrixA, const int* const matrixB,
                int* const matrixC, const int n) {
  // You can assume matrixC is initialized with zero
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        matrixC[i * n + j] += matrixA[i * n + k] * matrixB[k * n + j];
}

void printMatrix(const int* const M, const int n) {

	cout << "printing Matrix..." << endl;

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("%3d", M[i * n + j]);
		}
		cout << endl;
	}
	cout << endl;
}

int* const add(const int* const A, const int* const B, const int n) {
  int* const C = (int*)malloc(sizeof(int) * n * n);
  for(int i=0; i<n; i++) {
    int in = i * n;
    for(int j=0; j<n; j++) {
      C[in + j] = A[in + j] + B[in + j];
    }
  }
  return C;
}

void strassen(const int* const A, const int* const B, int* const C, const int n) {
  if(n <= 64) {
    for(int i=0; i<n; i++) {
      for(int k=0; k<n; k++) {
        int tmp = A[i*n + k];
        for(int j=0; j<n; j++) {
          C[i*n + j] += tmp * B[k*n + j];
        }
      }
    }
    return;
  }
  const int nn = n / 2;

  // allocating sub-matrices
  int* const A11 = (int*)malloc(sizeof(int) * nn * nn);
  int* const A12 = (int*)malloc(sizeof(int) * nn * nn);
  int* const A21 = (int*)malloc(sizeof(int) * nn * nn);
  int* const A22 = (int*)malloc(sizeof(int) * nn * nn);

  int* const B11 = (int*)malloc(sizeof(int) * nn * nn);
  int* const B12 = (int*)malloc(sizeof(int) * nn * nn);
  int* const B21 = (int*)malloc(sizeof(int) * nn * nn);
  int* const B22 = (int*)malloc(sizeof(int) * nn * nn);

  int* const C11 = (int*)malloc(sizeof(int) * nn * nn);
  int* const C12 = (int*)malloc(sizeof(int) * nn * nn);
  int* const C21 = (int*)malloc(sizeof(int) * nn * nn);
  int* const C22 = (int*)malloc(sizeof(int) * nn * nn);

  int* const M1 = (int*)calloc(nn * nn, sizeof(int));
  int* const M2 = (int*)calloc(nn * nn, sizeof(int));
  int* const M3 = (int*)calloc(nn * nn, sizeof(int));
  int* const M4 = (int*)calloc(nn * nn, sizeof(int));
  int* const M5 = (int*)calloc(nn * nn, sizeof(int));
  int* const M6 = (int*)calloc(nn * nn, sizeof(int));
  int* const M7 = (int*)calloc(nn * nn, sizeof(int));

  for(int i=0; i<n; i++) {
    if(i<nn) {
      for(int j=0; j<n; j++) {
        if(j<nn) {
          A11[i*nn + j] = A[i*n + j];
          B11[i*nn + j] = B[i*n + j];
        }
        else {
          A12[i*nn + j-nn] = A[i*n + j];
          B12[i*nn + j-nn] = B[i*n + j];
        }
      }
    }
    else {
      for(int j=0; j<n; j++) {
        if(j<nn) {
          A21[(i-nn)*nn + j] = A[i*n + j];
          B21[(i-nn)*nn + j] = B[i*n + j];
        }
        else {
          A22[(i-nn)*nn + j-nn] = A[i*n + j];
          B22[(i-nn)*nn + j-nn] = B[i*n + j];
        }
      }
    }
  }

}

void matmul_optimized(const int* const matrixA, const int* const matrixB,
                      int* const matrixC, const int n) {
  // TODO: Implement your code

}
