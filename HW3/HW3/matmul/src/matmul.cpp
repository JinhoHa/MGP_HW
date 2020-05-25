#include "matmul.h"
#include <omp.h>
using namespace std;

void matmul_ref(const int* const matrixA, const int* const matrixB,
                int* const matrixC, const int n) {
  // You can assume matrixC is initialized with zero
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        matrixC[i * n + j] += matrixA[i * n + k] * matrixB[k * n + j];
}

// void printMatrix(const int* const M, const int n) {
//
// 	cout << "printing Matrix..." << endl;
//
// 	for (int i = 0; i < n; i++) {
// 		for (int j = 0; j < n; j++) {
// 			printf("%3d", M[i * n + j]);
// 		}
// 		cout << endl;
// 	}
// 	cout << endl;
// }

const int THRESHOLD = 128;

int* const add(const int* const A, const int* const B, const int n) {
  int* const C = (int*)malloc(sizeof(int) * n * n);
  for(int i=0; i<n*n; i++) {
    C[i] = A[i] + B[i];
  }
  return C;
}

int* const sub(const int* const A, const int* const B, const int n) {
  int* const C = (int*)malloc(sizeof(int) * n * n);
  for(int i=0; i<n*n; i++) {
    C[i] = A[i] - B[i];
  }
  return C;
}

void strassen(const int* const A, const int* const B, int* const C, const int n) {
  if(n <= THRESHOLD) {
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

  ////////////////////////////////////////////////////////////
  // M1 := (A11 + A22)(B11 + B22)
  // M2 := (A21 + A22)(B11)
  // M3 := (A11)(B12 - B22)
  // M4 := (A22)(B21 - B11)
  // M5 := (A11 + A12)(B22)
  // M6 := (A21 - A11)(B11 + B12)
  // M7 := (A12 - A22)(B21 + B22)
  ///////////////////////////////////////////////////////////

  int* const T1 = add(A11, A22, nn);
  int* const T2 = add(B11, B22, nn);
  int* const T3 = add(A21, A22, nn);
  int* const T4 = sub(B12, B22, nn);
  int* const T5 = sub(B21, B11, nn);
  int* const T6 = add(A11, A12, nn);
  int* const T7 = sub(A21, A11, nn);
  int* const T8 = add(B11, B12, nn);
  int* const T9 = sub(A12, A22, nn);
  int* const T10 = add(B21, B22, nn);

  #pragma omp parallel sections
  {
    #pragma omp section
    {
      strassen(T1, T2, M1, nn);
    }
    #pragma omp section
    {
      strassen(T3, B11, M2, nn);
    }
    #pragma omp section
    {
      strassen(A11, T4, M3, nn);
    }
    #pragma omp section
    {
      strassen(A22, T5, M4, nn);
    }
    #pragma omp section
    {
      strassen(T6, B22, M5, nn);
    }
    #pragma omp section
    {
      strassen(T7, T8, M6, nn);
    }
    #pragma omp section
    {
      strassen(T9, T10, M7, nn);
    }
  }

  ////////////////////////////////////////////////////
  // C11 := M1 + M4 - M5 + M7
  // C12 := M3 + M5
  // C21 := M2 + M4
  // C22 := M1 - M2 + M3 + M6
  ////////////////////////////////////////////////////

  for(int i=0; i<nn*nn; i++) {
    C11[i] = M1[i] + M4[i] - M5[i] + M7[i];
    C12[i] = M3[i] + M5[i];
    C21[i] = M2[i] + M4[i];
    C22[i] = M1[i] - M2[i] + M3[i] + M6[i];
  }

  // merge C_ij to C
  for(int i=0; i<nn; i++) {
    for(int j=0; j<nn; j++) {
      C[i*n + j] = C11[i*nn + j];
    }
  }
  for(int i=0; i<nn; i++) {
    for(int j=0; j<nn; j++) {
      C[i*n + j+nn] = C12[i*nn + j];
    }
  }
  for(int i=0; i<nn; i++) {
    for(int j=0; j<nn; j++) {
      C[(i+nn)*n + j] = C21[i*nn + j];
    }
  }
  for(int i=0; i<nn; i++) {
    for(int j=0; j<nn; j++) {
      C[(i+nn)*n + j+nn] = C22[i*nn + j];
    }
  }

  free(A11); free(A12); free(A21); free(A22);
  free(B11); free(B12); free(B21); free(B22);
  free(C11); free(C12); free(C21); free(C22);
  free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7);
  free(T1); free(T2); free(T3); free(T4); free(T5);
  free(T6); free(T7); free(T8); free(T9); free(T10);

  return;
}

void matmul_optimized(const int* const matrixA, const int* const matrixB,
                      int* const matrixC, const int n) {
  // TODO: Implement your code
  if(n == 2048 || n == 4096) {
    strassen(matrixA, matrixB, matrixC, n);
    return;
  }

  // If n is not the shape of 2's exponential, make padding
  // The size of new matrices is nn
  int nn = n;
  int mod = 1;
  while (nn > THRESHOLD) {
    if (nn & 1) nn++;
    nn >>= 1;
    mod <<= 1;
  }
  nn *= mod;

  int* const A = (int*)calloc(nn * nn, sizeof(int));
  int* const B = (int*)calloc(nn * nn, sizeof(int));
  int* const C = (int*)calloc(nn * nn, sizeof(int));

  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++) {
      A[i*nn + j] = matrixA[i*n + j];
      B[i*nn + j] = matrixB[i*n + j];
      C[i*nn + j] = matrixC[i*n + j];
    }
  }

  strassen(A, B, C, nn);
  for(int i=0; i<n; i++) {
    for(int j=0; j<n; j++) {
      matrixC[i*n + j] = C[i*nn + j];
    }
  }
  free(A); free(B); free(C);

}
