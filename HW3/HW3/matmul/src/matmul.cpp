#include "matmul.h"
#include <thread>
#include <vector>
using namespace std;

const int NUM_THREAD = 16;

void matmul_ref(const int* const matrixA, const int* const matrixB,
                int* const matrixC, const int n) {
  // You can assume matrixC is initialized with zero
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        matrixC[i * n + j] += matrixA[i * n + k] * matrixB[k * n + j];
}

void rowwise(int tid, const int* const A, const int* const B, int* const C, const int n) {
	for (int i = tid; i < n; i += NUM_THREAD) {
		for (int k = 0; k < n; k++) {
			int a_ik = A[i*n + k];
			for (int j = 0; j < n; j++) {
				// C[i][j] += A[i][k] * B[k][j]
				C[i*n + j] += a_ik * B[k*n + j];
			}
		}
	}
}

void matmul_optimized(const int* const matrixA, const int* const matrixB,
                      int* const matrixC, const int n) {
  // TODO: Implement your code

	vector<thread> threads;
	for (int i = 0; i < NUM_THREAD; i++) {
		threads.push_back(thread(rowwise, i, matrixA, matrixB, matrixC, n));
	}

	for (thread& th : threads) {
		th.join();
	}

}
