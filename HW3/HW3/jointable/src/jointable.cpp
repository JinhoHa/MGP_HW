#include "jointable.h"

void jointable_ref(const int* const tableA, const int* const tableB,
                   std::vector<int>* const solution, const int R, const int S) {
  // You can assume matrixC is initialized with zero
  for (int i = 0; i < R; i++)
    for (int j = 0; j < S; j++)
      if (tableA[i] == tableB[j]) {
        solution->push_back(tableA[i]);
        break;
      }
}
void jointable_optimized(const int* const tableA, const int* const tableB,
                         std::vector<int>* const solution, const int R,
                         const int S) {
  // TODO: Implement your code
}