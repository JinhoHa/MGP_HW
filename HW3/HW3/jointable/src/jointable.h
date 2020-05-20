#ifndef JOINTABLE_H
#define JOINTABLE_H

#include <omp.h>
#include <algorithm>
#include <unordered_map>
#include <vector>

void jointable_ref(const int* const tableA, const int* const tableB,
                   std::vector<int>* const solution, const int R, const int S);
void jointable_optimized(const int* const tableA, const int* const tableB,
                         std::vector<int>* const solution, const int R,
                         const int S);

#endif