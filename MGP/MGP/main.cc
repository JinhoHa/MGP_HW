#include "better_locked_linked_list.h"
#include "locked_linked_list.h"

#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <vector>
#include "assert.h"

//PRNG: Pseudo Random Number Generator
//rand() is not thread-safe. So we use a simple near-random number instead
int PRNG(int seed1, int seed2) {
  int key = seed2;
  key ^= key << 16;
  key ^= key >> 5;
  key ^= key << 1;
  key ^= seed1;

  return key;
}


void init_worker(linked_list* LL, int tid, int num_threads, int num_items) {
  for (int i = 0; i < num_items; i += num_threads) {
    int key = i + tid;
    LL->insert(key);
  }
}

void test_worker(linked_list* LL, int tid, int num_threads, int num_ops, int num_items,
                 int additional_reads_per_op) {
  int seed = tid * tid + 3 * tid + 7;
  for (int i = 0; i < num_ops; i++) {
    int key = PRNG(seed, i) % num_items;
    LL->remove(key);
    // in a single thread-version, the below remove_chk has to be true
    //bool remove_chk = LL->contains(key);
    LL->insert(key);
    // in a single thread-version, the below insert_chk has to be true
    //bool insert_chk = LL->contains(key);
    for (int j = 0; j < additional_reads_per_op; j++) {
      int key = PRNG(j, seed) % num_items;
      LL->contains(key);
    }
  }
}

int main(int argc, char** argv) {
  ////////////////////////////////////////////////////////////////////////
  // parameters, arguments
  ////////////////////////////////////////////////////////////////////////
  if (argc != 6) {
    std::cout << "Usage: ./a.out N_items N_threads N_test_ops additional_reads_per_op "
                 "use_hand_over_hand_locking"
              << std::endl;
  }
  int num_items = atoi(argv[1]);
  int num_threads = atoi(argv[2]);
  int num_tests = atoi(argv[3]);
  int additional_reads_per_op = atoi(argv[4]);
  int use_hand_over_hand = atoi(argv[5]);
  ////////////////////////////////////////////////////////////////////////
  // initilaize linked list
  ////////////////////////////////////////////////////////////////////////
  linked_list* LL;
  if (use_hand_over_hand) {
    LL = new better_locked_linked_list();
  } else {
    LL = new locked_linked_list();
  }
  std::cout << "start filling" << std::endl;
  std::chrono::duration<double> diff;
  auto fill_start = std::chrono::steady_clock::now();
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; t++) {
    threads.push_back(std::thread(init_worker, LL, t, num_threads, num_items));
  }
  for (auto& t : threads) {
    t.join();
  }
  auto fill_end = std::chrono::steady_clock::now();
  diff = fill_end - fill_start;
  std::cout << "init list took " << diff.count() << " sec" << std::endl;
  ////////////////////////////////////////////////////////////////////////
  // test some queries
  ////////////////////////////////////////////////////////////////////////
  threads.clear();
  std::cout << "start test" << std::endl;
  auto test_start = std::chrono::steady_clock::now();
  for (int t = 0; t < num_threads; t++) {
    threads.push_back(
        std::thread(test_worker, LL, t, num_threads, num_tests / num_threads, num_items, additional_reads_per_op));
  }
  for (auto& t : threads) {
    t.join();
  }
  auto test_end = std::chrono::steady_clock::now();
  diff = test_end - test_start;
  std::cout << "test " << 4 * num_tests << " ops took " << diff.count()
            << " sec" << std::endl;
  ////////////////////////////////////////////////////////////////////////
  // check consistency
  ////////////////////////////////////////////////////////////////////////
  int chk = 0;
  for (int i = 0; i < num_items; i++) {
    bool contains = LL->contains(i);
    if (contains) chk++;
  }
  std::cout << "Sanity check: there are " << chk << " items left" << std::endl;
  return 0;
}
