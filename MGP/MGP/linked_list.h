#ifndef _LINKED_LIST_H_
#define _LINKED_LIST_H_
#include <iostream>
#include <mutex>
#include <thread>
// basic node structure
class linked_list {
 public:
  linked_list() {}
  virtual bool contains(int key) = 0;
  virtual bool insert(int key) = 0;
  virtual bool remove(int key) = 0;
};
#endif
