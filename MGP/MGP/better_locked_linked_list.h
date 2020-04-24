#ifndef _BETTER_LOCKED_LINKED_LIST_H_
#define _BETTER_LOCKED_LINKED_LIST_H_

#include <iostream>
#include <mutex>
#include <thread>
#include "linked_list.h"

// basic node structure
class better_locked_linked_list : public linked_list {
  class node {
    // TODO
	public:
		node(int key) : key(key) {
			next = nullptr;
		}
		int key;
		node* next;
		// every node has mutex, so that every node can be locked/unlocked respectively
		std::mutex node_mutex;
  };

  node* head;
	node* tail; // To figure out if a node is the end of the list, I will use tail instead of nullptr

 public:
  better_locked_linked_list() {
    // TODO
		this->head = new node(0);
		this->tail = new node(-1);
		this->head->next = this->tail;
  }

  bool contains(int key) {
    // TODO
		node* prev = head;
		prev->node_mutex.lock();
		node* curr = prev->next;
		curr->node_mutex.lock();

		// proceed with hand-by-hand locking
		while (curr != tail && curr->key < key) {
			// unlock the prev node first, and then lock the next node
			prev->node_mutex.unlock();
			prev = curr;
			curr = curr->next;
			curr->node_mutex.lock();
		}
		// unlock the nodes before return
		if (curr != tail && curr->key == key) {
			prev->node_mutex.unlock();
			curr->node_mutex.unlock();
			return true;
		}
		else {
			prev->node_mutex.unlock();
			curr->node_mutex.unlock();
			return false;
		}
  }

  bool insert(int key) {
    // TODO
		node* tmp = new node(key);
		node* prev = head;
		prev->node_mutex.lock();
		node* curr = prev->next;
		curr->node_mutex.lock();
		
		while (curr != tail && curr->key < key) {
			prev->node_mutex.unlock();
			prev = curr;
			curr = curr->next;
			curr->node_mutex.lock();
		}
		if (curr != tail && curr->key == key) {
			prev->node_mutex.unlock();
			curr->node_mutex.unlock();
			return false;
		}
		else {
			tmp->next = curr;
			prev->next = tmp;
			prev->node_mutex.unlock();
			curr->node_mutex.unlock();
			return true;
		}
  }

  bool remove(int key) {
    // TODO
		node* prev = head;
		prev->node_mutex.lock();
		node* curr = prev->next;
		curr->node_mutex.lock();

		while (curr != tail && curr->key < key) {
			prev->node_mutex.unlock();
			prev = curr;
			curr = curr->next;
			curr->node_mutex.lock();
		}

		if (curr != tail && curr->key == key) {
			prev->next = curr->next;
			delete curr;
			prev->node_mutex.unlock();
			return true;
		}
		else {
			prev->node_mutex.unlock();
			curr->node_mutex.unlock();
			return false;
		}
  }
};
#endif
