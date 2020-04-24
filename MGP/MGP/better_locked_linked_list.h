/*2017147594, Jinho Ha*/
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

public:
	better_locked_linked_list() {
		// TODO
		this->head = new node(0);
	}

	bool contains(int key) {
		// TODO
		node* prev = head;
		prev->node_mutex.lock();
		node* curr = prev->next;
		if (curr) { curr->node_mutex.lock(); }
		else {
			prev->node_mutex.unlock();
			return false;
		}

		// proceed with hand-by-hand locking
		while (curr->key < key) {
			// unlock the prev node first, and then lock the next node
			prev->node_mutex.unlock();
			prev = curr;
			curr = curr->next;
			if (curr) { curr->node_mutex.lock(); }
			else {
				prev->node_mutex.unlock();
				return false;
			}
		}
		// unlock the nodes before return
		if (curr != nullptr && curr->key == key) {
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

	}

	bool remove(int key) {
		return false;
		// TODO
	}
};
#endif
