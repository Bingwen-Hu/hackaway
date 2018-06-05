#ifndef THREAD_TREE_H
#define THREAD_TREE_H

typedef struct threadTree {
    int data;
    struct threadTree *left;
    struct threadTree *right;
    int threaded;  // flag variable to decide whether it has a thread.
} threadTree;

// here we will implement a right in-thread tree.
threadTree *insert_threadtree(threadTree *tree, int value);
threadTree *delete_threadtree(threadTree *tree, int value);
threadTree *destroy_threadtree(threadTree *tree);
void in_order_traversal_threadtree(threadTree *tree);

#endif // THREAD_TREE_H