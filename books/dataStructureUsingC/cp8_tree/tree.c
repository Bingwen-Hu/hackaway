#include "tree.h"
#include <stdlib.h>
#include <stdio.h>


bstree *insert_bstree(bstree *tree, int value) {
    if (tree == NULL) {
        tree = malloc(sizeof(bstree));
        tree->left = NULL;
        tree->right = NULL;
        tree->data = value;
        return tree;
    }

    if (tree->data == value) {
        return tree;
    }

    bstree *branch;
    if (value < tree->data) {
        branch = insert_bstree(tree->left, value);
        tree->left = branch;
    } else {
        branch = insert_bstree(tree->right, value);
        tree->right = branch;
    }
    return tree;
}

/* ----------- delete -----------------------
 * 1. delete a leaf node (node 2)
 * Just delete it!
 * 2. delete a node with one child (node 4)
 * The Child go up!
 * 3. delete a node with two children
 * like node 48, lift its right child
 * but node 31? two ways.
 * largest value in left subtree or smallest
 * value in right subtree.
 */ 
bstree *delete_bstree(bstree *tree, int value) {
    if (tree == NULL) {
        return tree;
    }
    if (value == tree->data) {
        if (tree->left == NULL && tree->right == NULL) {             // leaf
            free(tree);
            return NULL;
        } else if (tree->left == NULL) {         // right child
            bstree *child = tree->right;
            free(tree);
            return child;
        } else if (tree->right == NULL) {        // left child
            bstree *child = tree->left;
            free(tree);
            return child;
        } else {
            // two ways left subtree largest or right subtree smallest
            // adopt finding the left subtree largest value
            bstree *left_largest_parent = tree->left;
            bstree *left_largest = left_largest_parent->right;
            while (left_largest->right != NULL) {
                left_largest_parent = left_largest;
                left_largest = left_largest_parent->right;
            }
            tree->data = left_largest->data;
            free(left_largest);
            left_largest_parent->right = NULL;
            return tree;
        }
    } else if (value < tree->data) {
        bstree *left = delete_bstree(tree->left, value);
        tree->left = left;
    } else {
        bstree *right = delete_bstree(tree->right, value);
        tree->right = right;
    }
    return tree;
}
bstree mirror_image_bstree(bstree *tree);

int height_bstree(bstree *tree);
int internal_nodes_bstree(bstree *tree);
int external_nodes_bstree(bstree *tree);

int search_bstree(bstree *tree, int value) {
    if (tree == NULL) {
        return 0;  // buggy allowed
    }
    if (value == tree->data) {
        return 1;
    } else if (value < tree->data) {
        return search_bstree(tree->left, value);
    } else {
        return search_bstree(tree->right, value);
    }
}

int find_smallest_bstree(bstree *tree);
int find_largest_bstree(bstree *tree);

void pre_order_traversal(bstree *tree) {
    if (tree != NULL) {
        printf("%-3d", tree->data);
        pre_order_traversal(tree->left);
        pre_order_traversal(tree->right);
    }
}

void in_order_traversal(bstree *tree) {
    if (tree != NULL) {
        in_order_traversal(tree->left);
        printf("%-3d", tree->data);
        in_order_traversal(tree->right);
    }
}

void post_order_traversal(bstree *tree) {
    if (tree != NULL) {
        post_order_traversal(tree->left);
        post_order_traversal(tree->right);
        printf("%-3d", tree->data);
    }
}

int destroy_bstree(bstree *tree);