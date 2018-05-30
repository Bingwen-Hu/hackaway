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

bstree *delete_bstree(bstree *tree, int *value);
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