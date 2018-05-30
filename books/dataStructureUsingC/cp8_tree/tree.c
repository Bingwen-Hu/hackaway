#include "tree.h"
#include <stdlib.h>
#include <stdio.h>


void insert_bstree(bstree **tree, int value) {
    if (*tree == NULL) {
        *tree = malloc(sizeof(bstree));
        (*tree)->left = NULL;
        (*tree)->right = NULL;
        (*tree)->data = value;
    }

    if (value < (*tree)->data) {
        // go to left
        insert_bstree(&(*tree)->left, value);
    } else  if (value > (*tree)->data) {
        insert_bstree(&(*tree)->right, value);
    } else {
        printf("value %d already exists!\n", value);
    }
}

bstree *delete_bstree(bstree *tree, int *value);
bstree mirror_image_bstree(bstree *tree);

int height_bstree(bstree *tree);
int internal_nodes_bstree(bstree *tree);
int external_nodes_bstree(bstree *tree);
int search_bstree(bstree *tree, int value);
int find_smallest_bstree(bstree *tree);
int find_largest_bstree(bstree *tree);

void display_bstree(bstree *tree) {
    if (tree != NULL) {
        printf("%-3d", tree->data);
        display_bstree(tree->left);
        display_bstree(tree->right);
    }
}
int destroy_bstree(bstree *tree);