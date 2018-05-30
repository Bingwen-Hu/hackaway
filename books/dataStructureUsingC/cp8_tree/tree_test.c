#include "tree.h"
#include <stdio.h>

int main(int argc, char const *argv[])
{
    bstree *tree = NULL;
    int list[] = {10, 3, 4, 8, 2, 31, 48, 76, 43, 29};
    for (int i = 0; i < 10; i++) {
        tree = insert_bstree(tree, list[i]);
    }
    printf("3 is in the tree? %s\n", search_bstree(tree, 3) ? "Yes" : "No");
    printf("42 is in the tree? %s\n", search_bstree(tree, 42) ? "Yes" : "No");
    display_bstree(tree);
    return 0;
}
