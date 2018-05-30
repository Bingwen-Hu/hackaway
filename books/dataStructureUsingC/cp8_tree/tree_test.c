#include "tree.h"
#include <stdio.h>

int main(int argc, char const *argv[])
{
    bstree *tree = NULL;
    int list[] = {10, 3, 4, 8, 2, 3, 48, 76, 43, 29};
    for (int i = 0; i < 10; i++) {
        insert_bstree(&tree, list[i]);
    }
    display_bstree(tree);
    return 0;
}
