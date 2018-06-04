#include "tree.h"
#include <stdio.h>

/** --------- structure ---------------------
 *                  10
 *                /    \
 *               3      31
 *             /  \     / \
 *            2    4  29   48
 *                  \     /  \
 *                   8   43   76
 * ---------- traversal ---------------------
 *  pre-order: 10 3 2 4 8 31 29 48 43 76
 *   in-order: 2 3 4 8 10 29 31 43 48 76
 * post-order: 2 8 4 3 29 43 76 48 31 10
 * 
 * 
 */


int main(int argc, char const *argv[])
{
    bstree *tree = NULL;
    int list[] = {10, 3, 4, 8, 2, 31, 48, 76, 43, 29};
    for (int i = 0; i < 10; i++) {
        tree = insert_bstree(tree, list[i]);
    }

    puts("------------ test search -----------------");
    printf("3 is in the tree? %s\n", search_bstree(tree, 3) ? "Yes" : "No");
    printf("42 is in the tree? %s\n", search_bstree(tree, 42) ? "Yes" : "No");
    pre_order_traversal(tree);
    puts("");
    in_order_traversal(tree);
    puts("");
    post_order_traversal(tree);
    puts("");
    
    puts("------------ test delete -----------------");
    tree = delete_bstree(tree, 10);
    pre_order_traversal(tree);
    puts("");

    puts("--------test largest and smallest---------");
    printf("largest function ok? %s\n", find_largest_bstree(tree) == 76 ? "Yes" : "No");
    printf("smallest function ok? %s\n", find_smallest_bstree(tree) == 2 ? "Yes" : "No");


    puts("--------mirror a bstree---------");
    tree = mirror_image_bstree(tree);
    pre_order_traversal(tree);
    puts("");
    tree = mirror_image_bstree(tree);
    pre_order_traversal(tree);
    puts("");
    destroy_bstree(tree);

    puts("-------height of a bstree-------");
    int height = height_bstree(tree);
    printf("height of bstree is %d\n", height);

    return 0;
}
