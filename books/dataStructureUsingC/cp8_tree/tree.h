/** some basic
 * 
 * ------------- basic concept ---------------
 * Root: topmost node in a tree.
 * Subtree: treenodes under one tree node T is considered as T's subtrees
 * Leaf: A node has no children called leaf node.
 * Ancestor: parent's parent's ...
 * Descendant: children's children's ...
 * Level number: children is Parent + 1
 * Degree: Degree of a node is equal to the number of children that a node has.
 * * -----------------------------------------
 * 
 * ------------- six tree types --------------
 * General trees
 * Forests
 * Binary trees
 * Binary search trees
 * Expression trees
 * Tournament trees
 * -------------------------------------------
 */



// binary search tree
// all left node is less than root node
// all right node is larger than root node.

typedef struct node {
    int data;
    struct node *left;
    struct node *right;
} bstree;

bstree *insert_bstree(bstree *tree, int value);
bstree *delete_bstree(bstree *tree, int value);
bstree mirror_image_bstree(bstree *tree);

int height_bstree(bstree *tree);
int internal_nodes_bstree(bstree *tree);
int external_nodes_bstree(bstree *tree);
int search_bstree(bstree *tree, int value);
int find_smallest_bstree(bstree *tree);
int find_largest_bstree(bstree *tree);

void pre_order_traversal(bstree *tree);
void in_order_traversal(bstree *tree);
void post_order_traversal(bstree *tree);
int destroy_bstree(bstree *tree);