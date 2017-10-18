/* Tree
degree of node: the number of subtree a node owns
degree of tree; the largest degree of nodes

leaf node: degree == 0

level: root node is the first level
   =>  if one node is i level, the subtree of the node is i+1 level

depth aka height: the number of levels


forest: more than one tree is a forest
*/


#define MAX_TREE_SIZE 100

typedef int Element;
typedef struct Node{
  Element data;
  int parent;
}Node;

typedef struct {
  Node nodes[MAX_TREE_SIZE];
  int root, number;
}Tree;


