/* some attributes

1. on i_th level, number of nodes <= 2^(i-1)
2. if bitree depth == k, number of nodes <= 2^k-1
3. for any bitree, if leaf node is n0, number of nodes whose degree is 2 is n2
   then, n0 == n2 + 1
3.1 -> for any bitree, the number of nodes is 
   n = n0 + n1 + n2   (n0 means degree of node is 0)
3.2 -> for any bitree, the number of lines is
   lines = n - 1 = n1 + 2n2,   so
   n0 + n1 + n2 - 1 = n1 + 2n2 => n0 = n2 + 1
4. the depth of a complete binary tree with n nodes is floor(log(2, n))+1
   to certify, using attributes of full binary tree
5. for a complete binary tree with n nodes
5.1 if i = 1, then i is root; if i > 1, then i's parent node is floor(i/2)
5.2 if 2i>n, then i has no lchild, else i's lchild is 2i
5.3 if 2i+1>n, then i has no rchild, else i's rchild is 2i+1

*/

#include <stdio.h>
#include <stdlib.h>


typedef struct BiTNode{
  int data;
  struct BiTNode *lchild, *rchild;
}BiTNode, *BiTree;


void PreOrderTraverse(BiTree T){
  if (T == NULL){
    return;
  } 
  printf("%d ", T->data);
  PreOrderTraverse(T->lchild);
  PreOrderTraverse(T->rchild);
}


void InOrderTraverse(BiTree T){
  if (T == NULL){
    return;
  }
  InOrderTraverse(T->lchild);
  printf("%-3d", T->data);
  InOrderTraverse(T->rchild);
}


void PostOrderTraverse(BiTree T){
  if (T == NULL){
    return;
  }
  PostOrderTraverse(T->lchild);
  PostOrderTraverse(T->rchild);
  printf("%-3d", T->data);
}


BiTree createBiTree(BiTree T){
  int e;
  BiTNode *node;
  
  printf("input a number, -1 to exit: ");
  scanf("%d", &e);
  if (e == -1){
    T = NULL;
  } else {
    T = (BiTNode *)malloc(sizeof(BiTNode));
    T->data = e;
    printf("create left tree: ");
    T->lchild = createBiTree(T->lchild);
    printf("create right tree: ");
    T->rchild = createBiTree(T->rchild);
  }
  return T;
}

void destroyBiTree(BiTree T){
  if (T != NULL){
    destroyBiTree(T->lchild);
    destroyBiTree(T->rchild);
    free(T);
  }
}


void main(){

  BiTree t;
  t = createBiTree(t);
  printf("PreOrder: ");
  PreOrderTraverse(t);
  puts("");
  printf("InOrder: ");
  InOrderTraverse(t);
  puts("");
  printf("PostOrder: ");
  PostOrderTraverse(t);
  puts("");
  
  destroyBiTree(t);
}
