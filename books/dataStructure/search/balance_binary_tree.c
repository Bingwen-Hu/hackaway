#include <stdio.h>
#include <stdlib.h>


typedef struct BiTNode{
  int data;
  int bf;                       /* balance factor */
  struct BiTNode *lchild, *rchild;
}BiTNode, *BiTree;

void R_Rotate(BiTree *p){
  BiTree L;
  L = (*p)->lchild;
  (*p)->lchild = L->rchild;
  L->rchild = (*p);
  *p = L;
}

void L_Rotate(BiTree *p){
  BiTree R;
  R = (*p)->rchild;
  (*p)->rchild = R->lchild;
  R->lchild = (*p);
  *p = R;
}


#define LH 1
#define EH 0
#define RH -1

void LeftBalance(BiTree *T){
  BiTree L, lr;
  L = (*T)->lchild;
  switch(L->bf){
    case LH:
      (*T)->bf = L->bf = EH;
      R_Rotate(T);
      break;
    case RH:
      lr = L->rchild;
      switch(lr->bf){
        case LH: 
          (*T)->bf = RH;
          L->bf = EH;
          break;
        case EH: 
          (*T)->bf = L->bf = EH;
          break;
        case RH: 
          (*T)->bf = EH;
          L->bf = LH;
          break;
      }
      lr->bf = EH;
      L_Rotate(&(*T)->lchild);
      R_Rotate(T);
  }
}


/* Test note:
   still incomplete
 */
