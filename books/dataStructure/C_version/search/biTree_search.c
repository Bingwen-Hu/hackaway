#include <stdio.h>
#include <stdlib.h>

typedef struct BiTNode{
  int data;
  struct BiTNode *lchild, *rchild;
}BiTNode, *BiTree;


typedef enum {FALSE, TRUE} Status;


Status SearchBST(BiTree T, int key, BiTree parent, BiTree *p){
  if (!T){
    *p = parent;                /* why not NULL */
    return FALSE;
  } else if (key == T->data){
    *p = T;
    return TRUE;
  } else if (key < T->data){
    return SearchBST(T->lchild, key, T, p);
  } else {
    return SearchBST(T->rchild, key, T, p);
  }
}

Status InsertBST(BiTree *T, int key){
  BiTree p, s;
  if (!SearchBST(*T, key, NULL, &p)){
    s = (BiTree)malloc(sizeof(BiTNode));
    s->data = key;
    s->lchild = s->rchild = NULL;
    
    if (!p){
      *T = s;
    } else if (key < p->data){
      p->lchild = s;
    } else {
      p->rchild = s;
    }
    return TRUE;
  } 

  return FALSE;
}


Status ConstructBST(BiTree *T, int array[], int num){
  for (int i=0; i<num; i++){
    InsertBST(T, array[i]);
  }
  return TRUE;
}


void PreOrder(BiTree T){
  if (T == NULL){
    return;
  }
  
  printf("%-3d", T->data);
  PreOrder(T->lchild);
  PreOrder(T->rchild);
}

void InOrder(BiTree T){
  if (T == NULL){
    return;
  }
  InOrder(T->lchild);
  printf("%-3d", T->data);
  InOrder(T->rchild);
}


void destroyBiTree(BiTree T){
  if (T != NULL){
    destroyBiTree(T->lchild);
    destroyBiTree(T->rchild);
    free(T);
  }
}

Status Delete(BiTree *p){
  BiTree q, s;
  if ((*p)->rchild == NULL){
    q = *p;
    *p = (*p)->lchild;
    free(q);
  } else if ((*p)->lchild == NULL){
    q = *p;
    *p = (*p)->rchild;
    free(q);
  } else {
    q = *p; 
    s = (*p)->lchild;
    while (s->rchild){
      q = s;
      s = s->rchild;            /* find its forword */
    }
    (*p)->data = s->data;
    if (q!=*p){
      q->rchild = s->lchild;
    } else {
      q->lchild = s->lchild;
    }
    free(s);
  }
  return TRUE;
}

Status DeleteBST(BiTree *T, int key){
  if (!*T){
    return FALSE;
  } else {
    if (key == (*T)->data){
      return Delete(T);
    } else if (key < (*T)->data){
      return DeleteBST(&(*T)->lchild, key);
    } else {
      return DeleteBST(&(*T)->rchild, key);
    }
  }
  
}





void main(){
  BiTree T = NULL;
  int array[] = {62, 13, 75, 46, 93, 86, 92, 84, 54, 34};
  int length = 10;

  ConstructBST(&T, array, length);
  printf("PreOrder: ");
  PreOrder(T);
  puts("");
  printf("InOrder: ");
  InOrder(T);
  puts("");
  destroyBiTree(T);

}
