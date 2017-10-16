/* add an optional head pointer, we uniform the insertiona and
deletion on Linklist */

#include <stdio.h>
#include <stdlib.h>

typedef int ElemType;
typedef struct Node {
  ElemType data;
  struct Node *next;
} Node, *LinkList;

int InitNode(LinkList L, ElemType data){
  L = (LinkList) malloc(sizeof(Node));
  L->data = data;
  L->next = NULL;
  return 0;
}


int GetElem(LinkList L, int i, ElemType *e){
  int j;
  LinkList p;

  p = L->next;
  j = 0;
  while(p && j < i){
    p = p->next;
    j++;
  }
  
  if (!p || j>i)
    return 0;
  
  *e = p->data;
  return 1;
}

int InsertElem(LinkList L, int i, ElemType e){
  LinkList p = L->next;
  int j = 0;
  
  while (p && j < i){
    p = p->next;
    j++;
  }
  
  if (!p || j>i)
    return 0;

  LinkList new = (LinkList) malloc(sizeof(Node));
  new->data = e;
  new->next = p->next;
  p->next = new;
  return 1;  
}


int DeleteElem(LinkList L, ElemType i){

  LinkList p, q;
  p = L->next;
  int j = 0;

  while(p && j<i){
    q = p;
    p = p->next;
    j++;
  }
  
  if (!p || j>i)
    return 1;
  
  q->next = p->next;

  free(p);
}

void PrintList(LinkList L){
  LinkList p = L->next;
  int c = L->data;
  printf("Linklist: ");
  for (int i=0; i<c; i++){
    printf("%d\t", p->data);
    p = p->next;
  }
}



void main(){

  LinkList list, n1, n2;
  
  InitNode(list, 0);
  InitNode(n1, 49);
  InitNode(n2, 34);

  list->next = n1;
  list->data++;
  n1->next = n2;
  list->data++;

  PrintList(list);
  
}


