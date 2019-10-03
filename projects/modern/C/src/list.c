#include "list.h"


LinkList InitList_M(LinkList L, int length){
    LinkList p, q;
    ElemType r;

    L = (LinkList) malloc(sizeof(Node)); /* head node */
    L->data = 0;                  // store the length
    L->next = NULL;
    q = L;                        /* q as the end pointer */

    for (int i=0; i<length; i++){
        p = (LinkList) malloc(sizeof(Node));
        r = rand() % 42;
        p->data = r;
        p->next = NULL;
        q->next = p;
        q = p;                      /* go to the end */
        L->data++;
    }
    return L;
}


void PrintList(LinkList L){
    LinkList p = L->next;
    int c = L->data;
    printf("Linklist: ");
    for (int i=0; i<c; i++){
        printf("%d\t", p->data);
        p = p->next;
    }
    puts("");
}


void DestroyList(LinkList L){
    LinkList p; 
    while (L->next){
        p = L;
        L = L->next;
        free(p);
    }
    free(L);
}


ElemType GetElem(LinkList L, int index){
    LinkList p = L->next;
    ElemType e;

    if (index < 0 || L->data < index){
        puts("index invalid");
        return 0;
    }

    for (int i=0; i<index; i++){
        p = p->next;
    }
    e = p->data;
    return e;
}


void insertList(LinkList L, int index, ElemType e){
    LinkList p, q;
    p = L->next;
    q = L;
    
    for (int i=0; i<index; i++){
        q = p;
        p = p->next;
    }
    LinkList new = (LinkList) malloc(sizeof(Node));
    q->next = new;
    new->data = e;
    new->next = p;
    L->data++;

}

void deleteNode(LinkList L, int index){
    LinkList p, q;
    p = L->next;
    q = L;
    
    for(int i=0; i<index; i++){
        q = p;
        p = p->next;
    }
    q->next = p->next;
    free(p);
    L->data--;
}