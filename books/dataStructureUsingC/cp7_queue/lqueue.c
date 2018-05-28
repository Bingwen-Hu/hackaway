#include <stdlib.h>
#include <stdio.h>
#include "lqueue.h"


lqueue* create_lq() {
    lqueue *q = malloc(sizeof(lqueue));
    q->rear = NULL;
    q->front = NULL;
    return q;
}


lqueue* insert_lq(lqueue *q, int value) {
    node *new = malloc(sizeof(node));
    new->next = NULL;
    new->data = value;
    if (q->rear == NULL && q->front == NULL) {
        q->front = q->rear = new;
    } else {
        q->rear->next = new;
        q->rear = new;
    }
    return q;
}


lqueue* remove_lq(lqueue *q, int *value) {
    if (q->front == NULL) {
        printf("Queue is empty!\n");
        return q;
    } 
    
    node *p = q->front;
    *value = p->data;
    q->front = p->next;
    free(p);
    return q;
}


int peek_lq(lqueue *q) {
    if (q->front == NULL) {
        printf("Queue is empty!\n");
        return -1;
    }

    return q->front->data;
}


void destroy_lq(lqueue *q) {
    if (q->front == NULL)
        return;
    node *p = q->front;
    while (q->front != NULL) {
        p = q->front;
        q->front = p->next;
        free(p);
    }
    free(q);
}


void display_lq(lqueue *q) {
    node *n = q->front;
    while (n != NULL) {
        printf("%d%s", n->data, n->next != NULL ? "->" : "\n");
        n = n->next;
    }
}