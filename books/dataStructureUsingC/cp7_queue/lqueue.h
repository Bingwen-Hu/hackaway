#ifndef LQUEUE_H
#define LQUEUE_H

typedef struct node {
    int data;
    struct node *next;
} node;

typedef struct lqueue {
    node *front;
    node *rear;
} lqueue;

lqueue* create_lq();
lqueue* insert_lq(lqueue *q, int value);
lqueue* remove_lq(lqueue *q, int *value);
int peek_lq(lqueue *q);
void destroy_lq(lqueue *q);
void display_lq(lqueue *q);

#endif // LQUEUE_H