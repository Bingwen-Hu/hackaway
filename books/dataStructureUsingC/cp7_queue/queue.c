#include "queue.h"
#include <stdio.h>


void init(queue *q) {
    q->front = -1;
    q->rear = -1;
}
int peek(queue *q) {
    return q->data[q->front];
}

int insert_q(queue *q, int value) {
    if (q->rear == QUEUE_SIZE - 1) {
        printf("Insert Error! Queue is full!");
        return -1;
    }
    if (q->front == -1 && q->rear == -1) {
        q->front = q->rear = 0;
    } else {
        q->rear++;
    }
    q->data[q->rear] = value;
    return 1;
}

int remove_q(queue *q) {
    if (empty_q(q)) {
        printf("Remove Error! Queue is empty!");
        return -1;
    } 
    int value = q->data[q->front];
    q->front++;
    return value;
}


int empty_q(queue *q) {
    if (q->front == -1 || q->front > q->rear) {
        return 1;
    }
    return 0;
}