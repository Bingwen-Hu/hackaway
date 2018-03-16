#include "queue.h"
#include <stdio.h>

void InitQueue(Queue *Q){
    Q->size = 0;
}

void EnQueue(Queue *Q, int val){
    if (Q->size < MAXSIZE){
        Q->data[Q->size] = val;
        Q->size++;
    } else {
        puts("Queue Full! Enqueue Fail!");
    }
}

void DeQueue(Queue *Q, int *val){
    if (!QueueEmpty(Q)){
        *val = Q->data[0];
        Q->size--;
        for (int i=0; i < Q->size ; i++){
            Q->data[i] = Q->data[i+1];
        }
    } else {
        puts("Queue Empty! Dequeue Fail!");
    }
}
int QueueEmpty(Queue *Q){
    return Q->size == 0;
}