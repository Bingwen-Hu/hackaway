/* simple queue for BFS */
#pragma once

#define MAXSIZE 100

typedef struct Queue {
    int size;
    int data[MAXSIZE];
} Queue;

void InitQueue(Queue *Q);
void EnQueue(Queue *Q, int val);
void DeQueue(Queue *Q, int *val);
int QueueEmpty(Queue *Q);