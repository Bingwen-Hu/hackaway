#ifndef QUEUE_H_
#define QUEUE_H_

#define QUEUE_SIZE 200

typedef struct queue {
    int data[QUEUE_SIZE];
    int front;
    int rear;
} queue;

void init(queue *q);
int peek(queue *q);
int insert_q(queue *q, int value);
int remove_q(queue *q);


#endif // QUEUE_H_