#ifndef HEAP_H
#define HEAP_H

#define MAX_HEAP_SIZE 100

typedef struct heap {
    int data[MAX_HEAP_SIZE];
    int size;                   // heap size
    int length;                 // data length
} heap;

int parent(int i);
int left(int i);
int right(int i);


heap *max_heapify(heap *h, int i);
heap *max_heapify_down(heap *h, int i);
heap *build_max_heap(heap *h);
void heapsort(heap *h);
void display(heap *h);

#endif // HEAP_H