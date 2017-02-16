#include <stdio.h>
#include <stdlib.h>

#define PARENT(i) ((i+1)/2 - 1)
#define LEFT(i)   ((i+1) * 2 - 1)
#define RIGHT(i)  ((i+1) * 2)


void printHeap(int A[], int n);

void exchange(int *a, int *b){
    int c = *a;
    *a = *b;
    *b = c;
}

void maxHeapify(int A[], int n, int i){
    int l = LEFT(i);
    int r = RIGHT(i);
    int largest = i;
    if ((l <= n) && (A[l] > A[i])) {
        largest = l;
    }
    if ((r <= n) && (A[r] > A[largest])) {
        largest = r;
    }
    if (largest != i) {
        exchange(&A[i], &A[largest]);
        maxHeapify(A, n, largest);
    }
    return;
}
void buildMaxHeap(int A[], int n){
    for (int i=n/2; i>=0; i--){
        /* printf("n=%d, i=%d", n, i); */
        maxHeapify(A, n, i);
        /* printHeap(A, n); */
    }
}

/* this function still go wrong */
void heapSort(int A[], int n){
    buildMaxHeap(A, n);
    printHeap(A, n);
    int size = n;
    for(int i=n-1; i>0; i--){
        exchange(&A[0], &A[i]);
        size--;
        maxHeapify(A, size, 0);
    }
}
/* maxHeapInsert(); */


int main(int argc, char *argv[])
{
    /* test code */
    int heap[] = {4, 1, 3, 2, 16, 9, 10, 14, 8, 7};
    int n = 10;
    
    printHeap(heap, n);
    heapSort(heap, n);
    printHeap(heap, n);
    puts("function heapSort still go wrong");
    return 0;
}


void printHeap(int A[], int n){
    for (int i=0; i<n; i++)
        printf("%3d", A[i]);
    puts("");
}
