/** The heap data structure
Give array A, A[0] is the root, then

In a MAX-HEAP, the max-heap property is that for every node i other than the root.
A[parent(i)] >= A[i]

In a MIN-HEAP, the min-heap property is that for every node i other than the root.
A[parent(i)] <= A[i]

Very Important Procedures:
Max-Heapify, which runs in O(lg n)time, is the key to maintaining the max-heap property.
Buil-Max-Heap, which runs in linear time, produces a max-heap from an unordered input array.
Heapsort procedure, which runs in O(n lg n) time, sorts an array in place.
Max-Heap-Insert, Heap-Extract-Max, Heap-Increase-Key, and Heap-Maximum procedures, which
run in O(lg n) time, allow the heap data structure to implement a priority queue.

*/



typedef struct heap {
    int data[100];
    int length;
    int heap_size;
} Heap;

int parent(int i){
    return (i-1)/2;
}

int left(int i){
    return 2*i+1;
}

int right(int i){
    return 2*(i+1);
}

void swap(Heap *A, int i, int j){
    A->data[i] = A->data[i] + A->data[j];
    A->data[j] = A->data[i] - A->data[j];
    A->data[i] = A->data[i] - A->data[j];
}

/** sort the i element, although recursive but lg n time.
*/
void max_heapify(Heap *A, int i){
    int l = left(i);
    int r = right(i);
    int largest;
    if ((l < A->heap_size) && (A->data[l] > A->data[i])){
        largest = l;
    } else {
        largest = i;
    }
    if ((r < A->heap_size) && (A->data[r] > A->data[largest])){
        largest = r;
    }
    if (largest != i){
        swap(A, i, largest);
        max_heapify(A, largest);
    }
}

void build_max_heap(Heap *A){
    //A->heap_size = A->length;
    for (int i = (A->heap_size-1)/2; i >= 0; i--){
        max_heapify(A, i);
    }
}


void heapsort(Heap *A){
    build_max_heap(A);
    int heap_size = A->heap_size;           // save
    for (int i=A->heap_size-1; i>0; i--){
        swap(A, 0, i);
        A->heap_size--;                     // modify heap_size
        max_heapify(A, 0);
    }
    A->heap_size = heap_size;               // restore
}

/** here, heap_size is broken after sort. */
void println(Heap *A){
    printf("Heap size: %d\n", A->heap_size);
    for (int i = 0; i < A->heap_size; i++){
        printf("%d ", A->data[i]);
    }
    putchar('\n');
}

void main(){
    Heap A = {{4, 1, 3, 2, 16, 9, 10, 14, 8, 7}, 30, 10};
    heapsort(&A);
    println(&A);
}
