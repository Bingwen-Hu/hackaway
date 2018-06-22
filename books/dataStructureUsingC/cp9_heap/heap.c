#include "heap.h"
#include <stdio.h>

int pow2(int n) {
    int res = 1;
    for (int i = 0; i < n; i++) {
        res = 2 * res;
    }
    // printf("%d\n", res);
    return res;
}

int parent(int i) {
    if (i <= 0) {
        return -1;
    }
    return (i-1)/2;
}

int left(int i) {
    return 2*i+1;
}

int right(int i) {
    return 2*(i+1);
}

void heap_swap(heap *h, int i, int j) {
    int t = h->data[i];
    h->data[i] = h->data[j];
    h->data[j] = t;
}

heap *max_heapify(heap *h, int i) {
    int p = parent(i);
    if (p == -1) {
        return h;
    }

    if (h->data[p] < h->data[i]) {
        heap_swap(h, p, i);
        max_heapify(h, p);
    }

    return h;
}

heap *build_max_heap(heap *h) {
    int indece = h->length/2;
    for (int i = h->length-1; i > indece; i--) {
        h = max_heapify(h, i);
    }
    h->size = h->length;
    return h;
}

void display(heap *h) {
    int line = 1;
    for (int i = 0; i < h->length; i++) {
        printf("%5d\t", h->data[i]);
        if ((i + 1) == (pow2(line) - 1)) {
            putchar('\n');
            line++;
        }
    }
}

heap *max_heapify_down(heap *h, int i) {
    int l = left(i);
    int r = right(i);
    int largest;

    if ((l < h->size) && (h->data[l] > h->data[i])) {
        largest = l;
    } else {
        largest = i;
    }
    if ((r < h->size) && (h->data[r] > h->data[largest])) {
        largest = r;
    }
    if (largest != i) {
        heap_swap(h, i, largest);
        h = max_heapify_down(h, largest);
    }
    return h;
}

void heapsort(heap *h) {
    for (int i = 0; i < h->length; i++) {
        heap_swap(h, 0, h->size-1);
        h->size--;
        max_heapify_down(h, 0);
    }
    for (int i = 0; i < h->length; i++) {
        printf("%3d", h->data[i]);
    }
    puts("");
}