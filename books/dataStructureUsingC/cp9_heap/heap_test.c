#include "heap.h"
#include <stdio.h>



int main(int argc, char const *argv[])
{
    heap h = {
        .data = {24, 13, 6, 4, 7, 0, 1, 2, 42, 31},
        .length = 10,
        .size = 0,
    };

    // test max_heapify
    // display(&h);
    // heap *h2 = max_heapify(&h, 9);
    // puts("");
    // display(h2);

    // test max_heapify_down
    // heap *h2 = build_max_heap(&h);
    // h2->data[0] = -1;
    // h2 = max_heapify_down(h2, 0);
    // display(h2);

    heap *h2 = build_max_heap(&h);
    // display(h2);
    heapsort(h2);
    return 0;
}