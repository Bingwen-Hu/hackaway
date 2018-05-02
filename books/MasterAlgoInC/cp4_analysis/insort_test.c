#include "insort.h"
#include <stdio.h>

int compare(const int *key1, const int *key2) {
    return *key1 > *key2;
}

int main() {
    int data[] = {1, 2, 4, 3, 9, 6, 32, 12, 5, 7};
    int esize = sizeof(int);
    int size = sizeof(data) / esize;
    insort(data, size, esize, compare);

    for (int i = 0; i < size; i++) {
        printf("the %d-th element is: %d\n", i, data[i]);
    }
}



// note:
// 1 < lg n < n < nlg n < n^2 < n^2 lg n < n^3 < 2^n < 3^n < n!