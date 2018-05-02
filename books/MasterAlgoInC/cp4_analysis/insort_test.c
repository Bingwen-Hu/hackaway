#include "insort.h"
#include <stdio.h>

int compare(const double *key1, const double *key2) {
    return *key1 > *key2;
}

int main() {
    double data[] = {1., 2., 4., 3., 9., 6., 3.2, 1.2, 5., 7.};
    int esize = sizeof(double);
    int size = sizeof(data) / esize;
    insort(data, size, esize, compare);

    for (int i = 0; i < size; i++) {
        printf("the %d-th element is: %lf\n", i, data[i]);
    }
}