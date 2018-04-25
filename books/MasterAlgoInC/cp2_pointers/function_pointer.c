#include <stdio.h>



void swap(int *a, int *b){
    int t = *a;
    *a = *b;
    *b = t;
}

int main() {
    int a = 1, b = 3;
    printf("before swap, a is %d, b is %d\n", a, b);

    swap(&a, &b);
    printf("after swap, a is %d, b is %d\n", a, b);

    void (*fnptr)(int *a, int *b);
    fnptr = swap;
    fnptr(&a, &b);
    printf("after fnptr, a is %d, b is %d\n", a, b);
}