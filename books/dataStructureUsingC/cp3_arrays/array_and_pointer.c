#include <stdio.h>

int main() {
    int array[] = {1, 2, 3, 4, 6};
    int *ptr;
    ptr = &array[0];

    printf("address of array = %p %p %p\n", array, ptr, &array);
}

/* 
array: constant fixed address of the memory
ptr  : pointer to the address of the memory
&array: address of the first element of the memory
 */