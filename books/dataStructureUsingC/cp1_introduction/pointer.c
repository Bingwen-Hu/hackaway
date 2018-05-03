/* pointer using malloc */
#include <stdlib.h>
#include <stdio.h>

int main() {
    int *p;
    printf("pointer address is %p\n", p);
    
    p = malloc(sizeof(int));
    *p = 42;
    printf("pointer address is %p\n", p);
}