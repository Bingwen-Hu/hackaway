#include <stdio.h>
__global__ void myfirstkernel(void) {}

int main(void){
    myfirstkernel << <1, 1 >> >();
    printf("Hello Cuda!\n");
    return 0;
}
