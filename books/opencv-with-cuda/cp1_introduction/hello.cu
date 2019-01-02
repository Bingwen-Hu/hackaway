#include <stdio.h>
__global__ void myfirstkernel(void) {
    printf("Hello! I'm thread in block: %d\n", blockIdx.x);
}

int main(void){
    // A kernel call with 16 blocks and 1 thread per block
    myfirstkernel << <16, 1 >> >();

    cudaDeviceSynchronize();

    printf("All threads are finished!\n");
    return 0;
}
