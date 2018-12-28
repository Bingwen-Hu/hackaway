#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void gpuAdd(int d_a, int d_b, int *d_c)
{
    *d_c = d_a + d_b;
}

int main(void){
    int h_c; // define host variable to store answer
    int *d_c; // define device pointer
    cudaMalloc((void**)&d_c, sizeof(int));
    // kernel call by passing 1 and 4 as inputs and storing answer in d_c
    // << <1,1> >> means 1 block is executed with 1 thread per block
    gpuAdd <<<1,1 >>> (1, 4, d_c);
    // copy result from device memory to host memory
    cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    printf("1 + 4 = %d\n", h_c);
    cudaFree(d_c);
    return 0;
}
