#include <stdio>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void gpuAdd(int *d_a, int *d_b, int *d_c)
{
    *d_c = *d_a + *d_b;
}


int main(void){
    // define
    int h_a, h_b, h_c;
    int *d_a, *d_b, *d_c;

    // init
    h_a = 1;
    h_b = 4;
    
    // alloc
    cudaMalloc()

}
