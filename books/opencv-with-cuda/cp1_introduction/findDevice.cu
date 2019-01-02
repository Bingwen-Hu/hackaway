#include <stdio.h>
#include <cuda_runtime.h>


int main(void) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0){
        printf("There are no avaiable device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA capable device(s)\n", device_count);
    }
}
