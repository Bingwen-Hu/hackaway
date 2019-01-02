#include <cuda_runtime.h>
#include <stdio.h>


int main(){
    int device = 0;
    cudaDeviceProp device_property;
    cudaGetDeviceProperties(&device_property, device);
    printf("\nDevice %d: %s", device, device_property.name);

    int driver_version;
    int runtime_version;
    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);
    printf("\nCUDA driver Version / Runtime Version %.3g / %.3g", 
            driver_version/1000.0, runtime_version/1000.0);

    // thread property
    printf("\nMax number of threads per multiprocessor: %d\n", 
            device_property.maxThreadsPerMultiProcessor);
    printf("\nMax number of threads per block: %d\n",
            device_property.maxThreadsPerBlock);
    printf("\nMax dimension size of a thread block (x,y,z): (%d,%d,%d)\n",
            device_property.maxThreadsDim[0],
            device_property.maxThreadsDim[1],
            device_property.maxThreadsDim[2]);
    printf("\nMax dimension size of a grid size (x,y,z): (%d,%d,%d)\n",
            device_property.maxGridSize[0],
            device_property.maxGridSize[1],
            device_property.maxGridSize[2]);
    return 0;
}
