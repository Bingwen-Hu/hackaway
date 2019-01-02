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
    printf("\nCUDA driver Version / Runtime Version %i / %i", 
            driver_version, runtime_version);
    
    return 0;
}
