#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using namespace std;

int main() {
    cudaDeviceProp prop;
    int count;

    cudaGetDeviceCount(&count);

    for (int i = 0; i < count; i++) {
        cudaGetDeviceProperties(&prop, i);
        cout << "Device Number: " << i << endl;
        cout << "  Device name: " << prop.name << endl;
        cout << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << endl;
        cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth << endl;
        cout << "  Peak Memory Bandwidth (GB/s): " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << endl;
        cout << "  Max threads per block: " << prop.maxThreadsPerBlock << endl;
        cout << "  Max threads dim: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << endl;
        cout << "  Max grid size: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << endl;
        cout << "  Total global memory: " << prop.totalGlobalMem << endl;
        cout << "  Warp size: " << prop.warpSize << endl;
        cout << endl;
    }

    return 0;
}