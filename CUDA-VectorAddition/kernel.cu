#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <chrono>

using namespace std;

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd_d(double* a, double* b, double* c, int n)
{

    // The stride-based approach is often used when the number of data elements (n) 
    // is larger than the total number of threads launched (blockDim.x * gridDim.x).
        // In my case blockDim.x * gridDim.x = 1024 * 2147483647 = 2,199,023,255,552 threads
    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int i = index; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }

    // If the number of data elements is equal to or less than the total number of threads, 
    // you can use the unique global thread ID approach.
    // // Get our global thread ID
    //int id = blockIdx.x * blockDim.x + threadIdx.x;
    // Make sure we do not go out of bounds
    //if (id < n) {
    //    c[id] = a[id] + b[id];
    //}
}

// Vector addition function
void vecAdd_h(double* a, double* b, double* c, int n)
{
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void gpuVectorAddition() {
    // Size of vectors
    int N = 1 << 20; // 1M elements

    // Host input vectors
    double* h_a;
    double* h_b;
    //Host output vector
    double* h_c;

    // Device input vectors
    double* d_a;
    double* d_b;
    //Device output vector
    double* d_c;

    // Size, in bytes, of each vector
    size_t bytes = N * sizeof(double);

    // Allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    int i;
    // Initialize vectors on host
    for (i = 0; i < N; i++) {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
    }

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Number of threads in each thread block
    int blockSize = 256;

    // Number of thread blocks in grid
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Start recording
    auto start = std::chrono::high_resolution_clock::now();

    // Execute the kernel
    vecAdd_d << < numBlocks, blockSize >> > (d_a, d_b, d_c, N);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Stop recording
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // Print time
    printf("Time to calculate on Device: %f s\n", diff.count());

    // Copy array back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    for (i = 0; i < N; i++)
        sum += h_c[i];
    printf("Final result Device: %f\n", sum / N);

    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);
}
void cpuVectorAddition() {
    // Size of vectors
    int N = 1 << 20; // 1M elements

    // Host input vectors
    double* h_a;
    double* h_b;
    //Host output vector
    double* h_c;

    // Size, in bytes, of each vector
    size_t bytes = N * sizeof(double);

    // Allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);

    // Initialize vectors on host
    for (int i = 0; i < N; i++) {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
    }

    // Start recording
    auto start = std::chrono::high_resolution_clock::now();

    // Execute the function
    vecAdd_h(h_a, h_b, h_c, N);

    // Stop recording
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // Print time
    printf("Time to calculate: %f s\n", diff.count());

    // Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    for (int i = 0; i < N; i++)
        sum += h_c[i];
    printf("Final result: %f\n", sum / N);

    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);
}

int main(int argc, char* argv[])
{
    
    // Execute vector addition on device
    gpuVectorAddition();

    cout << endl;

    // Execute vector addition on device
    cpuVectorAddition();

    return 0;
}