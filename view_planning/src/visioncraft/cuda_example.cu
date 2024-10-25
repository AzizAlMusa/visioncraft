#include <cuda_runtime.h>

// CUDA kernel function to add two arrays
__global__ void addArrays(int* a, int* b, int* c, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

// Function to manage CUDA memory and run the kernel
void runAddArrays(int* a, int* b, int* c, int size) {
    int *d_a, *d_b, *d_c;

    // Allocate device memory
    cudaMalloc((void **)&d_a, size * sizeof(int));  // size is the number of elements
    cudaMalloc((void **)&d_b, size * sizeof(int));
    cudaMalloc((void **)&d_c, size * sizeof(int));

    // Copy input arrays from host to device
    cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with 1 block and 'size' threads
    addArrays<<<1, size>>>(d_a, d_b, d_c, size);

    // Wait for the kernel to finish executing
    cudaDeviceSynchronize();

    // Copy the result array from device to host
    cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
