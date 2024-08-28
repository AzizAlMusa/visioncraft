#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel definition
__global__ void advancedCudaKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * data[idx] + 2.0f; // More complex operation
    }
}

// CPU computation function
void cpuComputation(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = data[i] * data[i] + 2.0f; // Same operation as GPU for comparison
    }
}

// GPU computation function with timing
void run_advanced_cuda_example(int size) {
    float* d_data;
    float* h_data = new float[size];

    // Allocate memory on the GPU
    cudaMalloc(&d_data, size * sizeof(float));
    cudaMemset(d_data, 1, size * sizeof(float)); // Initialize to 1

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start);

    // Define block and grid sizes
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    // Launch the CUDA kernel
    advancedCudaKernel<<<numBlocks, blockSize>>>(d_data, size);

    // Record stop time
    cudaEventRecord(stop);

    // Wait for the event to complete
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Free the GPU memory
    cudaFree(d_data);

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    delete[] h_data;

    std::cout << "CUDA kernel execution time: " << elapsedTime / 1000.0 << " seconds" << std::endl; // Convert to seconds
}
