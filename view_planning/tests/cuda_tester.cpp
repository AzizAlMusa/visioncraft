#include <iostream>
#include <chrono>

// Forward declaration of the CUDA functions
void run_advanced_cuda_example(int size);
void cpuComputation(float* data, int size);

int main() {
    const int N = 1024 * 1024 * 32; // 32 million elements
    float* h_data = new float[N];

    // CPU computation timing
    auto start = std::chrono::high_resolution_clock::now();
    cpuComputation(h_data, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end - start;
    std::cout << "CPU computation time: " << cpu_duration.count() << " seconds" << std::endl;

    // GPU computation timing
    start = std::chrono::high_resolution_clock::now();
    run_advanced_cuda_example(N);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_duration = end - start;
    std::cout << "GPU computation time: " << gpu_duration.count() << " seconds" << std::endl;

    delete[] h_data;
    return 0;
}
