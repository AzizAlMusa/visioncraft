// raycasting.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <float.h>  // For FLT_MAX
#include <stdio.h>  // For debugging
#include "visioncraft/model.h"  // Ensure this header is included to define VoxelGridGPU

namespace visioncraft {

// Helper function to swap values (CUDA device compatible)
__device__ __host__ inline void custom_swap(float& a, float& b) {
    float temp = a;
    a = b;
    b = temp;
}

// Ray-box intersection function
__device__ bool rayBoxIntersection(const float* rayStart, const float* rayDir,
                                   const float box_min[3], const float box_max[3], float& t_near, float& t_far) {
    t_near = -FLT_MAX;
    t_far = FLT_MAX;

    for (int i = 0; i < 3; ++i) {
        if (fabsf(rayDir[i]) < 1e-6f) {  // Ray is parallel to the planes in this axis
            if (rayStart[i] < box_min[i] || rayStart[i] > box_max[i]) {
                return false;  // Ray is outside the box and parallel, no intersection
            }
        } else {
            float t1 = (box_min[i] - rayStart[i]) / rayDir[i];
            float t2 = (box_max[i] - rayStart[i]) / rayDir[i];
            if (t1 > t2) custom_swap(t1, t2);
            t_near = fmaxf(t_near, t1);
            t_far = fminf(t_far, t2);
            if (t_near > t_far) return false;  // No intersection
        }
    }

    return t_far >= t_near && t_far >= 0.0f;  // Valid intersection in positive ray direction
}

// DDA raycasting helper function with early termination
__device__ int3 ddaRaycast(const float* rayStart, const float* rayDir, const int* voxel_data,
                           int width, int height, int depth, float voxel_size, float far_plane,
                           const float box_min[3], const float box_max[3], bool& hit) {
    // Check for ray-box intersection
    float t_near, t_far;
    if (!rayBoxIntersection(rayStart, rayDir, box_min, box_max, t_near, t_far)) {
        hit = false;
        return make_int3(-1, -1, -1);
    }

    // Clamp t_near and t_far to the provided near and far planes
    t_near = fmaxf(t_near, 0.0f);
    t_far = fminf(t_far, far_plane);

    // Adjust the ray starting point if it starts outside the grid
    float newRayStart[3] = {
        rayStart[0] + t_near * rayDir[0],
        rayStart[1] + t_near * rayDir[1],
        rayStart[2] + t_near * rayDir[2]
    };

    // Initialize DDA variables (voxel stepping)
    int3 voxelIndex;
    voxelIndex.x = static_cast<int>((newRayStart[0] - box_min[0]) / voxel_size);
    voxelIndex.y = static_cast<int>((newRayStart[1] - box_min[1]) / voxel_size);
    voxelIndex.z = static_cast<int>((newRayStart[2] - box_min[2]) / voxel_size);

    // Clamp voxel indices to grid boundaries
    voxelIndex.x = max(0, min(voxelIndex.x, width - 1));
    voxelIndex.y = max(0, min(voxelIndex.y, height - 1));
    voxelIndex.z = max(0, min(voxelIndex.z, depth - 1));

    // Directional steps
    int stepX = (rayDir[0] > 0) ? 1 : -1;
    int stepY = (rayDir[1] > 0) ? 1 : -1;
    int stepZ = (rayDir[2] > 0) ? 1 : -1;

    // Avoid division by zero
    float invDirX = (fabsf(rayDir[0]) > 1e-6f) ? 1.0f / rayDir[0] : FLT_MAX;
    float invDirY = (fabsf(rayDir[1]) > 1e-6f) ? 1.0f / rayDir[1] : FLT_MAX;
    float invDirZ = (fabsf(rayDir[2]) > 1e-6f) ? 1.0f / rayDir[2] : FLT_MAX;

    // Compute tMax for each axis
    float voxelBorderX = box_min[0] + (voxelIndex.x + (stepX > 0 ? 1 : 0)) * voxel_size;
    float voxelBorderY = box_min[1] + (voxelIndex.y + (stepY > 0 ? 1 : 0)) * voxel_size;
    float voxelBorderZ = box_min[2] + (voxelIndex.z + (stepZ > 0 ? 1 : 0)) * voxel_size;

    float tMaxX = (voxelBorderX - newRayStart[0]) * invDirX;
    float tMaxY = (voxelBorderY - newRayStart[1]) * invDirY;
    float tMaxZ = (voxelBorderZ - newRayStart[2]) * invDirZ;

    // Compute tDelta for each axis
    float tDeltaX = fabsf(voxel_size * invDirX);
    float tDeltaY = fabsf(voxel_size * invDirY);
    float tDeltaZ = fabsf(voxel_size * invDirZ);

    hit = false;
    float t = t_near;

    // Main DDA loop
    while (t <= t_far) {
        // Compute voxelIdx safely
        int voxelIdx = voxelIndex.z * (width * height) + voxelIndex.y * width + voxelIndex.x;

        // Check if we are inside the grid
        if (voxelIndex.x >= 0 && voxelIndex.x < width &&
            voxelIndex.y >= 0 && voxelIndex.y < height &&
            voxelIndex.z >= 0 && voxelIndex.z < depth) {

            // Access voxel_data safely
            if (voxel_data[voxelIdx] == 1) {  // Assume 1 means occupied
                hit = true;
                return voxelIndex;  // Stop after first hit
            }
        } else {
            // If out of bounds, terminate the loop
            break;
        }

        // Determine which axis to step along
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                t = tMaxX;
                tMaxX += tDeltaX;
                voxelIndex.x += stepX;
            } else {
                t = tMaxZ;
                tMaxZ += tDeltaZ;
                voxelIndex.z += stepZ;
            }
        } else {
            if (tMaxY < tMaxZ) {
                t = tMaxY;
                tMaxY += tDeltaY;
                voxelIndex.y += stepY;
            } else {
                t = tMaxZ;
                tMaxZ += tDeltaZ;
                voxelIndex.z += stepZ;
            }
        }

        // Check if voxel indices are out of bounds
        if (voxelIndex.x < 0 || voxelIndex.x >= width ||
            voxelIndex.y < 0 || voxelIndex.y >= height ||
            voxelIndex.z < 0 || voxelIndex.z >= depth) {
            break;
        }
    }

    return make_int3(-1, -1, -1);  // Return invalid index if no hit
}

// Kernel to generate rays and perform raycasting
__global__ void generateRaysAndRaycastOnGPU(const float* position, const float* forward, const float* right, const float* up,
                                            float hfov, float vfov, int ds_width, int ds_height,
                                            float near_plane, float far_plane,
                                            const int* voxel_data, int width, int height, int depth, float voxel_size,
                                            const float* min_bound,
                                            int3* hit_voxels, unsigned int* hit_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = ds_width * ds_height;

    if (idx < total_pixels) {
        // Compute pixel coordinates (i, j)
        int i = idx % ds_width;
        int j = idx / ds_width;

        // Convert to Normalized Device Coordinates (NDC)
        float u = (2.0f * (i + 0.5f) / ds_width - 1.0f) * tanf(hfov * 0.5f * M_PI / 180.0f);
        float v = (1.0f - 2.0f * (j + 0.5f) / ds_height) * tanf(vfov * 0.5f * M_PI / 180.0f);

        // Compute the ray direction using forward, right, and up vectors
        float rayDir[3];
        rayDir[0] = forward[0] + u * right[0] + v * up[0];
        rayDir[1] = forward[1] + u * right[1] + v * up[1];
        rayDir[2] = forward[2] + u * right[2] + v * up[2];

        // Normalize the ray direction
        float norm = sqrtf(rayDir[0] * rayDir[0] + rayDir[1] * rayDir[1] + rayDir[2] * rayDir[2]);
        if (norm > 1e-6f) {
            rayDir[0] /= norm;
            rayDir[1] /= norm;
            rayDir[2] /= norm;
        } else {
            // Invalid ray direction, skip this ray
            return;
        }

        // Perform DDA raycasting and stop after the first hit
        bool hit = false;

        // Define box_min and box_max in the kernel
        float box_min[3] = { min_bound[0], min_bound[1], min_bound[2] };
        float box_max[3] = {
            min_bound[0] + width * voxel_size,
            min_bound[1] + height * voxel_size,
            min_bound[2] + depth * voxel_size
        };

        int3 hitVoxel = ddaRaycast(position, rayDir, voxel_data, width, height, depth, voxel_size, far_plane, box_min, box_max, hit);

        // If a hit occurred, store it using atomic operations
        if (hit) {
            // Get a unique index for this hit
            unsigned int hit_idx = atomicAdd(hit_count, 1);

            // Store the hit voxel index
            hit_voxels[hit_idx] = hitVoxel;
        }
    }
}

// Function to launch the kernel and handle memory allocations
void launchGenerateRaysOnGPU(const float* position, const float* forward, const float* right, const float* up,
                             float hfov, float vfov, int ds_width, int ds_height,
                             float near_plane, float far_plane,
                             const VoxelGridGPU& voxelGridGPU, int3* host_hit_voxels, unsigned int& host_hit_count) {
    // Prepare variables
    int total_pixels = ds_width * ds_height;

    cudaError_t err;

    // Allocate memory for the voxel grid on the GPU
    size_t voxel_data_size = voxelGridGPU.width * voxelGridGPU.height * voxelGridGPU.depth * sizeof(int);
    int* d_voxel_data;
    err = cudaMalloc((void**)&d_voxel_data, voxel_data_size);
    if (err != cudaSuccess) {
        printf("Error allocating d_voxel_data: %s\n", cudaGetErrorString(err));
        return;
    }

    // Copy voxel data to the GPU
    err = cudaMemcpy(d_voxel_data, voxelGridGPU.voxel_data, voxel_data_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Error copying voxel_data to GPU: %s\n", cudaGetErrorString(err));
        cudaFree(d_voxel_data);
        return;
    }

    // Allocate and copy min_bound to device
    float* d_min_bound;
    err = cudaMalloc((void**)&d_min_bound, 3 * sizeof(float));
    if (err != cudaSuccess) {
        printf("Error allocating d_min_bound: %s\n", cudaGetErrorString(err));
        cudaFree(d_voxel_data);
        return;
    }
    err = cudaMemcpy(d_min_bound, voxelGridGPU.min_bound, 3 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Error copying min_bound to GPU: %s\n", cudaGetErrorString(err));
        cudaFree(d_voxel_data);
        cudaFree(d_min_bound);
        return;
    }

    // Allocate memory for hit count on the GPU
    unsigned int* d_hit_count;
    err = cudaMalloc((void**)&d_hit_count, sizeof(unsigned int));
    if (err != cudaSuccess) {
        printf("Error allocating d_hit_count: %s\n", cudaGetErrorString(err));
        cudaFree(d_voxel_data);
        cudaFree(d_min_bound);
        return;
    }

    // Initialize hit count to zero
    err = cudaMemset(d_hit_count, 0, sizeof(unsigned int));
    if (err != cudaSuccess) {
        printf("Error setting d_hit_count to zero: %s\n", cudaGetErrorString(err));
        cudaFree(d_voxel_data);
        cudaFree(d_min_bound);
        cudaFree(d_hit_count);
        return;
    }

    // Allocate memory for hit voxels on the GPU
    int3* d_hit_voxels;
    err = cudaMalloc((void**)&d_hit_voxels, total_pixels * sizeof(int3));  // Max possible hits
    if (err != cudaSuccess) {
        printf("Error allocating d_hit_voxels: %s\n", cudaGetErrorString(err));
        cudaFree(d_voxel_data);
        cudaFree(d_min_bound);
        cudaFree(d_hit_count);
        return;
    }

    // Allocate memory for position, forward, right, and up vectors on the GPU
    float* d_position;
    float* d_forward;
    float* d_right;
    float* d_up;
    err = cudaMalloc((void**)&d_position, 3 * sizeof(float));
    err = cudaMalloc((void**)&d_forward, 3 * sizeof(float));
    err = cudaMalloc((void**)&d_right, 3 * sizeof(float));
    err = cudaMalloc((void**)&d_up, 3 * sizeof(float));

    // Copy data to the GPU
    err = cudaMemcpy(d_position, position, 3 * sizeof(float), cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_forward, forward, 3 * sizeof(float), cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_right, right, 3 * sizeof(float), cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_up, up, 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    generateRaysAndRaycastOnGPU<<<blocksPerGrid, threadsPerBlock>>>(d_position, d_forward, d_right, d_up, hfov, vfov,
                                                                    ds_width, ds_height, near_plane, far_plane,
                                                                    d_voxel_data, voxelGridGPU.width, voxelGridGPU.height,
                                                                    voxelGridGPU.depth, static_cast<float>(voxelGridGPU.voxel_size),
                                                                    d_min_bound,
                                                                    d_hit_voxels, d_hit_count);

    // Synchronize and check for errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        // Free GPU memory before returning
        cudaFree(d_voxel_data);
        cudaFree(d_min_bound);
        cudaFree(d_hit_voxels);
        cudaFree(d_hit_count);
        cudaFree(d_position);
        cudaFree(d_forward);
        cudaFree(d_right);
        cudaFree(d_up);
        return;
    }

    // Copy hit count back to host
    err = cudaMemcpy(&host_hit_count, d_hit_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Error copying hit_count to host: %s\n", cudaGetErrorString(err));
        // Free GPU memory before returning
        cudaFree(d_voxel_data);
        cudaFree(d_min_bound);
        cudaFree(d_hit_voxels);
        cudaFree(d_hit_count);
        cudaFree(d_position);
        cudaFree(d_forward);
        cudaFree(d_right);
        cudaFree(d_up);
        return;
    }

    // Copy only the actual hits back to host
    if (host_hit_count > 0) {
        err = cudaMemcpy(host_hit_voxels, d_hit_voxels, host_hit_count * sizeof(int3), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("Error copying hit_voxels to host: %s\n", cudaGetErrorString(err));
            // Free GPU memory before returning
            cudaFree(d_voxel_data);
            cudaFree(d_min_bound);
            cudaFree(d_hit_voxels);
            cudaFree(d_hit_count);
            cudaFree(d_position);
            cudaFree(d_forward);
            cudaFree(d_right);
            cudaFree(d_up);
            return;
        }
    }

    // Free GPU memory
    cudaFree(d_voxel_data);
    cudaFree(d_min_bound);
    cudaFree(d_hit_voxels);
    cudaFree(d_hit_count);
    cudaFree(d_position);
    cudaFree(d_forward);
    cudaFree(d_right);
    cudaFree(d_up);
}


}  // namespace visioncraft
