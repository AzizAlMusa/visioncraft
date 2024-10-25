#ifndef RAYCASTING_CUH
#define RAYCASTING_CUH

/**
 * @brief GPU-friendly Voxel Grid structure for raycasting.
 * 
 * This structure holds a flattened boolean array representing the voxel grid in GPU memory.
 * It also contains the grid dimensions and voxel size.
 */
struct VoxelGridGPU {
    bool* data;  ///< Flattened voxel data (true = occupied, false = free)
    int grid_size_x, grid_size_y, grid_size_z;  ///< Dimensions of the voxel grid
    float voxel_size;  ///< Size of each voxel

    /**
     * @brief Access function to retrieve the state of a voxel.
     * 
     * @param x The x index of the voxel.
     * @param y The y index of the voxel.
     * @param z The z index of the voxel.
     * @return true if the voxel is occupied, false if it's free.
     */
    __device__ bool getVoxel(int x, int y, int z) const {
        int idx = x + grid_size_x * (y + grid_size_y * z);  // Calculate flattened index
        return data[idx];  // Access the voxel state
    }
};

#endif // RAYCASTING_CUH
