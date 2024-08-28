import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualizeGrid(data_file, nbv=None, predicted_nbv=None, voxel_size=0.025):
    # Step 1: Read the Data from the text file
    data = np.loadtxt(data_file)

    # Step 2: Prepare the Data
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    occupancy = data[:, 3]

    # Determine voxel boundaries based on voxel size
    x_min, x_max = np.floor(np.min(x) / voxel_size) * voxel_size, np.ceil(np.max(x) / voxel_size) * voxel_size
    y_min, y_max = np.floor(np.min(y) / voxel_size) * voxel_size, np.ceil(np.max(y) / voxel_size) * voxel_size
    z_min, z_max = np.floor(np.min(z) / voxel_size) * voxel_size, np.ceil(np.max(z) / voxel_size) * voxel_size

    # Create a grid for the voxel data
    x_grid = np.arange(x_min, x_max + voxel_size, voxel_size)
    y_grid = np.arange(y_min, y_max + voxel_size, voxel_size)
    z_grid = np.arange(z_min, z_max + voxel_size, voxel_size)

    # Initialize voxel arrays
    colors = {}

    # Fill voxel colors based on occupancy
    for xi, yi, zi, oc in zip(x, y, z, occupancy):
        ix = int((xi - x_min) / voxel_size)
        iy = int((yi - y_min) / voxel_size)
        iz = int((zi - z_min) / voxel_size)
        if oc == 1.0:
            colors[(ix, iy, iz)] = 'blue'
        elif oc == 0.5:
            colors[(ix, iy, iz)] = 'yellow'

    # Step 3: Plot the Data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each voxel
    for (ix, iy, iz), color in colors.items():
        x_pos = x_grid[ix]
        y_pos = y_grid[iy]
        z_pos = z_grid[iz]
        ax.bar3d(x_pos, y_pos, z_pos, voxel_size, voxel_size, voxel_size, color=color, edgecolor='black')

    # Plot the NBV and Predicted NBV if provided
    center = np.array([x_grid.mean(), y_grid.mean(), z_grid.mean()])

    if nbv is not None:
        position = np.array(nbv[:3])
        direction = center - position
        ax.quiver(position[0], position[1], position[2], direction[0], direction[1], direction[2], length=0.1, normalize=True, color='green', linewidth=2)

    if predicted_nbv is not None:
        position = np.array(predicted_nbv[:3])
        direction = center - position
        ax.quiver(position[0], position[1], position[2], direction[0], direction[1], direction[2], length=0.1, normalize=True, color='red', linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
