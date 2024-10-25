import os
import time
import numpy as np
import torch
from torch.autograd import Variable
import nbvnet
import visioncraft_py
import regression_nbv_utils_archive as rnbv
# Global Configuration
CONFIG = {
    'voxel_resolution': 0.025,
    'dropout': 0.2,
    'model_weights_file': 'model/weights.pth',
    'nbv_positions_file': 'points_in_sphere.txt',
    'max_iterations': 10
}

# Check for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model():
    """Build and load the NBV-net model."""
    net = nbvnet.NBV_Net(CONFIG['dropout'])
    net.to(device)
    net.eval()
    net.load_state_dict(torch.load(CONFIG['model_weights_file'], map_location=device), strict=False)
    return net

def load_nbv_positions():
    """Load positions for NBV calculation."""
    return np.genfromtxt(CONFIG['nbv_positions_file'])

def get_positions(nbv_class, positions):
    """Convert class to corresponding pose."""
    return np.array(positions[nbv_class])

def predict_nbv(cube, net):
    """Perform a forward pass through the network and predict the NBV."""
    grids = torch.from_numpy(np.array([[cube]])).float().to(device)
    with torch.no_grad():
        start = time.time()
        output = net.forward(grids)
        end = time.time()
        print("Processing time:", end - start)
    output = output.cpu().numpy().squeeze()
    return output

def compute_full_nbv(bar_nbv):
    """Compute the full NBV with orientation."""
    origin = np.array([0, 0, 0])
    r = origin - bar_nbv
    r = r / np.linalg.norm(r)
    
    yaw = np.arctan2(r[1], r[0])
    pitch = -np.arcsin(r[2])
    roll = 0
    
    full_nbv = np.array([bar_nbv[0], bar_nbv[1], bar_nbv[2], yaw, pitch, roll])
    full_nbv_deg = full_nbv.copy()
    full_nbv_deg[3:] *= 180 / np.pi
    
    print('Full NBV:', full_nbv)
    print('Full NBV degrees:', full_nbv_deg)
    
    return full_nbv

def convert_exploration_map_to_cube(exploration_map_data):
    """Convert the exploration map data to a voxel grid (cube)."""
    # Extract positions and occupancies
    positions = np.array([[x, y, z] for x, y, z, _ in exploration_map_data])
    occupancies = np.array([value for _, _, _, value in exploration_map_data])

    # Define cube parameters
    cube_size = 32
    cube = np.zeros((cube_size, cube_size, cube_size))

    # Calculate indices
    pmin = positions.min(axis=0)
    voxel_resolution = CONFIG['voxel_resolution']
    indices = ((positions - pmin) / voxel_resolution).astype(int)

    # Filter valid indices
    valid_mask = np.all((indices >= 0) & (indices < cube_size), axis=1)
    indices = indices[valid_mask]
    occupancies = occupancies[valid_mask]

    # Fill the cube
    for idx, value in zip(indices, occupancies):
        cube[tuple(idx)] = value

    return cube

def main():
    """Main loop for iterative NBV prediction and simulation."""
    net = load_model()
    nbv_positions = load_nbv_positions()

    # Initialize the exploration simulator
    simulator = visioncraft_py.ExplorationSimulator()

    # Load the model into the simulator
    model_file = './objects/gorilla.ply'  # Update with your model path
    if not simulator.loadModel(model_file):
        print("Failed to load model")
        return

    scaling_factor = simulator.getScalingFactor()
    print(f"Scaling factor: {scaling_factor}")

    # Initialize the list of viewpoints
    all_nbvs = []

     # Set up the initial viewpoint
  
    # initial_position = np.random.uniform(-1, 1, size=3)  # Random initial position
    # initial_position = initial_position / np.linalg.norm(initial_position) * 4 / scaling_factor # Normalize to 0.4m from origin
    # print(f"Initial random viewpoint position: {initial_position}")
    
    # # Create initial Viewpoint object directed at the origin
    # look_at = np.array([0, 0, 0]).reshape(3, 1)
    
    # up_vector = np.array([0, 0, -1]).reshape(3, 1)

    # initial_viewpoint = visioncraft_py.Viewpoint()
    # initial_viewpoint.setPosition(initial_position)
    # initial_viewpoint.setLookAt(look_at, up_vector)
    # initial_viewpoint.setDownsampleFactor(4.0)

    # # Append the initial viewpoint to the viewpoints list
    # all_nbvs.append(initial_viewpoint)

    # # Update simulator with the initial viewpoint
    # simulator.setViewpoints(all_nbvs)
    # simulator.performRaycasting()
    # coverage_score = simulator.getCoverageScore()
    # print(f"Initial Coverage Score: {coverage_score}")

    for iteration in range(CONFIG['max_iterations']):
        print(f"Iteration: {iteration + 1}")

        # Get the exploration map data from the simulator
        exploration_map_data = simulator.getExplorationMapData()
     
        cube = convert_exploration_map_to_cube(exploration_map_data)
       
        output = predict_nbv(cube, net)
        class_nbv = np.unravel_index(np.argmax(output), output.shape)
        bar_nbv = get_positions(class_nbv, nbv_positions) * 5
        bar_nbv = bar_nbv.squeeze()
        
        import pdb; pdb.set_trace()
        rnbv.showGrid(cube, bar_nbv, bar_nbv)

        full_nbv = compute_full_nbv(bar_nbv)
        scaled_nbv = full_nbv[:3] / scaling_factor

        # Create a Viewpoint object
        position = scaled_nbv
        look_at = np.array([0, 0, 0]).reshape(3, 1)
        up_vector = np.array([0, 0, -1]).reshape(3, 1)

        viewpoint = visioncraft_py.Viewpoint()
        viewpoint.setPosition(position)
        viewpoint.setLookAt(look_at, up_vector)
        viewpoint.setDownsampleFactor(4.0)

        # Append to viewpoints
        all_nbvs.append(viewpoint)

        # Update simulator with new viewpoints
        simulator.setViewpoints(all_nbvs)
        simulator.performRaycasting()

        # Optionally, get the coverage score
        coverage_score = simulator.getCoverageScore()
        print(f"Coverage Score: {coverage_score}")

    print("Process completed.")


if __name__ == "__main__":
    main()
