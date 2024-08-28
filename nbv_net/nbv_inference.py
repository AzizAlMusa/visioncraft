import os
import time
import numpy as np
import torch
from torch.autograd import Variable
import regression_nbv_utils as rnbv
import temp as visualizer
import nbvnet

# Determine the root directory of the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Global Configuration
CONFIG = {
    'octomap_file': os.path.join(SCRIPT_DIR, 'data', 'exploration_map.txt'),
    'octomap_numpy_file': os.path.join(SCRIPT_DIR, 'data', 'exploration_map.npy'),
    'voxel_resolution': 0.025,
    'dropout': 0.2,
    'model_weights_file': os.path.join(SCRIPT_DIR, 'model', 'weights.pth'),
    'nbv_positions_file': os.path.join(SCRIPT_DIR, 'points_in_sphere.txt'),
    'nbv_txt_file': os.path.join(SCRIPT_DIR, 'log', 'nbv.txt'),
    'all_nbvs_txt_file': os.path.join(SCRIPT_DIR, 'log', 'all_nbvs.txt'),
    'scale_factor_file': os.path.join(SCRIPT_DIR, '..', 'view_planning', 'data', 'scaling_factor.txt'),
    'object_model_path': os.path.join(SCRIPT_DIR, '..', 'view_planning', 'models', 'cube.ply'),
    'cpp_simulator_path': os.path.join(SCRIPT_DIR, '..', 'view_planning', 'build', 'exploration_view_tests'),
    'max_iterations': 10
}

# Check for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def initialize_files():
    """Ensure necessary files are created."""
    if not os.path.exists(CONFIG['all_nbvs_txt_file']):
        with open(CONFIG['all_nbvs_txt_file'], 'w') as f:
            pass

def load_model():
    """Build and load the NBV-net model."""
    net = nbvnet.NBV_Net(CONFIG['dropout'])
    net.to(device)
    net.eval()
    net.load_state_dict(torch.load(CONFIG['model_weights_file'], map_location='cpu'), strict=False)
    return net

def load_nbv_positions():
    """Load positions for NBV calculation."""
    return np.genfromtxt(CONFIG['nbv_positions_file'])

def load_scale_factor():
    """Load the scale factor from the file."""
    with open(CONFIG['scale_factor_file'], 'r') as f:
        scale_factor = float(f.read().strip())
    print(f"Loaded scale factor: {scale_factor}")
    return scale_factor

def get_positions(nbv_class, positions):
    """Convert class to corresponding pose."""
    return np.array(positions[nbv_class])

def read_octomap_as_cube(octomap_file, voxel_resolution, map_shape=[32, 32, 32]):
    """Read octomap and convert it to a voxel grid."""
    raw_data = np.loadtxt(octomap_file, dtype=float)
    pmin = np.array([np.min(raw_data[:, 0]), np.min(raw_data[:, 1]), np.min(raw_data[:, 2])])
    positions = raw_data[:, 0:3]
    occupancy = raw_data[:, 3]
    
    if len(occupancy) != map_shape[0] * map_shape[1] * map_shape[2]:
        print(f"WARNING: {len(occupancy)} read elements. {map_shape[0] * map_shape[1] * map_shape[2]} expected")
    
    idx = (positions[:, 0] - pmin[0]) / voxel_resolution
    idy = (positions[:, 1] - pmin[1]) / voxel_resolution
    idz = (positions[:, 2] - pmin[2]) / voxel_resolution
    idx = idx.astype(np.int32)
    idy = idy.astype(np.int32)
    idz = idz.astype(np.int32)
    
    output_cube = np.zeros(map_shape)
    for i, p in enumerate(occupancy):
        output_cube[idx[i], idy[i], idz[i]] = p
    
    return output_cube

def predict_nbv(cube, net):
    """Perform a forward pass through the network and predict the NBV."""
    grids = torch.from_numpy(np.array([[cube]])).type(torch.FloatTensor)
    grids = Variable(grids).to(device)
    
    start = time.time()
    output = net.forward(grids)
    end = time.time()
    
    print("Processing time:", end - start)
    output = output.cpu().detach().numpy().squeeze()
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

def save_nbv(full_nbv, scale_factor):
    """Save the new NBV to a file with 6 numbers: x, y, z positions and 0, 0, 0 for lookAt."""
    # Scale the NBV positions by the scaling factor before saving
    scaled_nbv = full_nbv[:3] / scale_factor
    full_nbv_with_lookat = np.concatenate((scaled_nbv, np.array([0, 0, 0])))
    
    # Save to the single NBV file
    np.savetxt(CONFIG['nbv_txt_file'], [full_nbv_with_lookat], fmt='%.6f')

    # Append to the all NBVs file, ensuring it is added as a new line
    with open(CONFIG['all_nbvs_txt_file'], 'a') as f:
        f.write(' '.join(f'{num:.6f}' for num in full_nbv_with_lookat) + '\n')


def run_simulator():
    """Run the external C++ simulator."""
    os.system(f"{CONFIG['cpp_simulator_path']} {CONFIG['object_model_path']} {CONFIG['all_nbvs_txt_file']}")

def main():
    """Main loop for iterative NBV prediction and simulation."""
    initialize_files()
    net = load_model()
    nbv_positions = load_nbv_positions()
    
    # Run the initial simulation to generate the first exploration map
    run_simulator()

    # Load the scaling factor
    scale_factor = load_scale_factor()

    for iteration in range(CONFIG['max_iterations']):
        print(f"Iteration: {iteration + 1}")
        cube = read_octomap_as_cube(CONFIG['octomap_file'], CONFIG['voxel_resolution'])
        np.save(CONFIG['octomap_numpy_file'], cube)
        # visualizer.visualizeGrid(CONFIG['octomap_file'])

        output = predict_nbv(cube, net)
        class_nbv = np.where(output == np.amax(output))
        bar_nbv = get_positions(class_nbv, nbv_positions) * 10
        bar_nbv = bar_nbv.squeeze()

        # visualizer.visualizeGrid(CONFIG['octomap_file'], bar_nbv, bar_nbv)

        full_nbv = compute_full_nbv(bar_nbv)
        save_nbv(full_nbv, scale_factor)
        
        run_simulator()

    print("Process completed.")

if __name__ == "__main__":
    main()
