# config/sim_config.py
"""
Simulation Configuration
------------------------
Central configuration file for overlap_sim parameters.

Organizes constants for:
- Grid and simulation timing
- Physical field parameters
- Visualization and randomness
"""

from dataclasses import dataclass

@dataclass
class SimConfig:
    # --- Grid and Simulation ---
    grid_size: int = 100
    num_viewpoints: int = 8
    num_steps: int = 300
    step_size: float = 0.02 # step for vanilla optimizer
    seed: int = 0

    # Viewpoint insertion control
    insertion_mode = "manual"          # "manual", "auto_fixed", or "auto_adaptive"
    insert_interval = 50               # for auto_fixed
    velocity_thresh = 0.02             # for auto_adaptive
    stagnation_steps = 10              # for auto_adaptive


    # --- Optimizer ---
    optimizer: str = "vanilla"   # "vanilla" or "adam"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-3
    adam_gain: float = 1.0       # step gain scaling (like learning rate)


    # --- Physics Parameters ---
    k_attr: float = 10.0
    k_rep: float = 0.25
    sigma_rep: float = 20.0
    amp_rep: float = 300.0

    # --- Visibility / Need Map ---
    fov_radius: float = 20.0
    beta_vis: float = 20.0

    # --- Viewer Settings ---
    target_fps: int = 60
    show_vectors: bool = True
    enable_viewer: bool = True

    record_video = False
    video_path = "./videos/sample.mp4"

