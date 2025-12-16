# simulation/insertion_manager.py
"""
Viewpoint Insertion Manager
---------------------------
Controls when and how new viewpoints are added during the simulation.

Modes:
- manual: triggered by user input (via viewer)
- auto_fixed: add a viewpoint every `insert_interval` steps
- auto_adaptive: add when motion stagnates below velocity threshold

New viewpoint is placed at the global maximum of the current potential field.
"""

import numpy as np


class InsertionManager:
    def __init__(self, mode="manual", insert_interval=50,
                 velocity_thresh=0.02, stagnation_steps=10):
        """
        Parameters
        ----------
        mode : str
            'manual', 'auto_fixed', or 'auto_adaptive'
        insert_interval : int
            Add a viewpoint every N steps (for auto_fixed)
        velocity_thresh : float
            Threshold below which motion is considered stagnant (for auto_adaptive)
        stagnation_steps : int
            Number of consecutive low-motion steps before triggering adaptive insertion
        """
        self.mode = mode
        self.insert_interval = insert_interval
        self.velocity_thresh = velocity_thresh
        self.stagnation_steps = stagnation_steps

        self.step_counter = 0
        self.low_motion_count = 0

    # ----------------------------------------------------------
    def check_insertion(self, step, viewpoints, potential):
        """
        Determine if a new viewpoint should be added.

        Returns
        -------
        bool : True if new viewpoint should be inserted
        """
        self.step_counter += 1

        # --- manual mode (only triggered externally)
        if self.mode == "manual":
            return False

        # --- fixed interval mode
        if self.mode == "auto_fixed":
            if step % self.insert_interval == 0:
                return True
            return False

        # --- adaptive mode
        if self.mode == "auto_adaptive":
            if not hasattr(viewpoints, "velocities"):
                return False

            vel_mag = np.linalg.norm(viewpoints.velocities, axis=1)
            mean_vel = np.mean(vel_mag)

            if mean_vel < self.velocity_thresh:
                self.low_motion_count += 1
            else:
                self.low_motion_count = 0

            if self.low_motion_count >= self.stagnation_steps:
                self.low_motion_count = 0
                print(f"[InsertionManager] Adaptive trigger (mean velocity={mean_vel:.3f})")
                return True

        return False

    # ----------------------------------------------------------
    @staticmethod
    def select_best_insertion(potential, grid_size):
        """
        Selects the best position (x, y) for viewpoint insertion
        based on the global maximum of the potential map.
        """
        if potential is None or potential.size == 0:
            # fallback: center of grid
            return np.array([grid_size / 2, grid_size / 2])

        idx_flat = np.argmax(potential)
        iy, ix = np.unravel_index(idx_flat, potential.shape)
        return np.array([ix, iy], dtype=np.float64)
