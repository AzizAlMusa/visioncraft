# entities/viewpoint.py
"""
Viewpoint Entity
----------------
Defines the viewpoint structure that interacts with
the potential and repulsion fields.

Author: Abdulaziz (overlap_sim)
"""

try:
    import cupy as np
    GPU = True
except ImportError:
    import numpy as np
    GPU = False


class Viewpoints:
    def __init__(self, num=1, grid_size=100, seed=0, overlap_test=False):
        np.random.seed(seed)
        self.grid_size = grid_size

        # ============================================================
        # TEMPORARY TEST MODE: create two overlapping viewpoints
        # ============================================================
        if overlap_test and num == 2:
            # Center of the grid
            cx, cy = grid_size / 2.0, grid_size / 2.0

            # Overlap separation = 10% of FOV (approx) = 0.1 * grid_size / 10
            # Or equivalently, 10% of the grid dimension normalized.
            delta = 0.1 * grid_size / 2.0

            # Two viewpoints slightly offset along x-axis
            self.positions = np.array([
                [cx - delta, cy],
                [cx + delta, cy]
            ], dtype=np.float64)
        else:
            # Default random initialization
            self.positions = np.random.rand(num, 2) * grid_size

        # Initialize velocities
        self.velocities = np.zeros_like(self.positions)

        # ✅ Convert to CuPy arrays if GPU available
        if GPU:
            self.positions = np.asarray(self.positions)
            self.velocities = np.asarray(self.velocities)

        self.history = [self.positions.copy()]


    def step(self, forces, step_size=0.1):
        """Update viewpoint positions based on force field."""
        if len(forces) != len(self.positions):
            raise ValueError("forces shape mismatch")

        self.velocities = forces
        self.positions = (self.positions + step_size * self.velocities) % self.grid_size
        self.history.append(self.positions.copy())

    def add_viewpoint(self, position):
        """Add a new viewpoint (e.g., NBV insertion)."""
        if GPU:
            # Ensure GPU arrays remain consistent
            position = np.asarray(position)
            self.positions = np.concatenate([self.positions, position[None, :]], axis=0)
            zeros = np.zeros((1, 2), dtype=self.positions.dtype)
            self.velocities = np.concatenate([self.velocities, zeros], axis=0)
        else:
            self.positions = np.vstack([self.positions, position[None, :]])
            self.velocities = np.vstack([self.velocities, np.zeros((1, 2))])

    def __len__(self):
        return len(self.positions)
