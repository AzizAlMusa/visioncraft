# overlap_sim/map/base_map.py
"""
BaseMap
-------
Provides the spatial grid (X, Y) for the simulation domain.

Currently this is a uniform importance map (value = 1 everywhere),
but later it can be extended to include POI / edge weights.
"""

try:
    import cupy as np
    GPU = True
except ImportError:
    import numpy as np
    GPU = False


class BaseMap:
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.values = np.ones((grid_size, grid_size), dtype=np.float64)

        # Create meshgrid
        self.X, self.Y = np.meshgrid(
            np.arange(grid_size), np.arange(grid_size)
        )

        # ✅ Ensure CuPy arrays if GPU is available
        if GPU:
            self.X = np.asarray(self.X)
            self.Y = np.asarray(self.Y)
            self.values = np.asarray(self.values)

    def get_meshgrid(self):
        """Return the coordinate grids."""
        return self.X, self.Y

    def get_values(self):
        """Return the base importance map (uniform ones)."""
        return self.values
