# utils/wrap.py
import numpy as np

def wrap_distance(dx, size):
    """Shortest wrapped displacement on a torus of given size."""
    half = size / 2.0
    return np.where(np.abs(dx) > half, -np.sign(dx) * (size - np.abs(dx)), dx)
