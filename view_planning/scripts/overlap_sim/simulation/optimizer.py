# simulation/optimizer.py
"""
Optimizers for viewpoint motion
-------------------------------
Supports:
- Vanilla gradient ascent
- Adam optimizer (with dynamic resizing for added viewpoints)
"""

try:
    import cupy as np
    GPU = True
except ImportError:
    import numpy as np
    GPU = False


class Optimizer:
    def __init__(self, name="vanilla", step_size=0.01,
                 beta1=0.9, beta2=0.999, eps=1e-8, gain=5.0):
        """
        Parameters
        ----------
        name : str
            "vanilla" or "adam"
        step_size : float
            Base step size (learning rate)
        beta1, beta2 : float
            Adam momentum coefficients
        eps : float
            Numerical stability term
        gain : float
            Global scaling factor for Adam step magnitude
        """
        self.name = name.lower()
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.gain = gain

        # State for Adam optimizer
        self.m = None
        self.v = None
        self.t = 0

    # --------------------------------------------------------
    def reset(self, shape):
        """Reset internal moment estimates to zero."""
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    # --------------------------------------------------------
    def _resize_state(self, forces):
        """Ensure optimizer buffers (m, v) match the shape of current forces."""
        if self.m is None or self.v is None:
            self.reset(forces.shape)
            return

        old_n, new_n = self.m.shape[0], forces.shape[0]
        if new_n == old_n:
            return  # nothing to do

        if new_n > old_n:
            # Add zero momentum entries for new viewpoints
            extra_m = np.zeros((new_n - old_n, self.m.shape[1]), dtype=self.m.dtype)
            extra_v = np.zeros((new_n - old_n, self.v.shape[1]), dtype=self.v.dtype)
            self.m = np.concatenate([self.m, extra_m], axis=0)
            self.v = np.concatenate([self.v, extra_v], axis=0)
            print(f"[Optimizer] Expanded from {old_n}→{new_n} viewpoints (auto-resize)")
        else:
            # Viewpoints removed (trim buffers)
            self.m = self.m[:new_n]
            self.v = self.v[:new_n]
            print(f"[Optimizer] Trimmed from {old_n}→{new_n} viewpoints")

    # --------------------------------------------------------
    def step(self, pos, forces):
        """Update viewpoint positions according to selected optimizer."""
        # --- convert all to same backend type ---
        if GPU:
            pos = np.asarray(pos)
            forces = np.asarray(forces)

        # ------------------ Vanilla Gradient Ascent ------------------
        if self.name == "vanilla":
            return pos + self.step_size * forces

        # ------------------ Adam Optimizer ------------------
        elif self.name == "adam":
            # Resize internal state if number of viewpoints changed
            self._resize_state(forces)

            self.t += 1
            self.m = self.beta1 * self.m + (1 - self.beta1) * forces
            self.v = self.beta2 * self.v + (1 - self.beta2) * (forces ** 2)

            # Bias-corrected moment estimates
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)

            step = self.gain * m_hat / (np.sqrt(v_hat) + self.eps)
            return pos + step

        # ------------------ Invalid mode ------------------
        else:
            raise ValueError(f"Unknown optimizer '{self.name}'")
