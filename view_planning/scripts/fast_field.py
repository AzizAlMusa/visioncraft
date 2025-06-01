# === fast_field.py ============================================================
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from numba import jit, prange
from scipy.spatial import cKDTree

# -------------------------- helper: FFT kernel --------------------------------
def _make_radial_kernel(grid_size, fov_radius, beta):
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    X, Y = np.meshgrid(x, y, indexing='ij')
    dx = np.minimum(X, grid_size - X)
    dy = np.minimum(Y, grid_size - Y)
    r = np.hypot(dx, dy)
    return (1.0 / (1.0 + np.exp(beta * (r / fov_radius - 1.0)))).astype(np.float64)

# ------------------------------------------------------------------------------
class Field:
    """
    Fast drop-in replacement for the old Field class.
    Public API is identical; results are identical up to floating-point noise.
    """
    def __init__(self, grid_size, fov_radius, beta=20.0):
        self.grid_size  = grid_size
        self.fov_radius = fov_radius
        self.beta       = beta

        self.visibility = np.zeros((grid_size, grid_size), np.float64)
        self.potential  = np.zeros_like(self.visibility)

        # --- FFT pre-compute --------------------------------------------------
        self._K     = _make_radial_kernel(grid_size, fov_radius, beta)
        self._K_hat = fft2(self._K)
        self._fx, self._fy = self._make_grad_freqs()

        # --- caches & helpers -------------------------------------------------
        self._view_map  = np.zeros_like(self.visibility)
        self._V_last    = np.empty((0, 2))
        self._tree      = None
        self._rep_sigma = 10.0
        self._rep_amp   = 100.0

        # --- compatibility helpers (added) -----------------------------------
        xs = np.arange(grid_size)
        ys = np.arange(grid_size)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        self.field_points = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float64)

    # -------------------------------------------------------------------------
    # Public API (same signatures as before)
    def update_visibility(self, particles):
        self._ensure_view_map(particles)
        V_hat = fft2(self._view_map)
        self.visibility = np.real(ifft2(V_hat * self._K_hat))
        np.clip(self.visibility, 0.0, 1.0, out=self.visibility)

    def compute_potential(self, particles, alpha=1.0):
        self.potential = alpha * (1.0 - self.visibility)

    def compute_force(self, particles, k_attr=0.4, k_rep=1.0, alpha=1.0):
        V_hat = fft2(self._view_map)
        vis_hat = V_hat * self._K_hat
        dVdx = np.real(ifft2(1j * self._fx * vis_hat))
        dVdy = np.real(ifft2(1j * self._fy * vis_hat))

        idx = np.rint(particles).astype(int) % self.grid_size
        grad = np.stack((dVdx[idx[:, 0], idx[:, 1]],
                         dVdy[idx[:, 0], idx[:, 1]]), axis=1)
        attr = -alpha * grad
        rep  = self._compute_repulsion(particles)
        return k_attr * attr + k_rep * rep

    def compute_coverage_monte_carlo(self, particles, num_samples=10000,
                                     threshold=0.25):
        return _mc_cov_fast(particles, self.grid_size, self.fov_radius ** 2,
                            num_samples, threshold)

    # -------------------------------------------------------------------------
    # Compatibility helper for post-processing code
    def wrap_distance(self, diff):
        L = self.grid_size
        return (diff + L * 0.5) % L - L * 0.5

    # -------------------------------------------------------------------------
    # Internal helpers
    def _ensure_view_map(self, particles):
        if particles.shape == self._V_last.shape and np.allclose(particles, self._V_last):
            return
        self._view_map.fill(0.0)
        idx = np.rint(particles).astype(int) % self.grid_size
        self._view_map[idx[:, 0], idx[:, 1]] = 1.0
        self._V_last = particles.copy()
        self._tree   = cKDTree(particles)

    def _compute_repulsion(self, particles):
        if self._tree is None:
            return np.zeros_like(particles)

        pairs = self._tree.query_pairs(self._rep_sigma, output_type='set')
        rep = np.zeros_like(particles)
        σ2 = self._rep_sigma ** 2
        for i, j in pairs:
            dx = self._torus_delta(particles[i, 0], particles[j, 0])
            dy = self._torus_delta(particles[i, 1], particles[j, 1])
            d2 = dx * dx + dy * dy
            if d2 < 1e-9:
                continue
            coeff = self._rep_amp * np.exp(-d2 / (2 * σ2))
            f = coeff * np.array([dx, dy]) / np.sqrt(d2)
            rep[i] +=  f
            rep[j] += -f
        return rep

    def _torus_delta(self, a, b):
        d = a - b
        half = self.grid_size * 0.5
        if d >  half: d -= self.grid_size
        if d < -half: d += self.grid_size
        return d

    def _make_grad_freqs(self):
        k = 2 * np.pi * fftfreq(self.grid_size)[:, None]
        return k, k.T

# -------------------- Numba-accelerated Monte-Carlo ---------------------------
@jit(nopython=True, fastmath=True, parallel=True)
def _mc_cov_fast(particles, L, r2, N, thresh):
    M = particles.shape[0]
    covered = 0
    for s in prange(N):
        x = np.random.random() * L
        y = np.random.random() * L
        visible = 0
        for j in range(M):
            dx = x - particles[j, 0]
            dy = y - particles[j, 1]
            if dx > L/2: dx -= L
            if dx < -L/2: dx += L
            if dy > L/2: dy -= L
            if dy < -L/2: dy += L
            if dx * dx + dy * dy <= r2:
                visible += 1
            if visible > thresh:
                covered += 1
                break
    return covered / N
