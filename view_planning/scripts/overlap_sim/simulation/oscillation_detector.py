# simulation/oscillation_detector.py
import numpy as np

SMALL = 1e-6

def _finite_diff_force(sim, X, need_map, eps=1e-3):
    """
    Finite-difference Jacobian of F(X) wrt X (flattened 2N vector).
    Hold need_map fixed for the probe to get a meaningful local linearization.
    Returns J: (2N x 2N)
    """
    N = X.shape[0]
    base = sim.forces_given_positions(X, need_map=need_map)  # (N,2)
    f0 = base.reshape(-1)

    J = np.zeros((2*N, 2*N), dtype=np.float64)
    for j in range(2*N):
        Xp = X.copy()
        id_pt, id_dim = divmod(j, 2)
        Xp[id_pt, id_dim] = (Xp[id_pt, id_dim] + eps) % sim.grid_size
        fp = sim.forces_given_positions(Xp, need_map=need_map)

        # --- Ensure CPU (NumPy) arrays ---
        try:
            import cupy as cp
            if isinstance(fp, cp.ndarray):
                fp = cp.asnumpy(fp)
            if isinstance(f0, cp.ndarray):
                f0 = cp.asnumpy(f0)
        except ImportError:
            pass

        fp = fp.reshape(-1)
        J[:, j] = (fp - f0) / eps

    return J


def discrete_map_spectral_radius(J, step_size, beta=None):
    """
    Spectral radius of the linearized discrete update.
    - No momentum (Euler): x_{k+1} = x_k + step * F(x_k) => A = I + step*J
    - With momentum: v_{k+1} = beta v_k + (1-beta) F(x_k)
                      x_{k+1} = x_k + step * v_{k+1}
      State s = [x; v] in R^{4N}, build the companion linear map.

    Returns rho (float).
    """
    n2 = J.shape[0]
    I = np.eye(n2)

    if beta is None:
        A = I + step_size * J
        vals = np.linalg.eigvals(A)
        return np.max(np.abs(vals)).real

    # momentum case
    b = float(beta)
    # s_{k+1} = [ x + step*( b v + (1-b) F ) ; b v + (1-b) F ]
    # Linearize F ≈ J δx
    Z = np.zeros_like(J)
    # block matrix:
    # [ I + step*(1-b) J,   step*b I ]
    # [    (1-b) J       ,     b I   ]
    A11 = I + step_size * (1.0 - b) * J
    A12 = step_size * b * I
    A21 = (1.0 - b) * J
    A22 = b * I
    A = np.block([[A11, A12],
                  [A21, A22]])
    vals = np.linalg.eigvals(A)
    return np.max(np.abs(vals)).real


def local_instability_now(sim, X, need_map, step_size, beta=None, eps=1e-3, rho_thresh=1.0 + 1e-3):
    """
    Returns (bool, rho): True if spectral radius > 1 (local discrete-time instability).
    """
    J = _finite_diff_force(sim, X, need_map, eps=eps)
    rho = discrete_map_spectral_radius(J, step_size, beta=beta)
    return (rho > rho_thresh), float(rho)


# --------- lightweight time-series validator (optional, robust) ---------------

class OscillationTimeSeries:
    """
    Keep a short buffer of velocities; detect narrowband nonzero-frequency energy.
    Use alongside the local test to avoid false positives.
    """
    def __init__(self, N_viewpoints, buf_len=60):
        self.buf_len = int(buf_len)
        self.N = int(N_viewpoints)
        self.v_hist = np.zeros((self.buf_len, self.N, 2), dtype=np.float64)
        self.t = 0

    def push(self, velocities):
        velocities = np.asarray(velocities)
        n_new = velocities.shape[0]

        # Resize buffer if number of viewpoints changed
        if n_new != self.N:
            new_hist = np.zeros((self.buf_len, n_new, 2), dtype=np.float64)
            # Copy overlapping portion (min of old/new)
            minN = min(self.N, n_new)
            new_hist[:, :minN, :] = self.v_hist[:, :minN, :]
            self.v_hist = new_hist
            self.N = n_new
            print(f"[OscillationTimeSeries] Resized buffer for {n_new} viewpoints")

        i = self.t % self.buf_len
        self.v_hist[i] = velocities
        self.t += 1


    def is_oscillating(self, min_steps=30, power_ratio_thresh=4.0):
        if self.t < max(min_steps, self.buf_len // 2):
            return False
        # stack speed per viewpoint over time
        V = self.v_hist.copy()
        # detrend
        V -= V.mean(axis=0, keepdims=True)
        # power spectrum on speed magnitude
        spd = np.linalg.norm(V, axis=2)  # (T, N)
        # FFT along time axis
        S = np.fft.rfft(spd, axis=0)  # (F, N)
        P = (np.abs(S) ** 2)
        # ignore DC (bin 0), find peak/median ratio
        Pnz = P[1:]  # drop DC
        peak = Pnz.max(axis=0) + 1e-12
        median = np.median(Pnz, axis=0) + 1e-12
        ratios = peak / median
        # oscillation if enough viewpoints show narrowband peak
        frac = np.mean(ratios > power_ratio_thresh)
        return frac > 0.3  # >30% agents oscillating
