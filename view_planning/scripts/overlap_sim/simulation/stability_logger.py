# simulation/stability_logger.py

import os
import csv
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def _to_numpy(arr):
    """Convert NumPy/CuPy array (or list-like) to a NumPy float64 array."""
    if HAS_CUPY and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr).astype(np.float64)
    return np.asarray(arr, dtype=np.float64)


# ---------------------------------------------------------------------
# Global state for convergence + Lyapunov + reference tracking
# ---------------------------------------------------------------------

_prev_pos = None       # positions at previous step
_prev_step = None

_lyap_log_sum = 0.0    # running sum of log(spectral_radius)
_lyap_samples = 0      # how many times we've updated Lyap

# simple convergence detector
DELTA_POS_RMS_TOL = 1e-3   # grid units per step (RMS over viewpoints)
FORCE_MEAN_TOL    = 1e-3   # mean |F| per viewpoint
CONV_WINDOW       = 25     # consecutive steps under both thresholds

_conv_counter   = 0
_converged_step = None

# reference configuration once we first declare convergence
_ref_pos  = None
_ref_step = None


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------

def torus_spacing_metrics(positions, grid_size):
    """
    Compute pairwise spacing statistics on a 2D torus of side length `grid_size`.
    Returns min / mean / max spacing.
    """
    pos = _to_numpy(positions)
    N = pos.shape[0]
    if N < 2:
        return {
            "min_spacing": np.nan,
            "mean_spacing": np.nan,
            "max_spacing": np.nan,
        }

    dists = []
    half = grid_size / 2.0
    for i in range(N):
        for j in range(i + 1, N):
            delta = pos[i] - pos[j]
            # Wrap to shortest displacement on each axis
            for axis in range(2):
                if delta[axis] > half:
                    delta[axis] -= grid_size
                elif delta[axis] < -half:
                    delta[axis] += grid_size
            d = float(np.linalg.norm(delta))
            dists.append(d)

    dists = np.asarray(dists, dtype=np.float64)
    return {
        "min_spacing": float(dists.min()),
        "mean_spacing": float(dists.mean()),
        "max_spacing": float(dists.max()),
    }


def _torus_delta(current, reference, grid_size):
    """
    Shortest displacement current - reference on a 2D torus.
    Both arrays are (N,2). Returns (N,2).
    """
    delta = current - reference
    half = grid_size / 2.0
    for axis in range(2):
        da = delta[:, axis]
        da[da >  half] -= grid_size
        da[da < -half] += grid_size
        delta[:, axis] = da
    return delta


# ---------------------------------------------------------------------
# Jacobian + eigen diagnostics
# ---------------------------------------------------------------------

def numerical_jacobian(force_fn, positions, eps=1e-3):
    """
    Central-difference Jacobian of the continuous-time vector field F(V).

    Parameters
    ----------
    force_fn : callable
        Function taking positions (N,2) as NumPy array and returning forces (N,2).
    positions : array-like, shape (N, 2)
        Current viewpoint positions on the torus.
    eps : float
        Finite-difference step.

    Returns
    -------
    J : ndarray, shape (2N, 2N)
        Jacobian dF/dV flattened as usual (x1,y1,x2,y2,...).
    """
    V0 = _to_numpy(positions)
    N, d = V0.shape
    assert d == 2, "This helper assumes 2D positions per viewpoint."

    x0 = V0.reshape(-1)  # (2N,)
    D = x0.size
    J = np.zeros((D, D), dtype=np.float64)

    for k in range(D):
        dx = np.zeros_like(x0)
        dx[k] = eps

        x_plus = (x0 + dx).reshape(N, d)
        x_minus = (x0 - dx).reshape(N, d)

        f_plus = _to_numpy(force_fn(x_plus)).reshape(-1)
        f_minus = _to_numpy(force_fn(x_minus)).reshape(-1)

        J[:, k] = (f_plus - f_minus) / (2.0 * eps)

    return J


def eig_metrics(J, eta):
    """
    Compute continuous- and discrete-time stability diagnostics from Jacobian.

    Parameters
    ----------
    J : ndarray, shape (2N, 2N)
        Jacobian of continuous-time field F(V).
    eta : float
        Integration step size (Euler).

    Returns
    -------
    metrics : dict
        Contains max / min real parts of eigenvalues, spectral radius of I+eta J,
        and boolean stability flags.
    """
    eigvals = np.linalg.eigvals(J)
    real_parts = eigvals.real

    lambda_max_real = float(real_parts.max())
    lambda_min_real = float(real_parts.min())

    # eigenvalues of the one-step Euler map M(V) = V + eta F(V)
    mu = 1.0 + eta * eigvals
    spectral_radius = float(np.abs(mu).max())

    continuous_stable = bool(lambda_max_real < 0.0)
    discrete_stable   = bool(spectral_radius < 1.0)

    return {
        "lambda_max_real": lambda_max_real,
        "lambda_min_real": lambda_min_real,
        "spectral_radius_I_plus_etaJ": spectral_radius,
        "continuous_stable": continuous_stable,
        "discrete_stable": discrete_stable,
        "num_eig_pos_real": int((real_parts > 0.0).sum()),
    }


# ---------------------------------------------------------------------
# Main per-step logger
# ---------------------------------------------------------------------

def log_step_metrics(
    step,
    positions,
    forces,
    potential,
    need_map,
    cfg,
    force_fn,
    csv_path="./logs/stability_metrics.csv",
    eig_every=50,
    jacobian_eps=1e-3,
    print_to_console=True,
):
    """
    Generic per-step stability logger.

    Parameters
    ----------
    step : int
        Current simulation step.
    positions : array-like, shape (N,2)
        Current viewpoint positions (NumPy or CuPy).
    forces : array-like, shape (N,2)
        Current forces F(V) (NumPy or CuPy).
    potential : array-like
        Potential grid.
    need_map : array-like
        Current need map.
    cfg : SimConfig-like
        Holds grid_size, step_size, etc.
    force_fn : callable
        Function positions -> forces, used for Jacobian FD.
    csv_path : str
        Where to append CSV data.
    eig_every : int
        Compute Jacobian / eigenvalues every this many steps.
    jacobian_eps : float
        Finite-difference epsilon for Jacobian.
    print_to_console : bool
        If True, print the [Stab] line when eigen metrics are computed.

    Returns
    -------
    row : dict
        Metrics written for this step.
    """
    global _prev_pos, _prev_step
    global _lyap_log_sum, _lyap_samples
    global _conv_counter, _converged_step
    global _ref_pos, _ref_step

    pos_np = _to_numpy(positions)
    forces_np = _to_numpy(forces)
    pot_np = _to_numpy(potential)
    need_np = _to_numpy(need_map)

    N = pos_np.shape[0]
    grid_size = float(getattr(cfg, "grid_size", 1.0))

    # --- spacing stats -------------------------------------------------------
    spacing = torus_spacing_metrics(pos_np, grid_size)

    # --- force stats ---------------------------------------------------------
    if N > 0:
        fnorm = np.linalg.norm(forces_np, axis=1)
        mean_force = float(fnorm.mean())
        max_force = float(fnorm.max())
    else:
        mean_force = np.nan
        max_force = np.nan

    # --- potential / need stats ---------------------------------------------
    pot_mean = float(pot_np.mean())
    pot_min = float(pot_np.min())
    pot_max = float(pot_np.max())

    need_mean = float(need_np.mean())
    need_min = float(need_np.min())
    need_max = float(need_np.max())

    # --- delta-pos between consecutive steps (on torus) ----------------------
    delta_pos_rms = np.nan
    delta_pos_max = np.nan

    if (
        _prev_pos is not None
        and _prev_step is not None
        and _prev_step == step - 1
        and _prev_pos.shape == pos_np.shape
    ):
        delta = _torus_delta(pos_np, _prev_pos, grid_size)
        norms = np.linalg.norm(delta, axis=1)
        delta_pos_rms = float(np.sqrt((norms ** 2).mean()))
        delta_pos_max = float(norms.max())

    _prev_pos = pos_np.copy()
    _prev_step = int(step)

    # --- convergence detection ----------------------------------------------
    if (
        not np.isnan(delta_pos_rms)
        and mean_force < FORCE_MEAN_TOL
        and delta_pos_rms < DELTA_POS_RMS_TOL
    ):
        _conv_counter += 1
        if _converged_step is None and _conv_counter >= CONV_WINDOW:
            _converged_step = int(step)
            _ref_pos = pos_np.copy()
            _ref_step = int(step)
    else:
        _conv_counter = 0

    converged_flag = int(_converged_step is not None)
    converged_step = _converged_step if _converged_step is not None else -1

    # --- distance to first converged configuration --------------------------
    dist_ref_rms = np.nan
    dist_ref_max = np.nan
    if _ref_pos is not None and _ref_pos.shape == pos_np.shape:
        delta_ref = _torus_delta(pos_np, _ref_pos, grid_size)
        norms_ref = np.linalg.norm(delta_ref, axis=1)
        dist_ref_rms = float(np.sqrt((norms_ref ** 2).mean()))
        dist_ref_max = float(norms_ref.max())

    # --- Jacobian / eigen metrics -------------------------------------------
    lambda_max_real = np.nan
    lambda_min_real = np.nan
    spectral_radius = np.nan
    continuous_stable = False
    discrete_stable = False
    num_eig_pos_real = 0
    lyap_inst = np.nan
    lyap_avg = np.nan

    if eig_every > 0 and (step % eig_every == 0) and N > 0:
        J = numerical_jacobian(force_fn, pos_np, eps=jacobian_eps)
        em = eig_metrics(J, float(getattr(cfg, "step_size", 1.0)))

        lambda_max_real = em["lambda_max_real"]
        lambda_min_real = em["lambda_min_real"]
        spectral_radius = em["spectral_radius_I_plus_etaJ"]
        continuous_stable = em["continuous_stable"]
        discrete_stable = em["discrete_stable"]
        num_eig_pos_real = em["num_eig_pos_real"]

        # Lyapunov estimate from spectral radius
        lyap_inst = float(np.log(max(spectral_radius, 1e-12)))
        _lyap_log_sum += lyap_inst
        _lyap_samples += 1
        lyap_avg = _lyap_log_sum / max(_lyap_samples, 1)

        if print_to_console:
            cts_str = "OK" if continuous_stable else "UNST"
            disc_str = "OK" if discrete_stable else "UNST"
            print(
                f"[Stab] step={step:04d}  N={N}  "
                f"min_d={spacing['min_spacing']:.2f}  "
                f"maxReλ={lambda_max_real:.3f}  "
                f"ρ(I+ηJ)={spectral_radius:.3f}  "
                f"cts={cts_str}  disc={disc_str}  "
                f"Lyap_avg={lyap_avg:.3f}"
            )

    # --- assemble row --------------------------------------------------------
    row = {
        "step": int(step),
        "N": int(N),
        "min_spacing": spacing["min_spacing"],
        "mean_spacing": spacing["mean_spacing"],
        "max_spacing": spacing["max_spacing"],
        "mean_force": mean_force,
        "max_force": max_force,
        "pot_mean": pot_mean,
        "pot_min": pot_min,
        "pot_max": pot_max,
        "need_mean": need_mean,
        "need_min": need_min,
        "need_max": need_max,
        "lambda_max_real": lambda_max_real,
        "lambda_min_real": lambda_min_real,
        "spectral_radius_I_plus_etaJ": spectral_radius,
        "continuous_stable": continuous_stable,
        "discrete_stable": discrete_stable,
        "num_eig_pos_real": num_eig_pos_real,
        "lyap_inst": lyap_inst,
        "lyap_avg": lyap_avg,
        "delta_pos_rms": delta_pos_rms,
        "delta_pos_max": delta_pos_max,
        "converged_flag": converged_flag,
        "converged_step": converged_step,
        "dist_ref_rms": dist_ref_rms,
        "dist_ref_max": dist_ref_max,
    }

    # flatten positions into the row (pos_0_x, pos_0_y, ...)
    for i in range(N):
        row[f"pos_{i}_x"] = float(pos_np[i, 0])
        row[f"pos_{i}_y"] = float(pos_np[i, 1])

    # --- write CSV -----------------------------------------------------------
    if csv_path is not None:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        file_exists = os.path.exists(csv_path)
        fieldnames = sorted(row.keys())

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    return row
