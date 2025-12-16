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


def stability_from_jacobian(J, eta):
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

    mu = 1.0 + eta * eigvals  # eigenvalues of I + eta J
    spectral_radius = float(np.abs(mu).max())

    continuous_stable = bool(lambda_max_real < 0.0)
    discrete_stable = bool(spectral_radius < 1.0)
    
    return {
        "lambda_max_real": lambda_max_real,
        "lambda_min_real": lambda_min_real,
        "spectral_radius_I_plus_etaJ": spectral_radius,
        "continuous_stable": continuous_stable,
        "discrete_stable": discrete_stable,
        "num_eig_pos_real": int((real_parts > 0.0).sum()),
    }


def log_step_metrics(
    step,
    positions,
    forces,
    potential,
    need_map,
    cfg,
    force_fn,
    csv_path="./logs/local_metrics.csv",
    eig_every=50,
    jacobian_eps=1e-3,
):
    """
    Generic per-step stability logger.

    This is designed to be called from your main simulation loop and only
    assumes access to:
      - current positions and forces (continuous-time field F(V)),
      - scalar potential map (any shape),
      - need_map (coverage demand),
      - a force_fn that can be evaluated at arbitrary positions.

    Parameters
    ----------
    step : int
        Current simulation step.
    positions : array-like, shape (N,2)
        Current viewpoint positions (NumPy or CuPy).
    forces : array-like, shape (N,2)
        Current forces F(V) (NumPy or CuPy).
    potential : array-like, shape (H,W)
        Potential map on the grid (NumPy or CuPy).
    need_map : array-like, shape (H,W)
        Need map on the grid (NumPy or CuPy).
    cfg : SimConfig
        Configuration object with attributes (grid_size, k_attr, k_rep,
        sigma_rep, amp_rep, step_size).
    force_fn : callable
        Callback positions -> forces, consistent with your simulator.
    csv_path : str
        Path to CSV file where metrics will be appended.
    eig_every : int
        How often (in steps) to compute Jacobian eigen diagnostics.
    jacobian_eps : float
        Finite difference epsilon for numerical_jacobian.
    """
    pos_np = _to_numpy(positions)
    forces_np = _to_numpy(forces)
    pot_np = _to_numpy(potential)
    need_np = _to_numpy(need_map)

    N = pos_np.shape[0]
    grid_size = float(cfg.grid_size)

    # --- spacing + force stats --------------------------------------------
    spacing_stats = torus_spacing_metrics(pos_np, grid_size)
    force_norms = np.linalg.norm(forces_np, axis=1)
    mean_force = float(force_norms.mean())
    max_force = float(force_norms.max())

    # --- scalar summaries of potential / need -----------------------------
    mean_potential = float(pot_np.mean())
    var_potential = float(pot_np.var())
    mean_need = float(need_np.mean())
    max_need = float(need_np.max())

    # --- Jacobian-based diagnostics (occasionally) ------------------------
    eig_metrics = {
        "lambda_max_real": np.nan,
        "lambda_min_real": np.nan,
        "spectral_radius_I_plus_etaJ": np.nan,
        "continuous_stable": "",
        "discrete_stable": "",
        "num_eig_pos_real": -1,
    }

    if eig_every is not None and eig_every > 0 and (step % eig_every == 0):
        J = numerical_jacobian(force_fn, pos_np, eps=jacobian_eps)
        eig_metrics = stability_from_jacobian(J, eta=float(cfg.step_size))

        # Lightweight console summary so you see what's going on live
        print(
            f"[Stab] step={step:04d}  N={N}  "
            f"min_d={spacing_stats['min_spacing']:.2f}  "
            f"maxReλ={eig_metrics['lambda_max_real']:.3f}  "
            f"ρ(I+ηJ)={eig_metrics['spectral_radius_I_plus_etaJ']:.3f}  "
            f"cts={'OK' if eig_metrics['continuous_stable'] else 'UNST'}  "
            f"disc={'OK' if eig_metrics['discrete_stable'] else 'UNST'}"
        )

    # --- assemble row ------------------------------------------------------
    row = {
        "step": int(step),
        "N": int(N),
        "grid_size": grid_size,
        "k_attr": float(cfg.k_attr),
        "k_rep": float(cfg.k_rep),
        "sigma_rep": float(cfg.sigma_rep),
        "amp_rep": float(cfg.amp_rep),
        "eta": float(cfg.step_size),
        "min_spacing": spacing_stats["min_spacing"],
        "mean_spacing": spacing_stats["mean_spacing"],
        "max_spacing": spacing_stats["max_spacing"],
        "mean_force_norm": mean_force,
        "max_force_norm": max_force,
        "mean_potential": mean_potential,
        "var_potential": var_potential,
        "mean_need": mean_need,
        "max_need": max_need,
        "lambda_max_real": eig_metrics["lambda_max_real"],
        "lambda_min_real": eig_metrics["lambda_min_real"],
        "spectral_radius_I_plus_etaJ": eig_metrics["spectral_radius_I_plus_etaJ"],
        "continuous_stable": eig_metrics["continuous_stable"],
        "discrete_stable": eig_metrics["discrete_stable"],
        "num_eig_pos_real": eig_metrics["num_eig_pos_real"],
    }

    fieldnames = list(row.keys())

    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# -------------------------------------------------------------------------
# Optional helpers to quickly make thesis-ready plots from the CSV
# -------------------------------------------------------------------------
def load_metrics(csv_path):
    """
    Load the CSV written by `log_step_metrics` into a dict of numpy arrays.
    """
    import csv

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name, val in row.items():
                if name in ("continuous_stable", "discrete_stable"):
                    # booleans stored as 'True'/'False' or ''
                    if val == "":
                        cols[name].append(None)
                    else:
                        cols[name].append(val == "True")
                else:
                    try:
                        cols[name].append(float(val))
                    except ValueError:
                        cols[name].append(np.nan)

    return {k: np.array(v) for k, v in cols.items()}


def quick_plots(csv_path, out_prefix="./logs/stability"):
    """
    Generate a couple of standard plots:
      1) min spacing + maxRe(λ) vs step
      2) spectral radius of I+ηJ vs step (with ρ=1 line)

    These are ideal to drop into the stability section of your thesis.
    """
    import matplotlib.pyplot as plt

    data = load_metrics(csv_path)
    step = data["step"]

    # --- Plot 1: spacing + maxReλ ----------------------------------------
    fig, ax1 = plt.subplots(figsize=(6, 3))
    ax1.plot(step, data["min_spacing"], label="min spacing")
    ax1.set_xlabel("step")
    ax1.set_ylabel("min spacing")

    ax2 = ax1.twinx()
    ax2.plot(step, data["lambda_max_real"], linestyle="--", label="max Re(λ)")
    ax2.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
    ax2.set_ylabel("max Re(λ(J))")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_prefix + "_spacing_vs_lambda.png", dpi=200)
    plt.close(fig)

    # --- Plot 2: spectral radius of I+ηJ ---------------------------------
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(step, data["spectral_radius_I_plus_etaJ"], label="ρ(I+ηJ)")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("step")
    ax.set_ylabel("spectral radius")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_prefix + "_spectral_radius.png", dpi=200)
    plt.close(fig)
