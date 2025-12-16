# simulation/stability.py
import numpy as np


def _to_numpy(x):
    """Return a NumPy array (copying from CuPy if necessary)."""
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except ImportError:
        pass
    return np.asarray(x)


def _wrap_positions(X, grid_size):
    if grid_size is None:
        return X
    return X % grid_size


# ============================================================
# Finite-difference Jacobian (respects simulator backend)
# ============================================================
def finite_diff_jacobian(sim, V, need_map, h=0.1):
    """
    Central-difference Jacobian of total forces.

    h : finite-difference step (grid units).  Larger h reduces noise.
    Automatically mirrors simulator backend (NumPy or CuPy).
    """
    import importlib

    # simulator imports cupy as np when GPU backend is active
    sim_mod = importlib.import_module("simulation.simulator")
    sim_np = sim_mod.np  # either cupy or numpy

    V_host = _to_numpy(V)
    need_host = _to_numpy(need_map)

    N, d = V_host.shape[0], 2
    J = np.zeros((N * d, N * d), dtype=float)

    for j in range(N):
        for axis in range(d):
            e = np.zeros_like(V_host)
            e[j, axis] = h
            Vp = _wrap_positions(V_host + e, sim.grid_size)
            Vm = _wrap_positions(V_host - e, sim.grid_size)

            # --- convert to simulator backend (NumPy or CuPy) ---
            Vp_dev = sim_np.asarray(Vp)
            Vm_dev = sim_np.asarray(Vm)
            need_dev = sim_np.asarray(need_host)
            # ----------------------------------------------------

            Fp_dev = sim.forces_given_positions(Vp_dev, need_dev)
            Fm_dev = sim.forces_given_positions(Vm_dev, need_dev)

            # bring back to host for differencing
            Fp = _to_numpy(Fp_dev)
            Fm = _to_numpy(Fm_dev)

            J[:, j * d + axis] = ((Fp - Fm) / (2 * h)).reshape(-1)

    return J


# ============================================================
# Local metrics consistent with theoretical model
# ============================================================
def local_metrics(sim, V, need_map, step_size):
    """
    Compute predictive stability metrics from current snapshot.
    The Jacobian is scaled by step_size to align with discrete-time theory.
    """
    V = _to_numpy(V)
    need_map = _to_numpy(need_map)

    J = finite_diff_jacobian(sim, V, need_map)
    J *= step_size                      # scale to match simulator update Δt = step_size

    # continuous-time criterion
    Jsym = 0.5 * (J + J.T)
    lambda_sym_max = float(np.max(np.linalg.eigvalsh(Jsym)))

    # discrete-time criterion
    M = np.eye(J.shape[0]) + J          # now already includes step_size
    rho = float(np.max(np.abs(np.linalg.eigvals(M))))

    # optional safe-step estimate (only meaningful if contracting)
    eta_safe = None
    if lambda_sym_max < 0:
        eta_safe = float(2.0 / (-lambda_sym_max))

    # geometry proxy
    if len(V) >= 2:
        from scipy.spatial import cKDTree
        tree = cKDTree(V)
        dists, _ = tree.query(V, k=2)
        nn = dists[:, 1]
        rbar = float(np.mean(nn))
    else:
        rbar = np.inf
    spacing_ratio = rbar / float(sim.sigma_rep)

    return {
        "rho_I_plus_etaJ": rho,
        "lambda_sym_max": lambda_sym_max,
        "eta_safe": eta_safe,
        "spacing_ratio": spacing_ratio,
        "J": J,
        "Jsym": Jsym,
    }


# ============================================================
# Verdict logic with tolerances
# ============================================================
def stability_report(metrics, tol_lambda=0.5, tol_rho=0.05):
    """
    Translate raw metrics into qualitative verdict.
    tol_lambda : λ margin before declaring instability (continuous)
    tol_rho    : ρ margin before declaring instability (discrete)
    """
    rho = metrics["rho_I_plus_etaJ"]
    lam = metrics["lambda_sym_max"]
    eta_safe = metrics["eta_safe"]

    discrete_unstable = rho > 1.0 + tol_rho
    cont_noncontracting = lam >= tol_lambda

    verdict = "stable"
    if cont_noncontracting and discrete_unstable:
        verdict = "high-risk oscillatory"
    elif cont_noncontracting or discrete_unstable:
        verdict = "borderline"

    return {
        "verdict": verdict,
        "rho": rho,
        "lambda_sym_max": lam,
        "eta_safe": eta_safe,
        "spacing_ratio": metrics["spacing_ratio"],
    }
