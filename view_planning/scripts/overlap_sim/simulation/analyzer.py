# simulation/analyzer.py
"""
Live Stability Analyzer (GPU-aware, minimal touch)
--------------------------------------------------
What it does (each step):
- Logs: spacing, spacing_ratio (r̄/σ), mean speed, mean force, E_attr, E_rep, E_tot, dE/dt, need_mean, pot_var
- Prints: a one-line stability summary every `print_every` steps
- Optionally (small N, every `jacobian_every` steps): builds a finite-difference Jacobian J,
  computes eigenvalues λ(J) and (if eta given) eigenvalues of (I + eta J), then prints stability verdicts.

Design:
- CuPy/NumPy agnostic through `xp` backend
- Hook-in: pass a `force_fn(positions)` callback to recompute forces (using your existing physics path)
- Toroidal wrap respected for distances and perturbations
"""

# from __future__ import annotations
from typing import Dict, Any, Optional, Callable, Tuple

# ---- backend (CuPy if available) -------------------------------------------
try:
    import cupy as xp
    import cupy as cp
    GPU = True
except Exception:
    import numpy as xp
    import numpy as np
    GPU = False


def _asnumpy(a):
    if GPU:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    return a


def _to_xp(a):
    return xp.asarray(a)


def _wrap_delta(delta, grid_size):
    """Toroidal minimum-image displacement (componentwise)."""
    if grid_size is None:
        return delta
    half = grid_size / 2.0
    return xp.where(xp.abs(delta) > half, delta - xp.sign(delta) * grid_size, delta)


def _wrap_pos(P, grid_size):
    if grid_size is None:
        return P
    return xp.mod(P, grid_size)


def _pairwise_distances_torus(P, grid_size):
    """All-pairs wrapped distances, P:(N,2) -> (N,N)."""
    N = P.shape[0]
    if N == 0:
        return xp.zeros((0, 0), dtype=xp.float64)
    Pi = P[:, None, :]
    Pj = P[None, :, :]
    d = _wrap_delta(Pi - Pj, grid_size)
    return xp.sqrt(xp.sum(d * d, axis=2))


# ---------------------------------------------------------------------------
class Analyzer:
    """
    Analyzer(grid_size, k_attr, k_rep, sigma, amp,
             eta=None,
             print_every=10,
             jacobian_every=None,
             jacobian_fd_eps=1e-3,
             jacobian_max_N=10)

    Required per-step call:
      record(step, positions, velocities, potential, forces, need_map,
             force_fn=None)

    Optional:
      - To enable Jacobian/eigen probes, pass:
        * jacobian_every = e.g. 50
        * force_fn: a callable positions -> forces (same shape), using your current cfg/fields

      - eta: your simulator step size (for discrete-time eigen test)
    """

    def __init__(self,
                 grid_size: float,
                 k_attr: float,
                 k_rep: float,
                 sigma: float,
                 amp: float,
                 eta: Optional[float] = None,
                 print_every: int = 10,
                 jacobian_every: Optional[int] = None,
                 jacobian_fd_eps: float = 1e-3,
                 jacobian_max_N: int = 10):
        self.grid_size = float(grid_size)
        self.k_attr = float(k_attr)
        self.k_rep = float(k_rep)
        self.sigma = float(sigma)
        self.amp = float(amp)
        self.eta = float(eta) if eta is not None else None

        self.print_every = int(print_every)
        self.jacobian_every = int(jacobian_every) if jacobian_every else None
        self.jacobian_fd_eps = float(jacobian_fd_eps)
        self.jacobian_max_N = int(jacobian_max_N)

        self._rows = []
        self._prev_Etot = None
        self._prev_t = None
        self._last_status = None
        self._last_lambda_max = None
        self._last_rho = None
        # --- add after existing init fields ---
        self._win_E = []                 # ring buffer for E_tot
        self._win_size = 256             # FFT window
        self._osc_R_threshold = 6.0      # spectral dominance ratio threshold
        self._env_decay_eps = 1e-3       # near-zero slope considered non-decaying


    # ----------------------- public API -------------------------------------
    def record(self,
               step: int,
               positions,            # (N,2) np/cp
               velocities,           # (N,2) np/cp
               potential,            # (H,W) (CPU ok)
               forces,               # (N,2) np/cp
               need_map,             # (H,W) (CPU ok)
               force_fn: Optional[Callable[[xp.ndarray], xp.ndarray]] = None
               ) -> Dict[str, Any]:
        """
        Compute diagnostics, optionally Jacobian/eigens, and print live status.
        Returns the metrics dict for this step.
        """
        P = _to_xp(positions)
        V = _to_xp(velocities)
        F = _to_xp(forces)

        N = int(P.shape[0])
        # --- spacing ---
        rmat = _pairwise_distances_torus(P, self.grid_size)
        if rmat.size == 0:
            r_mean = 0.0
            r_min = 0.0
        else:
            mask = ~xp.eye(N, dtype=bool)
            r_mean = float(_asnumpy(xp.mean(rmat[mask]))) if mask.any() else 0.0
            r_min  = float(_asnumpy(xp.min(rmat[mask])))  if mask.any() else 0.0

        spacing_ratio = (r_mean / self.sigma) if self.sigma > 0 else 0.0

        # --- kinematics ---
        speed_mean = float(_asnumpy(xp.mean(xp.linalg.norm(V, axis=1)))) if N > 0 else 0.0
        force_mean = float(_asnumpy(xp.mean(xp.linalg.norm(F, axis=1)))) if N > 0 else 0.0

        # --- energy proxies ---
        E_attr = float(_asnumpy(potential).sum()) * self.k_attr
        E_rep = self._energy_rep_proxy(P)
        E_tot = E_attr + self.k_rep * E_rep

        # maintain energy window (CPU floats)
        self._win_E.append(E_tot)
        if len(self._win_E) > self._win_size:
            self._win_E.pop(0)

        if self._prev_Etot is None:
            dE_dt = 0.0
        else:
            dt = max(1, step - (self._prev_t or (step - 1)))
            dE_dt = float((E_tot - self._prev_Etot) / dt)
        self._prev_Etot, self._prev_t = E_tot, step

        # --- CPU stats for maps ---
        need_mean = float(_asnumpy(need_map).mean())
        pot_var   = float(_asnumpy(potential).var())

        metrics = dict(
            step=int(step), N=N,
            r_mean=r_mean, r_min=r_min, spacing_ratio=spacing_ratio,
            speed_mean=speed_mean, force_mean=force_mean,
            E_attr=E_attr, E_rep=E_rep, E_tot=E_tot, dE_dt=dE_dt,
            need_mean=need_mean, pot_var=pot_var
        )

        # --- Optional Jacobian/eigen probe ---
        lambda_max = None
        rho = None
        if (self.jacobian_every is not None
            and force_fn is not None
            and N > 0
            and N <= self.jacobian_max_N
            and (step % self.jacobian_every == 0)):
            J = self._finite_diff_jacobian(P, force_fn, self.jacobian_fd_eps)
            lambda_max = self._eig_max_real(J)
            rho = self._discrete_rho(J, self.eta) if self.eta is not None else None
            metrics["lambda_max_real"] = lambda_max if lambda_max is not None else xp.nan
            metrics["rho_I_plus_etaJ"] = rho if rho is not None else xp.nan
            self._last_lambda_max = lambda_max
            self._last_rho = rho

        self._rows.append(metrics)

        # --- live print (human-readable) every print_every ---
        if step % self.print_every == 0:
            status = self._compose_status(speed_mean, dE_dt, lambda_max, rho)
            self._last_status = status
            print(self._fmt_line(metrics, status))

        return metrics

    def save_csv(self, path: str):
        """Dump metrics to CSV (NumPy only at the boundary)."""
        import os
        import numpy as np
        if not self._rows:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # unify keys
        keys = sorted(set().union(*[r.keys() for r in self._rows]))
        M = np.array([[r.get(k, np.nan) for k in keys] for r in self._rows], dtype=float)
        header = ",".join(keys)
        np.savetxt(path, M, delimiter=",", header=header, comments="")

    # ----------------------- internals --------------------------------------
    def _energy_rep_proxy(self, P):
        r = _pairwise_distances_torus(P, self.grid_size)
        if r.size == 0:
            return 0.0
        N = r.shape[0]
        mask = xp.triu(xp.ones((N, N), dtype=bool), k=1)
        G = xp.exp(-(r * r) / (2.0 * self.sigma ** 2))
        return float(_asnumpy(self.amp * xp.sum(G[mask])))

    def _finite_diff_jacobian(self,
                              P: xp.ndarray,
                              force_fn: Callable[[xp.ndarray], xp.ndarray],
                              h: float) -> xp.ndarray:
        """
        Central-difference Jacobian of F at P.
        - P: (N,2)
        - force_fn: positions -> forces (N,2) using your existing pipeline
        Returns J: (2N, 2N)
        Cost ~ O(N) force evaluations per coordinate; keep N small.
        """
        N = int(P.shape[0])
        dim = 2 * N
        J = xp.zeros((dim, dim), dtype=xp.float64)

        # Baseline not required for central difference, but can warm caches
        # F0 = force_fn(P)

        for j in range(N):
            for ax in range(2):
                e = xp.zeros_like(P)
                e[j, ax] = h

                Pp = _wrap_pos(P + e, self.grid_size)
                Pm = _wrap_pos(P - e, self.grid_size)

                Fp = _to_xp(force_fn(_asnumpy(Pp)))  # callback may expect CPU; we convert back if needed
                Fm = _to_xp(force_fn(_asnumpy(Pm)))

                dF = (Fp - Fm) / (2.0 * h)  # (N,2)

                col = 2 * j + ax
                # place into J rows (i,coord) -> row = 2*i + coord
                for i in range(N):
                    J[2 * i + 0, col] = dF[i, 0]
                    J[2 * i + 1, col] = dF[i, 1]

        return J

    def _eig_max_real(self, J: xp.ndarray) -> float:
        """Max real part of eigenvalues of J (continuous-time criterion)."""
        # small matrices → safe to copy to CPU for robustness
        if GPU:
            import numpy as np
            vals = np.linalg.eigvals(_asnumpy(J))
        else:
            import numpy as np
            vals = np.linalg.eigvals(J)
        return float(vals.real.max())

    def _discrete_rho(self, J, eta: Optional[float]):
        """Spectral radius of I + eta J (discrete-time stability)."""
        if eta is None:
            return None
        I_plus = xp.eye(J.shape[0], dtype=J.dtype) + eta * J
        import numpy as np
        # Always compute eigenvalues on CPU (NumPy) for stability
        vals = np.linalg.eigvals(_asnumpy(I_plus))
        rho = float(np.max(np.abs(vals)))
        return rho


    def _compose_status(self, speed_mean: float, dE_dt: float,
                    lambda_max: Optional[float], rho: Optional[float]) -> str:
        # 1) Eigen-based verdicts (authoritative, local)
        if lambda_max is not None:
            ct = "locally-stable" if lambda_max < 0.0 else "locally-unstable"
            if rho is not None:
                dt = "DT-stable" if rho < 1.0 else "DT-unstable"
                return f"{ct} (λ_max={lambda_max:+.3e}, {dt}, ρ={rho:.3f})"
            return f"{ct} (λ_max={lambda_max:+.3e})"

        # 2) Windowed oscillator test on E_tot (global behavior)
        lc = self._detect_limit_cycle()
        if lc:
            return "limit-cycle"

        # 3) Transient vs settled (heuristic, but not calling it unstable)
        # use gentle thresholds so small jitters don't mislabel
        if abs(dE_dt) < 5e-4 and speed_mean < 5e-4:
            return "settled"
        return "transient"
    
    def _detect_limit_cycle(self) -> bool:
        """
        Detect intrinsic oscillation using windowed FFT of E_tot.
        Returns True if a strong, persistent spectral line exists and
        the envelope is not decaying.
        """
        import numpy as np
        W = self._win_size
        if len(self._win_E) < max(64, W // 2):
            return False

        x = np.asarray(self._win_E, dtype=np.float64)
        x = x - x.mean()
        # Hann window to reduce leakage
        w = np.hanning(len(x))
        X = np.fft.rfft(x * w)
        mag = np.abs(X)[1:]  # exclude DC

        if mag.size == 0:
            return False

        # Spectral dominance ratio: peak vs median remainder
        med = np.median(mag)
        if med <= 1e-12:
            return False
        R = float(np.max(mag) / med)

        # Envelope: RMS trend over the window halves
        mid = len(x) // 2
        rms1 = np.sqrt(np.mean(x[:mid] ** 2))
        rms2 = np.sqrt(np.mean(x[mid:] ** 2))
        slope = (rms2 - rms1)  # coarse "decay": negative means damping

        # Limit-cycle if there is a strong narrowband component (R large)
        # and no appreciable decay in amplitude (|slope| small)
        return (R >= self._osc_R_threshold) and (abs(slope) <= self._env_decay_eps)



    def _fmt_line(self, m: Dict[str, Any], status: str) -> str:
        return (f"[Stab] t={m['step']:6d} | N={m['N']:2d} | r̄/σ={m['spacing_ratio']:.2f} | "
                f"⟨|v|⟩={m['speed_mean']:.3e} | E={m['E_tot']:.3e} | dE/dt={m['dE_dt']:+.2e} | {status}")
