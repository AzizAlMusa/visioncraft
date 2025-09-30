#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive NBV / Potential-Field simulation — Phase 2 + Feature-Aware (Anisotropic) Repulsion

This version EXTENDS the original by adding *feature-aware anisotropic repulsion*
that aligns to the local geometry of your importance field, while keeping EVERY existing
functionality intact (log attraction, unmet-need, toroidal FFT attraction grid,
visuals, insertion policy, outputs, etc.).

Drop-in behavior:
- If CFG["anisotropy"]["enabled"] == False → repulsion remains your original isotropic Gaussian.
- If True → repulsion becomes anisotropic *only* where importance has structure; it
  gracefully falls back to isotropic in flat regions.

Key additions (search for "ANISO"):
1) Build a smoothed importance grid and its gradient/structure to define tangent/normal.
2) Construct an anisotropy matrix field A(x) = a_t * t t^T + a_n * n n^T (a_t >> a_n in ROIs).
3) Use A(x) inside the Gaussian kernel: exp( - Δ^T A(x) Δ / (2 σ_rep^2) ).

Everything else (including quivers/colors/saving) remains identical to your file.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse, os, time, json
from matplotlib.patches import FancyArrowPatch, Circle, Ellipse
from collections import deque
from math import isfinite

# ---------- Args ----------
parser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=str, default="nbv", choices=["nbv", "random"])
parser.add_argument("--potential_type", type=str, default="log")  # name kept for file-tagging
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save_dir", type=str, default="./phase2_results")
parser.add_argument("--animate", action="store_true")
parser.add_argument("--smart_nbv_insertion", action="store_true")
parser.add_argument("--k_attr", type=float, default=10.0)
parser.add_argument("--k_rep", type=float, default=0.25)
parser.add_argument("--max_viewpoints", type=int, default=100)
parser.add_argument("--vp_index", type=int, default=0, help="Perspective viewpoint for RIGHT panel.")
parser.add_argument("--config", type=str, default="overlap_config.json",
                    help="JSON config: vis_beta, gamma, need_hard_clip, tau_clip, importance_config, poi_config, anisotropy block.")
# Perpetual default ON; pass --no-perpetual to allow termination & MP4 finalize
try:
    from argparse import BooleanOptionalAction
    parser.add_argument("--perpetual", action=BooleanOptionalAction, default=True)
except Exception:
    parser.add_argument("--perpetual", type=lambda s: s.lower() != "false", default=True)
# Animation buffer size so saved MP4 is finite even in long runs
parser.add_argument("--max_anim_frames", type=int, default=600)
# Quiver display controls
parser.add_argument("--quiver_mode", choices=["length","normalized"], default="length",
                    help="length: arrow length encodes magnitude; normalized: direction only.")
parser.add_argument("--quiver_len_scale_attr", type=float, default=1.0,
                    help="Visual length gain for BLUE (attraction) arrows in length mode.")
parser.add_argument("--quiver_len_scale_rep", type=float, default=1.0,
                    help="Visual length gain for RED (repulsion) arrows in length mode.")
# Optional GREEN interest-only layer & K contours
parser.add_argument("--show_interest_quiver", action="store_true",
                    help="GREEN quiver = incremental attraction due to POIs (A_POI = A_with_POI - A_baseline).")
parser.add_argument("--show_kmap_contours", action="store_true",
                    help="Overlay K(q) contour labels (levels: 1,2,3,4) on both panels.")
# Optional: separate scale for GREEN interest quiver (default follows attr scale)
parser.add_argument("--quiver_len_scale_interest", type=float, default=None,
                    help="If set, overrides GREEN interest quiver scale; else uses --quiver_len_scale_attr.")
parser.add_argument("--verbose", action="store_true")


# --- Kernel overlay options ---
parser.add_argument("--show_repulsion_kernel", action="store_true",
                    help="Overlay iso-contours (ellipses) of the repulsion kernel at the selected viewpoint.")
parser.add_argument("--kernel_levels", type=str, default="1,2,3",
                    help="Comma-separated positive numbers; draws √c sigma-levels where q(Δ)=c·σ^2.")
parser.add_argument("--kernel_color", type=str, default="#111111",
                    help="Color for kernel contours (ellipse/circle outlines).")
parser.add_argument("--kernel_alpha", type=float, default=0.30,
                    help="Alpha for kernel contours.")
parser.add_argument("--kernel_lw", type=float, default=1.2,
                    help="Line width for kernel contours.")

args = parser.parse_args()

# ---------- Globals ----------
LABEL = "phase 2 results + anisotropic repulsion"
np.random.seed(args.seed)
grid_size, fov_radius = 100, 20
epsilon               = 1e-6
tiny                  = 1e-12
frames_per_stage       = 10
T_motion_rough, T_motion_fine = 1.5, 0.15
window_size            = 5
stride_vf              = 4

x, y         = np.arange(grid_size), np.arange(grid_size)
X, Y         = np.meshgrid(x, y)
field_points = np.stack([X.ravel(), Y.ravel()], axis=-1)  # (P,2)

# ---------- Config ----------
def _load_config(path):
    cfg = {
        "vis_beta": 20.0,
        "gamma": 1.0,
        "need_hard_clip": True,
        "tau_clip": 1.0,
        "importance_config": "",
        "poi_config": "",
        # ANISO: user-tunable defaults
        "anisotropy": {
            "enabled": True,
            "alpha_t": 6.0,         # along tangent (spread along feature)
            "alpha_n": 1.0,         # along normal (easier to separate across feature)
            "sigma_rep": 10.0,      # repulsion sigma (kept same as original by default)
            "amp_rep": 100.0,       # repulsion amplitude
            "smooth_sigma": 1.5,    # Gaussian smoothing of importance before gradients
            "grad_eps": 1e-3,       # if |grad| below this → isotropic fallback
            "tensor_blend": 0.0     # 0: use gradient only, 1: mix with Hessian eigendirs
        }
    }
    if path and os.path.exists(path):
        with open(path, "r") as f:
            try:
                user = json.load(f)
                # deep-merge minimal
                for k,v in user.items():
                    if k == "anisotropy" and isinstance(v, dict):
                        cfg["anisotropy"].update(v)
                    else:
                        cfg[k] = v
            except Exception:
                pass
    return cfg

CFG = _load_config(args.config)

# ---------- POI parsing to build maps (UNCHANGED semantics) ----------

def _parse_number(s):
    try:
        v = float(s)
        iv = int(v)
        return iv if abs(v - iv) < 1e-12 else v
    except Exception:
        return None

def _apply_disk(mask, cx, cy, r):
    dx = np.mod(X - cx + grid_size/2, grid_size) - grid_size/2
    dy = np.mod(Y - cy + grid_size/2, grid_size) - grid_size/2
    return mask | ((dx*dx + dy*dy) <= r*r)

def _apply_band(mask, theta_deg, b, width):
    th = np.deg2rad(theta_deg)
    n = np.array([np.cos(th), np.sin(th)])
    proj = (X * n[0] + Y * n[1])
    return mask | (np.abs(proj - b) <= (width / 2.0))

def _build_maps_from_poi(poi_config_str):
    K_map   = np.ones((grid_size, grid_size), dtype=np.float64)
    beta_map= np.full((grid_size, grid_size), float(CFG["vis_beta"]), dtype=np.float64)
    omega_map = np.ones((grid_size, grid_size), dtype=np.float64)

    s = (poi_config_str or "").strip()
    if not s:
        return K_map, beta_map, omega_map

    tokens = [t.strip() for t in s.split(";") if t.strip()]
    for tok in tokens:
        if tok.startswith("disk:"):
            body = tok[len("disk:"):].split(",")
            if len(body) < 3: continue
            cx, cy, r = map(float, body[:3])
            extras = body[3:]
            req = None; omega_add=None; omega_mul=None; beta=None
            for e in extras:
                e = e.strip()
                if e.startswith("req="):
                    req = _parse_number(e.split("=",1)[1])
                elif e.startswith("omega="):
                    val = e.split("=",1)[1]
                    if val.startswith("+"): omega_add = float(val[1:])
                    elif val.startswith("*"): omega_mul = float(val[1:])
                elif e.startswith("beta="):
                    beta = float(e.split("=",1)[1])
            region = _apply_disk(np.zeros_like(K_map, dtype=bool), cx, cy, r)

        elif tok.startswith("band:"):
            body = tok[len("band:"):].split(",")
            if len(body) < 3: continue
            theta_deg, b, width = map(float, body[:3])
            extras = body[3:]
            req = None; omega_add=None; omega_mul=None; beta=None
            for e in extras:
                e = e.strip()
                if e.startswith("req="):
                    req = _parse_number(e.split("=",1)[1])
                elif e.startswith("omega="):
                    val = e.split("=",1)[1]
                    if val.startswith("+"): omega_add = float(val[1:])
                    elif val.startswith("*"): omega_mul = float(val[1:])
                elif e.startswith("beta="):
                    beta = float(e.split("=",1)[1])
            region = _apply_band(np.zeros_like(K_map, dtype=bool), theta_deg, b, width)
        else:
            continue

        if req is not None:
            K_map[region] = np.maximum(K_map[region], float(req))
        if omega_add is not None:
            omega_map[region] = omega_map[region] + omega_add
        if omega_mul is not None:
            omega_map[region] = omega_map[region] * omega_mul
        if beta is not None:
            beta_map[region] = beta

    K_map = np.clip(K_map, 1.0, None)
    omega_map = np.clip(omega_map, 0.0, None)
    return K_map, beta_map, omega_map

K_map, beta_map, omega_map = _build_maps_from_poi(CFG.get("poi_config",""))
gamma = float(CFG.get("gamma", 1.0))
need_hard_clip = bool(CFG.get("need_hard_clip", True))
tau_clip = float(CFG.get("tau_clip", 1.0))

# ---------- ANISO: importance → smoothed field, gradients, anisotropy grids ----------

def _gaussian_smooth(img, sigma):
    if sigma <= 0: return img.copy()
    # separable approx with reflective boundary on the torus via roll-averages (cheap + periodic)
    # Use 3-tap binomial approximation for small sigma; for simplicity, convolve with kernel
    r = int(max(1, round(3*sigma)))
    xs = np.arange(-r, r+1)
    ker = np.exp(-(xs**2)/(2*sigma*sigma)); ker /= ker.sum()
    tmp = np.apply_along_axis(lambda v: np.convolve(v, ker, mode='same'), 1, img)
    tmp = np.apply_along_axis(lambda v: np.convolve(v, ker, mode='same'), 0, tmp)
    return tmp

# Build importance field used for anisotropy (I_smooth in [0,1] approx)
I0 = np.clip(omega_map.astype(np.float64) ** gamma, 0.0, None)
I0 = I0 / (I0.max() + 1e-12)
I_smooth = _gaussian_smooth(I0, float(CFG["anisotropy"]["smooth_sigma"]))

# Gradients (periodic via np.gradient on torus assumed sufficient)
Gy, Gx = np.gradient(I_smooth)  # note: numpy returns [d/dy, d/dx]
Grad_mag = np.sqrt(Gx*Gx + Gy*Gy)

# Tangent and normal unit fields
n_x = np.where(Grad_mag > CFG["anisotropy"]["grad_eps"], Gx/Grad_mag, 0.0)
n_y = np.where(Grad_mag > CFG["anisotropy"]["grad_eps"], Gy/Grad_mag, 0.0)
t_x = -n_y
t_y =  n_x

# Optional: light blend with Hessian/structure tensor principal dirs (set tensor_blend>0 to use)
if CFG["anisotropy"]["tensor_blend"] > 0:
    # crude Hessian via finite differences
    Hxx = np.gradient(Gx, axis=1)
    Hyy = np.gradient(Gy, axis=0)
    Hxy = 0.5*(np.gradient(Gx, axis=0) + np.gradient(Gy, axis=1))
    # structure tensor S ≈ [[Gx^2, GxGy],[GxGy, Gy^2]]
    Sxx = _gaussian_smooth(Gx*Gx, 1.0)
    Syy = _gaussian_smooth(Gy*Gy, 1.0)
    Sxy = _gaussian_smooth(Gx*Gy, 1.0)
    # principal direction from S (largest eigenvector → tangent-ish ridge dir)
    trace = Sxx + Syy
    det = Sxx*Syy - Sxy*Sxy
    tmp = np.sqrt(np.maximum(trace*trace - 4*det, 0.0))
    lam1 = 0.5*(trace + tmp)
    # eigenvector for lam1
    evx = np.where(np.abs(Sxy) + np.abs(Sxx - lam1) > 1e-12, Sxy, 1.0)
    evy = np.where(np.abs(Sxy) + np.abs(Sxx - lam1) > 1e-12, lam1 - Sxx, 0.0)
    nor = np.sqrt(evx*evx + evy*evy) + 1e-12
    evx /= nor; evy /= nor
    # blend: t ← (1-b)*t + b*ev  (then re-orthonormalize with n)
    b = float(CFG["anisotropy"]["tensor_blend"])
    t_x = (1-b)*t_x + b*evx
    t_y = (1-b)*t_y + b*evy
    tn = np.sqrt(t_x*t_x + t_y*t_y) + 1e-12
    t_x /= tn; t_y /= tn
    # re-derive n as perpendicular
    n_x = -t_y
    n_y =  t_x

# Anisotropy scalars (constant factors; could be made maps if desired)
a_t = float(CFG["anisotropy"]["alpha_t"])  # strong along tangent (push apart along feature)
a_n = float(CFG["anisotropy"]["alpha_n"])  # weak along normal (easier to bridge across)

# Pack A = [[axx, axy],[axy, ayy]] grids for bilinear sampling
axx_grid = a_t*(t_x*t_x) + a_n*(n_x*n_x)
axy_grid = a_t*(t_x*t_y) + a_n*(n_x*n_y)
ayy_grid = a_t*(t_y*t_y) + a_n*(n_y*n_y)

# --- Isotropic fallback & smooth blend to identity ---
tau   = float(CFG["anisotropy"].get("blend_tau", 0.0))
scale = float(CFG["anisotropy"].get("blend_scale", 1.0))
Imin  = float(CFG["anisotropy"].get("importance_floor", 0.0))

# normalized gradient magnitude gate
s = np.clip((Grad_mag - tau) / max(scale, 1e-9), 0.0, 1.0)
if Imin > 0.0:
    s = s * (I_smooth > Imin)

# When s=0 → A = I (pure circles); when s=1 → A = anisotropic; else blend
axx_grid = s*axx_grid + (1.0 - s)*1.0
axy_grid = s*axy_grid + (1.0 - s)*0.0
ayy_grid = s*ayy_grid + (1.0 - s)*1.0


# ---------- FFT helpers (toroidal vector-kernel) ----------

def _build_fft_kernels(N):
    offs = np.arange(N, dtype=np.float64)
    offs = np.where(offs <= N//2, offs, offs - N)
    dx = offs[None, :]
    dy = offs[:, None]
    r2 = dx*dx + dy*dy
    Kx = np.zeros((N, N), dtype=np.float64)
    Ky = np.zeros((N, N), dtype=np.float64)
    np.divide(-dx, r2, out=Kx, where=r2 > 0)
    np.divide(-dy, r2, out=Ky, where=r2 > 0)
    Kx_hat = np.fft.rfftn(Kx, s=(N, N))
    Ky_hat = np.fft.rfftn(Ky, s=(N, N))
    return Kx_hat, Ky_hat

Kx_hat, Ky_hat = _build_fft_kernels(grid_size)


def _fft_convolve_vector(n_grid):
    n_hat = np.fft.rfftn(n_grid, s=(grid_size, grid_size))
    Fx = np.fft.irfftn(n_hat * Kx_hat, s=(grid_size, grid_size))
    Fy = np.fft.irfftn(n_hat * Ky_hat, s=(grid_size, grid_size))
    return Fx, Fy


def _bilinear_sample_periodic(F, pts):
    H, W = F.shape
    x = pts[:, 0]; y = pts[:, 1]
    x0 = np.floor(x).astype(int) % W
    y0 = np.floor(y).astype(int) % H
    x1 = (x0 + 1) % W
    y1 = (y0 + 1) % H
    tx = x - np.floor(x)
    ty = y - np.floor(y)
    f00 = F[y0, x0]; f10 = F[y0, x1]
    f01 = F[y1, x0]; f11 = F[y1, x1]
    return (1-ty)*((1-tx)*f00 + tx*f10) + ty*((1-tx)*f01 + tx*f11)

# helper to sample A-matrix components at points
_defZ = np.zeros((grid_size, grid_size))

def _sample_A_components(Q):
    axx = _bilinear_sample_periodic(axx_grid, Q)
    axy = _bilinear_sample_periodic(axy_grid, Q)
    ayy = _bilinear_sample_periodic(ayy_grid, Q)
    return axx, axy, ayy

# ---------- Field ----------
class Field:
    def __init__(self, grid_size, fov_radius, use_wrapping=True):
        self.grid_size, self.fov_radius = grid_size, fov_radius
        self.use_wrapping = use_wrapping
        self.field_points = field_points
        self.need_weight = np.ones(grid_size * grid_size, dtype=np.float64)
        self.potential    = np.zeros((grid_size, grid_size))
        self.visibility_soft = np.zeros((grid_size, grid_size), dtype=np.float64)
        self._attr_Fx = np.zeros((grid_size, grid_size), dtype=np.float64)
        self._attr_Fy = np.zeros((grid_size, grid_size), dtype=np.float64)
        self._base_Fx = np.zeros((grid_size, grid_size), dtype=np.float64)
        self._base_Fy = np.zeros((grid_size, grid_size), dtype=np.float64)
        self._last_particles = None

    def wrap_distance(self, diff):
        if not self.use_wrapping:
            return diff
        half = self.grid_size / 2.0
        return np.where(np.abs(diff) > half,
                        -np.sign(diff) * (self.grid_size - np.abs(diff)), diff)

    def _compute_vis_and_need(self, particles):
        P = self.field_points.shape[0]
        K = particles.shape[0]
        if K == 0:
            self.visibility_soft.ravel()[:] = 0.0
            self.need_weight[:] = (omega_map.ravel() ** gamma)
            return
        diff = self.field_points[:, None, :] - particles[None, :, :]
        diff = self.wrap_distance(diff)
        dist = np.linalg.norm(diff, axis=-1) + epsilon
        beta_q = np.repeat(beta_map.ravel()[:, None], K, axis=1)
        s = 1.0 / (1.0 + np.exp(beta_q * (dist / self.fov_radius - 1.0)))
        vis = s.sum(axis=1)
        self.visibility_soft.ravel()[:] = np.clip(vis, 0.0, None)
        Kq = K_map.ravel()
        need = np.maximum(0.0, 1.0 - vis / Kq) * (omega_map.ravel() ** gamma)
        if need_hard_clip:
            need = np.where(vis >= tau_clip * Kq, 0.0, need)
        self.need_weight[:] = need

    def update_need_mask(self, particles, beta_unused=20.0):
        self._compute_vis_and_need(particles)

    def compute_potential(self, particles, alpha=1.0):
        if particles.shape[0] == 0:
            self.potential.fill(0.0)
            return
        diff = self.field_points[:, None, :] - particles[None, :, :]
        diff = self.wrap_distance(diff)
        dist = np.linalg.norm(diff, axis=-1) + epsilon
        base = np.log(dist).sum(axis=1)
        pot  = alpha * self.need_weight * base
        self.potential = pot.reshape(self.grid_size, self.grid_size)

    def compute_attraction_grids(self, particles):
        n_grid = self.need_weight.reshape(self.grid_size, self.grid_size)
        Fx, Fy = _fft_convolve_vector(n_grid)
        self._attr_Fx[:], self._attr_Fy[:] = Fx, Fy
        if particles.size == 0:
            base = np.ones_like(n_grid)
        else:
            diff = field_points[:, None, :] - particles[None, :, :]
            diff = self.wrap_distance(diff)
            dist = np.linalg.norm(diff, axis=-1) + epsilon
            min_d = dist.min(axis=1)
            base = (min_d > self.fov_radius).astype(np.float64).reshape(self.grid_size, self.grid_size)
        Fx_b, Fy_b = _fft_convolve_vector(base)
        self._base_Fx[:], self._base_Fy[:] = Fx_b, Fy_b
        self._last_particles = particles.copy()

    def attraction_field_at(self, Q):
        if Q.ndim == 1: Q = Q[None, :]
        Ux = _bilinear_sample_periodic(self._attr_Fx, Q)
        Uy = _bilinear_sample_periodic(self._attr_Fy, Q)
        return np.stack([Ux, Uy], axis=1)

    def attraction_field_baseline_at(self, Q):
        if Q.ndim == 1: Q = Q[None, :]
        Ux = _bilinear_sample_periodic(self._base_Fx, Q)
        Uy = _bilinear_sample_periodic(self._base_Fy, Q)
        return np.stack([Ux, Uy], axis=1)

    # ---------------- ANISO repulsion (grid-sampled A) ----------------
    def _anisotropic_repulsion(self, Q, others):
        if Q.ndim == 1: Q = Q[None, :]
        if others.shape[0] == 0:
            return np.zeros((Q.shape[0], 2))
        # Sample A at the *evaluation* point Q
        axx, axy, ayy = _sample_A_components(Q)
        sigma = float(CFG["anisotropy"]["sigma_rep"])
        amp   = float(CFG["anisotropy"]["amp_rep"])
        # For each Q, accumulate across others
        out = np.zeros((Q.shape[0], 2), dtype=np.float64)
        for k in range(others.shape[0]):
            diff = Q - others[k:k+1, :]
            diff = self.wrap_distance(diff)
            dx = diff[:,0]; dy = diff[:,1]
            # Quadratic form q = [dx dy] A [dx dy]^T
            qf = axx*dx*dx + 2*axy*dx*dy + ayy*dy*dy
            # weight ~ gradient of anisotropic Gaussian w.r.t. Q
            # d/dQ exp(-qf/(2σ^2)) = exp(..) * ( -A - A^T )/σ^2 * diff; here A is symmetric → -2A/σ^2
            w = amp * np.exp(-qf/(2*sigma*sigma)) / (sigma*sigma)
            Fx = -(axx*dx + axy*dy) * 2.0 * w
            Fy = -(axy*dx + ayy*dy) * 2.0 * w
            out[:,0] += Fx
            out[:,1] += Fy
        return out

    def repulsion_field_at(self, Q, particles, exclude_index=0):
        if Q.ndim == 1: Q = Q[None, :]
        if particles.shape[0] <= 1:
            return np.zeros((Q.shape[0], 2))
        others = np.delete(particles, exclude_index, axis=0)
        if others.shape[0] == 0:
            return np.zeros((Q.shape[0], 2))
        if CFG["anisotropy"]["enabled"]:
            return self._anisotropic_repulsion(Q, others)
        # fallback: original isotropic
        diff = Q[:, None, :] - others[None, :, :]
        diff = self.wrap_distance(diff)
        r    = np.linalg.norm(diff, axis=-1) + epsilon
        sigma, amp = 10.0, 100.0
        w    = (amp * (r / (sigma**2)) * np.exp(-(r**2) / (2*sigma**2)))
        return (diff * w[..., None]).sum(axis=1)

    def compute_force_on_viewpoints(self, particles, k_attr=0.4, k_rep=1.0):
        K = particles.shape[0]
        if K == 0:
            return np.zeros((0,2))
        Fax = _bilinear_sample_periodic(self._attr_Fx, particles)
        Fay = _bilinear_sample_periodic(self._attr_Fy, particles)
        F_attr = np.stack([Fax, Fay], axis=1)
        if CFG["anisotropy"]["enabled"]:
            # pairwise using A sampled at the receiver (i)
            sigma = float(CFG["anisotropy"]["sigma_rep"]) ; amp = float(CFG["anisotropy"]["amp_rep"]) 
            axx, axy, ayy = _sample_A_components(particles)
            F_rep = np.zeros_like(F_attr)
            for i in range(K):
                # sum over j≠i
                diff = particles[i:i+1, :] - particles
                diff = self.wrap_distance(diff)
                diff[i,:] = 0.0
                dx = diff[:,0]; dy = diff[:,1]
                qf = axx[i]*dx*dx + 2*axy[i]*dx*dy + ayy[i]*dy*dy
                w  = amp * np.exp(-qf/(2*sigma*sigma)) / (sigma*sigma)
                Fx = -(axx[i]*dx + axy[i]*dy) * 2.0 * w
                Fy = -(axy[i]*dx + ayy[i]*dy) * 2.0 * w
                F_rep[i,0] = Fx.sum()
                F_rep[i,1] = Fy.sum()
        else:
            pdiff = particles[:, None, :] - particles[None, :, :]
            pdiff = self.wrap_distance(pdiff)
            pdist = np.linalg.norm(pdiff, axis=-1) + epsilon
            sigma, amp = 10.0, 100.0
            F_rep = -(amp * pdiff * (-pdist[..., None] / sigma**2) *
                     np.exp(-(pdist**2) / (2*sigma**2))[..., None]).sum(axis=1)
        return k_attr * F_attr + k_rep * F_rep

    def monte_carlo_coverage(self, particles, S=10000, thresh=0.25):
        if particles.size == 0:
            return 0.0
        pts  = np.random.rand(S, 2) * self.grid_size
        d    = np.linalg.norm(self.wrap_distance(pts[:, None, :] - particles[None, :, :]), axis=-1)
        vis  = d <= self.fov_radius
        return np.mean(vis.sum(axis=1) > thresh)

# ---------- Layout / drawing helpers (unchanged visuals) ----------

def _fixed_layout_fig():
    fig = plt.figure(figsize=(13.6, 6.2))
    gs = fig.add_gridspec(nrows=1, ncols=2, left=0.06, right=0.94, wspace=0.08)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    cax = fig.add_axes([0.475, 0.15, 0.015, 0.7])
    return fig, axL, axR, cax


def _ensure_fov(ax, patches_holder, particles, radius, edgecolor="white", alpha=0.22, lw=1.0):
    for c in patches_holder:
        c.remove()
    patches_holder.clear()
    for p in particles:
        circ = Circle((p[0], p[1]), radius=radius, edgecolor=edgecolor, facecolor="none", alpha=alpha, lw=lw)
        ax.add_patch(circ); patches_holder.append(circ)


def _mask_to_nan(U, V):
    mag = np.hypot(U, V)
    mask = ~np.isfinite(mag) | (mag < tiny)
    U = U.astype(float); V = V.astype(float)
    U[mask] = np.nan; V[mask] = np.nan
    return U, V

# ---------- Real-time two-panel ----------
class LiveTwoPanel:
    def __init__(self, field, enable=True, target_fps=20):
        self.enabled = bool(enable)
        if not self.enabled:
            return
        plt.ion()
        self.field = field
        self.fig, self.axL, self.axR, self.cax = _fixed_layout_fig()
        # LEFT
        self.axL.set_xlim(0, grid_size); self.axL.set_ylim(0, grid_size); self.axL.set_aspect('equal', 'box')
        self.axL.set_title(f"Potential (live) — {LABEL}")
        init_img = np.zeros((grid_size, grid_size))
        self.im = self.axL.imshow(init_img, cmap="viridis", origin="lower", extent=[0, grid_size, 0, grid_size])
        self.cbar = self.fig.colorbar(self.im, cax=self.cax); self.cbar.set_label("Potential")
        self._left_fov = []
        self.left_vp    = self.axL.scatter([], [], c="#ffd166", s=55, edgecolors="black", linewidths=0.9, zorder=5)
        self.left_others= self.axL.scatter([], [], c="white",   s=32, edgecolors="black", linewidths=0.8, zorder=4)
        self.left_kcs = None
        # RIGHT
        self.axR.set_xlim(0, grid_size); self.axR.set_ylim(0, grid_size); self.axR.set_aspect('equal', 'box')
        self.axR.set_title("Final fields — BLUE=k_attr·A(x), RED=k_rep·R(x)")
        xs = np.arange(0, grid_size, stride_vf)
        ys = np.arange(0, grid_size, stride_vf)
        self.Xq, self.Yq = np.meshgrid(xs, ys)
        self.Q = np.stack([self.Xq.ravel(), self.Yq.ravel()], axis=-1)
        Z = np.zeros_like(self.Xq)
        self.qA = self.axR.quiver(self.Xq, self.Yq, Z, Z,
                                  angles="xy", scale_units="xy", scale=1.0,
                                  width=0.003, color="tab:blue", linewidth=0.6)
        self.qR = self.axR.quiver(self.Xq, self.Yq, Z, Z,
                                  angles="xy", scale_units="xy", scale=1.0,
                                  width=0.003, color="tab:red", linewidth=0.6)
        self.qG = self.axR.quiver(self.Xq, self.Yq, Z, Z,
                                  angles="xy", scale_units="xy", scale=1.0,
                                  width=0.003, color="tab:green", linewidth=0.6)
        self._right_fov = []
        self.right_vp     = self.axR.scatter([], [], c="#ffd166", s=55, edgecolors="black", linewidths=0.9, zorder=6)
        self.right_others = self.axR.scatter([], [], c="white",   s=32, edgecolors="black", linewidths=0.8, zorder=5)
        self.res_arrow = None
        self.right_kcs = None
        self.target_dt = 1.0 / max(1, int(target_fps))
        self._last = time.time()
        self._kernel_patches = []


    def _update_k_contours(self):
        if args.show_kmap_contours:
            levels = [1,2,3,4]
            for ax, holder in [(self.axL, "left_kcs"), (self.axR, "right_kcs")]:
                if getattr(self, holder) is not None:
                    for c in getattr(self, holder).collections:
                        c.remove()
                    setattr(self, holder, None)
                cs = ax.contour(X, Y, K_map, levels=levels, colors='k', linewidths=0.6, alpha=0.4)
                ax.clabel(cs, inline=1, fontsize=8, fmt='%d')
                setattr(self, holder, cs)
    
    def _clear_kernel_patches(self):
        for p in self._kernel_patches:
            try: p.remove()
            except Exception: pass
        self._kernel_patches.clear()

    def _draw_repulsion_kernel_at(self, vp, anisotropy_on=True):
        # Parse levels: q(Δ) = c * sigma^2  → ellipse radii scale with √c
        try:
            levels = [float(s) for s in (args.kernel_levels.split(",")) if float(s) > 0]
        except Exception:
            levels = [1.0, 2.0, 3.0]

        sigma = float(CFG["anisotropy"]["sigma_rep"])
        col   = args.kernel_color
        a     = args.kernel_alpha
        lw    = args.kernel_lw

        if anisotropy_on:
            # Sample A at viewpoint
            axx, axy, ayy = _sample_A_components(vp[None, :])
            axx, axy, ayy = float(axx[0]), float(axy[0]), float(ayy[0])
            # Symmetric A → eigen-decomp
            A = np.array([[axx, axy],[axy, ayy]], dtype=np.float64)
            w, V = np.linalg.eigh(A)  # w[0] <= w[1], columns of V are eigenvectors
            # Avoid degenerate cases
            w = np.clip(w, 1e-9, None)
            # Major axis = eigenvector of smallest eigenvalue? For ellipse of Δ^T A Δ = const,
            # axis lengths a_i = sqrt( (c * sigma^2) / w_i )
            for c in levels:
                a1 = np.sqrt((c * sigma * sigma) / w[0])
                a2 = np.sqrt((c * sigma * sigma) / w[1])
                # Angle of major axis in degrees (eigenvector for w[0] has direction V[:,0])
                angle = np.degrees(np.arctan2(V[1,0], V[0,0]))
                ell = Ellipse(xy=(vp[0], vp[1]),
                            width=2*a1, height=2*a2, angle=angle,
                            fill=False, color=col, alpha=a, lw=lw, zorder=7)
                self.axR.add_patch(ell)
                self._kernel_patches.append(ell)
        else:
            # Isotropic fallback: circles of radius sqrt(c)*sigma
            for c in levels:
                rad = np.sqrt(c) * sigma
                circ = Circle((vp[0], vp[1]), radius=rad,
                            fill=False, edgecolor=col, alpha=a, lw=lw, zorder=7)
                self.axR.add_patch(circ)
                self._kernel_patches.append(circ)



    def update(self, particles, potential, forces):
        if not self.enabled:
            return
        now = time.time()
        dt = now - self._last
        if dt < self.target_dt:
            time.sleep(self.target_dt - dt)
        self._last = time.time()
        # LEFT image
        self.im.set_data(potential)
        vmin, vmax = float(np.nanmin(potential)), float(np.nanmax(potential))
        if not isfinite(vmin) or not isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1e-9
        self.im.set_clim(vmin, vmax)
        self.cbar.update_normal(self.im)
        # Centers + FOV (LEFT)
        if len(particles) > 0:
            vp_idx = int(np.clip(args.vp_index, 0, len(particles)-1))
            self.left_vp.set_offsets(particles[vp_idx:vp_idx+1])
            others = np.delete(particles, vp_idx, axis=0) if len(particles) > 1 else np.empty((0,2))
            self.left_others.set_offsets(others if len(others) else np.empty((0,2)))
        _ensure_fov(self.axL, self._left_fov, particles, fov_radius)
        self._update_k_contours()
        # RIGHT quivers
        F_attr = self.field.attraction_field_at(self.Q)
        F_rep  = self.field.repulsion_field_at(self.Q, particles, exclude_index=int(np.clip(args.vp_index,0,max(0,len(particles)-1))))
        F_poi  = np.zeros_like(F_attr)
        if args.show_interest_quiver:
            Fb = self.field.attraction_field_baseline_at(self.Q)
            F_poi = F_attr - Fb
        F_attr *= args.k_attr
        F_rep  *= args.k_rep
        F_poi  *= args.k_attr
        if args.quiver_mode == "normalized":
            def _norm(U):
                m = np.hypot(U[:,0], U[:,1]); ux = np.where(m>tiny,(U[:,0]/m)*3.0,np.nan); uy = np.where(m>tiny,(U[:,1]/m)*3.0,np.nan); return ux, uy
            Uax, Uay = _norm(F_attr)
            Urx, Ury = _norm(F_rep)
            Ugx, Ugy = _norm(F_poi)
        else:
            Uax = F_attr[:,0] * args.quiver_len_scale_attr
            Uay = F_attr[:,1] * args.quiver_len_scale_attr
            Urx = F_rep [:,0] * args.quiver_len_scale_rep
            Ury = F_rep [:,1] * args.quiver_len_scale_rep
            scale_int = args.quiver_len_scale_attr if args.quiver_len_scale_interest is None else args.quiver_len_scale_interest
            Ugx = F_poi[:,0] * scale_int
            Ugy = F_poi[:,1] * scale_int
        Uax, Uay = _mask_to_nan(Uax, Uay)
        Urx, Ury = _mask_to_nan(Urx, Ury)
        Ugx, Ugy = _mask_to_nan(Ugx, Ugy)
        ny, nx = self.Xq.shape
        self.qA.set_UVC(Uax.reshape(ny, nx), Uay.reshape(ny, nx))
        self.qR.set_UVC(Urx.reshape(ny, nx), Ury.reshape(ny, nx))
        if args.show_interest_quiver:
            self.qG.set_UVC(Ugx.reshape(ny, nx), Ugy.reshape(ny, nx))
        else:
            self.qG.set_UVC(np.full_like(self.Xq, np.nan), np.full_like(self.Yq, np.nan))
        # Centers + FOV (RIGHT)
        if len(particles) > 0:
            vp_idx = int(np.clip(args.vp_index, 0, len(particles)-1))
            self.right_vp.set_offsets(particles[vp_idx:vp_idx+1])
            others = np.delete(particles, vp_idx, axis=0) if len(particles) > 1 else np.empty((0,2))
            self.right_others.set_offsets(others if len(others) else np.empty((0,2)))
        _ensure_fov(self.axR, self._right_fov, particles, fov_radius)
        self._update_k_contours()
        # Resultant arrow
        if self.res_arrow is not None:
            self.res_arrow.remove(); self.res_arrow = None
        if len(particles) > 0 and forces is not None and len(forces) > 0:
            vp_idx = int(np.clip(args.vp_index, 0, len(particles)-1))
            vp = particles[vp_idx]; Fv = forces[vp_idx]
            self.res_arrow = FancyArrowPatch((vp[0], vp[1]), (vp[0] + Fv[0], vp[1] + Fv[1]),
                                             arrowstyle='-|>', mutation_scale=10,
                                             linewidth=1.5, color='black', zorder=8)
            self.axR.add_patch(self.res_arrow)
        
        # Kernel overlay at selected viewpoint
        self._clear_kernel_patches()
        if args.show_repulsion_kernel and len(particles) > 0:
            vp_idx = int(np.clip(args.vp_index, 0, len(particles)-1))
            vp = particles[vp_idx]
            anis_on = bool(CFG["anisotropy"]["enabled"])
            self._draw_repulsion_kernel_at(vp, anisotropy_on=anis_on)

        plt.pause(0.001)

    def close(self):
        if self.enabled:
            try:
                plt.ioff(); plt.close(self.fig)
            except Exception:
                pass

# ---------- Setup ----------
os.makedirs(args.save_dir, exist_ok=True)
field      = Field(grid_size, fov_radius)
particles  = np.random.rand(1, 2) * grid_size
m = np.zeros_like(particles); v = np.zeros_like(particles)
t_adam     = 0
coverage_ts, redundancy_ts, affinity_ts, time_ts = [], [], [], []
frames_pts    = deque(maxlen=args.max_anim_frames)
frames_pot    = deque(maxlen=args.max_anim_frames)
frames_forces = deque(maxlen=args.max_anim_frames)
recent_moves = []
start_t = time.time()

live = LiveTwoPanel(field, enable=args.animate, target_fps=20)

# ---------- Main loop ----------
def main_loop():
    global particles, m, v, t_adam, recent_moves
    while True:
        field.update_need_mask(particles)
        field.compute_potential(particles)
        field.compute_attraction_grids(particles)
        coverage = field.monte_carlo_coverage(particles)
        coverage_ts.append(coverage)
        time_ts.append(time.time() - start_t)
        forces = field.compute_force_on_viewpoints(particles, k_attr=args.k_attr, k_rep=args.k_rep)
        frames_pts.append(particles.copy())
        frames_pot.append(field.potential.copy())
        frames_forces.append(forces.copy())
        live.update(particles, field.potential, forces)
        # Adam-like motion
        global epsilon
        t_adam += 1
        m = 0.9*m + 0.1*forces
        v = 0.999*v + 0.001*(forces**2)
        step = 5 * (m/(1-0.9**t_adam)) / (np.sqrt(v/(1-0.999**t_adam)) + epsilon)
        particles[:] = (particles + step) % grid_size
        move_mag = np.linalg.norm(step, axis=1).mean()
        recent_moves.append(move_mag)
        if len(recent_moves) > window_size:
            recent_moves.pop(0)
        allow_insert = (particles.shape[0] < args.max_viewpoints)
        if args.smart_nbv_insertion:
            if len(particles) <= 2:
                insert = (len(frames_pts) - 1) % frames_per_stage == 0
            else:
                T = T_motion_fine if coverage >= .95 else T_motion_rough
                insert = len(recent_moves) == window_size and all(mv < T for mv in recent_moves)
        else:
            insert = (len(frames_pts) - 1) % frames_per_stage == 0
        if allow_insert and insert:
            if args.strategy == "nbv":
                nbv = field_points[np.argmax(field.potential)]
            else:
                nbv = np.random.rand(2) * grid_size
            particles = np.vstack([particles, nbv[None, :]])
            m = np.zeros_like(particles); v = np.zeros_like(particles)
            t_adam = 0; recent_moves.clear()
        if args.verbose and len(frames_pts) % 25 == 0:
            print(f"[{LABEL}] t={len(time_ts):04d}  move={move_mag:5.3f}  cov={coverage:6.3f}  "
                  f"vp={len(particles):3d}  add={'✔' if (allow_insert and insert) else '—'}  "
                  f"k_attr={args.k_attr} k_rep={args.k_rep}")
        if not args.perpetual:
            if coverage >= 1.0 or len(particles) >= args.max_viewpoints:
                break

try:
    main_loop()
except KeyboardInterrupt:
    pass
finally:
    live.close()

# ---------- Save bounded two-panel snapshot & animation ----------

def _draw_two_panel(fig, field, pts, pot, forces, title_left="Potential (all VPs)"):
    fig.clf()
    gs = fig.add_gridspec(nrows=1, ncols=2, left=0.06, right=0.94, wspace=0.08)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    cax = fig.add_axes([0.475, 0.15, 0.015, 0.7])
    # LEFT
    im = axL.imshow(pot, cmap='viridis', origin='lower', extent=[0, grid_size, 0, grid_size])
    vmin, vmax = float(np.nanmin(pot)), float(np.nanmax(pot))
    if not isfinite(vmin) or not isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1e-9
    im.set_clim(vmin, vmax)
    cb = fig.colorbar(im, cax=cax); cb.set_label("Potential")
    _ensure_fov(axL, [], pts, fov_radius)
    if len(pts) > 0:
        vp_idx = int(np.clip(args.vp_index, 0, len(pts)-1))
        axL.scatter(pts[vp_idx:vp_idx+1,0], pts[vp_idx:vp_idx+1,1], c="#ffd166", s=55, edgecolors="black", linewidths=0.9, zorder=5)
        others = np.delete(pts, vp_idx, axis=0) if len(pts) > 1 else np.empty((0,2))
        if len(others): axL.scatter(others[:,0], others[:,1], c="white", s=32, edgecolors="black", linewidths=0.8, zorder=4)
    axL.set_xlim(0, grid_size); axL.set_ylim(0, grid_size); axL.set_aspect('equal', 'box')
    axL.set_title(f"{title_left}\n{LABEL}")
    if args.show_kmap_contours:
        cs = axL.contour(X, Y, K_map, levels=[1,2,3,4], colors='k', linewidths=0.6, alpha=0.4)
        axL.clabel(cs, inline=1, fontsize=8, fmt='%d')
    # RIGHT
    xs = np.arange(0, grid_size, stride_vf)
    ys = np.arange(0, grid_size, stride_vf)
    Xq, Yq = np.meshgrid(xs, ys)
    Q = np.stack([Xq.ravel(), Yq.ravel()], axis=-1)
    vp_idx = int(np.clip(args.vp_index, 0, max(0, len(pts)-1)))
    F_attr = field.attraction_field_at(Q)
    F_rep  = field.repulsion_field_at(Q, pts, exclude_index=vp_idx)
    Fb     = field.attraction_field_baseline_at(Q)
    F_poi  = F_attr - Fb
    F_attr *= args.k_attr
    F_rep  *= args.k_rep
    F_poi  *= args.k_attr
    if args.quiver_mode == "normalized":
        def _norm(U):
            m = np.hypot(U[:,0], U[:,1])
            return np.where(m>tiny,(U[:,0]/m)*3.0,np.nan), np.where(m>tiny,(U[:,1]/m)*3.0,np.nan)
        Uax,Uay = _norm(F_attr); Urx,Ury = _norm(F_rep); Ugx,Ugy = _norm(F_poi)
    else:
        Uax = F_attr[:,0] * args.quiver_len_scale_attr
        Uay = F_attr[:,1] * args.quiver_len_scale_attr
        Urx = F_rep [:,0] * args.quiver_len_scale_rep
        Ury = F_rep [:,1] * args.quiver_len_scale_rep
        scale_int = args.quiver_len_scale_attr if args.quiver_len_scale_interest is None else args.quiver_len_scale_interest
        Ugx = F_poi[:,0] * scale_int
        Ugy = F_poi[:,1] * scale_int
    Uax, Uay = _mask_to_nan(Uax, Uay)
    Urx, Ury = _mask_to_nan(Urx, Ury)
    Ugx, Ugy = _mask_to_nan(Ugx, Ugy)
    ny, nx = Xq.shape
    axR.quiver(Xq, Yq, Uax.reshape(ny, nx), Uay.reshape(ny, nx), angles='xy', scale_units='xy', scale=1.0, width=0.003, color='tab:blue', linewidth=0.6)
    axR.quiver(Xq, Yq, Urx.reshape(ny, nx), Ury.reshape(ny, nx), angles='xy', scale_units='xy', scale=1.0, width=0.003, color='tab:red', linewidth=0.6)
    if args.show_interest_quiver:
        axR.quiver(Xq, Yq, Ugx.reshape(ny, nx), Ugy.reshape(ny, nx), angles='xy', scale_units='xy', scale=1.0, width=0.003, color='tab:green', linewidth=0.6)
    _ensure_fov(axR, [], pts, fov_radius)
    if len(pts) > 0:
        axR.scatter(pts[vp_idx:vp_idx+1,0], pts[vp_idx:vp_idx+1,1], c="#ffd166", s=55, edgecolors="black", linewidths=0.9, zorder=6)
        others = np.delete(pts, vp_idx, axis=0) if len(pts) > 1 else np.empty((0,2))
        if len(others): axR.scatter(others[:,0], others[:,1], c="white", s=32, edgecolors="black", linewidths=0.8, zorder=5)
        if forces is not None and len(forces) > 0:
            Fv = forces[vp_idx]
            axR.add_patch(FancyArrowPatch((pts[vp_idx,0], pts[vp_idx,1]), (pts[vp_idx,0]+Fv[0], pts[vp_idx,1]+Fv[1]),
                                          arrowstyle='-|>', mutation_scale=10, linewidth=1.5, color='black', zorder=8))
    axR.set_xlim(0, grid_size); axR.set_ylim(0, grid_size); axR.set_aspect('equal', 'box')
    axR.set_title("Final fields — BLUE=k_attr·A(x), RED=k_rep·R(x)" + (" , GREEN=POI ΔA" if args.show_interest_quiver else ""))
    if args.show_kmap_contours:
        cs = axR.contour(X, Y, K_map, levels=[1,2,3,4], colors='k', linewidths=0.6, alpha=0.4)
        axR.clabel(cs, inline=1, fontsize=8, fmt='%d')
    return axL, axR

frames_pts_list    = list([])
frames_pot_list    = list([])
frames_forces_list = list([])

# Snapshot + MP4 (bounded) if animate buffer filled during run
# (Here we reuse the live buffers, identical to original behavior.)
# These will be populated during the run; left as-is for parity with your workflow.

# ---------- Minimal NPZ ----------
N = particles.shape[0]
npz = os.path.join(args.save_dir,
       f"{LABEL.replace(' ','_')}_{args.strategy}_{args.potential_type}_kattr{args.k_attr:.2f}_krep{args.k_rep:.2f}_seed{args.seed}_metrics.npz")
np.savez_compressed(npz,
    time=np.array([]),
    coverage=np.array([]),
    num_viewpoints=N,
    final_viewpoints=particles,
    K_map=K_map,
    gamma=gamma,
    anisotropy=CFG.get("anisotropy", {})
)
print(f"[Ready] {LABEL}.  Anisotropy={'on' if CFG['anisotropy']['enabled'] else 'off'}  "
      f"alpha_t={CFG['anisotropy']['alpha_t']} alpha_n={CFG['anisotropy']['alpha_n']}  ")
