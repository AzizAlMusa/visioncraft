#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NBV Potential-Field (Toroidal) with Edge-Boosted Attraction and Proximity-Ramped Anisotropic Repulsion

Goals satisfied:
1) Coverage everywhere (baseline ω=1 ensures attraction across the whole field).
2) Areas of interest require K(x)≥2 (set via poi_config with req=2).
3) Attraction prefers edges of interest (edge-boost, not replacement).
4) Repulsion is anisotropic and smoothly ramps with proximity to interest edges (toroidal convolution).
5) Anisotropy is a strict superset: alpha_t == alpha_n => EXACT isotropic forces.

Run:
    python overlap_anistropic.py
Requires:
    overlap_config.json  (JSON-only, no CLI).
"""

import os, json, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Ellipse
from collections import deque

# ----------------------- Load JSON -----------------------

DEFAULT_CONFIG_PATH = "overlap_config.json"

def load_cfg(path=DEFAULT_CONFIG_PATH):
    with open(path, "r") as f:
        cfg = json.load(f)

    # Top-level defaults
    cfg.setdefault("seed", 0)
    cfg.setdefault("save_dir", "./phase2_results")
    cfg.setdefault("animate", True)

    cfg.setdefault("grid_size", 100)
    cfg.setdefault("fov_radius", 20)

    cfg.setdefault("strategy", "nbv")     # "nbv" or "random"
    cfg.setdefault("potential_type", "log")
    cfg.setdefault("k_attr", 10.0)
    cfg.setdefault("k_rep", 0.25)
    cfg.setdefault("max_viewpoints", 100)
    cfg.setdefault("vp_index", 0)
    cfg.setdefault("perpetual", True)
    cfg.setdefault("max_anim_frames", 600)

    cfg.setdefault("quiver_mode", "length")  # "length" | "normalized"
    cfg.setdefault("quiver_len_scale_attr", 1.0)
    cfg.setdefault("quiver_len_scale_rep", 1.0)
    cfg.setdefault("show_interest_quiver", False)
    cfg.setdefault("show_kmap_contours", False)
    cfg.setdefault("quiver_len_scale_interest", None)

    cfg.setdefault("show_repulsion_kernel", False)
    cfg.setdefault("show_repulsion_kernel_all", False)
    cfg.setdefault("kernel_levels", "1,2,3")
    cfg.setdefault("kernel_color", "#111111")
    cfg.setdefault("kernel_alpha", 0.30)
    cfg.setdefault("kernel_lw", 1.2)

    cfg.setdefault("verbose", False)

    cfg.setdefault("vis_beta", 20.0)
    cfg.setdefault("gamma", 1.0)
    cfg.setdefault("need_hard_clip", True)
    cfg.setdefault("tau_clip", 1.0)

    cfg.setdefault("importance_config", "")
    cfg.setdefault("poi_config", "")

    cfg.setdefault("smart_nbv_insertion", False)
    cfg.setdefault("frames_per_stage", 10)
    cfg.setdefault("T_motion_rough", 1.5)
    cfg.setdefault("T_motion_fine", 0.15)
    cfg.setdefault("adam_step_gain", 5.0)

    # Interest overlay on right panel (optional)
    cfg.setdefault("show_interest_overlay_right", False)
    cfg.setdefault("interest_overlay_mode", "heat")  # "heat"|"contours"|"both"
    cfg.setdefault("interest_overlay_alpha", 0.25)
    cfg.setdefault("interest_contour_levels", "0.2,0.4,0.6,0.8")
    cfg.setdefault("interest_contour_color", "#6a5acd")
    cfg.setdefault("interest_contour_alpha", 0.65)
    cfg.setdefault("interest_colormap", "magma")

    # Edge-driven attraction config
    cfg.setdefault("interest_mode", "edge_grad")   # "area"|"edge_grad"|"edge_lap"|"edge_dog"
    cfg.setdefault("edge_power", 1.0)
    cfg.setdefault("edge_eps", 1e-6)
    cfg.setdefault("edge_norm", "max")            # "max"|"zscore"
    cfg.setdefault("edge_weight_scale", 1.0)
    cfg.setdefault("dog_sigma_inner", 1.5)
    cfg.setdefault("dog_sigma_outer", 4.0)
    cfg.setdefault("dog_positive_only", True)
    # Keep normal potential everywhere; edges get boosted
    cfg.setdefault("edge_mode", "boost")          # "boost"|"replace"
    cfg.setdefault("edge_gain", 1.0)
    cfg.setdefault("edge_mix_with_area", 0.0)     # used only when "replace"

    # Anisotropy defaults + proximity ramp
    an = cfg.setdefault("anisotropy", {})
    an.setdefault("enabled", True)
    an.setdefault("alpha_t", 6.0)
    an.setdefault("alpha_n", 1.0)
    an.setdefault("sigma_rep", 10.0)
    an.setdefault("amp_rep", 100.0)
    an.setdefault("smooth_sigma", 1.5)
    an.setdefault("grad_eps", 1e-3)
    an.setdefault("tensor_blend", 0.0)
    an.setdefault("blend_tau", 0.01)       # local gradient threshold → identity
    an.setdefault("blend_scale", 0.05)     # slope region
    an.setdefault("importance_floor", 0.20)

    # Proximity ramp for anisotropy (smooth, not short-sighted)
    an.setdefault("aniso_prox_sigma", 6.0)     # toroidal Gaussian on edge map
    an.setdefault("aniso_prox_gain", 4.0)      # scales proximity strength
    an.setdefault("aniso_prox_bias", 0.0)      # bias before logistic
    an.setdefault("aniso_prox_steep", 2.0)     # logistic steepness

    return cfg

CFG = load_cfg()

np.random.seed(int(CFG["seed"]))
os.makedirs(CFG["save_dir"], exist_ok=True)

LABEL = "PF (toroidal) — edge-boost + proximity-ramped anisotropic repulsion"
grid_size = int(CFG["grid_size"])
fov_radius = float(CFG["fov_radius"])
epsilon = 1e-9
tiny = 1e-12
stride_vf = 4

x = np.arange(grid_size)
y = np.arange(grid_size)
X, Y = np.meshgrid(x, y)
field_points = np.stack([X.ravel(), Y.ravel()], axis=-1)  # (P,2)

# ----------------------- Toroidal utils -----------------------

def wrap_distance(diff):
    """Shortest wrapped displacement per axis on a torus."""
    half = grid_size / 2.0
    return np.where(np.abs(diff) > half, -np.sign(diff) * (grid_size - np.abs(diff)), diff)

def roll2(A, dy, dx):
    return np.roll(np.roll(A, int(dy), axis=0), int(dx), axis=1)

def gaussian_kernel_1d(sigma):
    if sigma <= 0:
        return np.array([1.0], dtype=np.float64)
    r = int(max(1, round(3*sigma)))
    t = np.arange(-r, r+1, dtype=np.float64)
    k = np.exp(-(t*t)/(2*sigma*sigma))
    k /= np.sum(k)
    return k

def toroidal_gaussian_smooth(img, sigma):
    k = gaussian_kernel_1d(sigma)
    tmp = np.zeros_like(img, dtype=np.float64)
    for i, w in enumerate(k):
        dx = i - (len(k)//2)
        tmp += w * roll2(img, 0, dx)
    out = np.zeros_like(tmp, dtype=np.float64)
    for j, w in enumerate(k):
        dy = j - (len(k)//2)
        out += w * roll2(tmp, dy, 0)
    return out

def toroidal_gradient(img):
    fx = 0.5 * (roll2(img, 0, +1) - roll2(img, 0, -1))
    fy = 0.5 * (roll2(img, +1, 0) - roll2(img, -1, 0))
    return fx, fy

def bilinear_sample_periodic(F, pts):
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

# ----------------------- FFT attraction kernels -----------------------

def build_fft_kernels(N):
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

def fft_convolve_vector(n_grid, Kx_hat, Ky_hat, N):
    n_hat = np.fft.rfftn(n_grid, s=(N, N))
    Fx = np.fft.irfftn(n_hat * Kx_hat, s=(N, N))
    Fy = np.fft.irfftn(n_hat * Ky_hat, s=(N, N))
    return Fx, Fy

Kx_hat, Ky_hat = build_fft_kernels(grid_size)

# ----------------------- POI / importance maps -----------------------

def _parse_number(s):
    try:
        v = float(s); iv = int(v)
        return iv if abs(v - iv) < 1e-12 else v
    except Exception:
        return None

def _apply_disk(cx, cy, r):
    dx = np.mod(X - cx + grid_size/2, grid_size) - grid_size/2
    dy = np.mod(Y - cy + grid_size/2, grid_size) - grid_size/2
    return (dx*dx + dy*dy) <= r*r

def _apply_band(theta_deg, b, width):
    th = np.deg2rad(theta_deg)
    n = np.array([np.cos(th), np.sin(th)])
    proj = (X * n[0] + Y * n[1])
    return (np.abs(proj - b) <= (width / 2.0))

def build_maps_from_poi(poi_config_str, vis_beta_default):
    K_map    = np.ones((grid_size, grid_size), dtype=np.float64)
    beta_map = np.full((grid_size, grid_size), float(vis_beta_default), dtype=np.float64)
    omega_map= np.ones((grid_size, grid_size), dtype=np.float64)

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
                if e.startswith("req="):   req = _parse_number(e.split("=",1)[1])
                elif e.startswith("omega="):
                    val = e.split("=",1)[1]
                    if val.startswith("+"): omega_add = float(val[1:])
                    elif val.startswith("*"): omega_mul = float(val[1:])
                elif e.startswith("beta="): beta = float(e.split("=",1)[1])
            region = _apply_disk(cx, cy, r)

        elif tok.startswith("band:"):
            body = tok[len("band:"):].split(",")
            if len(body) < 3: continue
            theta_deg, b, width = map(float, body[:3])
            extras = body[3:]
            req = None; omega_add=None; omega_mul=None; beta=None
            for e in extras:
                e = e.strip()
                if e.startswith("req="):   req = _parse_number(e.split("=",1)[1])
                elif e.startswith("omega="):
                    val = e.split("=",1)[1]
                    if val.startswith("+"): omega_add = float(val[1:])
                    elif val.startswith("*"): omega_mul = float(val[1:])
                elif e.startswith("beta="): beta = float(e.split("=",1)[1])
            region = _apply_band(theta_deg, b, width)

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

K_map, beta_map, omega_map = build_maps_from_poi(CFG.get("poi_config",""), CFG["vis_beta"])
gamma         = float(CFG["gamma"])
need_hard_clip= bool(CFG["need_hard_clip"])
tau_clip      = float(CFG["tau_clip"])

# ----------------------- Importance, edges, and W_eff -----------------------

# Base area importance (uniform 1 unless POIs modulate omega)
W_area = np.clip(omega_map.astype(np.float64) ** gamma, 0.0, None)
if W_area.max() > 0:
    W_area = W_area / (W_area.max() + 1e-12)

# Smoothed importance (for edge extraction and anisotropy)
I0 = W_area  # already ω^γ normalized
I_smooth = toroidal_gaussian_smooth(I0, float(CFG["anisotropy"]["smooth_sigma"]))
Gx, Gy = toroidal_gradient(I_smooth)
Grad_mag = np.sqrt(Gx*Gx + Gy*Gy)

def toroidal_laplacian(F):
    return (
        roll2(F, 0, +1) + roll2(F, 0, -1) + roll2(F, +1, 0) + roll2(F, -1, 0)
        - 4.0 * F
    )

def normalize_edge(E, mode="max"):
    if mode == "zscore":
        mu = np.mean(E); sd = np.std(E) + 1e-12
        Z = (E - mu) / sd
        Z = np.maximum(0.0, Z)
        M = Z.max() + 1e-12
        return Z / M
    M = E.max() + 1e-12
    return E / M

# Edge maps
Lap = toroidal_laplacian(I_smooth)
I_inner = toroidal_gaussian_smooth(I0, float(CFG["dog_sigma_inner"]))
I_outer = toroidal_gaussian_smooth(I0, float(CFG["dog_sigma_outer"]))
DoG = I_inner - I_outer
if CFG["dog_positive_only"]:
    DoG = np.maximum(0.0, DoG)

edge_mode_name = str(CFG["interest_mode"]).lower()
edge_eps = float(CFG["edge_eps"])
edge_pow = float(CFG["edge_power"])
if edge_mode_name == "edge_grad":
    E = np.power(np.maximum(Grad_mag, edge_eps), edge_pow)
    W_edge = normalize_edge(E, CFG["edge_norm"])
elif edge_mode_name == "edge_lap":
    E = np.power(np.maximum(np.abs(Lap), edge_eps), edge_pow)
    W_edge = normalize_edge(E, CFG["edge_norm"])
elif edge_mode_name == "edge_dog":
    E = np.power(np.maximum(DoG, edge_eps), edge_pow)
    W_edge = normalize_edge(E, CFG["edge_norm"])
else:
    W_edge = None

# Keep normal potential everywhere; edges boost it
edge_mode = str(CFG.get("edge_mode", "boost")).lower()
edge_gain = float(CFG.get("edge_gain", 1.0))
mix = float(CFG.get("edge_mix_with_area", 0.0))

if W_edge is None or edge_mode == "replace":
    if W_edge is None:
        W_eff = W_area
    else:
        W_eff = (1.0 - mix) * W_edge + mix * W_area
else:
    # BOOST mode (recommended): areas outside interest keep W_area,
    # edges multiply it up, never to zero elsewhere.
    W_eff = W_area * (1.0 + edge_gain * W_edge)

W_eff *= float(CFG["edge_weight_scale"])
W_eff = np.clip(W_eff, 0.0, None)

# ----------------------- Anisotropy tensors A_eff(x) with proximity ramp -----------------------

grad_eps = float(CFG["anisotropy"]["grad_eps"])

# Tangent / normal unit directions
n_x = np.where(Grad_mag > grad_eps, Gx/(Grad_mag+1e-12), 0.0)
n_y = np.where(Grad_mag > grad_eps, Gy/(Grad_mag+1e-12), 0.0)
t_x = -n_y
t_y =  n_x

# Optional principal-direction blend
if float(CFG["anisotropy"]["tensor_blend"]) > 0.0:
    Sxx = toroidal_gaussian_smooth(Gx*Gx, 1.0)
    Syy = toroidal_gaussian_smooth(Gy*Gy, 1.0)
    Sxy = toroidal_gaussian_smooth(Gx*Gy, 1.0)
    trace = Sxx + Syy
    det   = Sxx*Syy - Sxy*Sxy
    tmp   = np.sqrt(np.maximum(trace*trace - 4*det, 0.0))
    lam1  = 0.5*(trace + tmp)
    evx = np.where(np.abs(Sxy) + np.abs(Sxx - lam1) > 1e-12, Sxy, 1.0)
    evy = np.where(np.abs(Sxy) + np.abs(Sxx - lam1) > 1e-12, lam1 - Sxx, 0.0)
    nor = np.sqrt(evx*evx + evy*evy) + 1e-12
    evx /= nor; evy /= nor
    b = float(CFG["anisotropy"]["tensor_blend"])
    t_x = (1-b)*t_x + b*evx
    t_y = (1-b)*t_y + b*evy
    tn = np.sqrt(t_x*t_x + t_y*t_y) + 1e-12
    t_x /= tn; t_y /= tn
    n_x = -t_y; n_y = t_x

alpha_t = float(CFG["anisotropy"]["alpha_t"])
alpha_n = float(CFG["anisotropy"]["alpha_n"])

# Raw tensor field A_raw(x) = alpha_t * t t^T + alpha_n * n n^T
axx_raw = alpha_t*(t_x*t_x) + alpha_n*(n_x*n_x)
axy_raw = alpha_t*(t_x*t_y) + alpha_n*(n_x*n_y)
ayy_raw = alpha_t*(t_y*t_y) + alpha_n*(n_y*n_y)

# Proximity map (smooth ramp near edges) via toroidal Gaussian on the edge field
# Use gradient magnitude as the seed for proximity unless another mode is chosen
prox_seed = W_edge if W_edge is not None else normalize_edge(Grad_mag, "max")
prox_sigma = float(CFG["anisotropy"]["aniso_prox_sigma"])
prox_map = toroidal_gaussian_smooth(prox_seed, prox_sigma)
# Normalize to [0,1]
prox_map = (prox_map - prox_map.min()) / (prox_map.max() - prox_map.min() + 1e-12)

# Logistic ramp: s_prox in [0,1], small far from edges, rises smoothly near edges
prox_gain  = float(CFG["anisotropy"]["aniso_prox_gain"])
prox_bias  = float(CFG["anisotropy"]["aniso_prox_bias"])
prox_k     = float(CFG["anisotropy"]["aniso_prox_steep"])
s_prox = 1.0 / (1.0 + np.exp(-prox_k * (prox_gain * (prox_map - prox_bias))))

# Superset normalization:
# If alpha_t == alpha_n => A_eff = I exactly.
# Else det-normalize A_raw, then ramp it with s_prox and fallback-to-identity in flat/noise regions.
eq_tol = 1e-12
if abs(alpha_t - alpha_n) < eq_tol:
    axx_eff_base = np.ones_like(axx_raw)
    axy_eff_base = np.zeros_like(axy_raw)
    ayy_eff_base = np.ones_like(ayy_raw)
else:
    detA = axx_raw*ayy_raw - axy_raw*axy_raw
    detA = np.where(detA <= 0, 1.0, detA)
    scale_det = np.sqrt(detA) + 1e-12
    axx_eff_base = axx_raw / scale_det
    axy_eff_base = axy_raw / scale_det
    ayy_eff_base = ayy_raw / scale_det

# Blend to identity based on: (a) local structure strength and (b) proximity ramp
# First: structure-based blend (avoid anisotropy in flat regions)
tau   = float(CFG["anisotropy"]["blend_tau"])
scale = float(CFG["anisotropy"]["blend_scale"])
Imin  = float(CFG["anisotropy"]["importance_floor"])
s_struct = np.clip((Grad_mag - tau) / max(scale, 1e-9), 0.0, 1.0)
if Imin > 0:
    s_struct = s_struct * (I_smooth > Imin)

# Final ramp multiplier
s_total = np.clip(s_struct * s_prox, 0.0, 1.0)

# A_eff(x) = (1 - s_total) * I + s_total * A_eff_base
axx_grid = s_total*axx_eff_base + (1.0 - s_total)*1.0
axy_grid = s_total*axy_eff_base + (1.0 - s_total)*0.0
ayy_grid = s_total*ayy_eff_base + (1.0 - s_total)*1.0

def sample_Aeff(Q):
    axx = bilinear_sample_periodic(axx_grid, Q)
    axy = bilinear_sample_periodic(axy_grid, Q)
    ayy = bilinear_sample_periodic(ayy_grid, Q)
    return axx, axy, ayy

# ----------------------- Field object -----------------------

class Field:
    def __init__(self, grid_size, fov_radius):
        self.grid_size = grid_size
        self.fov_radius= fov_radius
        self.need_weight = np.ones(grid_size * grid_size, dtype=np.float64)
        self.potential   = np.zeros((grid_size, grid_size), dtype=np.float64)
        self._attr_Fx    = np.zeros((grid_size, grid_size), dtype=np.float64)
        self._attr_Fy    = np.zeros((grid_size, grid_size), dtype=np.float64)
        self._base_Fx    = np.zeros((grid_size, grid_size), dtype=np.float64)
        self._base_Fy    = np.zeros((grid_size, grid_size), dtype=np.float64)

    def _compute_vis_and_need(self, particles):
        K = particles.shape[0]
        if K == 0:
            # Even with no VPs, maintain baseline pull (coverage everywhere)
            self.need_weight[:] = W_eff.ravel()
            return

        diff = field_points[:, None, :] - particles[None, :, :]
        diff = wrap_distance(diff)
        dist = np.linalg.norm(diff, axis=-1) + 1e-12

        beta_q = np.repeat(beta_map.ravel()[:, None], K, axis=1)
        s = 1.0 / (1.0 + np.exp(beta_q * (dist / self.fov_radius - 1.0)))
        vis = s.sum(axis=1)

        Kq = K_map.ravel()  # requirements (1 everywhere by default; req=2 in POIs)
        deficit = np.maximum(0.0, 1.0 - vis / Kq)
        if need_hard_clip:
            deficit = np.where(vis >= tau_clip * Kq, 0.0, deficit)

        # Apply the effective importance (area baseline + edge boost)
        self.need_weight[:] = deficit * W_eff.ravel()

    def update_need_mask(self, particles):
        self._compute_vis_and_need(particles)

    def compute_potential(self, particles, alpha=1.0):
        if particles.shape[0] == 0:
            self.potential.fill(0.0)
            return
        diff = field_points[:, None, :] - particles[None, :, :]
        diff = wrap_distance(diff)
        dist = np.linalg.norm(diff, axis=-1) + 1e-12
        base = np.log(dist).sum(axis=1)
        self.potential = (alpha * self.need_weight * base).reshape(grid_size, grid_size)

    def compute_attraction_grids(self, particles):
        n_grid = self.need_weight.reshape(grid_size, grid_size)
        Fx, Fy = fft_convolve_vector(n_grid, Kx_hat, Ky_hat, grid_size)
        self._attr_Fx[:], self._attr_Fy[:] = Fx, Fy

        # Baseline for "interest delta" quiver (optional)
        if particles.size == 0:
            base = np.ones_like(n_grid)
        else:
            diff = field_points[:, None, :] - particles[None, :, :]
            diff = wrap_distance(diff)
            dist = np.linalg.norm(diff, axis=-1) + 1e-12
            min_d = dist.min(axis=1)
            base = (min_d > self.fov_radius).astype(np.float64).reshape(grid_size, grid_size)
        Fx_b, Fy_b = fft_convolve_vector(base, Kx_hat, Ky_hat, grid_size)
        self._base_Fx[:], self._base_Fy[:] = Fx_b, Fy_b

    def attraction_field_at(self, Q):
        if Q.ndim == 1: Q = Q[None, :]
        Ux = bilinear_sample_periodic(self._attr_Fx, Q)
        Uy = bilinear_sample_periodic(self._attr_Fy, Q)
        return np.stack([Ux, Uy], axis=1)

    def attraction_field_baseline_at(self, Q):
        if Q.ndim == 1: Q = Q[None, :]
        Ux = bilinear_sample_periodic(self._base_Fx, Q)
        Uy = bilinear_sample_periodic(self._base_Fy, Q)
        return np.stack([Ux, Uy], axis=1)

    # Unified repulsion: superset (anisotropic → isotropic when A=I)
    def _pairwise_repulsion(self, Q, others):
        if Q.ndim == 1: Q = Q[None, :]
        if others.shape[0] == 0:
            return np.zeros((Q.shape[0], 2))
        sigma = float(CFG["anisotropy"]["sigma_rep"])
        amp   = float(CFG["anisotropy"]["amp_rep"])
        out = np.zeros((Q.shape[0], 2), dtype=np.float64)

        axx, axy, ayy = sample_Aeff(Q)  # sample at receivers
        for k in range(others.shape[0]):
            d = Q - others[k:k+1, :]
            d = wrap_distance(d)
            dx = d[:,0]; dy = d[:,1]
            q = axx*dx*dx + 2*axy*dx*dy + ayy*dy*dy
            coeff = (amp / (sigma*sigma)) * np.exp(-q/(2*sigma*sigma))
            Adx = axx*dx + axy*dy
            Ady = axy*dx + ayy*dy
            out[:,0] += coeff * Adx
            out[:,1] += coeff * Ady
        return out

    def repulsion_field_at(self, Q, particles, exclude_index=0):
        if Q.ndim == 1: Q = Q[None, :]
        if particles.shape[0] <= 1:
            return np.zeros((Q.shape[0], 2))
        others = np.delete(particles, exclude_index, axis=0)
        return self._pairwise_repulsion(Q, others)

    def compute_force_on_viewpoints(self, particles, k_attr=0.4, k_rep=1.0):
        K = particles.shape[0]
        if K == 0:
            return np.zeros((0,2))
        Fax = bilinear_sample_periodic(self._attr_Fx, particles)
        Fay = bilinear_sample_periodic(self._attr_Fy, particles)
        F_attr = np.stack([Fax, Fay], axis=1)
        F_rep = np.zeros_like(F_attr)
        for i in range(K):
            others = np.delete(particles, i, axis=0)
            F_rep[i] = self._pairwise_repulsion(particles[i:i+1,:], others)[0]
        return k_attr * F_attr + k_rep * F_rep

    def monte_carlo_coverage(self, particles, S=8000, thresh=0.25):
        if particles.size == 0:
            return 0.0
        pts = np.random.rand(S, 2) * grid_size
        d   = np.linalg.norm(wrap_distance(pts[:, None, :] - particles[None, :, :]), axis=-1)
        vis = d <= self.fov_radius
        return np.mean(vis.sum(axis=1) > thresh)

# ----------------------- Visualization -----------------------

def fixed_layout_fig():
    fig = plt.figure(figsize=(13.6, 6.2))
    gs = fig.add_gridspec(nrows=1, ncols=2, left=0.06, right=0.94, wspace=0.08)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    cax = fig.add_axes([0.475, 0.15, 0.015, 0.7])
    return fig, axL, axR, cax

def ensure_fov(ax, holder, particles, radius, edgecolor="white", alpha=0.22, lw=1.0):
    for c in holder:
        c.remove()
    holder.clear()
    for p in particles:
        circ = Circle((p[0], p[1]), radius=radius, edgecolor=edgecolor, facecolor="none", alpha=alpha, lw=lw)
        ax.add_patch(circ); holder.append(circ)

def mask_to_nan(U, V):
    mag = np.hypot(U, V)
    mask = ~np.isfinite(mag) | (mag < tiny)
    U = U.astype(float); V = V.astype(float)
    U[mask] = np.nan; V[mask] = np.nan
    return U, V

class LiveTwoPanel:
    def __init__(self, field, enable=True, target_fps=20):
        self.enabled = bool(enable)
        if not self.enabled: return
        plt.ion()
        self.field = field
        self.fig, self.axL, self.axR, self.cax = fixed_layout_fig()

        # LEFT
        self.axL.set_xlim(0, grid_size); self.axL.set_ylim(0, grid_size); self.axL.set_aspect('equal', 'box')
        self.axL.set_title(f"Potential (live) — {LABEL}")
        init_img = np.zeros((grid_size, grid_size))
        self.im = self.axL.imshow(init_img, cmap="viridis", origin="lower", extent=[0, grid_size, 0, grid_size])
        self.cbar = self.fig.colorbar(self.im, cax=self.cax); self.cbar.set_label("Potential")
        self._left_fov = []
        self.left_vp     = self.axL.scatter([], [], c="#ffd166", s=55, edgecolors="black", linewidths=0.9, zorder=5)
        self.left_others = self.axL.scatter([], [], c="white",   s=32, edgecolors="black", linewidths=0.8, zorder=4)
        self.left_kcs = None

        # RIGHT
        self.axR.set_xlim(0, grid_size); self.axR.set_ylim(0, grid_size); self.axR.set_aspect('equal', 'box')
        self.axR.set_title("Fields — BLUE=k_attr·A(x), RED=k_rep·R(x), GREEN=interest delta")
        xs = np.arange(0, grid_size, stride_vf)
        ys = np.arange(0, grid_size, stride_vf)
        self.Xq, self.Yq = np.meshgrid(xs, ys)
        self.Q = np.stack([self.Xq.ravel(), self.Yq.ravel()], axis=-1)
        Z = np.zeros_like(self.Xq)
        self.qA = self.axR.quiver(self.Xq, self.Yq, Z, Z, angles="xy", scale_units="xy", scale=1.0,
                                  width=0.003, color="tab:blue", linewidth=0.6)
        self.qR = self.axR.quiver(self.Xq, self.Yq, Z, Z, angles="xy", scale_units="xy", scale=1.0,
                                  width=0.003, color="tab:red", linewidth=0.6)
        self.qG = self.axR.quiver(self.Xq, self.Yq, Z, Z, angles="xy", scale_units="xy", scale=1.0,
                                  width=0.003, color="tab:green", linewidth=0.6)
        self._right_fov = []
        self.right_vp     = self.axR.scatter([], [], c="#ffd166", s=55, edgecolors="black", linewidths=0.9, zorder=6)
        self.right_others = self.axR.scatter([], [], c="white",   s=32, edgecolors="black", linewidths=0.8, zorder=5)
        self.res_arrow = None
        self.right_kcs = None
        self._kernel_patches = []
        self.target_dt = 1.0 / 20
        self._last = time.time()

        # Interest overlay (right)
        self.im_interest = None
        self.int_contours = None
        if CFG["show_interest_overlay_right"]:
            mode = str(CFG["interest_overlay_mode"]).lower()
            if mode in ("heat", "both"):
                self.im_interest = self.axR.imshow(
                    I_smooth, cmap=str(CFG["interest_colormap"]),
                    origin="lower", extent=[0, grid_size, 0, grid_size],
                    alpha=float(CFG["interest_overlay_alpha"]), zorder=1
                )
            if mode in ("contours", "both"):
                try:
                    levels = [float(v) for v in str(CFG["interest_contour_levels"]).split(",")]
                except Exception:
                    levels = [0.2, 0.4, 0.6, 0.8]
                self.int_contours = self.axR.contour(
                    X, Y, I_smooth, levels=levels,
                    colors=str(CFG["interest_contour_color"]),
                    linewidths=0.8, alpha=float(CFG["interest_contour_alpha"]), zorder=2
                )

    def _update_k_contours(self):
        if CFG["show_kmap_contours"]:
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

    def _draw_repulsion_kernel_at(self, vp):
        try:
            levels = [float(s) for s in (str(CFG["kernel_levels"]).split(",")) if float(s) > 0]
        except Exception:
            levels = [1.0, 2.0, 3.0]
        sigma = float(CFG["anisotropy"]["sigma_rep"])
        col   = CFG["kernel_color"]
        a     = float(CFG["kernel_alpha"])
        lw    = float(CFG["kernel_lw"])

        axx, axy, ayy = sample_Aeff(vp[None, :])
        axx, axy, ayy = float(axx[0]), float(axy[0]), float(ayy[0])
        A = np.array([[axx, axy],[axy, ayy]], dtype=np.float64)
        w, V = np.linalg.eigh(A)
        w = np.clip(w, 1e-9, None)
        for c in levels:
            a1 = np.sqrt((c * sigma * sigma) / w[0])
            a2 = np.sqrt((c * sigma * sigma) / w[1])
            angle = np.degrees(np.arctan2(V[1,0], V[0,0]))
            ell = Ellipse((vp[0], vp[1]), 2*a1, 2*a2, angle=angle,
                          fill=False, color=col, alpha=a, lw=lw, zorder=7)
            self.axR.add_patch(ell); self._kernel_patches.append(ell)

    def _draw_repulsion_kernels(self, particles):
        try:
            levels = [float(s) for s in (str(CFG["kernel_levels"]).split(",")) if float(s) > 0]
        except Exception:
            levels = [1.0, 2.0, 3.0]
        sigma = float(CFG["anisotropy"]["sigma_rep"])
        col   = CFG["kernel_color"]
        a     = float(CFG["kernel_alpha"])
        lw    = float(CFG["kernel_lw"])
        if particles is None or len(particles) == 0:
            return
        axx, axy, ayy = sample_Aeff(particles)
        for i in range(len(particles)):
            vp = particles[i]
            A = np.array([[float(axx[i]), float(axy[i])],
                          [float(axy[i]), float(ayy[i])]], dtype=np.float64)
            w, V = np.linalg.eigh(A)
            w = np.clip(w, 1e-9, None)
            angle = float(np.degrees(np.arctan2(V[1,0], V[0,0])))
            for c in levels:
                a1 = np.sqrt((c * sigma * sigma) / w[0])
                a2 = np.sqrt((c * sigma * sigma) / w[1])
                if abs(a1 - a2) < 1e-7:
                    circ = Circle((vp[0], vp[1]), radius=float(a1),
                                  fill=False, edgecolor=col, alpha=a, lw=lw, zorder=7)
                    self.axR.add_patch(circ); self._kernel_patches.append(circ)
                else:
                    ell = Ellipse((vp[0], vp[1]), 2*a1, 2*a2, angle=angle,
                                  fill=False, color=col, alpha=a, lw=lw, zorder=7)
                    self.axR.add_patch(ell); self._kernel_patches.append(ell)

    def update(self, particles, potential, forces):
        if not self.enabled: return
        now = time.time()
        dt  = now - self._last
        if dt < self.target_dt:
            time.sleep(self.target_dt - dt)
        self._last = time.time()

        # LEFT heatmap
        self.im.set_data(potential)
        vmin, vmax = float(np.nanmin(potential)), float(np.nanmax(potential))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1e-9
        self.im.set_clim(vmin, vmax)
        self.cbar.update_normal(self.im)

        # LEFT points & FOV
        if len(particles) > 0:
            vp_idx = int(np.clip(CFG["vp_index"], 0, len(particles)-1))
            self.left_vp.set_offsets(particles[vp_idx:vp_idx+1])
            others = np.delete(particles, vp_idx, axis=0) if len(particles) > 1 else np.empty((0,2))
            self.left_others.set_offsets(others if len(others) else np.empty((0,2)))
        ensure_fov(self.axL, self._left_fov, particles, fov_radius)
        self._update_k_contours()

        # RIGHT quivers
        F_attr = self.field.attraction_field_at(self.Q)
        F_rep  = self.field.repulsion_field_at(
            self.Q, particles,
            exclude_index=int(np.clip(CFG["vp_index"],0,max(0,len(particles)-1)))
        )
        Fb     = self.field.attraction_field_baseline_at(self.Q) if CFG["show_interest_quiver"] else 0.0

        F_attr *= CFG["k_attr"]
        F_rep  *= CFG["k_rep"]
        F_poi  = (F_attr - CFG["k_attr"]*Fb) if CFG["show_interest_quiver"] else 0.0

        if CFG["quiver_mode"] == "normalized":
            def _norm(U):
                m = np.hypot(U[:,0], U[:,1])
                ux = np.where(m>tiny,(U[:,0]/m)*3.0,np.nan)
                uy = np.where(m>tiny,(U[:,1]/m)*3.0,np.nan)
                return ux, uy
            Uax, Uay = _norm(F_attr)
            Urx, Ury = _norm(F_rep)
            Ugx, Ugy = _norm(F_poi if isinstance(F_poi, np.ndarray) else np.zeros_like(F_attr))
        else:
            Uax = F_attr[:,0] * CFG["quiver_len_scale_attr"]
            Uay = F_attr[:,1] * CFG["quiver_len_scale_attr"]
            Urx = F_rep [:,0] * CFG["quiver_len_scale_rep"]
            Ury = F_rep [:,1] * CFG["quiver_len_scale_rep"]
            scale_int = CFG["quiver_len_scale_attr"] if CFG["quiver_len_scale_interest"] is None else CFG["quiver_len_scale_interest"]
            if isinstance(F_poi, np.ndarray):
                Ugx = F_poi[:,0] * scale_int
                Ugy = F_poi[:,1] * scale_int
            else:
                Ugx = np.zeros_like(Uax); Ugy = np.zeros_like(Uay)

        Uax, Uay = mask_to_nan(Uax, Uay)
        Urx, Ury = mask_to_nan(Urx, Ury)
        Ugx, Ugy = mask_to_nan(Ugx, Ugy)
        ny, nx = self.Xq.shape
        self.qA.set_UVC(Uax.reshape(ny, nx), Uay.reshape(ny, nx))
        self.qR.set_UVC(Urx.reshape(ny, nx), Ury.reshape(ny, nx))
        if CFG["show_interest_quiver"]:
            self.qG.set_UVC(Ugx.reshape(ny, nx), Ugy.reshape(ny, nx))
        else:
            self.qG.set_UVC(np.full_like(self.Xq, np.nan), np.full_like(self.Yq, np.nan))

        # RIGHT points & FOV
        if len(particles) > 0:
            vp_idx = int(np.clip(CFG["vp_index"], 0, len(particles)-1))
            self.right_vp.set_offsets(particles[vp_idx:vp_idx+1])
            others = np.delete(particles, vp_idx, axis=0) if len(particles) > 1 else np.empty((0,2))
            self.right_others.set_offsets(others if len(others) else np.empty((0,2)))
        ensure_fov(self.axR, self._right_fov, particles, fov_radius)
        self._update_k_contours()

        # resultant arrow
        if self.res_arrow is not None:
            self.res_arrow.remove(); self.res_arrow = None
        if len(particles) > 0 and forces is not None and len(forces) > 0:
            vp_idx = int(np.clip(CFG["vp_index"], 0, len(particles)-1))
            vp = particles[vp_idx]; Fv = forces[vp_idx]
            self.res_arrow = FancyArrowPatch((vp[0], vp[1]), (vp[0]+Fv[0], vp[1]+Fv[1]),
                                             arrowstyle='-|>', mutation_scale=10,
                                             linewidth=1.5, color='black', zorder=8)
            self.axR.add_patch(self.res_arrow)

        # kernel overlay
        self._clear_kernel_patches()
        if CFG["show_repulsion_kernel"] and len(particles) > 0:
            if CFG.get("show_repulsion_kernel_all", False):
                self._draw_repulsion_kernels(particles)
            else:
                vp_idx = int(np.clip(CFG["vp_index"], 0, len(particles)-1))
                vp = particles[vp_idx]
                self._draw_repulsion_kernel_at(vp)

        plt.pause(0.001)

    def close(self):
        if self.enabled:
            try: plt.ioff(); plt.close(self.fig)
            except Exception: pass

# ----------------------- Main loop -----------------------

field      = Field(grid_size, fov_radius)
particles  = (np.random.rand(1, 2) * grid_size)  # start with 1 vp
m = np.zeros_like(particles)
v = np.zeros_like(particles)
t_adam     = 0

coverage_ts, time_ts = [], []
frames_pts    = deque(maxlen=int(CFG["max_anim_frames"]))
frames_pot    = deque(maxlen=int(CFG["max_anim_frames"]))
frames_forces = deque(maxlen=int(CFG["max_anim_frames"]))
recent_moves  = []
window_size   = 5

live = LiveTwoPanel(field, enable=bool(CFG["animate"]), target_fps=20)
start_t = time.time()

def main_loop():
    global particles, m, v, t_adam, recent_moves
    while True:
        field.update_need_mask(particles)
        field.compute_potential(particles)
        field.compute_attraction_grids(particles)

        coverage = field.monte_carlo_coverage(particles)
        coverage_ts.append(coverage)
        time_ts.append(time.time() - start_t)

        forces = field.compute_force_on_viewpoints(particles, k_attr=CFG["k_attr"], k_rep=CFG["k_rep"])

        frames_pts.append(particles.copy())
        frames_pot.append(field.potential.copy())
        frames_forces.append(forces.copy())

        live.update(particles, field.potential, forces)

        # Adam-like step
        t_adam += 1
        m = 0.9*m + 0.1*forces
        v = 0.999*v + 0.001*(forces**2)
        step = float(CFG["adam_step_gain"]) * (m/(1-0.9**t_adam)) / (np.sqrt(v/(1-0.999**t_adam)) + 1e-12)
        particles[:] = (particles + step) % grid_size

        move_mag = np.linalg.norm(step, axis=1).mean()
        recent_moves.append(move_mag)
        if len(recent_moves) > window_size:
            recent_moves.pop(0)

        allow_insert = (particles.shape[0] < int(CFG["max_viewpoints"]))
        if CFG["smart_nbv_insertion"]:
            if len(particles) <= 2:
                insert = (len(frames_pts) - 1) % int(CFG["frames_per_stage"]) == 0
            else:
                T = CFG["T_motion_fine"] if coverage >= .95 else CFG["T_motion_rough"]
                insert = len(recent_moves) == window_size and all(mv < T for mv in recent_moves)
        else:
            insert = (len(frames_pts) - 1) % int(CFG["frames_per_stage"]) == 0

        if allow_insert and insert:
            nbv = field_points[np.argmax(field.potential)] if CFG["strategy"] == "nbv" else np.random.rand(2) * grid_size
            particles = np.vstack([particles, nbv[None, :]])
            m = np.zeros_like(particles); v = np.zeros_like(particles)
            t_adam = 0; recent_moves.clear()

        if CFG["verbose"] and len(frames_pts) % 25 == 0:
            print(f"[{LABEL}] t={len(time_ts):04d} move={move_mag:5.3f} cov={coverage:6.3f} "
                  f"vp={len(particles):3d} add={'✔' if (allow_insert and insert) else '—'}  "
                  f"k_attr={CFG['k_attr']} k_rep={CFG['k_rep']}")

        if not CFG["perpetual"]:
            if coverage >= 1.0 or len(particles) >= int(CFG["max_viewpoints"]):
                break

try:
    main_loop()
except KeyboardInterrupt:
    pass
finally:
    live.close()

# ----------------------- Save snapshot -----------------------

npz = os.path.join(
    CFG["save_dir"],
    f"{LABEL.replace(' ','_')}_{CFG['strategy']}_{CFG['potential_type']}_"
    f"kattr{float(CFG['k_attr']):.2f}_krep{float(CFG['k_rep']):.2f}_seed{int(CFG['seed'])}_metrics.npz"
)
np.savez_compressed(
    npz,
    time=np.array(time_ts),
    coverage=np.array(coverage_ts),
    num_viewpoints=particles.shape[0],
    final_viewpoints=particles,
    K_map=K_map,
    gamma=gamma,
    anisotropy=CFG.get("anisotropy", {})
)
print(f"[Ready] {LABEL}. Anisotropy={'on' if CFG['anisotropy']['enabled'] else 'off'}  "
      f"alpha_t={CFG['anisotropy']['alpha_t']} alpha_n={CFG['anisotropy']['alpha_n']}")
