#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive NBV / Potential-Field simulation — Phase 2 results (REAL-TIME TWO-PANEL)
FAST EDITION: identical physics & visuals, accelerated with toroidal FFT for attraction fields.

What remains EXACTLY the same:
• Logarithmic attraction kernel & its gradient (physics unchanged).
• Gaussian inter-VP repulsion (unchanged).
• Toroidal wrapping (minimum-image).
• Two-panel visualization: sizes, colormap, quiver widths/colors, markers, titles.
• Insertion/termination semantics, saved snapshot/animation/NPZ formats & names.

What’s faster:
• The attraction field is computed once per frame over the whole grid via FFT-based convolution
  (periodic torus), then sampled for quivers and viewpoint forces.
• The “interest-only” GREEN layer is obtained by subtracting a cached legacy-baseline field grid
  (also FFT-convolved) — no per-query integration loops.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse, os, time, json
from matplotlib.patches import FancyArrowPatch, Circle
from collections import deque

# ---------- Args (JSON-only) ----------
# Load everything from old_config.json and create an `args` shim with defaults.
import json, os
from types import SimpleNamespace

CFG_PATH = "old_config.json"
if not os.path.exists(CFG_PATH):
    raise FileNotFoundError(f"Config file not found: {CFG_PATH}")

with open(CFG_PATH, "r") as f:
    CFG = json.load(f)

# Defaults mirror the old argparse defaults; CFG overrides any of them.
_defaults = dict(
    strategy="nbv",
    potential_type="log",
    seed=0,
    save_dir="./phase2_results",
    animate=True,
    smart_nbv_insertion=False,
    k_attr=10.0,
    k_rep=0.25,
    max_viewpoints=12,            # <= your requested cap
    vp_index=0,
    perpetual=True,
    max_anim_frames=600,
    quiver_mode="length",
    quiver_len_scale_attr=1.0,
    quiver_len_scale_rep=1.0,
    show_interest_quiver=False,   # <= no interest layer
    show_kmap_contours=False,
    quiver_len_scale_interest=None,
    verbose=False,

    # New JSON-driven sim/grid knobs (were hard-coded before)
    grid_size=100,
    fov_radius=20.0,
    frames_per_stage=10,
    T_motion_rough=1.5,
    T_motion_fine=0.15,
    window_size=5,
    stride_vf=4,
    epsilon=1e-6,
    tiny=1e-12,

    # Need/visibility fields
    vis_beta=20.0,
    gamma=1.0,
    need_hard_clip=True,
    tau_clip=1.0,

    # Interests/POIs off by default
    importance_config="",
    poi_config=""
)

_defaults.update(CFG)
args = SimpleNamespace(**_defaults)

# ---------- Globals ----------
LABEL = "phase 2 results"
np.random.seed(args.seed)
grid_size, fov_radius = 100, 20
epsilon               = 1e-6
frames_per_stage       = 10
T_motion_rough, T_motion_fine = 1.5, 0.15
window_size            = 5
stride_vf              = 4
tiny                   = 1e-12  # for masking zero-length arrows in quiver

x, y         = np.arange(grid_size), np.arange(grid_size)
X, Y         = np.meshgrid(x, y)
field_points = np.stack([X.ravel(), Y.ravel()], axis=-1)  # (P,2)

# ---------- Globals ----------
LABEL = "phase 2 results"

# Seed from JSON
np.random.seed(args.seed)

# Grid & sim constants from JSON
grid_size          = int(args.grid_size)
fov_radius         = float(args.fov_radius)
epsilon            = float(args.epsilon)
frames_per_stage   = int(args.frames_per_stage)
T_motion_rough     = float(args.T_motion_rough)
T_motion_fine      = float(args.T_motion_fine)
window_size        = int(args.window_size)
stride_vf          = int(args.stride_vf)
tiny               = float(args.tiny)

# Meshgrid & points
x, y         = np.arange(grid_size), np.arange(grid_size)
X, Y         = np.meshgrid(x, y)
field_points = np.stack([X.ravel(), Y.ravel()], axis=-1)  # (P,2)

# ---------- POI parsing to build maps ----------
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
    # Line: n·q = b, where n = [cosθ, sinθ]; band is |n·q - b| <= width/2 (approx toroidal band)
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
        # forms:
        #  disk:cx,cy,r,req=2,omega=+2,beta=18
        #  band:theta_deg,b,width,req=3,omega=*1.2
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
            beta_map[region] = beta  # last-writer-wins

    # safety
    K_map = np.clip(K_map, 1.0, None)
    omega_map = np.clip(omega_map, 0.0, None)
    return K_map, beta_map, omega_map

K_map, beta_map, omega_map = _build_maps_from_poi(CFG.get("poi_config",""))
gamma = float(CFG.get("gamma", 1.0))
need_hard_clip = bool(CFG.get("need_hard_clip", True))
tau_clip = float(CFG.get("tau_clip", 1.0))

# ---------- FFT helpers (toroidal vector-kernel) ----------
# Replace your _build_fft_kernels with this
def _build_fft_kernels(N):
    """
    Toroidal kernel for y[i] = sum_j x[j] * h[i-j] with
      h(Δ) = -(Δ)/||Δ||^2, h(0,0)=0  (vector log-gradient).
    Build Δx,Δy as wrapped offsets at the actual array indices, so h[0,0] is zero-lag.
    No fftshift/ifftshift needed.
    """
    # wrapped offsets: 0,1,2,...,N//2, -(N//2-1),...,-2,-1  (for even N places +N/2 at +N/2)
    offs = np.arange(N, dtype=np.float64)
    offs = np.where(offs <= N//2, offs, offs - N)   # Δ in circular coords

    dx = offs[None, :]    # columns (x)
    dy = offs[:, None]    # rows    (y)
    r2 = dx*dx + dy*dy

    Kx = np.zeros((N, N), dtype=np.float64)
    Ky = np.zeros((N, N), dtype=np.float64)
    # h_x = -(Δx)/r^2, h_y = -(Δy)/r^2  (origin left at [0,0])
    np.divide(-dx, r2, out=Kx, where=r2 > 0)
    np.divide(-dy, r2, out=Ky, where=r2 > 0)

    # Direct FFT — kernel is already aligned s.t. index [0,0] is Δ=0
    Kx_hat = np.fft.rfftn(Kx, s=(N, N))
    Ky_hat = np.fft.rfftn(Ky, s=(N, N))
    return Kx_hat, Ky_hat


Kx_hat, Ky_hat = _build_fft_kernels(grid_size)

def _fft_convolve_vector(n_grid):
    """
    Given scalar need map n(x,y), return vector field (Fx,Fy) = n * K (toroidal conv) via FFT.
    """
    n_hat = np.fft.rfftn(n_grid, s=(grid_size, grid_size))
    Fx = np.fft.irfftn(n_hat * Kx_hat, s=(grid_size, grid_size))
    Fy = np.fft.irfftn(n_hat * Ky_hat, s=(grid_size, grid_size))
    return Fx, Fy

def _bilinear_sample_periodic(F, pts):
    """
    Bilinear sampling of grid F (HxW) at floating pts[:,2] with toroidal wrap.
    pts in [0,N). Returns values shape (len(pts),).
    """
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

# ---------- Field ----------
class Field:
    def __init__(self, grid_size, fov_radius, use_wrapping=True):
        self.grid_size, self.fov_radius = grid_size, fov_radius
        self.use_wrapping = use_wrapping
        self.field_points = field_points  # (P,2)
        # n(q) weight map (previously need_mask): same semantics for attraction integration
        self.need_weight = np.ones(grid_size * grid_size, dtype=np.float64)
        self.potential    = np.zeros((grid_size, grid_size))
        self.visibility_soft = np.zeros((grid_size, grid_size), dtype=np.float64)  # retained for parity

        # cached per-frame attraction grids (with POIs) and baseline legacy grids (outside-FOV only)
        self._attr_Fx = np.zeros((grid_size, grid_size), dtype=np.float64)
        self._attr_Fy = np.zeros((grid_size, grid_size), dtype=np.float64)
        self._base_Fx = np.zeros((grid_size, grid_size), dtype=np.float64)
        self._base_Fy = np.zeros((grid_size, grid_size), dtype=np.float64)
        self._last_particles = None  # for baseline cache validity

    def wrap_distance(self, diff):
        if not self.use_wrapping:
            return diff
        half = self.grid_size / 2.0
        return np.where(np.abs(diff) > half,
                        -np.sign(diff) * (self.grid_size - np.abs(diff)), diff)

    def _compute_vis_and_need(self, particles):
        """
        Compute vis(q) using local beta_map, then unmet-need:
            n(q) = max(0, 1 - vis(q)/K(q)) * omega(q)^gamma
        Optional hard-clip: if vis(q) >= tau*K(q) ⇒ n(q)=0
        """
        P = self.field_points.shape[0]
        K = particles.shape[0]
        if K == 0:
            self.visibility_soft.ravel()[:] = 0.0
            self.need_weight[:] = (omega_map.ravel() ** gamma)
            return

        diff = self.field_points[:, None, :] - particles[None, :, :]  # (P,K,2)
        diff = self.wrap_distance(diff)
        dist = np.linalg.norm(diff, axis=-1) + epsilon                # (P,K)

        # local beta at each q broadcast over viewpoints
        beta_q = np.repeat(beta_map.ravel()[:, None], K, axis=1)      # (P,K)
        s = 1.0 / (1.0 + np.exp(beta_q * (dist / self.fov_radius - 1.0)))
        vis = s.sum(axis=1)                                           # (P,)

        self.visibility_soft.ravel()[:] = np.clip(vis, 0.0, None)

        Kq = K_map.ravel()
        need = np.maximum(0.0, 1.0 - vis / Kq) * (omega_map.ravel() ** gamma)
        if need_hard_clip:
            need = np.where(vis >= tau_clip * Kq, 0.0, need)

        self.need_weight[:] = need

    def update_need_mask(self, particles, beta_unused=20.0):
        """
        Retained signature; internally computes local-β visibility and unmet-need n(q).
        """
        self._compute_vis_and_need(particles)

    def compute_potential(self, particles, alpha=1.0):
        """
        LEFT-panel potential image (unchanged form), but weighted by local n(q):
            P(x) = n(x) * sum_v log(dist(x,v))
        """
        if particles.shape[0] == 0:
            self.potential.fill(0.0)
            return
        diff = self.field_points[:, None, :] - particles[None, :, :]
        diff = self.wrap_distance(diff)
        dist = np.linalg.norm(diff, axis=-1) + epsilon                # (P,K)
        base = np.log(dist).sum(axis=1)                               # (P,)
        pot  = alpha * self.need_weight * base
        self.potential = pot.reshape(self.grid_size, self.grid_size)

    # === FAST: per-frame attraction grids via FFT (toroidal convolution) ===
    def compute_attraction_grids(self, particles):
        """
        Build two vector fields over the entire grid:
          • _attr_Fx, _attr_Fy : using POI-weighted need_weight (current behavior)
          • _base_Fx, _base_Fy : using legacy hard outside-FOV mask (for GREEN interest = attr - base)
        """
        # with-POI attraction
        n_grid = self.need_weight.reshape(self.grid_size, self.grid_size)
        Fx, Fy = _fft_convolve_vector(n_grid)
        self._attr_Fx[:], self._attr_Fy[:] = Fx, Fy

        # baseline legacy: 1 outside any FOV, else 0
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

    # === Fields for visualization at arbitrary grid points Q (exact same physics) ===
    def attraction_field_at(self, Q):
        """BLUE field at Q (M,2): sample from cached FFT grid (unscaled)."""
        if Q.ndim == 1: Q = Q[None, :]
        Ux = _bilinear_sample_periodic(self._attr_Fx, Q)
        Uy = _bilinear_sample_periodic(self._attr_Fy, Q)
        return np.stack([Ux, Uy], axis=1)

    def attraction_field_baseline_at(self, Q):
        """Legacy baseline field at Q (unscaled), from cached FFT baseline grid."""
        if Q.ndim == 1: Q = Q[None, :]
        Ux = _bilinear_sample_periodic(self._base_Fx, Q)
        Uy = _bilinear_sample_periodic(self._base_Fy, Q)
        return np.stack([Ux, Uy], axis=1)

    def repulsion_field_at(self, Q, particles, exclude_index=0):
        """RED field at Q (M,2): repulsion from all viewpoints EXCEPT exclude_index. Unscaled."""
        if Q.ndim == 1: Q = Q[None, :]
        if particles.shape[0] <= 1:
            return np.zeros((Q.shape[0], 2))
        others = np.delete(particles, exclude_index, axis=0)
        if others.shape[0] == 0:
            return np.zeros((Q.shape[0], 2))
        diff = Q[:, None, :] - others[None, :, :]                      # (M,K-1,2)
        diff = self.wrap_distance(diff)
        r    = np.linalg.norm(diff, axis=-1) + epsilon                 # (M,K-1)
        sigma, amp = 10.0, 100.0
        w    = (amp * (r / (sigma**2)) * np.exp(-(r**2) / (2*sigma**2)))
        F_rep = (diff * w[..., None]).sum(axis=1)                      # (M,2)
        return F_rep

    # === Forces used for motion on actual viewpoints ===
    def compute_force_on_viewpoints(self, particles, k_attr=0.4, k_rep=1.0):
        """
        Attraction sampled from cached FFT grid at each VP (bilinear on torus).
        Repulsion unchanged (pairwise Gaussian).
        """
        K = particles.shape[0]
        if K == 0:
            return np.zeros((0,2))

        # attraction (unscaled) sampled at VP positions
        Fax = _bilinear_sample_periodic(self._attr_Fx, particles)
        Fay = _bilinear_sample_periodic(self._attr_Fy, particles)
        F_attr = np.stack([Fax, Fay], axis=1)

        # repulsion (unchanged)
        pdiff = particles[:, None, :] - particles[None, :, :]         # (K,K,2)
        pdiff = self.wrap_distance(pdiff)
        pdist = np.linalg.norm(pdiff, axis=-1) + epsilon              # (K,K)
        sigma, amp = 10.0, 100.0
        F_rep = -(amp * pdiff * (-pdist[..., None] / sigma**2) *
                 np.exp(-(pdist**2) / (2*sigma**2))[..., None]).sum(axis=1)      # (K,2)

        return k_attr * F_attr + k_rep * F_rep

    # === Monte-Carlo coverage estimate (for termination / logging) ===
    def monte_carlo_coverage(self, particles, S=10000, thresh=0.25):
        if particles.size == 0:
            return 0.0
        pts  = np.random.rand(S, 2) * self.grid_size
        d    = np.linalg.norm(self.wrap_distance(pts[:, None, :] - particles[None, :, :]), axis=-1)
        vis  = d <= self.fov_radius
        return np.mean(vis.sum(axis=1) > thresh)

# ---------- Layout / drawing helpers ----------
def _fixed_layout_fig():
    # Panels same size via GridSpec; colorbar in its own axis
    fig = plt.figure(figsize=(13.6, 6.2))
    gs = fig.add_gridspec(nrows=1, ncols=2, left=0.06, right=0.94, wspace=0.08)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    cax = fig.add_axes([0.475, 0.15, 0.015, 0.7])  # dedicated colorbar axis
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
        self.left_kcs = None  # K contour set

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
        # Optional POI incremental attraction (GREEN)
        self.qG = self.axR.quiver(self.Xq, self.Yq, Z, Z,
                                  angles="xy", scale_units="xy", scale=1.0,
                                  width=0.003, color="tab:green", linewidth=0.6)
        self._right_fov = []
        self.right_vp     = self.axR.scatter([], [], c="#ffd166", s=55, edgecolors="black", linewidths=0.9, zorder=6)
        self.right_others = self.axR.scatter([], [], c="white",   s=32, edgecolors="black", linewidths=0.8, zorder=5)
        self.res_arrow = None
        self.right_kcs = None  # K contour set

        self.target_dt = 1.0 / max(1, int(target_fps))
        self._last = time.time()

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

    def update(self, particles, potential, forces):
        if not self.enabled:
            return
        # throttle
        now = time.time()
        dt = now - self._last
        if dt < self.target_dt:
            time.sleep(self.target_dt - dt)
        self._last = time.time()

        # LEFT image
        self.im.set_data(potential)
        vmin, vmax = float(np.nanmin(potential)), float(np.nanmax(potential))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
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

        # Quivers (RIGHT) — exact fields (from cached FFT grids)
        F_attr = self.field.attraction_field_at(self.Q)  # unscaled
        F_rep  = self.field.repulsion_field_at(self.Q, particles, exclude_index=int(np.clip(args.vp_index,0,max(0,len(particles)-1))))  # unscaled
        F_poi  = np.zeros_like(F_attr)
        if args.show_interest_quiver:
            Fb = self.field.attraction_field_baseline_at(self.Q)       # legacy
            F_poi = F_attr - Fb                                        # incremental due to POIs/weights

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

        Uax, Uay = _mask_to_nan(Uax, Uay)  # avoid MPL warnings
        Urx, Ury = _mask_to_nan(Urx, Ury)
        Ugx, Ugy = _mask_to_nan(Ugx, Ugy)

        ny, nx = self.Xq.shape
        self.qA.set_UVC(Uax.reshape(ny, nx), Uay.reshape(ny, nx))
        self.qR.set_UVC(Urx.reshape(ny, nx), Ury.reshape(ny, nx))
        # GREEN only if toggled
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

        # Resultant arrow on followed vp
        if self.res_arrow is not None:
            self.res_arrow.remove()
            self.res_arrow = None
        if len(particles) > 0 and forces is not None and len(forces) > 0:
            vp_idx = int(np.clip(args.vp_index, 0, len(particles)-1))
            vp = particles[vp_idx]; Fv = forces[vp_idx]
            self.res_arrow = FancyArrowPatch((vp[0], vp[1]),
                                             (vp[0] + Fv[0], vp[1] + Fv[1]),
                                             arrowstyle='-|>', mutation_scale=10,
                                             linewidth=1.5, color='black', zorder=8)
            self.axR.add_patch(self.res_arrow)

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

# Rolling buffers for finite saved animation
frames_pts    = deque(maxlen=args.max_anim_frames)
frames_pot    = deque(maxlen=args.max_anim_frames)
frames_forces = deque(maxlen=args.max_anim_frames)

recent_moves = []
start_t = time.time()

live = LiveTwoPanel(field, enable=args.animate, target_fps=20)

# ---------- Main loop (restored semantics) ----------
def main_loop():
    global particles, m, v, t_adam, recent_moves
    while True:
        # Update unmet-need weights & potential (unchanged formulas)
        field.update_need_mask(particles)
        field.compute_potential(particles)

        # FAST: build attraction grids (with-POI and legacy baseline) via FFT
        field.compute_attraction_grids(particles)

        # Coverage estimate (MC) — unchanged heuristic
        coverage = field.monte_carlo_coverage(particles)
        coverage_ts.append(coverage)
        time_ts.append(time.time() - start_t)

        # Forces for motion (attraction sampled from grid; repulsion unchanged)
        forces = field.compute_force_on_viewpoints(particles, k_attr=args.k_attr, k_rep=args.k_rep)

        # Push into rolling buffers
        frames_pts.append(particles.copy())
        frames_pot.append(field.potential.copy())
        frames_forces.append(forces.copy())

        # Realtime display
        live.update(particles, field.potential, forces)

        # Adam-like motion
        t_adam += 1
        m = 0.9*m + 0.1*forces
        v = 0.999*v + 0.001*(forces**2)
        step = 5 * (m/(1-0.9**t_adam)) / (np.sqrt(v/(1-0.999**t_adam)) + epsilon)
        particles[:] = (particles + step) % grid_size

        move_mag = np.linalg.norm(step, axis=1).mean()
        recent_moves.append(move_mag)
        if len(recent_moves) > window_size:
            recent_moves.pop(0)

        # Periodic or smart insertion (STRICT cap)
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
            # reset optimizer state on resize
            m = np.zeros_like(particles); v = np.zeros_like(particles)
            t_adam = 0; recent_moves.clear()

        if args.verbose and len(frames_pts) % 25 == 0:
            print(f"[{LABEL}] t={len(time_ts):04d}  move={move_mag:5.3f}  cov={coverage:6.3f}  "
                  f"vp={len(particles):3d}  add={'✔' if (allow_insert and insert) else '—'}  "
                  f"k_attr={args.k_attr} k_rep={args.k_rep}")

        # Termination (only if not perpetual)
        if not args.perpetual:
            if coverage >= 1.0 or len(particles) >= args.max_viewpoints:
                break

# Run loop
try:
    main_loop()
except KeyboardInterrupt:
    pass
finally:
    live.close()

# ---------- Save bounded two-panel snapshot & animation ----------
def _draw_two_panel(fig, field, pts, pot, forces, title_left="Potential (all VPs)"):
    fig.clf()
    # fixed layout
    gs = fig.add_gridspec(nrows=1, ncols=2, left=0.06, right=0.94, wspace=0.08)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    cax = fig.add_axes([0.475, 0.15, 0.015, 0.7])

    # LEFT
    im = axL.imshow(pot, cmap='viridis', origin='lower', extent=[0, grid_size, 0, grid_size])
    vmin, vmax = float(np.nanmin(pot)), float(np.nanmax(pot))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
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

    # RIGHT (use cached FFT grids just like live path)
    xs = np.arange(0, grid_size, stride_vf)
    ys = np.arange(0, grid_size, stride_vf)
    Xq, Yq = np.meshgrid(xs, ys)
    Q = np.stack([Xq.ravel(), Yq.ravel()], axis=-1)

    vp_idx = int(np.clip(args.vp_index, 0, max(0, len(pts)-1)))
    F_attr = field.attraction_field_at(Q)                                  # unscaled
    F_rep  = field.repulsion_field_at(Q, pts, exclude_index=vp_idx)        # unscaled
    Fb     = field.attraction_field_baseline_at(Q)                         # legacy
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
    axR.quiver(Xq, Yq, Uax.reshape(ny, nx), Uay.reshape(ny, nx),
               angles='xy', scale_units='xy', scale=1.0, width=0.003, color='tab:blue', linewidth=0.6)
    axR.quiver(Xq, Yq, Urx.reshape(ny, nx), Ury.reshape(ny, nx),
               angles='xy', scale_units='xy', scale=1.0, width=0.003, color='tab:red', linewidth=0.6)
    if args.show_interest_quiver:
        axR.quiver(Xq, Yq, Ugx.reshape(ny, nx), Ugy.reshape(ny, nx),
                   angles='xy', scale_units='xy', scale=1.0, width=0.003, color='tab:green', linewidth=0.6)

    _ensure_fov(axR, [], pts, fov_radius)
    if len(pts) > 0:
        axR.scatter(pts[vp_idx:vp_idx+1,0], pts[vp_idx:vp_idx+1,1], c="#ffd166", s=55, edgecolors="black", linewidths=0.9, zorder=6)
        others = np.delete(pts, vp_idx, axis=0) if len(pts) > 1 else np.empty((0,2))
        if len(others): axR.scatter(others[:,0], others[:,1], c="white", s=32, edgecolors="black", linewidths=0.8, zorder=5)
        if forces is not None and len(forces) > 0:
            Fv = forces[vp_idx]
            axR.add_patch(FancyArrowPatch((pts[vp_idx,0], pts[vp_idx,1]),
                                          (pts[vp_idx,0]+Fv[0], pts[vp_idx,1]+Fv[1]),
                                          arrowstyle='-|>', mutation_scale=10,
                                          linewidth=1.5, color='black', zorder=8))
    axR.set_xlim(0, grid_size); axR.set_ylim(0, grid_size); axR.set_aspect('equal', 'box')
    axR.set_title("Final fields — BLUE=k_attr·A(x), RED=k_rep·R(x)" + (" , GREEN=POI ΔA" if args.show_interest_quiver else ""))
    if args.show_kmap_contours:
        cs = axR.contour(X, Y, K_map, levels=[1,2,3,4], colors='k', linewidths=0.6, alpha=0.4)
        axR.clabel(cs, inline=1, fontsize=8, fmt='%d')
    return axL, axR

# Convert rolling deques to lists (bounded by max_anim_frames)
frames_pts_list    = list(frames_pts)
frames_pot_list    = list(frames_pot)
frames_forces_list = list(frames_forces)

if args.animate and len(frames_pts_list) > 0:
    # Snapshot (last frame)
    fig = plt.figure(figsize=(13.6, 6.2))
    _draw_two_panel(fig, field, frames_pts_list[-1], frames_pot_list[-1], frames_forces_list[-1],
                    title_left="Final potential (with FOV holes & POIs)")
    snap = os.path.join(args.save_dir,
           f"{LABEL.replace(' ','_')}_{args.strategy}_{args.potential_type}_seed{args.seed}_snapshots.png")
    plt.savefig(snap, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[Saved] {LABEL} — Snapshots ➜ {snap}")

    # Animation across the bounded buffer
    fig = plt.figure(figsize=(13.6, 6.2))
    def _animate(i):
        _draw_two_panel(fig, field, frames_pts_list[i], frames_pot_list[i], frames_forces_list[i],
                        title_left=f"Potential (t={i:03d})")
        return []
    ani = animation.FuncAnimation(fig, _animate,
                                  frames=len(frames_pts_list), interval=100, blit=False)
    mp4 = os.path.join(args.save_dir,
          f"{LABEL.replace(' ','_')}_{args.strategy}_{args.potential_type}_seed{args.seed}_anim.mp4")
    ani.save(mp4, writer='ffmpeg', fps=10, dpi=220); plt.close()
    print(f"[Saved] {LABEL} — Animation ➜ {mp4}")

# ---------- Minimal NPZ (structure placeholders kept simple) ----------
N = particles.shape[0]
npz = os.path.join(args.save_dir,
       f"{LABEL.replace(' ','_')}_{args.strategy}_{args.potential_type}_kattr{args.k_attr:.2f}_krep{args.k_rep:.2f}_seed{args.seed}_metrics.npz")
np.savez_compressed(npz,
    time=np.array(time_ts),
    coverage=np.array(coverage_ts),
    num_viewpoints=N,
    final_viewpoints=particles,
    K_map=K_map,
    gamma=gamma
)
print(f"[Done] {LABEL}.  Perpetual={args.perpetual}  Viewpoints={N}/{args.max_viewpoints}  "
      f"Frames kept={len(frames_pts_list)}/{args.max_anim_frames}  "
      f"QuiverMode={args.quiver_mode}  k_attr={args.k_attr} k_rep={args.k_rep}  vp_index={args.vp_index}  "
      f"POIs={'on' if CFG.get('poi_config','').strip() else 'off'}")
