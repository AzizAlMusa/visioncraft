#!/usr/bin/env python3
# old_overlap.py — 10s @ 60fps MP4 export (clean & deterministic)

import os, json, signal, argparse
import numpy as np
import matplotlib
# Use a non-interactive backend to avoid window/GUI issues during export
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

# ---------------------- I/O ----------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

# ---------------------- wrap ----------------------
def torus_wrap(diff, size, use_wrap):
    if not use_wrap:
        return diff
    half = size / 2.0
    return np.where(np.abs(diff) > half,
                    -np.sign(diff) * (size - np.abs(diff)), diff)

# ---------------------- importance ----------------------
def parse_importance_config(cfg, grid_size):
    omega = np.ones((grid_size, grid_size), dtype=np.float32)
    if (not cfg) or (str(cfg).lower() == "none"):
        return omega
    xs, ys = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    xs = xs.astype(np.float32); ys = ys.astype(np.float32)
    for token in str(cfg).split(';'):
        token = token.strip()
        if not token:
            continue
        if token.startswith('disk:'):
            cx, cy, r, amp = map(float, token[len('disk:'):].split(','))
            d2 = (xs - cx)**2 + (ys - cy)**2
            sigma = max(r/2.0, 1.0)
            omega += amp * np.exp(-d2 / (2*sigma**2)).astype(np.float32)
        elif token.startswith('band:'):
            ths, b, width, amp = token[len('band:'):].split(',')
            th = float(ths.replace('deg',''))
            b = float(b); width = float(width); amp = float(amp)
            n = np.array([np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))], dtype=np.float32)
            d = (n[0]*xs + n[1]*ys - b)
            sigma = max(float(width)/2.0, 1.0)
            omega += float(amp) * np.exp(-(d**2) / (2*sigma**2)).astype(np.float32)
    return omega

# ---------------------- field ----------------------
class Field:
    def __init__(self, cfg):
        self.cfg = cfg
        self.size = int(cfg["grid_size"])
        self.R = float(cfg["fov_radius"])
        self.beta = float(cfg["vis_beta"])
        self.need_hard_clip = bool(cfg["need_hard_clip"])
        self.need_clip_threshold = float(cfg["need_clip_threshold"])
        self.vis_requirement = max(1.0, float(cfg.get("vis_requirement", 1.0)))
        self.use_wrap = True  # always wrap

        x = np.arange(self.size, dtype=np.float32); y = np.arange(self.size, dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        self.field_points = np.stack([X.ravel(), Y.ravel()], axis=-1)  # (N^2,2) float32

        self.visibility = np.zeros((self.size, self.size), dtype=np.float32)
        self.potential  = np.zeros((self.size, self.size), dtype=np.float32)

        self.omega = parse_importance_config(cfg["importance_config"], self.size).astype(np.float32)
        self.omega_gamma = float(cfg["importance_gamma"])
        self._omega_flat = (np.maximum(self.omega, 0.0) ** self.omega_gamma).ravel().astype(np.float32)

        self._need_flat = None

    # --- visibility & unmet-need ---
    def update_visibility(self, V):
        diff = self.field_points[:, None, :] - V[None, :, :].astype(np.float32)
        diff = torus_wrap(diff, self.size, self.use_wrap)
        dist = np.linalg.norm(diff, axis=-1) + 1e-12
        s = 1.0 / (1.0 + np.exp(self.beta * (dist / self.R - 1.0)))
        vis = np.clip(s.sum(axis=1), 0.0, self.vis_requirement)
        self.visibility.ravel()[:] = vis.astype(np.float32)
        self._need_flat = None

    def need_flat(self, use_visibility=True):
        if not use_visibility:
            return self._omega_flat
        if self._need_flat is not None:
            return self._need_flat
        vis_k = (self.visibility.ravel() / self.vis_requirement).astype(np.float32)
        need = np.clip(1.0 - vis_k, 0.0, 1.0).astype(np.float32) * self._omega_flat
        if self.need_hard_clip:
            mask = (self.visibility.ravel() >= self.need_clip_threshold*self.vis_requirement)
            need = np.where(mask, 0.0, need).astype(np.float32)
        self._need_flat = need
        return need

    # --- potential for left panel ---
    def compute_potential(self, V, alpha=1.0):
        diff = self.field_points[:, None, :] - V[None, :, :].astype(np.float32)
        diff = torus_wrap(diff, self.size, self.use_wrap)
        dist = np.linalg.norm(diff, axis=-1) + 1e-12
        pot  = alpha * self.need_flat(use_visibility=True) * np.log(dist).sum(axis=1)
        self.potential = pot.reshape(self.size, self.size).astype(np.float32)

    # Forces on VP[0] only
    def forces_on_primary(self, V, k_attr, k_rep, alpha=1.0, rep_sigma=10.0, rep_amp=100.0):
        if V.shape[0] == 0:
            z = np.zeros((1,2), dtype=np.float32)
            return z, z, z
        v0 = V[0:1, :].astype(np.float32)
        # attraction
        diff = self.field_points - v0
        diff = torus_wrap(diff, self.size, self.use_wrap)
        dist = np.linalg.norm(diff, axis=-1) + 1e-12
        w = (alpha * self.need_flat(use_visibility=True)).astype(np.float32)
        F_attr0 = (w[:, None] * diff / (dist[:, None]**2)).sum(axis=0, keepdims=True)
        # repulsion from others
        if V.shape[0] <= 1:
            F_rep0 = np.zeros_like(F_attr0)
        else:
            others = V[1:, :].astype(np.float32)
            pdiff = v0 - others
            pdiff = torus_wrap(pdiff, self.size, self.use_wrap)
            r2 = np.sum(pdiff**2, axis=-1) + 1e-12
            coeff = (rep_amp / (rep_sigma**2)) * np.exp(-r2 / (2*rep_sigma**2))
            F_rep0 = (coeff[:, None] * pdiff).sum(axis=0, keepdims=True)
        F_attr0 *= k_attr; F_rep0 *= k_rep
        return F_attr0.astype(np.float32), F_rep0.astype(np.float32), (F_attr0+F_rep0).astype(np.float32)

    # Forces on ALL viewpoints (when move_others=true)
    def forces_on_all(self, V, k_attr, k_rep, alpha=1.0, rep_sigma=10.0, rep_amp=100.0):
        M = V.shape[0]
        if M == 0:
            return np.zeros((0,2), dtype=np.float32), np.zeros((0,2), dtype=np.float32), np.zeros((0,2), dtype=np.float32)
        V = V.astype(np.float32)
        # attraction
        diff = self.field_points[:, None, :] - V[None, :, :]
        diff = torus_wrap(diff, self.size, self.use_wrap)
        dist = np.linalg.norm(diff, axis=-1) + 1e-12
        w    = (alpha * self.need_flat(use_visibility=True))[:, None]
        F_attr = (w[..., None] * diff / (dist[..., None]**2)).sum(axis=0)
        # repulsion
        pdiff = V[:, None, :] - V[None, :, :]
        pdiff = torus_wrap(pdiff, self.size, self.use_wrap)
        r2    = np.sum(pdiff**2, axis=-1) + 1e-12
        np.fill_diagonal(pdiff[...,0], 0.0); np.fill_diagonal(pdiff[...,1], 0.0)
        np.fill_diagonal(r2, 1.0)
        coeff = (rep_amp / (rep_sigma**2)) * np.exp(-r2 / (2*rep_sigma**2))
        F_rep = (coeff[..., None] * pdiff).sum(axis=1)
        F_attr *= k_attr; F_rep *= k_rep
        return F_attr.astype(np.float32), F_rep.astype(np.float32), (F_attr+F_rep).astype(np.float32)

    # --- VP[0]-perspective probe field (attr & rep) ---
    def probe_vector_field_primary(self, V, include_attr, include_rep, use_visibility,
                                   k_attr, k_rep, alpha=1.0, rep_sigma=10.0, rep_amp=100.0,
                                   half_window=50.0, grid=18, need_eps=1e-6, max_sources=2000):
        center = np.array([self.size/2, self.size/2], dtype=np.float32)
        xs = np.linspace(center[0]-half_window, center[0]+half_window, grid).astype(np.float32)
        ys = np.linspace(center[1]-half_window, center[1]+half_window, grid).astype(np.float32)
        GX, GY = np.meshgrid(xs, ys)
        GP = np.stack([GX.ravel(), GY.ravel()], axis=-1).astype(np.float32)

        U_attr = np.zeros(GP.shape[0], dtype=np.float32); V_attr = np.zeros(GP.shape[0], dtype=np.float32)
        U_rep  = np.zeros(GP.shape[0], dtype=np.float32); V_rep  = np.zeros(GP.shape[0], dtype=np.float32)

        need_grid = (np.clip(1.0 - self.visibility/self.vis_requirement, 0.0, 1.0) *
                     (self.omega ** self.omega_gamma)).astype(np.float32)
        if self.need_hard_clip:
            need_grid = np.where(self.visibility >= self.need_clip_threshold*self.vis_requirement, 0.0, need_grid)
        ix = np.clip(np.round(GX).astype(int), 0, self.size-1)
        iy = np.clip(np.round(GY).astype(int), 0, self.size-1)
        local_need = need_grid[iy, ix].ravel()

        if include_attr:
            need = self.need_flat(use_visibility=use_visibility)
            idx = np.where(need > need_eps)[0]
            if idx.size > 0:
                if idx.size > max_sources:
                    w = need[idx]; p = w / (w.sum()+1e-12)
                    sel = np.random.choice(idx, size=max_sources, replace=False, p=p)
                    w_sel = need[sel]
                    scale = (w.sum() / (w_sel.sum()+1e-12)).astype(np.float32)
                    src_xy = self.field_points[sel, :].astype(np.float32)
                    src_w  = (w_sel * scale).astype(np.float32)
                else:
                    src_xy = self.field_points[idx, :].astype(np.float32)
                    src_w  = need[idx].astype(np.float32)
                diff_p = src_xy[None, :, :] - GP[:, None, :]
                diff_p = torus_wrap(diff_p, self.size, self.use_wrap)
                dist_p = np.linalg.norm(diff_p, axis=-1) + 1e-12
                Fp = (src_w[None, :, None] * diff_p / (dist_p[..., None]**2)).sum(axis=1)
                if bool(self.cfg.get("vf_zero_inside_need", True)):
                    Fp *= (local_need > 0.0).astype(np.float32)[:, None]
                U_attr += (k_attr * Fp[:,0]); V_attr += (k_attr * Fp[:,1])

        if include_rep and V.shape[0] > 1:
            others = V[1:, :].astype(np.float32)
            diff_v = GP[:, None, :] - others[None, :, :]
            diff_v = torus_wrap(diff_v, self.size, self.use_wrap)
            r2 = np.sum(diff_v**2, axis=-1) + 1e-12
            coeff = (rep_amp / (rep_sigma**2)) * np.exp(-r2 / (2*rep_sigma**2))
            Fr = (coeff[..., None] * diff_v).sum(axis=1)
            U_rep += (k_rep * Fr[:,0]); V_rep += (k_rep * Fr[:,1])

        return GX, GY, U_attr.reshape(GX.shape), V_attr.reshape(GY.shape), U_rep.reshape(GX.shape), V_rep.reshape(GY.shape)

# ---------------------- plotting ----------------------
def setup_figure(field, cfg):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios':[1, 1]})

    # LEFT
    imL = axL.imshow(np.zeros_like(field.potential), cmap='viridis', origin='lower',
                     extent=[0,field.size,0,field.size])
    imImpL = axL.imshow(field.omega, cmap='magma', origin='lower',
                        extent=[0,field.size,0,field.size],
                        alpha=float(cfg["importance_viz_alpha"]))
    fov_patches_L = []
    scatL = axL.scatter([], [], c='deepskyblue', s=34, edgecolors='white',
                        linewidths=0.6, zorder=3)
    qvpL = axL.quiver([], [], [], [], angles='xy', scale_units='xy',
                      scale=float(cfg["vp_force_scale"]), width=float(cfg["vp_force_width"]),
                      color='cyan', zorder=6)
    txtL = axL.text(0.02, 0.98, "", transform=axL.transAxes,
                    va='top', ha='left', fontsize=9, color='cyan')

    axL.set_xlim(0,field.size); axL.set_ylim(0,field.size)
    axL.set_xlabel("x"); axL.set_ylabel("y")
    axL.set_title("Potential & VP[0] resultant")

    # RIGHT
    imR = None
    if str(cfg["vf_overlay"]) != "none":
        bgR = field.omega if str(cfg["vf_overlay"])=="importance" else np.zeros_like(field.omega)
        imR = axR.imshow(bgR, cmap='gray', origin='lower',
                         extent=[0,field.size,0,field.size],
                         alpha=min(0.4, float(cfg["importance_viz_alpha"])+0.15))
        axR.set_facecolor("#0a0a0a")

    qAttr = None
    qRep  = None
    scatR = axR.scatter([], [], c=['yellow'], s=36, edgecolors='black',
                        linewidths=0.6, zorder=4)
    qvpR = axR.quiver([], [], [], [], angles='xy', scale_units='xy',
                      scale=float(cfg["vp_force_scale"]), width=float(cfg["vp_force_width"]),
                      color='cyan', zorder=7)
    txtR = axR.text(0.02, 0.98, "", transform=axR.transAxes,
                    va='top', ha='left', fontsize=9, color='cyan')

    center = np.array([field.size/2, field.size/2])
    hw = float(cfg["vf_window"])
    axR.add_patch(plt.Rectangle((center[0]-hw, center[1]-hw), 2*hw, 2*hw,
                                fill=False, edgecolor='white', lw=1.0, alpha=0.8, zorder=5))
    axR.set_xlim(max(0, center[0]-hw), min(field.size, center[0]+hw))
    axR.set_ylim(max(0, center[1]-hw), min(field.size, center[1]+hw))
    axR.set_xlabel("x"); axR.set_ylabel("y")
    axR.set_title("Vector field (VP[0] perspective)")

    fig.canvas.draw()
    return fig, axL, axR, imL, imImpL, fov_patches_L, scatL, qvpL, txtL, imR, qAttr, qRep, scatR, qvpR, txtR

def update_left(axL, field, V, pot, cfg, imL, imImpL, fov_patches, scatL, qvpL, txtL, F0):
    imL.set_data(pot)
    imL.set_clim(vmin=float(np.nanmin(pot)), vmax=float(np.nanmax(pot)))
    imImpL.set_alpha(float(cfg["importance_viz_alpha"]))
    for p in fov_patches:
        p.remove()
    fov_patches.clear()
    for p in V:
        circ = plt.Circle((p[0], p[1]), field.R, edgecolor='white', facecolor='none', alpha=0.2, lw=1)
        axL.add_patch(circ); fov_patches.append(circ)
    scatL.set_offsets(V)
    qvpL.set_offsets(V[0:1, :])
    qvpL.set_UVC(F0[0:1,0], F0[0:1,1])
    txtL.set_text(f"|F0|={float(np.linalg.norm(F0)):0.3f}")

def update_right(axR, field, V, cfg, imR, qAttr, qRep, scatR, qvpR, txtR, F0, quiver_cache):
    if imR is not None and str(cfg["vf_overlay"]) == "need":
        need_now = (np.clip(1.0 - field.visibility/field.vis_requirement, 0.0, 1.0) *
                    (field.omega ** field.omega_gamma))
        if field.need_hard_clip:
            need_now = np.where(field.visibility >= field.need_clip_threshold*field.vis_requirement, 0.0, need_now)
        imR.set_data(need_now)

    it = quiver_cache["it"]
    if (it % int(cfg["vf_update_every"])) == 0 or qAttr is None or qRep is None:
        GX, GY, Ua, Va, Ur, Vr = field.probe_vector_field_primary(
            V,
            include_attr=bool(cfg["vf_include_attr"]),
            include_rep =bool(cfg["vf_include_rep"]) and (len(V) > 1),
            use_visibility=bool(cfg["vf_use_visibility"]),
            k_attr=float(cfg["k_attr"]), k_rep=float(cfg["k_rep"]),
            alpha=1.0,
            rep_sigma=float(cfg["rep_sigma"]), rep_amp=float(cfg["rep_amp"]),
            half_window=float(cfg["vf_window"]), grid=int(cfg["vf_grid"]),
            need_eps=float(cfg["vf_need_eps"]), max_sources=int(cfg["vf_max_sources"])
        )
        quiver_cache.update({"GX":GX, "GY":GY, "Ua":Ua, "Va":Va, "Ur":Ur, "Vr":Vr})
    else:
        GX, GY = quiver_cache["GX"], quiver_cache["GY"]
        Ua, Va = quiver_cache["Ua"], quiver_cache["Va"]
        Ur, Vr = quiver_cache["Ur"], quiver_cache["Vr"]

    if qAttr is None:
        magnitude = np.hypot(Ua, Va)
        qAttr = axR.quiver(GX, GY, Ua, Va, magnitude,
                           cmap='Greys',
                           scale=float(cfg["vf_quiver_scale_attr"]),
                           width=float(cfg["vf_quiver_width_attr"]),
                           pivot='mid', zorder=3)
    else:
        qAttr.set_UVC(Ua, Va)

    if qRep is None:
        qRep = axR.quiver(GX, GY, Ur, Vr, angles='xy', scale_units='xy',
                          scale=float(cfg["vf_quiver_scale_rep"]), width=float(cfg["vf_quiver_width_rep"]),
                          pivot='mid', color='#e91e63', zorder=4)
    else:
        qRep.set_UVC(Ur, Vr)

    if len(V) > 0:
        colors = ['yellow'] + ['#666666']*(len(V)-1)
        scatR.set_offsets(V)
        if hasattr(scatR, "set_facecolors"): scatR.set_facecolors(colors)
        else: scatR.set_color(colors)
    else:
        scatR.set_offsets(np.empty((0,2)))

    qvpR.set_offsets(V[0:1, :])
    qvpR.set_UVC(F0[0:1,0], F0[0:1,1])
    txtR.set_text(f"|F0|={float(np.linalg.norm(F0)):0.3f}")

    return qAttr, qRep

# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="old_config.json")
    args = ap.parse_args()

    cfg = load_json(args.config)
    np.random.seed(int(cfg["seed"]))
    ensure_dir(cfg["save_dir"])

    # --- Fixed-length capture: 10s @ 60fps ---
    fps     = int(cfg.get("video_fps", 60))
    seconds = int(cfg.get("video_seconds", 10))
    max_frames = fps * seconds
    out_path   = os.path.join(cfg["save_dir"], cfg.get("video_filename", "simulation_1080p60.mp4"))
    dpi_val    = int(cfg.get("video_dpi", 100))

    field = Field(cfg)
    V = (np.random.rand(1,2).astype(np.float32) * field.size)

    # We simulate, cache exactly 600 frames, then render/export once.
    frames_pts, frames_pot = [], []

    stop = False
    def sigint(_s, _f):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, sigint)

    it = 0
    m = np.zeros_like(V); v = np.zeros_like(V)      # Adam for VP[0]
    m_all = None; v_all = None                      # Adam for all (when move_others=true)
    quiver_cache = {"it": 0}

    # ----- SIMULATION LOOP (headless; just cache frames) -----
    while True:
        field.update_visibility(V)
        field.compute_potential(V)

        # Forces
        if bool(cfg["move_others"]):
            F_attr_all, F_rep_all, F_tot_all = field.forces_on_all(
                V,
                k_attr=float(cfg["k_attr"]), k_rep=float(cfg["k_rep"]),
                alpha=1.0, rep_sigma=float(cfg["rep_sigma"]), rep_amp=float(cfg["rep_amp"])
            )
            F_tot0 = F_tot_all[0:1, :]
        else:
            F_attr0, F_rep0, F_tot0 = field.forces_on_primary(
                V,
                k_attr=float(cfg["k_attr"]), k_rep=float(cfg["k_rep"]),
                alpha=1.0, rep_sigma=float(cfg["rep_sigma"]), rep_amp=float(cfg["rep_amp"])
            )

        # Cache the current frame
        frames_pts.append(V.copy())
        frames_pot.append(field.potential.copy())

        # Motion update(s)
        if bool(cfg["move_others"]):
            if (m_all is None) or (m_all.shape[0] != V.shape[0]):
                m_all = np.zeros_like(V); v_all = np.zeros_like(V)
            g = F_tot_all.astype(np.float32)
            m_all = 0.9*m_all + 0.1*g
            v_all = 0.999*v_all + 0.001*(g*g)
            mhat = m_all/(1-0.9**(it+1)); vhat = v_all/(1-0.999**(it+1))
            step = float(cfg["adam_step"]) * mhat / (np.sqrt(vhat) + 1e-8)
            V = (V + step) % field.size
        else:
            g0 = F_tot0.astype(np.float32)
            m = 0.9*m + 0.1*g0
            v = 0.999*v + 0.001*(g0*g0)
            mhat = m/(1-0.9**(it+1)); vhat = v/(1-0.999**(it+1))
            step = float(cfg["adam_step"]) * mhat / (np.sqrt(vhat) + 1e-8)
            V[0:1, :] = (V[0:1, :] + step) % field.size

        # NBV insertion (kept as in config; has no effect if max_viewpoints=1)
        can_insert = (len(V) < int(cfg["max_viewpoints"]))
        insert = (it % int(cfg["frames_per_stage"]) == 0) and can_insert if bool(cfg["smart_nbv_insertion"]) \
                 else (it % int(cfg["frames_per_stage"]) == 0) and can_insert
        if insert:
            nbv = field.field_points[np.argmax(field.potential)].astype(np.float32)
            V = np.vstack([V, nbv[None,:]])
            if bool(cfg["move_others"]):
                m_all = np.vstack([m_all, np.zeros((1,2), dtype=np.float32)])
                v_all = np.vstack([v_all, np.zeros((1,2), dtype=np.float32)])

        it += 1
        if stop or it >= max_frames:
            break

    # ----- RENDER/EXPORT (single pass) -----
    fig, axL, axR, imL, imImpL, fov_patches, scatL, qvpL, txtL, imR, qAttr, qRep, scatR, qvpR, txtR = setup_figure(field, cfg)
    # 1080p canvas (inches = pixels / dpi), here we keep default (12x5 @ dpi) for speed; scale if needed:
    # fig.set_size_inches(19.2, 10.8)  # uncomment to hard-set 1920x1080 at dpi=100

    quiver_cache2 = {"it": 0, "qAttr": None, "qRep": None}

    def draw(i):
        V_i = frames_pts[i]
        pot_i = frames_pot[i]
        field.update_visibility(V_i)
        # Left
        F0 = np.zeros((1,2), dtype=np.float32)
        update_left(axL, field, V_i, pot_i, cfg, imL, imImpL, fov_patches, scatL, qvpL, txtL, F0)
        # Right
        quiver_cache2["it"] = i
        qA, qR = update_right(axR, field, V_i, cfg, imR,
                              quiver_cache2.get("qAttr"), quiver_cache2.get("qRep"),
                              scatR, qvpR, txtR, F0, quiver_cache2)
        quiver_cache2["qAttr"], quiver_cache2["qRep"] = qA, qR
        return []

    ani = animation.FuncAnimation(fig, draw, frames=len(frames_pts), interval=1000/fps, blit=False)

    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=18000, extra_args=["-pix_fmt", "yuv420p"])
    ani.save(out_path, writer=writer, dpi=dpi_val, savefig_kwargs=dict(facecolor=fig.get_facecolor()))
    print("[Video] saved:", out_path)

    # Optional: 2 snapshots (first/last)
    def snapshot(Vsnap, pot, fn):
        figS, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5), gridspec_kw={'width_ratios':[1,1]})
        # left
        ax0.imshow(pot, cmap='viridis', origin='lower', extent=[0,field.size,0,field.size])
        ax0.imshow(field.omega, cmap='magma', origin='lower', extent=[0,field.size,0,field.size],
                   alpha=float(cfg["importance_viz_alpha"]))
        ax0.add_patch(plt.Circle((Vsnap[0,0], Vsnap[0,1]), field.R, edgecolor='white', facecolor='none', alpha=0.2, lw=1))
        ax0.scatter(Vsnap[:,0], Vsnap[:,1], c='deepskyblue', s=34, edgecolors='white', linewidths=0.6, zorder=3)
        ax0.set_xlim(0,field.size); ax0.set_ylim(0,field.size)
        ax0.set_title("Snapshot")
        # right
        need_now = (np.clip(1.0 - field.visibility/field.vis_requirement, 0.0, 1.0) *
                    (field.omega ** field.omega_gamma))
        if field.need_hard_clip:
            need_now = np.where(field.visibility >= field.need_clip_threshold*field.vis_requirement, 0.0, need_now)
        ax1.imshow(need_now, cmap='magma', origin='lower', extent=[0,field.size,0,field.size], alpha=0.45)
        GX, GY, Ua, Va, Ur, Vr = field.probe_vector_field_primary(
            Vsnap,
            include_attr=bool(cfg["vf_include_attr"]),
            include_rep =bool(cfg["vf_include_rep"]) and (len(Vsnap) > 1),
            use_visibility=bool(cfg["vf_use_visibility"]),
            k_attr=float(cfg["k_attr"]), k_rep=float(cfg["k_rep"]),
            alpha=1.0,
            rep_sigma=float(cfg["rep_sigma"]), rep_amp=float(cfg["rep_amp"]),
            half_window=float(cfg["vf_window"]), grid=int(cfg["vf_grid"]),
            need_eps=float(cfg["vf_need_eps"]), max_sources=int(cfg["vf_max_sources"])
        )
        ax1.quiver(GX, GY, Ua, Va, angles='xy', scale_units='xy',
                   scale=float(cfg["vf_quiver_scale_attr"]), width=float(cfg["vf_quiver_width_attr"]),
                   pivot='mid', color='black', zorder=3)
        ax1.quiver(GX, GY, Ur, Vr, angles='xy', scale_units='xy',
                   scale=float(cfg["vf_quiver_scale_rep"]), width=float(cfg["vf_quiver_width_rep"]),
                   pivot='mid', color='red', zorder=4)
        colors = ['yellow'] + ['#666666']*(len(Vsnap)-1)
        ax1.scatter(Vsnap[:,0], Vsnap[:,1], c=colors, s=36, edgecolors='black', linewidths=0.6, zorder=4)
        ax1.set_title("Vector field")
        figS.tight_layout()
        figS.savefig(fn, dpi=300, bbox_inches='tight'); plt.close(figS)

    snapshot(frames_pts[0], frames_pot[0], os.path.join(cfg["save_dir"], "snap_initial.png"))
    snapshot(frames_pts[-1], frames_pot[-1], os.path.join(cfg["save_dir"], "snap_final.png"))
    plt.close('all')

if __name__ == "__main__":
    main()
