#!/usr/bin/env python3
"""
Silhouette PNGs  →  Matplotlib marker Paths (with holes)

What’s added in this version
----------------------------
• Preprocess each icon so its min(width, height) matches **goku**’s min-side,
  with aspect ratio preserved. (GLOBAL_PADDING still applies uniformly.)
• Everything else (contour extraction, hole handling, orientation, demo) stays the same.

Notes
-----
• Uses the PNG alpha channel you already have (transparent = background/holes).
• Holes are preserved via multiple subpaths and correct winding (outers CCW, holes CW).
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.path import Path
from skimage import measure

# ---------------- Tunables ----------------
ICON_DIR_IN      = "./icons_clean"  # folder with your cleaned, transparent PNGs
REF_ICON_NAME    = "goku"           # the icon whose min-side is the reference
TARGET_MIN_SIDE  = 1.0              # kept for API compatibility (not used in match-to-goku path)
GLOBAL_PADDING   = 0.90             # uniform shrink after matching (1.0 = no shrink)
MIN_CONTOUR_LEN  = 12               # ignore tiny specks

# ---------------- Utilities ----------------
def _to_float01(arr):
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    return ((arr - mn) / (mx - mn + 1e-12)).clip(0, 1)

def _polygon_area_signed(verts):
    # Shoelace; positive if CCW
    x = verts[:, 0]
    y = verts[:, 1]
    x1 = np.roll(x, -1)
    y1 = np.roll(y, -1)
    return 0.5 * np.sum(x * y1 - x1 * y)

def _centroid_polygon(verts):
    # Green’s-theorem centroid (robust for concave simple polygons)
    a = _polygon_area_signed(verts)
    if abs(a) < 1e-12:
        return verts.mean(axis=0)  # degenerate fallback
    x = verts[:, 0]; y = verts[:, 1]
    x1 = np.roll(x, -1); y1 = np.roll(y, -1)
    cross = x * y1 - x1 * y
    cx = np.sum((x + x1) * cross) / (6.0 * a)
    cy = np.sum((y + y1) * cross) / (6.0 * a)
    return np.array([cx, cy], dtype=np.float64)

def _nesting_depth(polys):
    """
    Depth = how many polygons contain this polygon's centroid.
    Even depth → outer ring; odd depth → hole ring.
    """
    from matplotlib.path import Path as MPath
    paths = [MPath(p, closed=True) for p in polys]
    cents = [_centroid_polygon(p) for p in polys]
    depths = []
    for i, c in enumerate(cents):
        d = 0
        for j, pj in enumerate(paths):
            if i == j:
                continue
            if pj.contains_point(c):
                d += 1
        depths.append(d)
    return depths

# ---------------- Contour extraction helpers (RAW, no scaling yet) ----------------
def _extract_contour_polys_from_rgba(rgba):
    """Return list of polygons from alpha mask. Coordinates in image space; Y flipped upright."""
    alpha = _to_float01(rgba[..., 3])
    mask  = alpha > 0.5
    contours = measure.find_contours(mask.astype(float), 0.5)
    polys = []
    for c in contours:
        if len(c) < MIN_CONTOUR_LEN:
            continue
        # c is (row, col); convert to (x, y) and flip Y so up is +Y
        verts = np.stack([c[:, 1], -c[:, 0]], axis=1)
        polys.append(verts)
    return polys

def _raw_min_side_from_polys(polys):
    pts = np.vstack(polys)
    (x0, y0) = pts.min(axis=0)
    (x1, y1) = pts.max(axis=0)
    w, h = (x1 - x0), (y1 - y0)
    return float(max(1e-12, min(w, h)))

def _raw_min_side_from_rgba(rgba):
    polys = _extract_contour_polys_from_rgba(rgba)
    if not polys:
        return 1.0
    return _raw_min_side_from_polys(polys)

# ---------------- RGBA → Path (with scaling to a target min-side) ----------------
def _polys_to_path_oriented(polys):
    """Orient outers CCW, holes CW; then pack into a compound Path."""
    depths = _nesting_depth(polys)
    pieces = []
    for pg, d in zip(polys, depths):
        want_ccw = (d % 2 == 0)            # even depth = outer
        is_ccw   = _polygon_area_signed(pg) > 0
        if is_ccw != want_ccw:
            pg = pg[::-1].copy()
        pieces.append(pg)

    verts_all, codes_all = [], []
    for pg in pieces:
        codes = np.full(len(pg), Path.LINETO, dtype=np.uint8)
        codes[0] = Path.MOVETO
        verts_all.append(pg)
        codes_all.append(codes)
        verts_all.append(pg[[0]])
        codes_all.append(np.array([Path.CLOSEPOLY], dtype=np.uint8))

    return Path(np.vstack(verts_all), np.concatenate(codes_all))

def load_icon_as_marker_with_target(png_path, ref_min_side, global_padding=GLOBAL_PADDING):
    """
    1) Extract polygons from PNG alpha.
    2) Center all rings together.
    3) Uniformly scale so min(width, height) == ref_min_side (aspect preserved).
    4) Apply GLOBAL_PADDING.
    5) Return an oriented compound Path.
    """
    rgba  = np.array(Image.open(png_path).convert("RGBA"))
    polys = _extract_contour_polys_from_rgba(rgba)
    if not polys:
        raise ValueError(f"No usable contours in {png_path}")

    # center all rings together
    all_pts = np.vstack(polys)
    all_pts -= all_pts.mean(axis=0, keepdims=True)

    # current min-side and uniform scale to match reference
    (x0, y0) = all_pts.min(axis=0)
    (x1, y1) = all_pts.max(axis=0)
    cur_min  = float(max(1e-12, min(x1 - x0, y1 - y0)))
    s        = (ref_min_side / cur_min) * global_padding
    all_pts *= s

    # re-split to per-ring arrays
    polys_scaled, idx = [], 0
    for p in polys:
        n = len(p)
        polys_scaled.append(all_pts[idx:idx+n])
        idx += n

    return _polys_to_path_oriented(polys_scaled)

# ---------------- Legacy API (kept intact, not used in main) ----------------
def rgba_to_marker_path(rgba, target_min_side=TARGET_MIN_SIDE, global_padding=GLOBAL_PADDING):
    """
    Legacy: convert RGBA to Path by normalizing each image to TARGET_MIN_SIDE.
    Kept for compatibility; main flow below uses goku-referenced sizing instead.
    """
    polys = _extract_contour_polys_from_rgba(rgba)
    if not polys:
        raise ValueError("No contours found. Check input alpha.")

    # normalize: center + scale to target_min_side
    all_pts = np.vstack(polys)
    all_pts -= all_pts.mean(axis=0, keepdims=True)
    (x0, y0) = all_pts.min(axis=0); (x1, y1) = all_pts.max(axis=0)
    cur_min  = float(max(1e-12, min(x1 - x0, y1 - y0)))
    s        = (target_min_side / cur_min) * global_padding
    all_pts *= s

    # re-split and pack
    polys_scaled, idx = [], 0
    for p in polys:
        n = len(p)
        polys_scaled.append(all_pts[idx:idx+n])
        idx += n
    return _polys_to_path_oriented(polys_scaled)

def load_icon_as_marker(png_path, target_min_side=TARGET_MIN_SIDE, global_padding=GLOBAL_PADDING):
    rgba = np.array(Image.open(png_path).convert("RGBA"))
    return rgba_to_marker_path(rgba, target_min_side=target_min_side, global_padding=global_padding)

# ---------------- Folder loader that MATCHES GOKU ----------------
def markerize_folder_match_goku(folder=ICON_DIR_IN, ref_name=REF_ICON_NAME, global_padding=GLOBAL_PADDING):
    # 1) compute goku's raw min-side
    ref_png  = os.path.join(folder, f"{ref_name}.png")
    ref_rgba = np.array(Image.open(ref_png).convert("RGBA"))
    ref_min  = _raw_min_side_from_rgba(ref_rgba)

    # 2) scale every icon uniformly to match that min-side
    markers = {}
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".png"):
            continue
        name = os.path.splitext(fname)[0]
        path = os.path.join(folder, fname)
        markers[name] = load_icon_as_marker_with_target(path, ref_min, global_padding=global_padding)
        print(f"[marker] {name}  -> matched to {ref_name} min-side={ref_min:.3f}")
    return markers

# --- Optional PathPatch-based renderer (kept; not used by default demo) ---
from matplotlib.patches import PathPatch
from matplotlib import transforms

def scatter_with_holes(ax, xs, ys, paths, s=1600, facecolors=None,
                       edgecolor='black', linewidth=0.9, zorder=3):
    """
    Draw compound Path markers (with holes) as PathPatches.
    This version scales in data space; if you ever see aspect distortion,
    either set `ax.set_aspect('equal', adjustable='box')` or switch to a
    pure display-space transform.
    """
    if isinstance(paths, Path):
        paths = [paths] * len(xs)
    if facecolors is None:
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        facecolors = [cycle[i % len(cycle)] for i in range(len(xs))]

    L_pts = float(np.sqrt(s))
    L_px  = L_pts * (ax.figure.dpi / 72.0)

    for (x, y, p, fc) in zip(xs, ys, paths, facecolors):
        xd, yd = ax.transData.transform((x, y))
        x2, y2 = ax.transData.inverted().transform((xd + L_px, yd + L_px))
        sx = (x2 - x) or 1e-12
        sy = (y2 - y) or 1e-12
        k  = min(sx, sy)  # uniform scale to preserve icon aspect

        trans = transforms.Affine2D().scale(k, k).translate(x, y) + ax.transData
        patch = PathPatch(p, facecolor=fc, edgecolor=edgecolor, lw=linewidth,
                          transform=trans, zorder=zorder)
        ax.add_patch(patch)
    return ax

# ---------------- Demo ----------------
def demo_scatter(markers):
    xs = np.arange(len(markers))
    ys = np.linspace(0.2, 0.8, len(markers))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for i, (k, m) in enumerate(markers.items()):
        # Using scatter keeps aspect independent of axis scaling (size in points)
        coll = ax.scatter([i], [ys[i]], marker=m, s=2200, label=k,
                          edgecolor='black', linewidths=0.75)
        # On some Matplotlib builds, this is ignored for PathCollection; harmless to try.
        try:
            coll.set_fillrule('evenodd')
        except Exception:
            pass
    ax.set_xlim(-0.5, len(markers) - 0.5)
    ax.set_ylim(0.15, 0.85)
    ax.set_xticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.set_title("Markers with holes (compound paths), min-side matched to goku")
    fig.tight_layout()
    plt.show()

# ---------------- Main ----------------
if __name__ == "__main__":
    # Build markers with per-icon uniform scaling so min-side == goku's
    markers = markerize_folder_match_goku(ICON_DIR_IN, ref_name=REF_ICON_NAME, global_padding=GLOBAL_PADDING)
    demo_scatter(markers)
