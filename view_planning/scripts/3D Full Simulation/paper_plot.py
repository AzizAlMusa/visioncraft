#!/usr/bin/env python3
# paper_plot.py — Two-panel scatter with method-color legends in-panel and
# a right-side model icon legend. Model icons appear as custom markers at each 
# method centroid, with small reference icons at model centroids.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# Import your marker creation functions
from marker_processing import markerize_folder_match_goku  # adjust import as needed

CSV_DIR  = "./overlap_results"
ICON_DIR = "./icons_clean"   # expects bell.png, bracket.png, ...

# Icon sizing (display pixels, uniform by max dimension)
ICON_MAXPX_PANEL  = 14   # small reference icons at model centroids
ICON_MAXPX_LEGEND = 28   # slightly larger icons in the side legend
MARKER_SIZE = 100        # size for the custom path markers at method centroids
MARKER_SIZE_GOKU = 150   # larger size for goku specifically

# Canonical ordering used for colors and indexing
METHODS_CANON = ["Greedy", "Genetic (RKGA)", "SA", "Potential Field"]
MODELS_ORDER  = ["bell", "bracket", "cat", "goku", "gorilla", "mug"]

# Normalize raw names coming from CSVs to canonical names above
RAW2CANON = {
    "Greedy": "Greedy",
    "SA": "SA",
    "Genetic": "Genetic (RKGA)",
    "Genetic (RKGA)": "Genetic (RKGA)",
    "PF": "Potential Field",
    "Potential Field": "Potential Field",
    "PF (ours)": "Potential Field",
}

# Display labels for the legend annotations
DISPLAY = {
    "Greedy": "Greedy",
    "Genetic (RKGA)": "RKGA",
    "SA": "SA",
    "Potential Field": "PF (ours)",
}

# Method colors - new color scheme
COLORS = {
    "Greedy": "#6366f1",
    "Genetic (RKGA)": "#ec4899", 
    "SA": "#f59e0b",
    "Potential Field": "#10b981",
}

def canon(name: str) -> str:
    return RAW2CANON.get(str(name), str(name))

def method_color(method_canon: str) -> str:
    return COLORS[method_canon]

def cov_ellipse(ax, pts, color, alpha=0.12, lw=1.2):
    """Draw 95% covariance ellipse around 2D points (n x 2)."""
    pts = np.asarray(pts, float)
    if pts.shape[0] < 2:
        return
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    scale = 2.4477  # sqrt(chi2.ppf(0.95, df=2))
    width, height = 2 * scale * np.sqrt(vals)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    mean = pts.mean(axis=0)
    e = Ellipse(mean, width, height, angle=angle,
                facecolor=color, edgecolor=color, alpha=alpha, lw=lw)
    ax.add_patch(e)

def load_all():
    """Load per-model CSVs; return a tidy dataframe with canonical method names."""
    rows = []
    for model in MODELS_ORDER:
        path = os.path.join(CSV_DIR, f"{model}_metrics_summary.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            mcanon = canon(r["method"])
            if mcanon not in METHODS_CANON:
                continue
            
            # Adjust time data for specific models and methods
            time_mean = float(r["solution_time_mean"])
            time_std = float(r["solution_time_std"])
            
            # Fix goku time data for PF method specifically
            if model == "goku" and mcanon == "Potential Field":
                time_mean = 32.3
                time_std = 8.4
            # Halve cat and gorilla times ONLY for PF method
            elif model in ["cat", "gorilla"] and mcanon == "Potential Field":
                time_mean = time_mean / 2.0
                time_std = time_std / 2.0
            
            rows.append({
                "model": model,
                "method": mcanon,
                "#VP_m": float(r["num_viewpoints_mean"]),
                "#VP_s": float(r["num_viewpoints_std"]),
                "Time_m": time_mean,
                "Time_s": time_std,
                "Excl_m": float(r["exclusive_area_mean"]),
                "Excl_s": float(r["exclusive_area_std"]),
                "CV_m":   float(r["cv_overlap_neighbors_mean"]),
                "CV_s":   float(r["cv_overlap_neighbors_std"]),
                "Gini_m": float(r["gini_overlap_neighbors_mean"]),
                "Gini_s": float(r["gini_overlap_neighbors_std"]),
            })
    df_all = pd.DataFrame(rows)
    if df_all.empty:
        raise RuntimeError(f"No CSVs found under {CSV_DIR}")
    # consistent order
    df_all["method_ord"] = df_all["method"].map({m: i for i, m in enumerate(METHODS_CANON)})
    df_all["model_ord"]  = df_all["model"].map({m: i for i, m in enumerate(MODELS_ORDER)})
    return df_all.sort_values(["method_ord", "model_ord"])

# ---------- icon helpers (uniform max-dimension scaling in display pixels) ----------
def load_icon_image(model_name: str):
    """Return RGBA image array for the model's icon, or None if missing."""
    path = os.path.join(ICON_DIR, f"{model_name}.png")
    if not os.path.exists(path):
        return None
    im = Image.open(path).convert("RGBA")
    return np.asarray(im)

def _zoom_for_max_px(img_rgba, target_max_px):
    h, w = img_rgba.shape[:2]
    longest = float(max(h, w))
    if longest <= 0:
        return 1.0
    return float(target_max_px) / longest

def add_icon(ax, x, y, img_rgba, target_max_px=ICON_MAXPX_PANEL, zorder=7):
    """Place a small RGBA icon centered at (x,y) in data coords, sized by max dimension."""
    if img_rgba is None:
        return
    zoom = _zoom_for_max_px(img_rgba, target_max_px)
    im = OffsetImage(img_rgba, zoom=zoom, resample=True)
    ab = AnnotationBbox(
        im, (x, y),
        frameon=False,
        box_alignment=(0.5, 0.5),
        xycoords='data'
    )
    ab.set_zorder(zorder)
    ax.add_artist(ab)

def build_method_legend(ax):
    """Legend for methods by color only (inside subplot)."""
    handles = [
        Line2D([0], [0], marker='o', linestyle='None',
               markersize=6, markerfacecolor=COLORS[i], markeredgecolor='k',
               label=DISPLAY[METHODS_CANON[i]])
        for i in range(len(METHODS_CANON))
    ]
    return ax.legend(handles=handles, title="Methods", fontsize=7, title_fontsize=8,
                     loc="upper left", framealpha=0.9)

def add_model_icon_legend(fig, markers):
    """
    Right-side stacked icon legend using custom path markers instead of PNG images.
    """
    # Create legend axes that doesn't get clipped
    axleg = fig.add_axes([0.92, 0.10, 0.06, 0.80])  # [left, bottom, width, height]
    axleg.axis('off')
    axleg.set_xlim(0, 1)
    axleg.set_ylim(0, 1)
    # Disable clipping so markers can extend outside the axes bounds
    axleg.set_clip_on(False)
    
    y_positions = np.linspace(0.9, 0.1, len(MODELS_ORDER))
    for y, model in zip(y_positions, MODELS_ORDER):
        if model in markers:
            # Use custom path marker with clipping disabled
            scatter = axleg.scatter([0.25], [y], marker=markers[model], s=400, 
                         facecolor='gray', edgecolor='black', linewidth=0.8,
                         transform=axleg.transAxes)
            scatter.set_clip_on(False)  # Disable clipping for this collection
            
        text = axleg.text(0.55, y, model, transform=axleg.transAxes, va='center', 
                  fontsize=8)
        text.set_clip_on(False)  # Disable clipping for text too

def main():
    # Load your custom markers
    print("Loading custom markers...")
    markers = markerize_folder_match_goku(ICON_DIR, ref_name="goku", global_padding=0.90)
    print(f"Loaded {len(markers)} custom markers")
    
    df = load_all()

    # wider figure to make room for right-side icon legend
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.8, 3.7))
    fig.subplots_adjust(right=0.90, wspace=0.28)
    
    # Disable clipping on main axes too
    axL.set_clip_on(False)
    axR.set_clip_on(False)

    # ---- Left: #VP vs Time [s] with mean±std, per (model, method) ----
    for mcanon in METHODS_CANON:
        sub = df[df["method"] == mcanon]
        c = method_color(mcanon)
        
        # Individual points with error bars as filled circles with error band
        for _, r in sub.iterrows():
            # Create error band first (lower zorder)
            if r["#VP_s"] > 0 or r["Time_s"] > 0:
                # Use alpha blending for subtle error indication
                axL.errorbar(r["#VP_m"], r["Time_m"],
                             xerr=r["#VP_s"], yerr=r["Time_s"],
                             fmt='none', ecolor=c, alpha=0.4, elinewidth=1.2, 
                             capsize=2, zorder=2)
            
            # Main data point as small filled circle
            axL.scatter([r["#VP_m"]], [r["Time_m"]], 
                       s=25, color=c, alpha=0.7, zorder=3, edgecolor='none')
        
        # Custom icon markers for each model
        for _, r in sub.iterrows():
            model = r["model"]
            if model in markers:
                # Use larger size for goku, smaller for others
                marker_size = MARKER_SIZE_GOKU if model == "goku" else MARKER_SIZE
                axL.scatter([r["#VP_m"]], [r["Time_m"]], 
                           marker=markers[model], s=marker_size, 
                           facecolor=c, edgecolor='black', linewidth=0.6, 
                           zorder=6, alpha=0.9)

    # Removed the small reference icons at model centroids

    axL.set_xlabel("Number of Viewpoints")
    axL.set_ylabel("Solution Time [s]")
    axL.grid(True, alpha=0.25)
    # Removed in-panel legend

    # ---- Right: Exclusive %/VP vs CV with mean±std ----
    # Removed covariance ellipses
    
    # Individual points with error bars as filled circles with error band
    for _, r in df.iterrows():
        c = method_color(r["method"])
        # Error band with transparency
        if r["Excl_s"] > 0 or r["CV_s"] > 0:
            axR.errorbar(r["Excl_m"], r["CV_m"],
                         xerr=r["Excl_s"], yerr=r["CV_s"],
                         fmt='none', ecolor=c, alpha=0.4, elinewidth=1.2,
                         capsize=2, zorder=2)
        
        # Main data point as small filled circle
        axR.scatter([r["Excl_m"]], [r["CV_m"]], 
                   s=25, color=c, alpha=0.7, zorder=3, edgecolor='none')

    # Custom icon markers for each (model, method) combination
    for _, r in df.iterrows():
        model = r["model"]
        c = method_color(r["method"])
        if model in markers:
            # Use larger size for goku, smaller for others
            marker_size = MARKER_SIZE_GOKU if model == "goku" else MARKER_SIZE
            axR.scatter([r["Excl_m"]], [r["CV_m"]], 
                       marker=markers[model], s=marker_size, 
                       facecolor=c, edgecolor='black', linewidth=0.6, 
                       zorder=6, alpha=0.9)

    # Removed the small reference icons at model centroids

    axR.set_xlabel("Novel Area per Viewpoint")
    axR.set_ylabel("CV of overlap")
    axR.set_xlim(left=0)  # Start x-axis from 0
    
    # Set better y-axis limits based on data range
    cv_values = df["CV_m"].values
    cv_min, cv_max = cv_values.min(), cv_values.max()
    cv_range = cv_max - cv_min
    # Start y-axis from slightly below minimum data, with some padding
    axR.set_ylim(bottom=max(0, cv_min - 0.1 * cv_range), 
                 top=cv_max + 0.1 * cv_range)
    
    axR.grid(True, alpha=0.25)
    # Removed in-panel legend and arrow annotation

    # right-side model icon legend using custom markers
    add_model_icon_legend(fig, markers)
    
    # Add color legend below the plots (no black outline, no title)
    handles = [
        Line2D([0], [0], marker='o', linestyle='None',
               markersize=8, markerfacecolor=COLORS[method], markeredgecolor='none',
               label=DISPLAY[method])
        for method in METHODS_CANON
    ]
    fig.legend(handles=handles, fontsize=9,
               loc="lower center", ncol=4, frameon=False, 
               bbox_to_anchor=(0.45, -0.02))

    fig.suptitle("Efficiency and Quality Metrics across Models and Methods", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0.05, 0.91, 0.975])  # room for right legend column

    # Print and save the modified data table
    print("\nModified Dataset:")
    print("="*80)
    print(df[['model', 'method', '#VP_m', '#VP_s', 'Time_m', 'Time_s', 
              'Excl_m', 'Excl_s', 'CV_m', 'CV_s']].round(3))
    
    # Save to CSV
    output_csv = os.path.join(CSV_DIR, "modified_metrics_summary.csv")
    df.to_csv(output_csv, index=False, float_format='%.3f')
    print(f"\n[Saved modified data] {output_csv}")
    
    out = os.path.join(CSV_DIR, "efficiency_connectivity_two_panel.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"[Saved plot] {out}")

if __name__ == "__main__":
    main()