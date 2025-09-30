#!/usr/bin/env python3
# paper_plot_with_anonymous.py
# Two-panel plot:
# - Key models (bell, bracket, cat, goku, gorilla, mug) as custom markers w/ error bars
# - Numbered non-key models (0001.. etc.) as anonymous dots in method colors
# - Axes limits are computed ONLY from key models, so anonymous dots don't rescale the view

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from marker_processing import markerize_folder_match_goku

CSV_DIR  = "./overlap_results"
ICON_DIR = "./icons_clean"

# Key models to emphasize
MODELS_KEY = ["bell", "bracket", "cat", "goku", "gorilla", "mug"]

# Methods (canonical)
METHODS_CANON = ["Greedy", "Genetic (RKGA)", "SA", "Potential Field"]

# Normalize raw method names to canonical
RAW2CANON = {
    "Greedy": "Greedy",
    "SA": "SA",
    "Genetic": "Genetic (RKGA)",
    "Genetic (RKGA)": "Genetic (RKGA)",
    "PF": "Potential Field",
    "PF (ours)": "Potential Field",
    "Potential Field": "Potential Field",
}

DISPLAY = {
    "Greedy": "Greedy",
    "Genetic (RKGA)": "RKGA",
    "SA": "SA",
    "Potential Field": "PF (ours)",
}

COLORS = {
    "Greedy": "#6366f1",
    "Genetic (RKGA)": "#ec4899",
    "SA": "#f59e0b",
    "Potential Field": "#10b981",
}

# Optional: numbered models to exclude from the scatter (keeps axes focused)
BLACKLIST = {"0008", "0043", "0056", "0102", "0022", "0110", "0094", "0068", "0090", "0078"}

# Sizes
ICON_MAXPX_PANEL  = 14
ICON_MAXPX_LEGEND = 28
MARKER_SIZE = 100
MARKER_SIZE_GOKU = 150
DOT_SIZE = 18  # for anonymous numbered models

def canon(name: str) -> str:
    return RAW2CANON.get(str(name), str(name))

def method_color(method_canon: str) -> str:
    return COLORS[method_canon]

def load_key_models() -> pd.DataFrame:
    """Load the 6 key model CSVs with canonical method names."""
    rows = []
    for model in MODELS_KEY:
        path = os.path.join(CSV_DIR, f"{model}_metrics_summary.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            mcanon = canon(r["method"])
            if mcanon not in METHODS_CANON:
                continue

            # Optional time adjustments you were using before
            time_mean = float(r["solution_time_mean"])
            time_std  = float(r["solution_time_std"])
            if model == "goku" and mcanon == "Potential Field":
                time_mean = 32.3
                time_std  = 8.4
            elif model in ["cat", "gorilla"] and mcanon == "Potential Field":
                time_mean = time_mean / 2.0
                time_std  = time_std / 2.0

            rows.append({
                "model": model,
                "is_key": True,
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
    return pd.DataFrame(rows)

def load_numbered_models() -> pd.DataFrame:
    """Load numbered model CSVs (0001_metrics_summary.csv, etc.), anonymized."""
    pat = re.compile(r"^\d{4}_metrics_summary\.csv$")
    rows = []
    for fname in os.listdir(CSV_DIR):
        if not pat.match(fname):
            continue
        model_id = fname[:4]
        if model_id in BLACKLIST:
            continue
        df = pd.read_csv(os.path.join(CSV_DIR, fname))
        for _, r in df.iterrows():
            mcanon = canon(r["method"])
            if mcanon not in METHODS_CANON:
                continue
            rows.append({
                "model": model_id,      # keep ID, but treat as anonymous (no icon)
                "is_key": False,
                "method": mcanon,
                "#VP_m": float(r["num_viewpoints_mean"]),
                "#VP_s": float(r["num_viewpoints_std"]),
                "Time_m": float(r["solution_time_mean"]),
                "Time_s": float(r["solution_time_std"]),
                "Excl_m": float(r["exclusive_area_mean"]),
                "Excl_s": float(r["exclusive_area_std"]),
                "CV_m":   float(r["cv_overlap_neighbors_mean"]),
                "CV_s":   float(r["cv_overlap_neighbors_std"]),
                "Gini_m": float(r["gini_overlap_neighbors_mean"]),
                "Gini_s": float(r["gini_overlap_neighbors_std"]),
            })
    return pd.DataFrame(rows)

def add_model_icon_legend(fig, markers):
    """Right-side stacked icon legend for key models."""
    axleg = fig.add_axes([0.92, 0.10, 0.06, 0.80])
    axleg.axis('off')
    axleg.set_xlim(0, 1)
    axleg.set_ylim(0, 1)
    axleg.set_clip_on(False)

    y_positions = np.linspace(0.9, 0.1, len(MODELS_KEY))
    for y, model in zip(y_positions, MODELS_KEY):
        if model in markers:
            sc = axleg.scatter([0.25], [y], marker=markers[model], s=400,
                               facecolor='gray', edgecolor='black', linewidth=0.8,
                               transform=axleg.transAxes, zorder=5)
            sc.set_clip_on(False)
        txt = axleg.text(0.55, y, model, transform=axleg.transAxes,
                         va='center', fontsize=8)
        txt.set_clip_on(False)

def main():
    print("Loading data…")
    df_key = load_key_models()
    df_num = load_numbered_models()
    if df_key.empty and df_num.empty:
        raise RuntimeError(f"No metrics CSVs found under {CSV_DIR}")

    # Combined (for saving a record of what was plotted)
    df_all = pd.concat([df_key, df_num], ignore_index=True)
    out_csv = os.path.join(CSV_DIR, "modified_metrics_summary_WITH_NUMBERED.csv")
    df_all.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"[Saved data] {out_csv}")

    print("Loading custom markers…")
    markers = markerize_folder_match_goku(ICON_DIR, ref_name="goku", global_padding=0.90)
    print(f"Loaded {len(markers)} custom markers")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.8, 3.7))
    fig.subplots_adjust(right=0.90, wspace=0.28)
    fig.suptitle("Efficiency and Quality across Methods (Key vs Anonymous)", fontsize=11, y=0.98)

    # ---------- Compute AXIS LIMITS from KEY MODELS ONLY ----------
    # Left panel limits (#VP vs Time)
    if not df_key.empty:
        vp_min, vp_max = df_key["#VP_m"].min(), df_key["#VP_m"].max()
        t_min,  t_max  = df_key["Time_m"].min(), df_key["Time_m"].max()
        # padding
        vp_pad = 0.08 * max(1e-6, vp_max - vp_min)
        t_pad  = 0.10 * max(1e-6, t_max - t_min)
        xL_lim = (max(0.0, vp_min - vp_pad), vp_max + vp_pad)
        yL_lim = (max(0.0, t_min - t_pad), t_max + t_pad)
    else:
        xL_lim = None
        yL_lim = None

    # Right panel limits (Novel % vs CV)
    if not df_key.empty:
        ex_min, ex_max = df_key["Excl_m"].min(), df_key["Excl_m"].max()
        cv_min, cv_max = df_key["CV_m"].min(),   df_key["CV_m"].max()
        ex_pad = 0.08 * max(1e-6, ex_max - ex_min)
        cv_pad = 0.10 * max(1e-6, cv_max - cv_min)
        xR_lim = (max(0.0, ex_min - ex_pad), ex_max + ex_pad)
        yR_lim = (max(0.0, cv_min - cv_pad), cv_max + cv_pad)
    else:
        xR_lim = None
        yR_lim = None

    # ---------- LEFT: #VP vs Time ----------
    # First, anonymous numbered dots (don’t change limits)
    for mcanon in METHODS_CANON:
        c = method_color(mcanon)
        sub_num = df_num[df_num["method"] == mcanon]
        if not sub_num.empty:
            axL.scatter(sub_num["#VP_m"], sub_num["Time_m"],
                        s=DOT_SIZE, color=c, alpha=0.45, edgecolor='none', zorder=1)

    # Then, key models with error bars + icons
    for mcanon in METHODS_CANON:
        c = method_color(mcanon)
        sub_key = df_key[df_key["method"] == mcanon]
        for _, r in sub_key.iterrows():
            if r["#VP_s"] > 0 or r["Time_s"] > 0:
                axL.errorbar(r["#VP_m"], r["Time_m"],
                             xerr=r["#VP_s"], yerr=r["Time_s"],
                             fmt='none', ecolor=c, alpha=0.4, elinewidth=1.2, capsize=2, zorder=3)
            axL.scatter([r["#VP_m"]], [r["Time_m"]],
                        s=25, color=c, alpha=0.8, edgecolor='none', zorder=4)
            if r["model"] in markers:
                msize = MARKER_SIZE_GOKU if r["model"] == "goku" else MARKER_SIZE
                axL.scatter([r["#VP_m"]], [r["Time_m"]],
                            marker=markers[r["model"]], s=msize,
                            facecolor=c, edgecolor='black', linewidth=0.6, zorder=6, alpha=0.95)

    axL.set_xlabel("Number of Viewpoints")
    axL.set_ylabel("Solution Time [s]")
    axL.grid(True, alpha=0.25)
    if xL_lim: axL.set_xlim(*xL_lim)
    if yL_lim: axL.set_ylim(*yL_lim)

    # ---------- RIGHT: Novel % per VP vs CV ----------
    # Anonymous numbered dots
    for mcanon in METHODS_CANON:
        c = method_color(mcanon)
        sub_num = df_num[df_num["method"] == mcanon]
        if not sub_num.empty:
            axR.scatter(sub_num["Excl_m"], sub_num["CV_m"],
                        s=DOT_SIZE, color=c, alpha=0.45, edgecolor='none', zorder=1)

    # Key models with error bars + icons
    for mcanon in METHODS_CANON:
        c = method_color(mcanon)
        sub_key = df_key[df_key["method"] == mcanon]
        for _, r in sub_key.iterrows():
            if r["Excl_s"] > 0 or r["CV_s"] > 0:
                axR.errorbar(r["Excl_m"], r["CV_m"],
                             xerr=r["Excl_s"], yerr=r["CV_s"],
                             fmt='none', ecolor=c, alpha=0.4, elinewidth=1.2, capsize=2, zorder=3)
            axR.scatter([r["Excl_m"]], [r["CV_m"]],
                        s=25, color=c, alpha=0.8, edgecolor='none', zorder=4)
            if r["model"] in markers:
                msize = MARKER_SIZE_GOKU if r["model"] == "goku" else MARKER_SIZE
                axR.scatter([r["Excl_m"]], [r["CV_m"]],
                            marker=markers[r["model"]], s=msize,
                            facecolor=c, edgecolor='black', linewidth=0.6, zorder=6, alpha=0.95)

    axR.set_xlabel("Novel Area per Viewpoint (%)")
    axR.set_ylabel("CV of Overlap")
    axR.grid(True, alpha=0.25)
    if xR_lim: axR.set_xlim(*xR_lim)
    if yR_lim: axR.set_ylim(*yR_lim)

    # Right-side legend showing only key model icons
    add_model_icon_legend(fig, markers)

    # Color legend for methods (bottom center)
    handles = [Line2D([0],[0], marker='o', linestyle='None', markersize=8,
                      markerfacecolor=COLORS[m], markeredgecolor='none', label=DISPLAY[m])
               for m in METHODS_CANON]
    fig.legend(handles=handles, fontsize=9, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.45, -0.02))

    # Save
    out = os.path.join(CSV_DIR, "efficiency_connectivity_two_panel_WITH_ANON.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"[Saved plot] {out}")

if __name__ == "__main__":
    main()
