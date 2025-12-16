#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment C (repulsion sweep): stability vs steps for each k_rep

Reads CSVs named `stability_C_*.csv` from ../logs and produces:

  - spacing_vs_step.png
      step → min & mean spacing for each k_rep

  - lambda_vs_step_clipped.png
      step → max Re(λ(J)) for each k_rep (clipped, transients removed)

  - lambda_heatmap.png
      2D heatmap: x = step, y = log10(k_rep), color = max Re(λ(J))

  - summary_vs_krep.png
      final spacing & final max Re(λ) vs k_rep (log x-axis)

  - lambda_vs_spacing.png
      final max Re(λ) vs final mean spacing (one point per k_rep)

Console summary prints one line per k_rep.

You can tweak:
  MIN_STEP_FOR_EIG     : ignore eigen data before this step
  LAMBDA_PLOT_CLIP     : clip y-axis for line plots
"""

import os
import glob
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


LOG_DIR = os.path.join("..", "logs")
OUT_DIR = os.path.join(".", "figures_experiment_C_ratio")

MIN_STEP_FOR_EIG = 20      # ignore eigenvalues before this step
LAMBDA_PLOT_CLIP = 100.0   # y-range clip for line plots and heatmap


# ---------------------------------------------------------------------
# CSV loading & grouping
# ---------------------------------------------------------------------
def load_csv(path):
    """Load a stability CSV into a dict of numpy arrays."""
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name, val in row.items():
                if name in ("continuous_stable", "discrete_stable"):
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


def collect_by_krep(pattern="stability_C_*.csv"):
    """
    Group experiment-C CSVs by k_rep (using the k_rep column in the CSV).

    Returns
    -------
    data_by_k : dict[float, dict]
        Mapping k_rep -> { 'paths': [...], 'data': [data_dict, ...] }.
        Typically each k_rep has exactly one CSV.
    """
    pattern_path = os.path.join(LOG_DIR, pattern)
    files = sorted(glob.glob(pattern_path))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern_path}")

    data_by_k = defaultdict(lambda: {"paths": [], "data": []})

    for path in files:
        data = load_csv(path)
        k_vals = data["k_rep"]
        k_unique = np.unique(k_vals[~np.isnan(k_vals)])
        if len(k_unique) != 1:
            print(f"[WARN] File {path} has varying k_rep values: {k_unique}")
        k = float(k_unique[0]) if len(k_unique) > 0 else np.nan
        data_by_k[k]["paths"].append(path)
        data_by_k[k]["data"].append(data)

    return data_by_k


def last_non_nan_after_step(data, key, min_step=0):
    """Return last non-NaN value of column `key` after step >= min_step."""
    vals = data[key]
    steps = data["step"]
    mask = (~np.isnan(vals)) & (steps >= min_step)
    if not np.any(mask):
        return np.nan
    return float(vals[mask][-1])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data_by_k = collect_by_krep("stability_C_*.csv")
    k_vals_sorted = sorted(data_by_k.keys())

    # ------------------------------------------------------------------
    # Plot 1: min & mean spacing vs step (one curve per k_rep)
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    for k in k_vals_sorted:
        data = data_by_k[k]["data"][0]
        step = data["step"]
        min_spacing = data["min_spacing"]
        mean_spacing = data.get("mean_spacing", None)

        label_base = f"k_rep={k:g}"
        if mean_spacing is not None:
            plt.plot(step, mean_spacing, linestyle="-", alpha=0.7,
                     label=f"{label_base} (mean)")
            plt.plot(step, min_spacing, linestyle="--", alpha=0.5,
                     label=f"{label_base} (min)")
        else:
            plt.plot(step, min_spacing, label=f"{label_base} (min)")

    plt.xlabel("Step")
    plt.ylabel("Spacing (torus)")
    plt.title("Experiment C: Spacing vs step (min & mean)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "spacing_vs_step.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Saved] {out_path}")

    # ------------------------------------------------------------------
    # Plot 2: lambda_max_real vs step (clipped, transient removed)
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    for k in k_vals_sorted:
        data = data_by_k[k]["data"][0]
        step = data["step"]
        lam = data["lambda_max_real"]
        mask = (~np.isnan(lam)) & (step >= MIN_STEP_FOR_EIG)
        if not np.any(mask):
            continue
        lam_use = np.clip(lam[mask], -LAMBDA_PLOT_CLIP, LAMBDA_PLOT_CLIP)
        plt.plot(step[mask], lam_use, label=f"k_rep={k:g}")

    plt.axhline(0.0, linestyle=":", linewidth=1.0, color="gray")
    plt.xlabel("Step")
    plt.ylabel("max Re(λ(J)) (clipped)")
    plt.title(f"Experiment C: Max real eigenvalue vs step (step ≥ {MIN_STEP_FOR_EIG})")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "lambda_vs_step_clipped.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Saved] {out_path}")

    # ------------------------------------------------------------------
    # Plot 3: lambda heatmap (k_rep vs step)
    # ------------------------------------------------------------------
    # Find common step range (intersection) so we can build a grid
    step_sets = []
    for k in k_vals_sorted:
        data = data_by_k[k]["data"][0]
        steps = data["step"]
        step_sets.append(set(steps.astype(int)))

    common_steps = sorted(set.intersection(*step_sets))
    common_steps = [s for s in common_steps if s >= MIN_STEP_FOR_EIG]
    common_steps = np.array(common_steps, dtype=int)
    if common_steps.size == 0:
        print("[WARN] No common steps for heatmap after MIN_STEP_FOR_EIG")
    else:
        n_k = len(k_vals_sorted)
        n_t = len(common_steps)
        lambda_grid = np.full((n_k, n_t), np.nan, dtype=float)

        for i, k in enumerate(k_vals_sorted):
            data = data_by_k[k]["data"][0]
            step = data["step"].astype(int)
            lam = data["lambda_max_real"]
            for j, s in enumerate(common_steps):
                idx = np.where(step == s)[0]
                if idx.size == 0:
                    continue
                val = lam[idx[-1]]
                if np.isnan(val):
                    continue
                lambda_grid[i, j] = val

        lambda_grid = np.clip(lambda_grid, -LAMBDA_PLOT_CLIP, LAMBDA_PLOT_CLIP)

        # y-axis will be log10(k_rep)
        k_arr = np.array(k_vals_sorted, dtype=float)
        y_vals = np.log10(k_arr)
        # Build extent for imshow
        x_min, x_max = common_steps[0], common_steps[-1]
        y_min, y_max = y_vals.min(), y_vals.max()

        plt.figure(figsize=(9, 4))
        im = plt.imshow(
            lambda_grid,
            aspect="auto",
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            cmap="coolwarm",
        )
        plt.colorbar(im, label="max Re(λ(J)) (clipped)")
        plt.axhline(0, color="none")  # just to keep imshow happy in some backends

        # nice y ticks: actual k_rep values
        plt.yticks(y_vals, [f"{k:g}" for k in k_arr])
        plt.xlabel("Step")
        plt.ylabel("k_rep (log10 scale on axis)")
        plt.title(f"Experiment C: Stability heatmap (max Re(λ(J)), step ≥ {MIN_STEP_FOR_EIG})")
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, "lambda_heatmap.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[Saved] {out_path}")

    # ------------------------------------------------------------------
    # Summary vs k_rep (final values)
    # ------------------------------------------------------------------
    print("\nSummary at final step (per k_rep), ignoring early transients:")
    print(" k_rep | final_min_spacing | final_mean_spacing | final_lambda_max_real | final_spectral_radius | num_eig_pos")
    print("------+--------------------+--------------------+------------------------+------------------------+------------")

    k_list = []
    final_min_spacing = []
    final_mean_spacing = []
    final_lambda_max = []

    for k in k_vals_sorted:
        data = data_by_k[k]["data"][0]
        min_sp = float(data["min_spacing"][-1])
        mean_sp = float(data["mean_spacing"][-1])
        lam_last = last_non_nan_after_step(data, "lambda_max_real", MIN_STEP_FOR_EIG)
        rho_last = last_non_nan_after_step(data, "spectral_radius_I_plus_etaJ", MIN_STEP_FOR_EIG)
        num_pos = last_non_nan_after_step(data, "num_eig_pos_real", MIN_STEP_FOR_EIG)

        k_list.append(k)
        final_min_spacing.append(min_sp)
        final_mean_spacing.append(mean_sp)
        final_lambda_max.append(lam_last)

        print(
            f"{k:5.1f} | {min_sp:18.3f} | {mean_sp:18.3f} | "
            f"{lam_last:22.3f} | {rho_last:22.3f} | {int(num_pos) if not np.isnan(num_pos) else -1:10d}"
        )

    k_arr = np.array(k_list)
    final_min_spacing = np.array(final_min_spacing)
    final_mean_spacing = np.array(final_mean_spacing)
    final_lambda_max = np.array(final_lambda_max)

    # ------------------------------------------------------------------
    # Plot 4: final spacing & lambda vs k_rep
    # ------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(k_arr, final_mean_spacing, marker="o", label="final mean spacing")
    ax1.plot(k_arr, final_min_spacing, marker="^", linestyle=":", label="final min spacing")
    ax1.set_xscale("log")
    ax1.set_xlabel("k_rep (log scale)")
    ax1.set_ylabel("Spacing (torus)")

    ax2 = ax1.twinx()
    ax2.plot(k_arr, final_lambda_max, marker="s", linestyle="--", label="final max Re(λ)")
    ax2.axhline(0.0, linestyle=":", linewidth=1.0, color="gray")
    ax2.set_ylabel("Final max Re(λ(J))")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best", fontsize=8)

    plt.title("Experiment C: Final spacing and stability vs k_rep")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "summary_vs_krep.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Saved] {out_path}")

    # ------------------------------------------------------------------
    # Plot 5: λ_max vs mean spacing (one point per k_rep)
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    for k, sp, lam in zip(k_arr, final_mean_spacing, final_lambda_max):
        plt.scatter(sp, lam)
        plt.text(sp, lam, f"k={k:g}", fontsize=8, ha="left", va="bottom")

    plt.axhline(0.0, linestyle=":", linewidth=1.0, color="gray")
    plt.xlabel("Final mean spacing")
    plt.ylabel("Final max Re(λ(J))")
    plt.title("Experiment C: Stability vs spacing")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "lambda_vs_spacing.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
