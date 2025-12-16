#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Postprocessing for Experiment A (stability vs number of viewpoints)

Reads CSVs named `stability_A_*.csv` from ../logs and produces:
  - spacing_vs_step.png       : step vs min & mean spacing for each N
  - lambda_vs_step_clipped.png: step vs lambda_max_real (clipped, transient removed)
  - summary_vs_N.png          : final spacing and final lambda_max_real vs N
  - lambda_vs_spacing.png     : final lambda_max_real vs final mean_spacing (one dot per N)

Also prints a numerical summary table to the console.

You can tweak:
  MIN_STEP_FOR_EIG     : ignore eigen data before this step
  LAMBDA_PLOT_CLIP     : clip y-axis for lambda_vs_step plot to +/- this value
"""

import os
import glob
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


LOG_DIR = os.path.join("..", "logs")
OUT_DIR = os.path.join(".", "figures_experiment_A")

MIN_STEP_FOR_EIG = 20     # ignore eigenvalues before this step
LAMBDA_PLOT_CLIP = 100.0  # y-axis clip for lambda_vs_step plot


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


def collect_by_N(pattern="stability_A_*.csv"):
    """Group all experiment-A CSVs by N."""
    pattern_path = os.path.join(LOG_DIR, pattern)
    files = sorted(glob.glob(pattern_path))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern_path}")

    data_by_N = defaultdict(lambda: {"paths": [], "data": []})

    for path in files:
        data = load_csv(path)
        N_vals = data["N"]
        N_unique = np.unique(N_vals[~np.isnan(N_vals)])
        if len(N_unique) != 1:
            print(f"[WARN] File {path} has varying N values: {N_unique}")
        N = int(N_unique[0]) if len(N_unique) > 0 else -1
        data_by_N[N]["paths"].append(path)
        data_by_N[N]["data"].append(data)

    return data_by_N


def last_non_nan_after_step(data, key, min_step=0):
    """Return last non-NaN value of column `key` after step >= min_step."""
    vals = data[key]
    steps = data["step"]
    mask = (~np.isnan(vals)) & (steps >= min_step)
    if not np.any(mask):
        return np.nan
    return float(vals[mask][-1])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data_by_N = collect_by_N("stability_A_*.csv")
    Ns = sorted(data_by_N.keys())

    # ------------------------------------------------------------------
    # Plot 1: min & mean spacing vs step (one curve per N)
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    for N in Ns:
        data = data_by_N[N]["data"][0]
        step = data["step"]
        min_spacing = data["min_spacing"]
        mean_spacing = data.get("mean_spacing", None)

        if mean_spacing is not None:
            plt.plot(step, mean_spacing, linestyle="-", alpha=0.7, label=f"N={N} (mean)")
            plt.plot(step, min_spacing, linestyle="--", alpha=0.5, label=f"N={N} (min)")
        else:
            plt.plot(step, min_spacing, label=f"N={N} (min)")

    plt.xlabel("Step")
    plt.ylabel("Spacing (torus)")
    plt.title("Experiment A: Spacing vs step (min & mean)")
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
    for N in Ns:
        data = data_by_N[N]["data"][0]
        step = data["step"]
        lam = data["lambda_max_real"]
        mask = (~np.isnan(lam)) & (step >= MIN_STEP_FOR_EIG)
        if not np.any(mask):
            continue
        lam_use = lam[mask]
        lam_use = np.clip(lam_use, -LAMBDA_PLOT_CLIP, LAMBDA_PLOT_CLIP)
        plt.plot(step[mask], lam_use, label=f"N={N}")

    plt.axhline(0.0, linestyle=":", linewidth=1.0, color="gray")
    plt.xlabel("Step")
    plt.ylabel("max Re(λ(J)) (clipped)")
    plt.title(f"Experiment A: Max real eigenvalue vs step (step ≥ {MIN_STEP_FOR_EIG})")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "lambda_vs_step_clipped.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Saved] {out_path}")

    # ------------------------------------------------------------------
    # Summary vs N (final values)
    # ------------------------------------------------------------------
    print("\nSummary at final step (per N), ignoring early transients:")
    print(" N | final_min_spacing | final_mean_spacing | final_lambda_max_real | final_spectral_radius | num_eig_pos")
    print("---+--------------------+--------------------+------------------------+------------------------+------------")

    N_list = []
    final_min_spacing = []
    final_mean_spacing = []
    final_lambda_max = []

    for N in Ns:
        data = data_by_N[N]["data"][0]
        min_sp = float(data["min_spacing"][-1])
        mean_sp = float(data["mean_spacing"][-1])
        lam_last = last_non_nan_after_step(data, "lambda_max_real", MIN_STEP_FOR_EIG)
        rho_last = last_non_nan_after_step(data, "spectral_radius_I_plus_etaJ", MIN_STEP_FOR_EIG)
        num_pos = last_non_nan_after_step(data, "num_eig_pos_real", MIN_STEP_FOR_EIG)

        N_list.append(N)
        final_min_spacing.append(min_sp)
        final_mean_spacing.append(mean_sp)
        final_lambda_max.append(lam_last)

        print(
            f"{N:2d} | {min_sp:18.3f} | {mean_sp:18.3f} | "
            f"{lam_last:22.3f} | {rho_last:22.3f} | {int(num_pos) if not np.isnan(num_pos) else -1:10d}"
        )

    N_arr = np.array(N_list)
    final_min_spacing = np.array(final_min_spacing)
    final_mean_spacing = np.array(final_mean_spacing)
    final_lambda_max = np.array(final_lambda_max)

    # ------------------------------------------------------------------
    # Plot 3: final spacing & lambda vs N
    # ------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(N_arr, final_mean_spacing, marker="o", label="final mean spacing")
    ax1.plot(N_arr, final_min_spacing, marker="^", linestyle=":", label="final min spacing")
    ax1.set_xlabel("Number of viewpoints N")
    ax1.set_ylabel("Spacing (torus)")

    ax2 = ax1.twinx()
    ax2.plot(N_arr, final_lambda_max, marker="s", linestyle="--", label="final max Re(λ)")
    ax2.axhline(0.0, linestyle=":", linewidth=1.0, color="gray")
    ax2.set_ylabel("Final max Re(λ(J))")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best", fontsize=8)

    plt.title("Experiment A: Final spacing and stability vs N")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "summary_vs_N.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Saved] {out_path}")

    # ------------------------------------------------------------------
    # Plot 4: λ_max vs mean spacing (one point per N)
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    for N, sp, lam in zip(N_arr, final_mean_spacing, final_lambda_max):
        plt.scatter(sp, lam)
        plt.text(sp, lam, f"N={N}", fontsize=8, ha="left", va="bottom")

    plt.axhline(0.0, linestyle=":", linewidth=1.0, color="gray")
    plt.xlabel("Final mean spacing")
    plt.ylabel("Final max Re(λ(J))")
    plt.title("Experiment A: Stability vs spacing")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "lambda_vs_spacing.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
