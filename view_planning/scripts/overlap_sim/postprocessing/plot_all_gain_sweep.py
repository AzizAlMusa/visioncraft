import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# ---------------------------------------------------------
# Load all files: ./postprocessing script, ./logs data
# ---------------------------------------------------------
files = sorted(glob.glob("../logs/exp1_gain_ratio_sweep_N*.csv"))

def extract_N(fname):
    base = os.path.basename(fname)
    return int(base.split("_N")[1].split(".")[0])

Ns = [extract_N(f) for f in files]

# ---------------------------------------------------------
# Color map for N curves
# ---------------------------------------------------------
colors = plt.cm.viridis(np.linspace(0, 1, len(Ns)))

# ---------------------------------------------------------
# 1. Re(max λ) line plot
# ---------------------------------------------------------
plt.figure(figsize=(11, 6))
ax = plt.gca()

for fname, N, c in zip(files, Ns, colors):
    df = pd.read_csv(fname)

    gamma = df["gamma"].values
    lam = df["lambda_max_real_median"].values

    ax.plot(gamma, lam, color=c, linewidth=2, label=f"N={N}")

ax.axhline(0, color="black", linestyle="--", linewidth=1.2)

ax.set_xscale("log")
ax.set_xlim(1e-3, 1e3)
ax.set_ylim(-5, 5)

ax.set_title("Re(max λ) vs γ")
ax.set_xlabel("γ = k_attr / k_rep")
ax.set_ylabel("Re(max λ)")

ax.grid(True, alpha=0.25)
ax.legend(title="Viewpoints")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 2. Spectral radius line plot
# ---------------------------------------------------------
plt.figure(figsize=(11, 6))
ax = plt.gca()

for fname, N, c in zip(files, Ns, colors):
    df = pd.read_csv(fname)

    gamma = df["gamma"].values
    rho = df["spectral_radius_median"].values

    ax.plot(gamma, rho, color=c, linewidth=2, label=f"N={N}")

ax.axhline(1, color="black", linestyle="--", linewidth=1.2)

ax.set_xscale("log")
ax.set_xlim(1e-3, 1e3)
ax.set_ylim(0, 2)

ax.set_title("Spectral Radius vs γ")
ax.set_xlabel("γ = k_attr / k_rep")
ax.set_ylabel("Spectral radius")

ax.grid(True, alpha=0.25)
ax.legend(title="Viewpoints")

plt.tight_layout()
plt.show()
