import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
df = pd.read_csv("../logs/exp1_gain_ratio_summary.csv")

gamma = df["gamma"].values
rho = df["spectral_radius_median"].values

# ---------------------------------------------------------
# Visible window for spectral radius
# ---------------------------------------------------------
rho_low, rho_high = 0.8, 1.2
in_range = (rho >= rho_low) & (rho <= rho_high)
below = rho < rho_low
above = rho > rho_high

# ---------------------------------------------------------
# Stability classification
# ---------------------------------------------------------
stable = rho < 1
unstable = rho >= 1

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
ax = plt.gca()

# -----------------------
# In-range stable / unstable
# -----------------------
ax.scatter(gamma[in_range & stable], rho[in_range & stable],
           s=18, color="green", label="stable")

ax.scatter(gamma[in_range & unstable], rho[in_range & unstable],
           s=18, color="red", label="unstable")

# -----------------------
# Overflow points (clamped visually)
# -----------------------
ax.scatter(gamma[below], np.full(np.sum(below), rho_low),
           marker='v', color="red", s=25)

ax.scatter(gamma[above], np.full(np.sum(above), rho_high),
           marker='^', color="red", s=25)

# -----------------------
# Stability cutoff line: rho = 1
# -----------------------
ax.axhline(1, color="black", linewidth=1.4, linestyle="--")

# -----------------------
# Axes settings
# -----------------------
ax.set_xscale("log")
ax.set_xlim(1e-3, 1e3)
ax.set_ylim(rho_low, rho_high)

ax.set_xlabel("γ = k_attr / k_rep")
ax.set_ylabel("Spectral radius")
ax.set_title("Spectral radius vs γ (stability highlighted)")

ax.grid(True, alpha=0.25)
ax.legend()

plt.tight_layout()
plt.show()
