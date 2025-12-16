import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
df = pd.read_csv("../logs/exp1_gain_ratio_summary.csv")

gamma = df["gamma"].values
lam = df["lambda_max_real_median"].values

# ---------------------------------------------------------
# Visible window for lambda
# ---------------------------------------------------------
lam_low, lam_high = -3, 3
in_range = (lam >= lam_low) & (lam <= lam_high)
below = lam < lam_low
above = lam > lam_high

# ---------------------------------------------------------
# Stability classification
# ---------------------------------------------------------
stable = lam < 0
unstable = lam >= 0

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
ax = plt.gca()

# -----------------------
# Plot in-range stable / unstable
# -----------------------
ax.scatter(gamma[in_range & stable], lam[in_range & stable],
           s=18, color="green", label="stable")

ax.scatter(gamma[in_range & unstable], lam[in_range & unstable],
           s=18, color="red", label="unstable")

# -----------------------
# Overflow points (clamped visually)
# -----------------------
ax.scatter(gamma[below], np.full(np.sum(below), lam_low),
           marker='v', color="red", s=25)

ax.scatter(gamma[above], np.full(np.sum(above), lam_high),
           marker='^', color="red", s=25)

# -----------------------
# Stability cutoff line: Re(λ) = 0
# -----------------------
ax.axhline(0, color="black", linewidth=1.4, linestyle="--")

# -----------------------
# Axes settings
# -----------------------
ax.set_xscale("log")
ax.set_xlim(1e-3, 1e3)
ax.set_ylim(lam_low, lam_high)

ax.set_xlabel("γ = k_attr / k_rep")
ax.set_ylabel("Re(max λ)")
ax.set_title("Re(max λ) vs γ (stability highlighted)")

ax.grid(True, alpha=0.25)
ax.legend()

plt.tight_layout()
plt.show()
