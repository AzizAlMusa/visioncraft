import csv
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "../logs/stability_metrics.csv"   # adjust if needed

steps = []
dist_ref_rms = []
delta_pos_rms = []
lambda_max = []
rho = []

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps.append(int(row["step"]))
        dist_ref_rms.append(float(row["dist_ref_rms"]))
        delta_pos_rms.append(float(row["delta_pos_rms"]))
        lambda_max.append(float(row["lambda_max_real"]))
        rho.append(float(row["spectral_radius_I_plus_etaJ"]))

steps = np.array(steps)
dist_ref_rms = np.array(dist_ref_rms)
delta_pos_rms = np.array(delta_pos_rms)
lambda_max = np.array(lambda_max)
rho = np.array(rho)

# Find a perturbation episode automatically:
# look for the first step where dist_ref_rms rises above a tiny threshold
baseline = np.nanmedian(dist_ref_rms[dist_ref_rms > 0]) if np.any(dist_ref_rms > 0) else 0.0
thresh = baseline + 1e-3  # tweak if needed

pert_idx = None
for i in range(len(steps)):
    if dist_ref_rms[i] > thresh:
        pert_idx = i
        break

if pert_idx is None:
    print("No obvious perturbation detected (dist_ref_rms never rose above threshold).")
    exit()

# Look at a window around the perturbation
window = 200
mask = (steps >= steps[pert_idx]) & (steps <= steps[pert_idx] + window)

t_rel = steps[mask] - steps[pert_idx]
d = dist_ref_rms[mask]

# Estimate empirical contraction rate from log(d)
valid = d > 1e-6
if valid.sum() > 5:
    coeffs = np.polyfit(t_rel[valid], np.log(d[valid]), 1)
    lyap_emp = coeffs[0]
    print(f"Empirical decay rate (Lyapunov-style) ≈ {lyap_emp:.4f} per step")
else:
    lyap_emp = None
    print("Not enough data to fit an empirical decay rate.")

plt.figure()
plt.plot(steps, dist_ref_rms, label="dist_ref_rms (to ref eq)")
plt.axvline(steps[pert_idx], linestyle="--", label="perturb start")
plt.xlabel("step")
plt.ylabel("RMS distance to reference config")
plt.legend()
plt.tight_layout()
plt.show()
