import numpy as np
import matplotlib.pyplot as plt

# Methods
methods = ['Gaussian', 'Inverse', 'Linear', 'Log', 'Quadratic']

# Metrics to compare
metrics = ['Coverage AUC', 'Redundancy AUC', 'Affinity AUC', 'Final Coverage', 'Final Redundancy', 'Final Affinity']

# Raw data (order matches metrics)
data = np.array([
    [88.8596, 9.1998, 114.6963, 0.9546, 0.0498, 1.0528],   # Gaussian
    [84.8054, 12.6414, 120.9353, 0.9545, 0.0520, 1.0552],  # Inverse
    [52.7039, 30.4853, 189.1265, 0.5054, 0.3258, 1.9902],  # Linear
    [71.8280, 24.0649, 140.5554, 0.7375, 0.2617, 1.3712],  # Log
    [51.4269, 31.3644, 193.1895, 0.5037, 0.3276, 1.9923],  # Quadratic
])

# Rough max values for normalization
max_vals = np.array([100, 40, 200, 1, 0.35, 2])  
min_vals = np.array([0, 0, 0, 0, 0, 0])

# Normalize between 0 and 1
norm_data = (data - min_vals) / (max_vals - min_vals)

# Radar plot setup
num_metrics = len(metrics)
angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for i, method in enumerate(methods):
    values = norm_data[i].tolist()
    values += values[:1]  # close the plot
    ax.plot(angles, values, label=method)
    ax.fill(angles, values, alpha=0.1)

ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
ax.set_ylim(0, 1)
ax.set_title("Comparison of View Planning Methods (Normalized)", size=15, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.show()
