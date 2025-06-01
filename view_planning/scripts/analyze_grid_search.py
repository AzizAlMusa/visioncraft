import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

results_dir = "./results2/grid_search"

# === Load all .npz files ===
data_list = []

for root, _, files in os.walk(results_dir):
    for file in files:
        if file.endswith(".npz"):
            path = os.path.join(root, file)
            try:
                d = np.load(path)
                k_attr = float(root.split("kattr")[1].split("_")[0])
                k_rep = float(root.split("krep")[1])
                coverage = float(d["coverage"][-1])
                redundancy = float(d["redundancy"][-1])
                num_viewpoints = int(d["num_viewpoints"])
                sim_time = float(d.get("simulation_time_sec", 0.0))
                
                data_list.append({
                    "k_attr": k_attr,
                    "k_rep": k_rep,
                    "coverage": coverage,
                    "redundancy": redundancy,
                    "viewpoints": num_viewpoints,
                    "time": sim_time
                })
            except Exception as e:
                print(f"[Skip] {path}: {e}")
                continue

# === Convert to DataFrame ===
df = pd.DataFrame(data_list)

# === Normalize for scoring ===
df["coverage_norm"] = df["coverage"]
df["redundancy_norm"] = df["redundancy"]
df["viewpoints_norm"] = df["viewpoints"] / df["viewpoints"].max()
df["time_norm"] = df["time"] / df["time"].max()

# === Compute composite score (time replaces iterations) ===
df["score"] = (
    2.0 * df["coverage_norm"]                 # promote coverage
    - 1.0 * df["redundancy_norm"]             # penalize redundancy
    - 3.0 * df["viewpoints_norm"]             # strongly penalize viewpoint count
    - 2.0 * df["time_norm"]                   # penalize longer time
)

# === Sort by score descending ===
df_sorted = df.sort_values("score", ascending=False)

# === Print Top 10 ===
print(f"{'Rank':<5} {'k_attr':<7} {'k_rep':<7} {'Coverage':<10} {'Redundancy':<12} {'#Viewpoints':<13} {'Time(s)':<10} {'Score':<6}")
print("-"*85)
for rank, (_, row) in enumerate(df_sorted.head(50).iterrows(), 1):
    print(f"{rank:<5} {row.k_attr:<7.2f} {row.k_rep:<7.2f} {row.coverage:<10.4f} {row.redundancy:<12.4f} {row.viewpoints:<13} {row.time:<10.2f} {row.score:.3f}")

# === 3D Scatter Plot (Color = Time) ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(
    df["viewpoints"], df["coverage"], df["redundancy"],
    c=df["time"], cmap="viridis", s=60, edgecolor='k'
)
ax.set_xlabel("#Viewpoints")
ax.set_ylabel("Coverage")
ax.set_zlabel("Redundancy")
cbar = fig.colorbar(sc, pad=0.1)
cbar.set_label("Simulation Time (sec)")
ax.set_title("Grid Search Results (Color = Time)")
plt.tight_layout()
plt.show()
