# debug_point_cloud.py
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../../build/python_bindings")
from visioncraft_py import Model

model_path = "../../models/gorilla.ply"

m = Model()
ok = m.loadModel(model_path, 250000)
print("loadModel:", ok)

pc = m.get_point_cloud_array()  # Nx3, world coords
pc = np.asarray(pc)
print("raw pc shape:", pc.shape)

center = np.array(m.getCenter())
pc_centered = pc - center

r = np.linalg.norm(pc_centered, axis=1)
print("radius min/max/mean:", r.min(), r.max(), r.mean())

# Quick 2D projection for sanity
idx = np.random.choice(pc_centered.shape[0], size=min(20000, pc_centered.shape[0]), replace=False)
sub = pc_centered[idx]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].scatter(sub[:,0], sub[:,1], s=1)
axes[0].set_title("XY projection")
axes[1].scatter(sub[:,0], sub[:,2], s=1)
axes[1].set_title("XZ projection")
axes[2].scatter(sub[:,1], sub[:,2], s=1)
axes[2].set_title("YZ projection")

for ax in axes:
    ax.set_aspect("equal")

plt.tight_layout()
plt.show()
