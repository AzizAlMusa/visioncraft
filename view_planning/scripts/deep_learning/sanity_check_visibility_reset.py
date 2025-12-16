# sanity_check_visibility_reset.py
import sys
import numpy as np
sys.path.append("../../build/python_bindings")

from visioncraft_py import Model, Viewpoint, VisibilityManager

model_path = "../../models/gorilla.ply"

m = Model()
ok = m.loadModel(model_path, 250000)
print("loadModel:", ok)

vm = VisibilityManager(m)
print("initial coverage:", vm.getCoverageScore())

# one random viewpoint
center = np.array(m.getCenter())
pos = center + np.array([0, 0, 400.0])
vp = Viewpoint.from_lookat(pos.tolist(), center.tolist())
vp.setNearPlane(300.0)
vp.setFarPlane(900.0)
vp.setDownsampleFactor(2.0)

vm.trackViewpoint(vp)
vp.performRaycastingOnGPU(m)

cov1 = vm.getCoverageScore()
print("coverage after 1 view:", cov1)

# now reset
vm.untrackAllViewpoints()
cov2 = vm.getCoverageScore()
print("coverage after untrackAllViewpoints:", cov2)
