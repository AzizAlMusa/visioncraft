import sys, time, numpy as np, os
sys.path.append("../build/python_bindings")
from visioncraft_py import Model, Viewpoint, VisibilityManager, Visualizer


def canonical_top_right_viewpoint(radius=120.0):
    pos = [radius, radius, radius]  # +X +Y +Z
    vp = Viewpoint.from_lookat(pos, [0.0, 0.0, 0.0])
    vp.setDownsampleFactor(8.0)
    vp.setNearPlane(50.0)
    vp.setFarPlane(200.0)
    return vp


def rotated_viewpoint_around_z(base_viewpoint, degrees=90.0):
    """Rotate a viewpoint around the Z axis by the given angle in degrees."""
    angle = np.deg2rad(degrees)
    Rz = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle),  np.cos(angle), 0.0],
        [0.0,            0.0,           1.0]
    ])
    pos = np.dot(Rz, base_viewpoint.getPosition())  # rotate its position
    vp = Viewpoint.from_lookat(pos, [0.0, 0.0, 0.0])
    vp.setDownsampleFactor(8.0)
    vp.setNearPlane(base_viewpoint.getNearPlane())
    vp.setFarPlane(base_viewpoint.getFarPlane())
    return vp


# --- Setup ---
visualizer = Visualizer()
visualizer.initializeWindow("VisionCraft — 2-View Visibility")
visualizer.setBackgroundColor([1.0, 1.0, 1.0])        # white
visualizer.setViewpointFrustumColor([0.0, 0.5, 0.6])  # teal
# visualizer.setCameraPose(
#     [530.35999917, 186.438878, 158.50448711],
#     [35.11318952, 4.58285662, 0.38645901],
#     [-0.26366948, -0.11470131, 0.95776929]
# )

visualizer.setCameraPose(
    [530.21795036, 167.84260618, 180.33750178],
    [34.97114071, -14.0134152,   22.21947368],
    [-0.26366948, -0.11470131,  0.95776929]
)



# --- Model + Visibility Manager ---
model = Model()
model.loadModel("../models/cat.ply", 200000)
vm = VisibilityManager(model)

# --- Viewpoints ---
vp1 = canonical_top_right_viewpoint(120.0)
vp2 = rotated_viewpoint_around_z(vp1, -90.0)  # 90° around Z axis

# First viewpoint raycast
vm.trackViewpoint(vp1)
vp1.performRaycastingOnGPU(model)

# Second viewpoint raycast
# vm.trackViewpoint(vp2)
# vp2.performRaycastingOnGPU(model)

# --- Coloring ---
# Red → unseen (0)
# Green → seen once (1)
# Dark green → seen twice or more (2+)
visualizer.addVoxelMapProperty(
    model,
    "visibility",
    [1.0, 0.0, 0.0],   # baseColor (unseen)
    [0.0, 1.0, 0.0],   # propertyColor (seen)
    0.0, 1.0           # map 0→2 to red→green→dark green
)

# --- Visualization ---
visualizer.addViewpoint(vp1, True, True, False, False)
# visualizer.addViewpoint(vp2, True, True, False, False)



# --- Render and keep window open (async) ---
visualizer.startAsyncRendering()

try:
    while True:
        time.sleep(1.0)
        pos, focal, up = visualizer.getCameraPose()
        print(f"Camera pos: {pos}, focal: {focal}, up: {up}")
except KeyboardInterrupt:
    print("Saving figure...")
    os.makedirs("../figures", exist_ok=True)
    visualizer.stopAsyncRendering()
    visualizer.saveScreenshot("../figures/cat_visibility_two_views.png", magnification=4)
    visualizer.startAsyncRendering()
finally:
    visualizer.stopAsyncRendering()
