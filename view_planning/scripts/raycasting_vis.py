import sys, time, numpy as np, os
sys.path.append("../build/python_bindings")
from visioncraft_py import Model, Viewpoint, VisibilityManager, Visualizer




def hex_to_rgb_normalized(hex_color: str) -> np.ndarray:
    """
    Convert hex color (e.g., "#00B7EB" or "00B7EB") to normalized RGB (0–1 range).
    Returns a NumPy array of shape (3,).
    """
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError("Hex color must be 6 digits (e.g., '#FF00FF')")
    
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return np.array(rgb, dtype=np.float64) / 255.0


def canonical_top_right_viewpoint(radius=120.0):
    pos = [1.6*radius, 0, 0]  # +X +Y +Z
    vp = Viewpoint.from_lookat(pos, [0.0, 0.0, 0.0])
    vp.setDownsampleFactor(16.0)
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

# visualizer.setCameraPose(
#     [-303.74937583, -279.25557283,  -15.632392558],
#     [46.63075466, 10.80677806,  1.23434538],
#     [-0.03765598, -0.01261634,  0.99921112]
# )

visualizer.setCameraPose(
    [ 6.8178571,  -434.04052063,   58.1254131],
    [76.0562778,  11.71839452, -2.635765],
    [-0.00198908,  0.1353636,   0.99079399]
)





# --- Model + Visibility Manager ---

# --- Model + Visibility Manager ---
model = Model()
model.loadModel("../models/cat.ply", 200000)
vm = VisibilityManager(model)

# --- Viewpoints ---
vp1 = canonical_top_right_viewpoint(120.0)

vm.trackViewpoint(vp1)
vp1.performRaycastingOnGPU(model)


# Second viewpoint raycast
# vm.trackViewpoint(vp2)
# vp2.performRaycastingOnGPU(model)


# --- Visualization ---
visualizer.addViewpoint(vp1, True, True, False, False)
# visualizer.addViewpoint(vp2, True, True, False, False)

color = hex_to_rgb_normalized("#FED527")   # bright cyan

visualizer.addVoxelMapProperty(
    model,
    "visibility",
    [1.0, 0.0, 0.0],   # baseColor (unseen)
    [0.0, 1.0, 0.0],   # propertyColor (seen)
    -1.0, -1.0           # map 0→2 to red→green→dark green
)

visualizer.showRays(vp1, color)
# visualizer.showRaysParallel(vp1)

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
    visualizer.saveScreenshot("../figures/ray_vis_side.png", magnification=4)
    visualizer.startAsyncRendering()
finally:
    visualizer.stopAsyncRendering()
