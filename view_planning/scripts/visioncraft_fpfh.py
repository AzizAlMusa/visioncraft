# fpfh_entropy_clusters.py
import sys, os, time
import numpy as np
import open3d as o3d
import matplotlib.cm as cm

# If you want VisionCraft integration:
sys.path.append("../build/python_bindings")
from visioncraft_py import Model, Viewpoint, VisibilityManager, Visualizer


# ---------- utils ----------
def ensure_normals(pcd, radius, max_nn=60):
    if not pcd.has_normals() or len(pcd.normals) != len(pcd.points):
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        pcd.normalize_normals()
    return pcd

def compute_fpfh(pcd, radius, max_nn=100):
    feat = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    # (33, N) -> (N, 33)
    X = np.asarray(feat.data, dtype=np.float64).T
    # sanitize
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

def histogram_entropy(X, eps=1e-12):
    """X is (N,33) raw FPFH bins (non-neg). Returns H in bits, shape (N,)."""
    s = X.sum(axis=1, keepdims=True)
    s[s < eps] = 1.0
    P = X / s
    # clip to avoid log(0)
    P = np.clip(P, eps, 1.0)
    H = -np.sum(P * np.log2(P), axis=1)  # 0..log2(33)
    return H

def otsu_threshold_1d(x, bins=128):
    # histogram on range
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax <= xmin + 1e-12:
        return xmin
    hist, edges = np.histogram(x, bins=bins, range=(xmin, xmax))
    p = hist.astype(np.float64)
    if p.sum() == 0:
        return 0.5 * (xmin + xmax)
    p /= p.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    omega = np.cumsum(p)
    mu_k  = np.cumsum(p * centers)
    mu_t  = mu_k[-1]
    sigma_b2 = (mu_t * omega - mu_k)**2 / (omega * (1.0 - omega) + 1e-16)
    idx = np.nanargmax(sigma_b2)
    return float(centers[idx])

def spatial_mode_filter(points_xyz, labels, knn=16, iters=1):
    """Simple kNN majority vote to de-speckle labels."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    kdt = o3d.geometry.KDTreeFlann(pcd)
    lab = labels.copy()
    for _ in range(iters):
        new = lab.copy()
        for i in range(points_xyz.shape[0]):
            k, idx, _ = kdt.search_knn_vector_3d(points_xyz[i], knn)
            if k > 0:
                votes = np.bincount(lab[idx], minlength=2)
                new[i] = int(np.argmax(votes))
        lab = new
    return lab

def color_and_show(points_xyz, entropy, title="FPFH Entropy Map"):
    """
    Display entropy as a smooth gradient:
      low entropy (feature-rich)  -> #F27202 (orange)
      high entropy (bland/flat)   -> #F5CAE9 (pink)
    """
    # Normalize entropy to [0,1]
    e_min, e_max = float(np.min(entropy)), float(np.max(entropy))
    e_norm = (entropy - e_min) / (e_max - e_min + 1e-12)

    # Convert hex → normalized RGB
    def hex_to_rgb(hex_str):
        h = hex_str.lstrip('#')
        return np.array([int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float64)

    c_low  = hex_to_rgb("#F27202")  # low entropy  -> orange (features)
    c_high = hex_to_rgb("#F5CAE9")  # high entropy -> pink   (bland)

    # Linear interpolation between the two colors
    colors = (1.0 - e_norm[:, None]) * c_low + e_norm[:, None] * c_high

    # Build and visualize point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"[Open3D] Entropy range: min={e_min:.3f}, max={e_max:.3f}")
    try:
        o3d.visualization.draw([pcd], title=title)
    except Exception:
        o3d.visualization.draw_geometries(
            [pcd], window_name=title, width=960, height=720
        )

# ---------- VisionCraft helpers ----------
def hex_to_rgb01(hex_str):
    h = hex_str.lstrip('#')
    return [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]

COL_FEATURE = hex_to_rgb01("#FDBC0F")  # orange  (feature / low H) 
COL_BLAND   = hex_to_rgb01("#188DEA")  # pink    (bland  / high H)

def canonical_top_right_viewpoint(radius=120.0):
    pos = [radius, radius, radius]  # +X +Y +Z
    vp = Viewpoint.from_lookat(pos, [0.0, 0.0, 0.0])
    vp.setDownsampleFactor(8.0)
    vp.setNearPlane(50.0)
    vp.setFarPlane(200.0)
    return vp

def paint_into_visioncraft(model_path, pts, entropy, labels,
                           num_samples=200000,
                           window_title="VisionCraft — FPFH Entropy (meta-voxels)"):
    """
    Project Open3D per-point entropy/labels onto VisionCraft meta-voxels via 1-NN,
    write 'fpfh_entropy' (float) and 'fpfh_cluster' (int: 1=feature, 0=bland),
    then visualize with your custom colors.
    """
    # --- VisionCraft: load model & generate metavoxels ---
    model = Model()
    model.loadModel(model_path, num_samples)
    model.generateVoxelMap()

    # Build KDTree over the source points we computed entropy/labels on
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(pts)
    kdt = o3d.geometry.KDTreeFlann(pcd_src)

    # Iterate metavoxels, assign from nearest source point
    vm = model.getVoxelMap()  # dict: key(tuple) -> MetaVoxel
    for key_tuple, mv in vm.items():
        c = mv.getPosition()
        c_np = np.array([float(c[0]), float(c[1]), float(c[2])], dtype=np.float64)
        k, idx, _ = kdt.search_knn_vector_3d(c_np, 1)
        if k > 0:
            i = idx[0]
            model.setVoxelProperty(key_tuple, "fpfh_entropy", float(entropy[i]))
            model.setVoxelProperty(key_tuple, "fpfh_cluster", int(labels[i]))
        else:
            # If no neighbor found (unlikely), mark bland
            model.setVoxelProperty(key_tuple, "fpfh_entropy", float(0.0))
            model.setVoxelProperty(key_tuple, "fpfh_cluster", int(0))

    # --- (Optional) visibility for context ---
    vmgr = VisibilityManager(model)
    vp1 = canonical_top_right_viewpoint(120.0)
    vmgr.trackViewpoint(vp1)
    vp1.performRaycastingOnGPU(model)

    # --- Visualize with your two hex colors ---
    vis = Visualizer()
    vis.initializeWindow(window_title)
    vis.setBackgroundColor([1.0, 1.0, 1.0])
    vis.setViewpointFrustumColor([0.0, 0.5, 0.6])

    # IMPORTANT: baseColor == value 0, propertyColor == value 1
    vis.addVoxelMapProperty(
        model,
        "fpfh_cluster",
        COL_BLAND,    # 0 → pink
        COL_FEATURE,  # 1 → orange
        0.0, 1.0
    )
    # vis.addViewpoint(vp1, True, True, False, False)
    vis.setCameraPose(
    [221.76315766,  84.6960134,   63.28084895],
    [30.82407358, 14.58264472,  2.31950428],
    [-0.26366948, -0.11470131,  0.95776929]
    )

    vis.startAsyncRendering()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        os.makedirs("../figures", exist_ok=True)
        vis.stopAsyncRendering()
        vis.saveScreenshot("../figures/cat_metavoxel_fpfh_entropy.png", magnification=4)
        vis.startAsyncRendering()
    finally:
        vis.stopAsyncRendering()


# ---------- main pipeline ----------
if __name__ == "__main__":
    # --- Load a point cloud (Open3D-only). Replace with your file path. ---
    model_path = "../models/cat.ply"
    pcd = o3d.io.read_point_cloud(model_path)
    if len(pcd.points) == 0:
        raise RuntimeError("Point cloud is empty. Check the path.")

    # Scale-aware radii
    dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    avg = float(dists.mean()) if dists.size else 1.0
    normal_radius = max(2.0 * avg, 1e-3)
    fpfh_radius   = max(5.0 * avg, 1e-3)

    # Normals + FPFH
    ensure_normals(pcd, normal_radius, max_nn=60)
    X = compute_fpfh(pcd, fpfh_radius, max_nn=100)  # (N,33)

    # Entropy and 1-D split
    H = histogram_entropy(X)                        # (N,)
    thr = otsu_threshold_1d(H, bins=128)
    # 1 = feature-rich (low entropy), 0 = bland (high entropy)
    raw_labels = (H < thr).astype(np.int32)

    # Spatial de-speckle
    pts = np.asarray(pcd.points)
    labels = spatial_mode_filter(pts, raw_labels, knn=16, iters=1)

    # Open3D entropy heatmap (orange↔︎pink gradient)
    color_and_show(pts, H, title="FPFH Entropy — Open3D")

    # ---- VisionCraft projection + viz (same split/colors) ----
    paint_into_visioncraft(model_path, pts, H, labels, num_samples=200000)
