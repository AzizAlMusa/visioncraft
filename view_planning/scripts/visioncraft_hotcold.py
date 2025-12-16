# seed_hotfield_visioncraft.py
import numpy as np
import open3d as o3d
import matplotlib.cm as cm
import heapq

import sys, os, time
sys.path.append("../build/python_bindings")
from visioncraft_py import Model, Viewpoint, VisibilityManager, Visualizer

# =========================
# Config
# =========================
POINT_CLOUD_PATH = "../models/cat.ply"
SEED_MODE        = "auto"                 # "manual" or "auto"
MANUAL_SEEDS_IDX = [100, 2000, 5000]      # if SEED_MODE=="manual", give exactly 3 indices
KNN_GRAPH_K      = 16
GAUSS_SCALE_C    = 40.0                   # sigma = C * mean_nn_spacing
FALLOFF          = "gaussian"             # "gaussian" or "softinv"
SOFTINV_ALPHA    = 2.0
COMBINE          = "max"                  # "max" | "sum" | "softmax"
SOFTMAX_BETA     = 8.0
RANDOM_SUBSAMPLE = None                   # e.g., 120000; None = all

# =========================
# Utilities (Open3D)
# =========================
def mean_nn_spacing(pts, k=8):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    kdt = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    for i in range(len(pts)):
        k_, idx, _ = kdt.search_knn_vector_3d(pts[i], k+1)  # include self
        if k_ > 1:
            nn = pts[idx[1:]] - pts[i]
            dists.append(np.linalg.norm(nn, axis=1).min())
    return float(np.mean(dists)) if dists else 1.0

def pca_curvature(pts, k=KNN_GRAPH_K):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    kdt = o3d.geometry.KDTreeFlann(pcd)
    curv = np.zeros(len(pts), dtype=np.float64)
    for i, p in enumerate(pts):
        k_, idx, _ = kdt.search_knn_vector_3d(p, k)
        if k_ < 3:
            curv[i] = 0.0
            continue
        nbrs = pts[idx]
        C = np.cov((nbrs - nbrs.mean(axis=0)).T)
        evals = np.linalg.eigvalsh(C)  # ascending
        s = evals.sum()
        curv[i] = float(evals[0] / s) if s > 1e-12 else 0.0
    return curv

def build_knn_graph(pts, k=KNN_GRAPH_K):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    kdt = o3d.geometry.KDTreeFlann(pcd)
    n = len(pts)
    nbrs = [[] for _ in range(n)]
    for i in range(n):
        k_, idx, _ = kdt.search_knn_vector_3d(pts[i], k+1)
        if k_ > 1:
            for j in idx[1:]:
                w = np.linalg.norm(pts[j] - pts[i])
                nbrs[i].append((j, w))
                nbrs[j].append((i, w))
    return nbrs

def multi_source_dijkstra(nbrs, seed_idx):
    n = len(nbrs)
    dist = np.full(n, np.inf, dtype=np.float64)
    pq = []
    for s in seed_idx:
        dist[s] = 0.0
        heapq.heappush(pq, (0.0, s))
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in nbrs[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist

def distance_to_heat(d, sigma, falloff="gaussian", alpha=2.0):
    d = np.asarray(d, dtype=np.float64)
    if falloff == "gaussian":
        H = np.exp(-(d * d) / (2.0 * sigma * sigma + 1e-12))
    else:
        H = 1.0 / (1.0 + (d / (sigma + 1e-12)) ** alpha)
    return np.clip(H, 0.0, 1.0)

def combine_fields(fields, mode="max", beta=8.0):
    F = np.stack(fields, axis=1)  # (N,S)
    if mode == "max":
        H = F.max(axis=1)
    elif mode == "sum":
        H = F.sum(axis=1)
        H = (H - H.min()) / (H.max() - H.min() + 1e-12)
    else:  # softmax
        G = np.exp(beta * F)
        H = G.sum(axis=1)
        H = H / (H.max() + 1e-12)
    return np.clip(H, 0.0, 1.0)

def pick_seeds(pts, mode="auto", manual_idx=None):
    if mode == "manual":
        if manual_idx is None or len(manual_idx) != 3:
            raise ValueError("Provide exactly three indices in MANUAL_SEEDS_IDX.")
        return manual_idx
    curv = pca_curvature(pts, k=KNN_GRAPH_K)
    s0 = int(np.argmax(curv))
    d0 = np.linalg.norm(pts - pts[s0], axis=1)
    s1 = int(np.argmax(d0))
    d1 = np.minimum(d0, np.linalg.norm(pts - pts[s1], axis=1))
    s2 = int(np.argmax(d1))
    return [s0, s1, s2]

def colorize_open3d(pts, scalars01, cmap_name="gnuplot2", title="Seed hotfield — gnuplot2"):
    cmap = cm.get_cmap(cmap_name)
    colors = cmap(np.clip(scalars01, 0.0, 1.0))[:, :3]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    try:
        o3d.visualization.draw([pcd], title=title)
    except Exception:
        o3d.visualization.draw_geometries([pcd], window_name=title, width=1080, height=820)

# =========================
# Visioncraft projection
# =========================
def project_hotfield_to_visioncraft(model_path, pts, H, num_samples=200000, window_title="Visioncraft — seed_hotfield"):
    """
    1-NN project point-level H into metavoxels:
      - seed_hotfield : float in [0,1]
      - seed_hot_bin  : int in {0,1,2} (tertiles) for current discrete shader
    """
    # Load model & generate meta-voxels
    model = Model()
    model.loadModel(model_path, num_samples)
    model.generateVoxelMap()

    # KD over source points
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(pts)
    kdt = o3d.geometry.KDTreeFlann(pcd_src)

    # Tertile edges (for discrete Visioncraft coloring)
    t1, t2 = np.quantile(H, [1/3, 2/3]) if len(H) > 3 else (0.33, 0.66)

    vm = model.getVoxelMap()  # dict-like
    for key_tuple, mv in vm.items():
        c = mv.getPosition()
        c_np = np.array([float(c[0]), float(c[1]), float(c[2])], dtype=np.float64)
        k, idx, _ = kdt.search_knn_vector_3d(c_np, 1)
        if k == 0:
            model.setVoxelProperty(key_tuple, "seed_hotfield", 0.0)
            model.setVoxelProperty(key_tuple, "seed_hot_bin", 0)
            continue
        i = idx[0]
        h = float(H[i])
        model.setVoxelProperty(key_tuple, "seed_hotfield", h)
        # bin: 0=cold, 1=warm, 2=hot
        if h < t1:      b = 0
        elif h < t2:    b = 1
        else:           b = 2
        model.setVoxelProperty(key_tuple, "seed_hot_bin", int(b))

    # Context: track a canonical viewpoint (also populates 'visibility' if desired)
    vmgr = VisibilityManager(model)
    vp = Viewpoint.from_lookat([120.0, 120.0, 120.0], [0.0, 0.0, 0.0])
    vp.setDownsampleFactor(8.0)
    vp.setNearPlane(50.0); vp.setFarPlane(200.0)
    vmgr.trackViewpoint(vp)
    vp.performRaycastingOnGPU(model)

    # Visualize
    vis = Visualizer()
    vis.initializeWindow(window_title)
    vis.setBackgroundColor([1.0, 1.0, 1.0])

    # Discrete red/green/dark-green mapping (seed_hot_bin = 0/1/2)
    # Continuous gnuplot2 mapping (auto min/max from data)
    vis.addVoxelMapProperty(model, "seed_hotfield",
                        [0,0,0], [1,1,1],
                        -1.0, -1.0)



    # If you want to see redundancy too, uncomment:
    # vis.addVoxelMapProperty(model, "visibility",
    #                         [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0, 2.0)
    vis.setCameraPose(
    [221.76315766,  84.6960134,   63.28084895],
    [30.82407358, 14.58264472,  2.31950428],
    [-0.26366948, -0.11470131,  0.95776929]
    )

    vis.startAsyncRendering()
    try:
        while True:
            time.sleep(1.0)
            pos, focal, up = vis.getCameraPose()
            print(f"Camera pos: {pos}, focal: {focal}, up: {up}")
    except KeyboardInterrupt:
        os.makedirs("../figures", exist_ok=True)
        vis.stopAsyncRendering()
        vis.saveScreenshot("../figures/seed_hotfield_metavoxel.png", magnification=4)
        vis.startAsyncRendering()
    finally:
        vis.stopAsyncRendering()

# =========================
# Main
# =========================
if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud(POINT_CLOUD_PATH)
    if len(pcd.points) == 0:
        raise RuntimeError("Empty point cloud. Check POINT_CLOUD_PATH.")
    pts = np.asarray(pcd.points)

    if RANDOM_SUBSAMPLE is not None and RANDOM_SUBSAMPLE < len(pts):
        idx = np.random.choice(len(pts), size=RANDOM_SUBSAMPLE, replace=False)
        pts = pts[idx]

    mnn   = mean_nn_spacing(pts, k=8)
    sigma = GAUSS_SCALE_C * mnn

    seeds = pick_seeds(pts, mode=SEED_MODE, manual_idx=MANUAL_SEEDS_IDX)
    print(f"[info] Seeds (indices): {seeds}")

    print("[info] Building kNN graph...")
    nbrs = build_knn_graph(pts, k=KNN_GRAPH_K)

    # per-seed heat, then combine (intrinsic distances)
    fields = []
    for s in seeds:
        d_s = multi_source_dijkstra(nbrs, [s])
        H_s = distance_to_heat(d_s, sigma, falloff=FALLOFF, alpha=SOFTINV_ALPHA)
        fields.append(H_s)
    H = combine_fields(fields, mode=COMBINE, beta=SOFTMAX_BETA)
    H = (H - H.min()) / (H.max() - H.min() + 1e-12)

    print(f"[info] Heat stats: min={H.min():.4f}, max={H.max():.4f}, mean={H.mean():.4f}")

    # -------- Open3D preview (gnuplot2) --------
    colorize_open3d(pts, H, cmap_name="gnuplot2", title="Seed hotfield — gnuplot2")
    # colorize_open3d(pts, H, cmap_name="cool",      title="Seed hotfield — cool")  # (kept commented)

    # -------- Project & visualize in Visioncraft --------
    project_hotfield_to_visioncraft(POINT_CLOUD_PATH, pts, H, num_samples=200000,
                                    window_title="Visioncraft — seed_hotfield")
