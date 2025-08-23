#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import os
import glob
from scipy.stats import entropy

# --------- Dir helpers ---------
def has_seeds(dir_path: str) -> bool:
    return any(os.path.isdir(p) for p in glob.glob(os.path.join(dir_path, "seed_*")))

def list_model_dirs(method_root: str):
    """
    If method_root directly contains seed_* => legacy layout: treat it as one 'model' (name = leaf dir).
    Else => return subdirs under method_root that contain seed_*.
    """
    if os.path.isdir(method_root) and has_seeds(method_root):
        return [method_root]
    candidates = [p for p in glob.glob(os.path.join(method_root, "*")) if os.path.isdir(p)]
    return sorted([p for p in candidates if has_seeds(p)])

def model_name_from_dir(model_dir: str, method_root: str) -> str:
    rel = os.path.relpath(model_dir, method_root)
    leaf = rel.split(os.sep)[0]
    name = leaf if leaf not in (".", "") else os.path.basename(model_dir.rstrip(os.sep))
    # In legacy layout, model_dir == method_root; name becomes the leaf of method_root (e.g., 'greedy')
    return name

def list_npz(model_dir: str, strategy: str):
    return sorted(glob.glob(os.path.join(model_dir, "seed_*", f"{strategy}_result.npz")))

def resolve_model_dir(method_root: str, model_name: str):
    """
    Prefer multi-model: <method_root>/<model_name>
    Fallback to legacy: method_root (if it has seed_*)
    """
    cand = os.path.join(method_root, model_name)
    if os.path.isdir(cand) and has_seeds(cand):
        return cand
    if os.path.isdir(method_root) and has_seeds(method_root):
        # legacy single 'model' at root
        # only return if the requested model_name matches that leaf (so we don't mis-assign)
        if model_name == os.path.basename(os.path.normpath(method_root)):
            return method_root
    return None

# --------- Metric helpers ---------
def auc_coverage(time_arr: np.ndarray, cov_arr: np.ndarray) -> float:
    if time_arr.size >= 2 and cov_arr.size == time_arr.size:
        return float(np.trapz(cov_arr, x=time_arr))
    return 0.0

def _gini(x: np.ndarray) -> float:
    """Gini coefficient for a nonnegative 1D array; returns 0 if all zeros or len<1."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    s = x.sum()
    if s <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = (np.arange(1, n + 1) * x).sum()
    g = (2.0 * cum) / (n * s) - (n + 1.0) / n
    return float(max(0.0, min(1.0, g)))

def compute_exclusive_entropy_and_neighbor_overlap_stats(assign: np.ndarray):
    """
    assign: bool/int array, either (points x viewpoints) or (viewpoints x points).

    Returns (scalars):
      exclusive_mean:        mean (%) unique contribution per viewpoint
      entropy_mean:          mean normalized overlap entropy over nonzero neighbors
      cv_overlap_mean:       mean CV of overlap over nonzero neighbors
      gini_overlap_mean:     mean Gini of overlap over nonzero neighbors
    """
    # Ensure shape: points x viewpoints
    if assign.shape[0] < assign.shape[1]:
        assign = assign.T
    num_points, N = assign.shape

    # Exclusive contributions: points seen by exactly one viewpoint
    seen_counts = assign.sum(axis=1)  # per point
    exclusive_mask = (seen_counts == 1)
    exclusive_contrib = np.zeros(N, dtype=float)
    if exclusive_mask.any():
        idxs = np.where(exclusive_mask)[0]
        vps = assign[idxs].argmax(axis=1)  # unique viewer of that point
        np.add.at(exclusive_contrib, vps, 1.0)
    exclusive_contrib = (exclusive_contrib / max(1, num_points)) * 100.0
    exclusive_mean = float(np.mean(exclusive_contrib)) if N > 0 else 0.0

    # Overlap matrix (IoU on boolean OR)
    overlap = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        vi = assign[:, i].astype(bool)
        for j in range(i, N):
            vj = assign[:, j].astype(bool)
            inter = np.logical_and(vi, vj).sum()
            union = np.logical_or(vi, vj).sum()
            overlap[i, j] = overlap[j, i] = inter / (union + 1e-12)
    np.fill_diagonal(overlap, 1.0)

    # Per-viewpoint neighbor-only stats
    denom = np.log(max(2, N - 1))  # for normalized entropy
    ent_vals, cv_vals, gi_vals = [], [], []

    for i in range(N):
        row = np.delete(overlap[i], i)   # remove self
        nbrs = row[row > 0]              # only nonzero neighbors

        # Normalized entropy over nonzero neighbors
        if nbrs.size == 0 or denom == 0:
            ent_vals.append(0.0)
        else:
            p = nbrs / (nbrs.sum() + 1e-12)
            ent_vals.append(float(entropy(p)) / denom)

        # CV over nonzero neighbors
        if nbrs.size == 0:
            cv_vals.append(0.0)
        else:
            mu = nbrs.mean()
            sd = nbrs.std(ddof=0)
            cv_vals.append(float(sd / (mu + 1e-12)))

        # Gini over nonzero neighbors
        gi_vals.append(_gini(nbrs))

    entropy_mean       = float(np.mean(ent_vals)) if ent_vals else 0.0
    cv_overlap_mean    = float(np.mean(cv_vals))  if cv_vals  else 0.0
    gini_overlap_mean  = float(np.mean(gi_vals))  if gi_vals  else 0.0

    return exclusive_mean, entropy_mean, cv_overlap_mean, gini_overlap_mean

def aggregate_for_method(method_root: str, model_name: str, strategy: str):
    model_dir = resolve_model_dir(method_root, model_name)
    if not model_dir:
        return None

    files = list_npz(model_dir, strategy)
    if not files:
        return None

    num_viewpoints_list = []
    cov_auc_list = []
    sol_time_list = []
    excl_mean_list = []
    ent_mean_list = []
    cv_nb_list = []
    gi_nb_list = []

    for path in files:
        data = np.load(path)
        time = data["time"]
        coverage = data["coverage"]
        num_viewpoints = int(data["num_viewpoints"])
        assign = data["viewpoint_point_assignments"]

        num_viewpoints_list.append(num_viewpoints)
        cov_auc_list.append(auc_coverage(time, coverage))
        sol_time_list.append(float(time[-1]) if time.size else 0.0)

        excl_mean, ent_mean, cv_nb, gi_nb = compute_exclusive_entropy_and_neighbor_overlap_stats(assign)
        excl_mean_list.append(excl_mean)
        ent_mean_list.append(ent_mean)
        cv_nb_list.append(cv_nb)
        gi_nb_list.append(gi_nb)

    def mean_std(x):
        arr = np.array(x, dtype=float)
        return float(np.mean(arr)), float(np.std(arr))

    return {
        "n_seeds": len(files),
        "num_viewpoints": mean_std(num_viewpoints_list),
        "coverage_auc":   mean_std(cov_auc_list),
        "solution_time":  mean_std(sol_time_list),
        "exclusive_area": mean_std(excl_mean_list),
        "overlap_entropy":mean_std(ent_mean_list),
        "cv_overlap_neighbors": mean_std(cv_nb_list),
        "gini_overlap_neighbors": mean_std(gi_nb_list),
    }

# --------- Model discovery across all methods ---------
def discover_all_models(method_roots):
    """
    Union of model names discovered under each method root.
    In legacy layout, the 'model name' is the leaf folder of the method root (e.g., 'greedy').
    """
    models = set()
    for root in method_roots:
        for mdir in list_model_dirs(root):
            name = model_name_from_dir(mdir, root)
            models.add(name)
    return sorted(models)

# --------- Main ---------
def main():
    ap = argparse.ArgumentParser(description="Compare metrics across methods for ALL models.")
    ap.add_argument("--dir_greedy",   type=str, default="./greedy")
    ap.add_argument("--dir_genetic",  type=str, default="./genetic")
    ap.add_argument("--dir_sa",       type=str, default="./sa")
    ap.add_argument("--dir_potential",type=str, default="./potential_field")
    ap.add_argument("--out_csv_dir",  type=str, default="./overlap_results")
    args = ap.parse_args()

    os.makedirs(args.out_csv_dir, exist_ok=True)

    methods = [
        ("Greedy", args.dir_greedy, "greedy"),
        ("Genetic", args.dir_genetic, "genetic"),
        ("SA", args.dir_sa, "sa"),
        ("Potential Field", args.dir_potential, "potential_field"),
    ]

    # Discover all model names present in any method directory
    all_models = discover_all_models([m[1] for m in methods])
    if not all_models:
        print("No models found.")
        return

    for model in all_models:
        rows = []
        print(f"\n=== Model: {model} ===")
        print(
            f"{'Method':16s}  {'Seeds':>5s}  "
            f"{'#VP (mean±std)':>20s}  {'AUC (mean±std)':>20s}  {'Time s (mean±std)':>22s}  "
            f"{'Exclusive %/vp (mean±std)':>28s}  {'Entropy (mean±std)':>22s}  "
            f"{'CV overlap (mean±std)':>24s}  {'Gini overlap (mean±std)':>26s}"
        )

        for label, root, strategy in methods:
            agg = aggregate_for_method(root, model, strategy)
            if not agg:
                continue

            n = agg["n_seeds"]
            vp_m, vp_s = agg["num_viewpoints"]
            auc_m, auc_s = agg["coverage_auc"]
            t_m, t_s   = agg["solution_time"]
            ex_m, ex_s = agg["exclusive_area"]
            en_m, en_s = agg["overlap_entropy"]
            cvn_m, cvn_s = agg["cv_overlap_neighbors"]
            gin_m, gin_s = agg["gini_overlap_neighbors"]

            print(
                f"{label:16s}  {n:5d}  "
                f"{vp_m:8.2f}±{vp_s:<8.2f}  {auc_m:8.3f}±{auc_s:<8.3f}  {t_m:8.3f}±{t_s:<8.3f}  "
                f"{ex_m:8.3f}±{ex_s:<8.3f}            {en_m:6.3f}±{en_s:<6.3f}  "
                f"{cvn_m:8.3f}±{cvn_s:<8.3f}          {gin_m:6.3f}±{gin_s:<6.3f}"
            )

            rows.append({
                "method": label,
                "n_seeds": n,
                "num_viewpoints_mean": vp_m,
                "num_viewpoints_std":  vp_s,
                "coverage_auc_mean":   auc_m,
                "coverage_auc_std":    auc_s,
                "solution_time_mean":  t_m,
                "solution_time_std":   t_s,
                "exclusive_area_mean": ex_m,
                "exclusive_area_std":  ex_s,
                "overlap_entropy_mean": en_m,
                "overlap_entropy_std":  en_s,
                "cv_overlap_neighbors_mean":   cvn_m,
                "cv_overlap_neighbors_std":    cvn_s,
                "gini_overlap_neighbors_mean": gin_m,
                "gini_overlap_neighbors_std":  gin_s,
            })

        if not rows:
            print("  (no data)")
            continue

        out_csv = os.path.join(args.out_csv_dir, f"{model}_metrics_summary.csv")
        df = pd.DataFrame(rows)
        num_cols = df.select_dtypes(include=[float, int]).columns
        df[num_cols] = df[num_cols].round(6)
        df.to_csv(out_csv, index=False)
        print(f"[Saved] {out_csv}")

if __name__ == "__main__":
    main()
