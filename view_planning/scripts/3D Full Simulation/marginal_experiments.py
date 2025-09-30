#!/usr/bin/env python3
# deep_analysis_filtered.py
# Same as deep_analysis.py, but supports excluding specific "bad" models
# from all summaries (while always keeping named models).
#
# Outputs (under ./overlap_results):
#   bad_models.csv
#   per_model_metrics_all.csv      (unfiltered)
#   per_model_metrics.csv          (filtered)
#   per_method_summary_mean_std.csv
#   per_method_summary_mean_std_BOUNDED.csv
#   pf_time_outliers.csv

import os, glob, argparse
import numpy as np
import pandas as pd
from scipy.stats import entropy

# ----------------- Helpers for model discovery -----------------
def has_seeds(dir_path: str) -> bool:
    return any(os.path.isdir(p) for p in glob.glob(os.path.join(dir_path, "seed_*")))

def list_model_dirs(method_root: str):
    if os.path.isdir(method_root) and has_seeds(method_root):
        return [method_root]
    candidates = [p for p in glob.glob(os.path.join(method_root, "*")) if os.path.isdir(p)]
    return sorted([p for p in candidates if has_seeds(p)])

def model_name_from_dir(model_dir: str, method_root: str) -> str:
    rel = os.path.relpath(model_dir, method_root)
    leaf = rel.split(os.sep)[0]
    return leaf if leaf not in (".", "") else os.path.basename(model_dir.rstrip(os.sep))

def list_npz(model_dir: str, strategy: str):
    return sorted(glob.glob(os.path.join(model_dir, "seed_*", f"{strategy}_result.npz")))

def discover_all_models(method_roots):
    models = set()
    for root in method_roots:
        for mdir in list_model_dirs(root):
            models.add(model_name_from_dir(mdir, root))
    return sorted(models)

# ----------------- Core metric helpers -----------------
def gini(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if x.size == 0: return 0.0
    s = x.sum()
    if s <= 0: return 0.0
    x = np.sort(x)
    n = x.size
    cum = (np.arange(1, n+1) * x).sum()
    g = (2.0 * cum) / (n * s) - (n + 1.0) / n
    return float(np.clip(g, 0.0, 1.0))

def coverage_auc(time_arr: np.ndarray, cov_arr: np.ndarray) -> float:
    if time_arr.size >= 2 and cov_arr.size == time_arr.size:
        return float(np.trapz(cov_arr, x=time_arr))
    return 0.0

def per_seed_stats(assign: np.ndarray):
    """
    Returns:
      exclusive_mean_pct, overlap_mean, overlap_cv, overlap_entropy_norm, overlap_gini,
      n_points, n_viewpoints, n_pairs_sampled
    """
    assign = np.asarray(assign)
    if assign.shape[0] < assign.shape[1]:
        assign = assign.T
    P, V = assign.shape
    if V == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, P, V, 0

    seen_counts = assign.sum(axis=1)
    exclusive_mask = (seen_counts == 1)

    exclusive_contrib = np.zeros(V, dtype=float)
    if exclusive_mask.any():
        idxs = np.where(exclusive_mask)[0]
        vps = assign[idxs].argmax(axis=1)
        np.add.at(exclusive_contrib, vps, 1.0)
    exclusive_contrib = (exclusive_contrib / max(1, P)) * 100.0
    exclusive_mean_pct = float(np.mean(exclusive_contrib)) if V > 0 else 0.0

    # Overlap IoU between viewpoint pairs
    overlap = np.zeros((V, V), dtype=np.float64)
    for i in range(V):
        vi = assign[:, i].astype(bool)
        for j in range(i, V):
            vj = assign[:, j].astype(bool)
            inter = np.logical_and(vi, vj).sum()
            union = np.logical_or(vi, vj).sum()
            overlap[i, j] = overlap[j, i] = inter / (union + 1e-12)
    np.fill_diagonal(overlap, 1.0)

    denom = np.log(max(2, V - 1))
    ent_vals, cv_vals, gi_vals, mean_vals = [], [], [], []
    n_pairs_total = 0

    for i in range(V):
        nbrs = np.delete(overlap[i], i)
        nbrs = nbrs[nbrs > 0]
        n_pairs_total += nbrs.size

        mean_vals.append(float(np.mean(nbrs)) if nbrs.size else 0.0)

        if nbrs.size == 0 or denom == 0:
            ent_vals.append(0.0)
        else:
            p = nbrs / (nbrs.sum() + 1e-12)
            ent_vals.append(float(entropy(p)) / denom)

        if nbrs.size == 0:
            cv_vals.append(0.0)
        else:
            mu = float(np.mean(nbrs))
            sd = float(np.std(nbrs, ddof=0))
            cv_vals.append(sd / (mu + 1e-12))

        gi_vals.append(gini(nbrs))

    return (
        float(np.mean(exclusive_contrib)) if V > 0 else 0.0,
        float(np.mean(mean_vals)) if mean_vals else 0.0,
        float(np.mean(cv_vals))   if cv_vals   else 0.0,
        float(np.mean(ent_vals))  if ent_vals  else 0.0,
        float(np.mean(gi_vals))   if gi_vals   else 0.0,
        P, V, n_pairs_total
    )

def mean_std(arr):
    arr = np.asarray(arr, float)
    return float(np.mean(arr)) if arr.size else 0.0, float(np.std(arr)) if arr.size else 0.0

# ----------------- Aggregation across seeds (per model) -----------------
def aggregate_model(method_root: str, model_name: str, strategy: str):
    # try multi-model layout first
    model_dir = os.path.join(method_root, model_name)
    if not (os.path.isdir(model_dir) and has_seeds(model_dir)):
        # legacy layout fallback: allow root==single-model when names match
        if os.path.basename(os.path.normpath(method_root)) == model_name and has_seeds(method_root):
            model_dir = method_root
        else:
            return None

    files = list_npz(model_dir, strategy)
    if not files:
        return None

    vp_list, auc_list, time_list = [], [], []
    excl_list, ovl_list, cv_list, ent_list, gini_list = [], [], [], [], []

    npoints_sum = 0
    nviewpoints_sum = 0
    npairs_sum = 0
    nseeds = 0

    for path in files:
        data = np.load(path)
        time  = np.asarray(data["time"])
        cover = np.asarray(data["coverage"])
        vp    = int(data["num_viewpoints"])
        assign = np.asarray(data["viewpoint_point_assignments"])

        excl, ovl_m, ovl_cv, ent, gi, P, V, n_pairs = per_seed_stats(assign)

        vp_list.append(vp)
        auc_list.append(coverage_auc(time, cover))
        time_list.append(float(time[-1]) if time.size else 0.0)
        excl_list.append(excl)
        ovl_list.append(ovl_m)
        cv_list.append(ovl_cv)
        ent_list.append(ent)
        gini_list.append(gi)

        npoints_sum     += int(P)
        nviewpoints_sum += int(V)
        npairs_sum      += int(n_pairs)
        nseeds          += 1

    vp_m, vp_s   = mean_std(vp_list)
    auc_m, auc_s = mean_std(auc_list)
    t_m,  t_s    = mean_std(time_list)
    ex_m, ex_s   = mean_std(excl_list)
    ov_m, ov_s   = mean_std(ovl_list)   # actual neighbor overlap mean (0..1)
    cv_m, cv_s   = mean_std(cv_list)
    en_m, en_s   = mean_std(ent_list)
    gi_m, gi_s   = mean_std(gini_list)

    return {
        "n_seeds": nseeds,
        "samples_points": npoints_sum,
        "samples_viewpoints": nviewpoints_sum,
        "samples_pairs": npairs_sum,
        "num_viewpoints_mean": vp_m,
        "num_viewpoints_std":  vp_s,
        "coverage_auc_mean":   auc_m,
        "coverage_auc_std":    auc_s,
        "solution_time_mean":  t_m,
        "solution_time_std":   t_s,
        "exclusive_area_mean": ex_m,
        "exclusive_area_std":  ex_s,
        "overlap_mean":        ov_m,   # 0..1
        "overlap_std":         ov_s,
        "cv_overlap_mean":     cv_m,
        "cv_overlap_std":      cv_s,
        "overlap_entropy_mean": en_m,
        "overlap_entropy_std":  en_s,
        "gini_overlap_mean":   gi_m,
        "gini_overlap_std":    gi_s,
    }

# ----------------- Main routine -----------------
def main():
    ap = argparse.ArgumentParser(description="Deep analysis (filtered) with overlap mean + bounded summaries")
    ap.add_argument("--dir_greedy",   type=str, default="./greedy")
    ap.add_argument("--dir_genetic",  type=str, default="./genetic")
    ap.add_argument("--dir_sa",       type=str, default="./sa")
    ap.add_argument("--dir_potential",type=str, default="./potential_field")
    ap.add_argument("--out_dir",      type=str, default="./overlap_results")
    ap.add_argument("--pf_outlier_topk", type=int, default=12)
    # Exclude models list (comma-separated 4-digit IDs); named models are always kept
    ap.add_argument("--exclude_models", type=str,
                    default="0008,0043,0056,0102,0022,0110,0094,0068,0090,0078")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    methods = [
        ("Greedy",          args.dir_greedy,    "greedy"),
        ("Genetic (RKGA)",  args.dir_genetic,   "genetic"),
        ("SA",              args.dir_sa,        "sa"),
        ("Potential Field", args.dir_potential, "potential_field"),
    ]

    # union of models visible under any method root
    all_models = discover_all_models([m[1] for m in methods])
    if not all_models:
        print("No models found.")
        return

    # ---- per-model table (unfiltered) ----
    rows = []
    for model in all_models:
        for label, root, strat in methods:
            agg = aggregate_model(root, model, strat)
            if not agg:
                continue
            rows.append({
                "model": model,
                "method": label,
                **agg
            })

    if not rows:
        print("No data aggregated.")
        return

    per_model_all = pd.DataFrame(rows).sort_values(["method", "model"]).reset_index(drop=True)

    # Save unfiltered reference
    out_all = os.path.join(args.out_dir, "per_model_metrics_all.csv")
    per_model_all.to_csv(out_all, index=False, float_format="%.6f")
    print(f"[Saved] {out_all}")

    # ---- apply blacklist filter (except named models) ----
    named_models = {"cat","goku","gorilla","bracket","bell","mug"}
    exclude_set = set(s.strip() for s in args.exclude_models.split(",") if s.strip())
    # keep named models even if listed in exclude_set
    mask_exclude = per_model_all["model"].astype(str).isin(exclude_set) & (~per_model_all["model"].isin(named_models))
    per_model = per_model_all[~mask_exclude].copy()

    # Persist the effective blacklist used
    effective_bad = sorted(set(per_model_all["model"].astype(str)) & exclude_set - named_models)
    pd.DataFrame({"bad_model": effective_bad}).to_csv(os.path.join(args.out_dir, "bad_models.csv"),
                                                     index=False)
    print(f"[Saved] {os.path.join(args.out_dir, 'bad_models.csv')}  ({len(effective_bad)} models)")

    # Save filtered per-model table
    out1 = os.path.join(args.out_dir, "per_model_metrics.csv")
    per_model.to_csv(out1, index=False, float_format="%.6f")
    print(f"[Saved] {out1}")

    # ---- PF time outliers (on filtered set) ----
    pf = per_model[per_model["method"] == "Potential Field"].copy()
    if not pf.empty:
        pf_out = pf.sort_values("solution_time_mean", ascending=False)[
            ["model","solution_time_mean","solution_time_std",
             "num_viewpoints_mean","num_viewpoints_std",
             "samples_points","samples_viewpoints","samples_pairs","n_seeds",
             "overlap_mean","cv_overlap_mean","exclusive_area_mean"]
        ]
        out_pf = os.path.join(args.out_dir, "pf_time_outliers.csv")
        pf_out.head(args.pf_outlier_topk).to_csv(out_pf, index=False, float_format="%.6f")
        print(f"[Saved] {out_pf}")
    else:
        print("[info] No Potential Field rows found; PF outliers not written.")

    # ---- per-method summary (means & stds across *models*, filtered) ----
    def summarize(group: pd.DataFrame):
        d = {
            "n_models": group["model"].nunique(),
            "n_seeds_total": int(group["n_seeds"].sum()),
            "sum_points": int(group["samples_points"].sum()),
            "sum_viewpoints": int(group["samples_viewpoints"].sum()),
            "sum_pairs": int(group["samples_pairs"].sum()),

            "mean_num_viewpoints": group["num_viewpoints_mean"].mean(),
            "std_num_viewpoints":  group["num_viewpoints_mean"].std(ddof=0),

            "mean_solution_time":  group["solution_time_mean"].mean(),
            "std_solution_time":   group["solution_time_mean"].std(ddof=0),

            "mean_cv_overlap":     group["cv_overlap_mean"].mean(),
            "std_cv_overlap":      group["cv_overlap_mean"].std(ddof=0),

            "mean_novel_area":     group["exclusive_area_mean"].mean(),
            "std_novel_area":      group["exclusive_area_mean"].std(ddof=0),

            "mean_overlap":        group["overlap_mean"].mean(),
            "std_overlap":         group["overlap_mean"].std(ddof=0),
        }
        return pd.Series(d)

    per_method = per_model.groupby("method", sort=False).apply(summarize).reset_index()

    out2 = os.path.join(args.out_dir, "per_method_summary_mean_std.csv")
    per_method.to_csv(out2, index=False, float_format="%.6f")
    print(f"[Saved] {out2}")

    # ---- bounded summary (no negative lower bounds) ----
    def bounded_row(r):
        def bounds(mean, std, lo, hi):
            if (mean is None) or (std is None) or (np.isnan(mean)) or (np.isnan(std)):
                return np.nan, np.nan
            return max(lo, mean - std), min(hi, mean + std)

        m = r.copy()
        m["novel_area_lower"], m["novel_area_upper"] = bounds(r["mean_novel_area"], r["std_novel_area"], 0.0, 100.0)
        m["overlap_lower"],    m["overlap_upper"]    = bounds(r["mean_overlap"],    r["std_overlap"],    0.0, 1.0)
        m["cv_lower"],         m["cv_upper"]         = bounds(r["mean_cv_overlap"], r["std_cv_overlap"], 0.0, float("inf"))
        m["#vp_lower"],        m["#vp_upper"]        = bounds(r["mean_num_viewpoints"], r["std_num_viewpoints"], 0.0, float("inf"))
        m["time_lower"],       m["time_upper"]       = bounds(r["mean_solution_time"],  r["std_solution_time"],  0.0, float("inf"))
        return m

    bounded = per_method.apply(bounded_row, axis=1)
    out3 = os.path.join(args.out_dir, "per_method_summary_mean_std_BOUNDED.csv")
    bounded.to_csv(out3, index=False, float_format="%.6f")
    print(f"[Saved] {out3}")

    print("\nDone.")

if __name__ == "__main__":
    main()
