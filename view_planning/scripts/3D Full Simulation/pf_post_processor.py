#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import os
import glob
import shutil

# -------- Helpers --------

def has_seeds(dir_path: str) -> bool:
    """True if dir_path contains any seed_* directories."""
    return any(os.path.isdir(p) for p in glob.glob(os.path.join(dir_path, "seed_*")))

def list_model_dirs(cpp_root: str):
    """
    If cpp_root directly contains seed_* => old layout: treat cpp_root as one 'model'.
    Else => new layout: return subdirs under cpp_root that contain seed_*.
    """
    if has_seeds(cpp_root):
        return [cpp_root]
    candidates = [p for p in glob.glob(os.path.join(cpp_root, "*")) if os.path.isdir(p)]
    return sorted([p for p in candidates if has_seeds(p)])

def model_name_from_dir(model_dir: str, cpp_root: str) -> str:
    """Readable model name from directory path."""
    rel = os.path.relpath(model_dir, cpp_root)
    leaf = rel.split(os.sep)[0]
    return leaf if leaf not in (".", "") else os.path.basename(model_dir.rstrip(os.sep))

def list_seed_dirs(model_dir: str):
    seeds = sorted(
        glob.glob(os.path.join(model_dir, "seed_*")),
        key=lambda p: int(os.path.basename(p).split("_")[-1])
                      if os.path.basename(p).split("_")[-1].isdigit() else 1_000_000
    )
    return seeds

def load_one_seed(cpp_seed_dir: str):
    """
    Load all required CSVs from one C++ seed directory.

    Expected files (from the original script):
      - time_series.csv  -> columns: time, coverage, redundancy, affinity
      - viewpoint_point_assignments.csv  (bool)
      - viewpoint_contribution_hist.csv  (int; also used to count num_viewpoints)
      - point_redundancy_hist.csv        (int)
      - viewpoint_overlap_matrix.csv     (float, comma-separated)
      - functional_isolation_flags.csv   (bool)
      - final_viewpoints.csv             (float, comma-separated, header present)
      - potential_field_viewpoints.csv   (copied through):contentReference[oaicite:1]{index=1}
    """
    # time_series
    ts = pd.read_csv(os.path.join(cpp_seed_dir, 'time_series.csv'))
    coverage   = ts['coverage'].values
    redundancy = ts['redundancy'].values
    affinity   = ts['affinity'].values
    time_ts    = ts['time'].values
    num_iterations = len(coverage)

    # num_viewpoints from contribution histogram row count
    contrib_csv = os.path.join(cpp_seed_dir, 'viewpoint_contribution_hist.csv')
    N = len(pd.read_csv(contrib_csv, header=None))

    # other artifacts
    assign   = np.loadtxt(os.path.join(cpp_seed_dir, 'viewpoint_point_assignments.csv'),
                          delimiter=',', dtype=bool)
    contrib  = np.loadtxt(contrib_csv, dtype=np.int32)
    red_hist = np.loadtxt(os.path.join(cpp_seed_dir, 'point_redundancy_hist.csv'),
                          dtype=np.int32)
    overlap  = np.loadtxt(os.path.join(cpp_seed_dir, 'viewpoint_overlap_matrix.csv'),
                          delimiter=',', dtype=np.float32)
    isolation= np.loadtxt(os.path.join(cpp_seed_dir, 'functional_isolation_flags.csv'),
                          dtype=bool)
    final_vp = np.loadtxt(os.path.join(cpp_seed_dir, 'final_viewpoints.csv'),
                          delimiter=',', skiprows=1, dtype=float)

    return {
        "coverage": coverage,
        "redundancy": redundancy,
        "affinity": affinity,
        "time": time_ts,
        "num_viewpoints": int(N),
        "num_iterations": int(num_iterations),
        "viewpoint_point_assignments": assign,
        "viewpoint_contribution_hist": contrib,
        "point_redundancy_hist": red_hist,
        "viewpoint_overlap_matrix": overlap,
        "functional_isolation_flags": isolation,
        "final_viewpoints": final_vp
    }

def save_one_seed(out_seed_dir: str, payload: dict, cpp_seed_dir: str):
    os.makedirs(out_seed_dir, exist_ok=True)
    # NPZ
    np.savez_compressed(
        os.path.join(out_seed_dir, 'potential_field_result.npz'),
        **payload
    )
    print(f"[Saved] {os.path.join(out_seed_dir, 'potential_field_result.npz')}")
    # Copy viewpoints CSV
    src_csv = os.path.join(cpp_seed_dir, 'potential_field_viewpoints.csv')
    dst_csv = os.path.join(out_seed_dir, 'potential_field_viewpoints.csv')
    if os.path.exists(src_csv):
        shutil.copy(src_csv, dst_csv)
        print(f"[Saved] {dst_csv}")

# -------- Main --------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp_root", type=str, default="./cpp_output",
                        help="Root of C++ outputs. Supports ./cpp_output/seed_* "
                             "or ./cpp_output/<model>/seed_*.")
    parser.add_argument("--out_root", type=str, default="./potential_field",
                        help="Root to write processed outputs.")
    parser.add_argument("--models", type=str, default=None,
                        help="Optional comma-separated list of model names to include. "
                             "If omitted, processes all models found.")
    args = parser.parse_args()

    cpp_root = os.path.abspath(args.cpp_root)
    out_root = os.path.abspath(args.out_root)

    # Discover model directories
    model_dirs = list_model_dirs(cpp_root)
    if not model_dirs:
        print(f"[info] No seeds found under {cpp_root}. Nothing to do.")
        return

    # Filter by --models if provided
    allow = None
    if args.models:
        allow = set([m.strip() for m in args.models.split(",") if m.strip()])

    for model_dir in model_dirs:
        model_name = model_name_from_dir(model_dir, cpp_root)
        if allow and model_name not in allow:
            continue

        print(f"[info] Model: {model_name}")
        seed_dirs = list_seed_dirs(model_dir)
        if not seed_dirs:
            print(f"[info]   No seeds in {model_dir}; skipping.")
            continue

        # Output base: out_root/<model>/
        model_out_dir = os.path.join(out_root, model_name)
        os.makedirs(model_out_dir, exist_ok=True)

        for seed_dir in seed_dirs:
            seed_leaf = os.path.basename(seed_dir)  # e.g., seed_0
            out_seed_dir = os.path.join(model_out_dir, seed_leaf)
            try:
                payload = load_one_seed(seed_dir)  # loads exactly what the original script did:contentReference[oaicite:2]{index=2}
                save_one_seed(out_seed_dir, payload, seed_dir)
            except Exception as e:
                print(f"[warn] Failed to process {seed_dir}: {e}")

    print("[info] Done.")

if __name__ == "__main__":
    main()
