#!/bin/bash
set -euo pipefail

# Absolute path to this script's directory (handles spaces)
SIM_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Fixed locations for your repo layout
GENETIC_PY="$SIM_ROOT/genetic.py"                  # <- script lives here
GENETIC_DIR="$SIM_ROOT/genetic"                    # <- results go here: genetic/<model>/seed_0
MODEL_DIR="$SIM_ROOT/../../models/abc"             # <- models live here (two levels up)

# Models & seeds
MODELS=$(seq -w 0001 0064)
SEEDS=(0)

# Safety: ensure files/dirs exist
[[ -f "$GENETIC_PY" ]] || { echo "Missing: $GENETIC_PY"; exit 1; }
mkdir -p "$GENETIC_DIR"

# Globs that match nothing become empty arrays
shopt -s nullglob

echo "SIM_ROOT   : $SIM_ROOT"
echo "GENETIC_PY : $GENETIC_PY"
echo "GENETIC_DIR: $GENETIC_DIR"
echo "MODEL_DIR  : $(realpath -m "$MODEL_DIR")"
echo "Running genetic simulations with skipping logic..."

for m in $MODELS; do
  model_path="$MODEL_DIR/${m}.stl"

  # 1) Skip if model file is missing
  if [[ ! -f "$model_path" ]]; then
    echo "→ Skip ${m}: model not found at $model_path"
    continue
  fi

  for s in "${SEEDS[@]}"; do
    out_dir="$GENETIC_DIR/${m}/seed_${s}"

    # 2) Skip if already processed (has ≥1 csv and ≥1 npz)
    if [[ -d "$out_dir" ]]; then
      csv_matches=("$out_dir"/*.csv)
      npz_matches=("$out_dir"/*.npz)
      if (( ${#csv_matches[@]} >= 1 && ${#npz_matches[@]} >= 1 )); then
        echo "→ Skip ${m} (seed ${s}): already processed (${#csv_matches[@]} csv, ${#npz_matches[@]} npz) at $out_dir"
        continue
      fi
    fi

    echo "▶ Run ${m}.stl (seed ${s})"
    (
      cd "$SIM_ROOT"  # run where genetic.py resides so its relative imports work
      python3 "$GENETIC_PY" --seed "$s" --model "$model_path"
    )
  done
done

echo "All eligible genetic simulations completed."
echo "Results under: $GENETIC_DIR/<model_id>/seed_0/"
