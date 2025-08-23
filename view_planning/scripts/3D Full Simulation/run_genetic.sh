#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# -----------------------------
# Config
# -----------------------------
MODELS=(gorilla cat mug bracket goku bell)
SEED_START=0
SEED_END=9

# -----------------------------
# Helpers
# -----------------------------
usage() {
  cat <<EOF
Usage:
  $(basename "$0") [--redo MODEL SEED] [--resume] [--models m1,m2,...]

Modes:
  --redo MODEL SEED     Re-run only the specified model/seed pair (e.g., --redo gorilla 8) and exit.
  --resume              For each model, detect highest completed seed and resume from (that - 1) to ${SEED_END}.
                        Re-runs the (that - 1) seed even if completed; skips other completed seeds.
  --models m1,m2,...    Limit to a subset of models (comma-separated). Default: ${MODELS[*]}

Notes:
- Completion for a seed = existence of: genetic/<model>/seed_<n>/genetic_result.npz
- Default behavior (no flags): run all models and seeds, skipping already-completed seeds.
EOF
}

# Return 0 if result exists (completed), else 1
is_completed() {
  local model="$1" seed="$2"
  [[ -f "genetic/${model}/seed_${seed}/genetic_result.npz" ]]
}

# Get the highest completed seed index for a model, or -1 if none
highest_completed_seed() {
  local model="$1"
  local highest="-1"
  for (( s=SEED_START; s<=SEED_END; s++ )); do
    if is_completed "$model" "$s"; then
      highest="$s"
    fi
  done
  echo "$highest"
}

run_one() {
  local model="$1" seed="$2"
  echo "  Seed ${seed}"
  python3 genetic.py --seed "${seed}" --model "../../models/${model}.stl"
}

run_range_skip_completed() {
  local model="$1" start_s="$2" end_s="$3" rerun_seed="$4"
  echo "Model: ${model}.stl"
  for (( s=start_s; s<=end_s; s++ )); do
    # Re-run the designated rerun_seed even if completed; skip others if completed
    if [[ "$s" -ne "$rerun_seed" ]] && is_completed "$model" "$s"; then
      echo "  Seed ${s} (already completed) -> skip"
      continue
    fi
    run_one "$model" "$s"
  done
}

# -----------------------------
# Args
# -----------------------------
REDO_MODEL=""
REDO_SEED=""
RESUME="0"
LIMIT_MODELS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --redo)
      [[ $# -ge 3 ]] || { echo "Error: --redo requires MODEL and SEED"; usage; exit 1; }
      REDO_MODEL="$2"
      REDO_SEED="$3"
      shift 3
      ;;
    --resume)
      RESUME="1"
      shift
      ;;
    --models)
      [[ $# -ge 2 ]] || { echo "Error: --models requires a comma-separated list"; usage; exit 1; }
      IFS=',' read -r -a LIMIT_MODELS <<< "$2"
      shift 2
      ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

# Apply model filter if provided
if [[ ${#LIMIT_MODELS[@]} -gt 0 ]]; then
  MODELS=("${LIMIT_MODELS[@]}")
fi

# -----------------------------
# Mode: redo single pair
# -----------------------------
if [[ -n "$REDO_MODEL" ]]; then
  if [[ "$REDO_SEED" =~ ^[0-9]+$ ]] && (( REDO_SEED >= SEED_START && REDO_SEED <= SEED_END )); then
    echo "Re-running only: ${REDO_MODEL} seed ${REDO_SEED}"
    echo "Model: ${REDO_MODEL}.stl"
    run_one "$REDO_MODEL" "$REDO_SEED"
    echo "Done."
    exit 0
  else
    echo "Seed out of range (${SEED_START}-${SEED_END}): ${REDO_SEED}"; exit 1
  fi
fi

echo "Running genetic simulations..."

# -----------------------------
# Mode: resume
# -----------------------------
if [[ "$RESUME" == "1" ]]; then
  for m in "${MODELS[@]}"; do
    h=$(highest_completed_seed "$m")
    # Start at (h - 1) but not below SEED_START; if none completed (h=-1), start at SEED_START
    if (( h >= 0 )); then
      start=$(( h - 1 ))
      (( start < SEED_START )) && start=$SEED_START
      # We'll re-run exactly 'start' even if completed; skip other completed
      run_range_skip_completed "$m" "$start" "$SEED_END" "$start"
    else
      # no completions: run from SEED_START, skip none
      run_range_skip_completed "$m" "$SEED_START" "$SEED_END" "-1"
    fi
  done
  echo "All resume runs completed."
  echo "Results stored under: genetic/<model>/seed_*/"
  exit 0
fi

# -----------------------------
# Default mode: run all, skip completed
# -----------------------------
for m in "${MODELS[@]}"; do
  echo "Model: ${m}.stl"
  for s in $(seq "$SEED_START" "$SEED_END"); do
    if is_completed "$m" "$s"; then
      echo "  Seed ${s} (already completed) -> skip"
      continue
    fi
    run_one "$m" "$s"
  done
done

echo "All simulations completed."
echo "Results stored under: genetic/<model>/seed_*/"
