#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

MODELS=(gorilla cat mug bracket goku bell)
SEEDS=$(seq 0 9)

echo "Running simulated annealing (SA) simulations..."
for m in "${MODELS[@]}"; do
  echo "Model: ${m}.stl"
  for s in $SEEDS; do
    echo "  Seed $s"
    python3 simulated_annealing.py --seed "$s" --model "../../models/${m}.stl"
  done
done

echo "Done. Results in: sa/<model>/seed_*/"
