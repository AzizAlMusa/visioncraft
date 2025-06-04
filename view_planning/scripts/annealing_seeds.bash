#!/bin/bash

SA_DIR="./results2/sa_seeds"
mkdir -p "$SA_DIR"

for SEED in $(seq 0 10); do
  echo "Running simulated annealing simulation for seed $SEED"
  python figure2_annealing.py \
    --save_dir "$SA_DIR/seed${SEED}" \
    --seed $SEED \
    --animate
done
