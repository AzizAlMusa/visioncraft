#!/bin/bash

GREEDY_DIR="./results2/greedy_seeds"
mkdir -p "$GREEDY_DIR"

for SEED in $(seq 0 10); do
  echo "Running greedy simulation for seed $SEED"
  python figure2_greedy.py \
    --strategy greedy \
    --save_dir "$GREEDY_DIR/seed${SEED}" \
    --seed $SEED \
    --verbose \
    --animate
done
