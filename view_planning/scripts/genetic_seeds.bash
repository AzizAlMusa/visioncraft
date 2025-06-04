#!/bin/bash

GENETIC_DIR="./results2/genetic_seeds"
mkdir -p "$GENETIC_DIR"

for SEED in $(seq 0 10); do
  echo "Running genetic simulation for seed $SEED"
  python figure2_genetic.py \
    --strategy rkga \
    --save_dir "$GENETIC_DIR/seed${SEED}" \
    --seed $SEED \
    --animate
done
