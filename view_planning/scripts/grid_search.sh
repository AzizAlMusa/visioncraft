#!/bin/bash

SAVE_DIR="./results2/grid_search"
mkdir -p $SAVE_DIR

for K_ATTR in $(seq 0 1.0 10); do
  for K_REP in $(seq 0 1.0 10); do
    python figure2_data.py \
      --strategy nbv \
      --smart_nbv_insertion \
      --potential_type log \
      --k_attr $K_ATTR \
      --k_rep $K_REP \
      --save_dir "$SAVE_DIR/kattr${K_ATTR}_krep${K_REP}" \
      --seed 0
  done
done
