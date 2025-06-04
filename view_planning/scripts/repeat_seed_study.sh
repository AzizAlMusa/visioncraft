#!/bin/bash

OPT_K_ATTR=10.0
OPT_K_REP=0.25
NBV_DIR="./results2/nbv_seeds"
RAND_DIR="./results2/random_seeds"
mkdir -p $NBV_DIR
mkdir -p $RAND_DIR

for SEED in $(seq 0 10); do
  python figure2_data.py \
    --strategy nbv \
    --smart_nbv_insertion \
    --potential_type log \
    --k_attr $OPT_K_ATTR \
    --k_rep $OPT_K_REP \
    --save_dir "$NBV_DIR/seed${SEED}" \
    --seed $SEED \
    --verbose

  python figure2_data.py \
    --strategy random \
    --smart_nbv_insertion \
    --potential_type log \
    --k_attr $OPT_K_ATTR \
    --k_rep $OPT_K_REP \
    --save_dir "$RAND_DIR/seed${SEED}" \
    --seed $SEED \
    --verbose
done
