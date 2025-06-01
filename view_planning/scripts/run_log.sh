#!/bin/bash

for SEED in {0..9}
do
  echo "Running log with seed $SEED"
  python figure1_data.py --potential_type log --seed $SEED --save_dir ./results
done
