#!/bin/bash

cd "$(dirname "$0")"

echo "Running post processing for cpp output to npz..."

for seed in {0..9}; do
    echo "Running seed $seed..."
    python3 pf_post_processor.py --seed $seed
done

echo "All post processing completed."