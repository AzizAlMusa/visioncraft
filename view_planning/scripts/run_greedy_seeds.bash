#!/bin/bash

# Automated experiment runner for greedy view planning algorithm
# Runs 30 seeds for ONLY greedy strategy

# Configuration
OPT_K_ATTR=0.4
OPT_K_REP=1.0
BASE_DIR="./results2"
GREEDY_DIR="$BASE_DIR/greedy_seeds"

# Create directories
mkdir -p $GREEDY_DIR

# Number of seeds to run
NUM_SEEDS=10

echo "Starting automated greedy experiments with $NUM_SEEDS seeds..."
echo "Results will be saved to: $GREEDY_DIR"
echo "Parameters: k_attr=$OPT_K_ATTR, k_rep=$OPT_K_REP"
echo ""

# Progress tracking
current_run=0

# Function to show progress
show_progress() {
    current_run=$((current_run + 1))
    percentage=$((current_run * 100 / NUM_SEEDS))
    echo "Progress: [$current_run/$NUM_SEEDS] ($percentage%)"
}

# Run greedy experiments
echo "=== Running Greedy Strategy ==="
for SEED in $(seq 0 $((NUM_SEEDS-1))); do
    echo "Running greedy with seed $SEED..."
    
    python figure2_greedy.py \
        --strategy greedy \
        --k_attr $OPT_K_ATTR \
        --k_rep $OPT_K_REP \
        --save_dir "$GREEDY_DIR/seed${SEED}" \
        --seed $SEED \
        --verbose
    
    show_progress
done

echo ""
echo "=== Experiments Complete ==="
echo "Greedy results saved to: $GREEDY_DIR"
echo ""
echo "Run post-processing with:"
echo "python post_process_greedy.py --results_dir $BASE_DIR --k_attr $OPT_K_ATTR --k_rep $OPT_K_REP"