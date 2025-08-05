#!/bin/bash

set -e  # Exit on any error

# Sweep DE_LOGNORM_MEAN from 0.1 to 1.0 in increments of 0.1
BASE_OUT="data/ABCD_V1V2"
mkdir -p "$BASE_OUT"
for DE_MEAN in $(seq 0.1 0.1 1.5); do
    OUTDIR="$BASE_OUT/DE_mean_${DE_MEAN}"
    echo "Running simulate_counts.py with DE_LOGNORM_MEAN=$DE_MEAN -> $OUTDIR"
    python3 scripts/simulation/simulate_counts.py \
        --n-genes 2000 \
        --identities '["A", "B", "C", "D"]' \
        --activity-topics '["V1", "V2"]' \
        --cells-per-identity 3000 \
        --dirichlet-params '[8,8,8]' \
        --activity-frac 0.3 \
        --n-de 250 \
        --de-mean $DE_MEAN \
        --de-sigma 0.4 \
        --out "$OUTDIR"
done

echo "\nAll simulate_counts.py runs complete!" 