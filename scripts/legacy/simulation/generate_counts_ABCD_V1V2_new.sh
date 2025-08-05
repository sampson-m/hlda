#!/bin/bash

set -e

# Configuration for ABCD_V1V2 with new Dirichlet prior [8,2,2]
IDENTITIES='["A", "B", "C", "D"]'
ACTIVITY_TOPICS='["V1", "V2"]'
DIRICHLET_PARAMS='[8, 2, 2]'  # New Dirichlet prior
ACTIVITY_FRAC=0.30
CELLS_PER_IDENTITY=3000
N_GENES=2000
N_DE=400

# Output directory for new data
OUTPUT_DIR="data/ABCD_V1V2_new"

echo "Generating ABCD_V1V2 data with Dirichlet prior [8,2,2]"
echo "Output directory: $OUTPUT_DIR"
echo "DE means: 0.1, 0.2, 0.3, 0.4, 0.5"
echo ""

# Generate data for each DE mean
for DE_MEAN in 0.1 0.2 0.3 0.4 0.5; do
    echo "=== Generating DE_mean_$DE_MEAN ==="
    
    # Create output directory
    OUT_PATH="$OUTPUT_DIR/DE_mean_$DE_MEAN"
    mkdir -p "$OUT_PATH"
    
    # Run simulation
    python3 scripts/simulation/simulate_counts.py \
        --identities "$IDENTITIES" \
        --activity-topics "$ACTIVITY_TOPICS" \
        --dirichlet-params "$DIRICHLET_PARAMS" \
        --activity-frac $ACTIVITY_FRAC \
        --cells-per-identity $CELLS_PER_IDENTITY \
        --n-genes $N_GENES \
        --n-de $N_DE \
        --de-mean $DE_MEAN \
        --de-sigma 0.4 \
        --out "$OUT_PATH" \
        --seed 42
    
    echo "✓ Generated DE_mean_$DE_MEAN"
    echo ""
done

echo "=== Data generation complete! ==="
echo "Generated data in: $OUTPUT_DIR"
echo "DE means: 0.1, 0.2, 0.3, 0.4, 0.5"
echo "Dirichlet prior: [8, 2, 2]" 