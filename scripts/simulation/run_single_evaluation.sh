#!/bin/bash

set -e

# Configuration for single DE_MEAN evaluation
DATA_ROOT="data/AB_V1"
EST_ROOT="estimates/AB_V1"
CONFIG_FILE="dataset_identities.yaml"
DE_MEAN="DE_mean_0.75"
TOPIC_CONFIG="3_topic_fit"

# Paths
SIMDIR="$DATA_ROOT/$DE_MEAN"
OUTDIR="$EST_ROOT/$DE_MEAN/$TOPIC_CONFIG"
COUNTS_CSV="$SIMDIR/counts.csv"
TRUE_THETA="$SIMDIR/theta.csv"
TRUE_BETA="$SIMDIR/gene_means.csv"
TRAIN_CSV="$SIMDIR/filtered_counts_train.csv"
TEST_CSV="$SIMDIR/filtered_counts_test.csv"

echo "=== Running evaluation for $DE_MEAN ==="

# Check if required files exist
if [ ! -f "$COUNTS_CSV" ]; then
    echo "Error: $COUNTS_CSV not found. Please run simulation first."
    exit 1
fi

if [ ! -f "$TRAIN_CSV" ]; then
    echo "Error: $TRAIN_CSV not found. Please run train/test split first."
    exit 1
fi

if [ ! -d "$OUTDIR" ]; then
    echo "Error: $OUTDIR not found. Please run model fitting first."
    exit 1
fi

# Run model evaluation
echo "Running model evaluation..."
python3 scripts/shared/evaluate_models.py \
    --counts_csv "$TRAIN_CSV" \
    --test_csv "$TEST_CSV" \
    --output_dir "$OUTDIR" \
    --n_extra_topics 1 \
    --dataset AB_V1 \
    --config_file "$CONFIG_FILE"

# Run comprehensive analysis
echo "Running comprehensive analysis..."
python3 scripts/shared/analyze_all_fits.py \
    --base_dir "$EST_ROOT/$DE_MEAN" \
    --output_dir "$EST_ROOT/$DE_MEAN/3_topic_fit/model_comparison" \
    --topic_configs "3" \
    --config_file "$CONFIG_FILE"

# Generate combined theta scatter plots and noise analysis using new approach
echo "Generating combined scatter plots and noise analysis..."
python3 scripts/simulation/combined_evaluation.py \
    --simulation_dir "$SIMDIR" \
    --output_dir "$OUTDIR" \
    --models HLDA LDA NMF \
    --topic_config "3_topic_fit" \
    --libsize_mean 1500 \
    --n_identity_topics 2

echo "Evaluation complete for $DE_MEAN!"
echo "Results saved to: $EST_ROOT/$DE_MEAN/"
echo "Comprehensive analysis plots saved to: $EST_ROOT/$DE_MEAN/model_comparison/"
echo "Combined theta scatter plot saved to: $OUTDIR/combined_theta_scatter.png"
echo "Noise analysis saved to: $DATA_ROOT/noise_analysis/"