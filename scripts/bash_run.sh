#!/bin/bash

# This assumes you have your data files ready and are running from the root directory

# Set your data paths here (relative to root directory)
COUNTS_CSV="data/pbmc/filtered_counts_train.csv"
TEST_CSV="data/pbmc/filtered_counts_test.csv"
OUTPUT_DIR="estimates/pbmc/9_topic_fit"

# PBMC cell types (identity topics)
IDENTITY_TOPICS="T cells,CD19+ B,CD56+ NK,CD34+,Dendritic,CD14+ Monocyte"

# Number of extra activity topics (V1, V2)
N_EXTRA_TOPICS=3

echo "Running 9-topic fit with updated parameters..."
echo "Working directory: $(pwd)"
echo "Counts file: $COUNTS_CSV"
echo "Test file: $TEST_CSV"
echo "Output directory: $OUTPUT_DIR"
echo "Identity topics: $IDENTITY_TOPICS"
echo "Extra topics: $N_EXTRA_TOPICS"

echo "Start time: $(date)"

# Check if we're in the right directory (should have scripts/ subdirectory)
if [ ! -d "scripts" ]; then
    echo "Error: scripts/ directory not found. Please run this script from the root directory."
    exit 1
fi

# Check if data files exist
if [ ! -f "$COUNTS_CSV" ]; then
    echo "Error: Counts file not found at $COUNTS_CSV"
    exit 1
fi

if [ ! -f "$TEST_CSV" ]; then
    echo "Error: Test file not found at $TEST_CSV"
    exit 1
fi

# Run all models in sequence (from scripts directory)
cd scripts
python3 run_all_models.py \
    --counts_csv "../$COUNTS_CSV" \
    --test_csv "../$TEST_CSV" \
    --output_dir "../$OUTPUT_DIR" \
    --identity_topics "$IDENTITY_TOPICS" \
    --n_extra_topics $N_EXTRA_TOPICS

echo "Done! Check the output directory for results." 