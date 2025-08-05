#!/bin/bash

# Single entry point script for running HLDA, LDA, and NMF pipelines
# Supports both PBMC and glioma datasets

set -e  # Exit on any error

# Function to print usage
print_usage() {
    echo "Usage: $0 [pbmc|glioma] [options]"
    echo ""
    echo "Datasets:"
    echo "  pbmc    - Run PBMC pipeline with heldout cell counts"
    echo "  glioma  - Run glioma pipeline with topic configurations"
    echo ""
    echo "PBMC Options:"
    echo "  --input_csv INPUT_CSV          Path to full count matrix CSV"
    echo "  --heldout_counts COUNTS        Comma-separated heldout counts (default: 1000,1500)"
    echo "  --topic_configs CONFIGS        Comma-separated topic counts (default: 7,8,9)"
    echo "  --base_output_dir DIR          Base output directory (default: estimates/pbmc)"
    echo ""
    echo "Glioma Options:"
    echo "  --train_csv TRAIN_CSV          Path to training count matrix CSV"
    echo "  --test_csv TEST_CSV            Path to test count matrix CSV"
    echo "  --topic_configs CONFIGS        Comma-separated topic counts (default: 13,14,15,16)"
    echo "  --base_output_dir DIR          Base output directory (default: estimates/glioma)"
    echo ""
    echo "Common Options:"
    echo "  --config_file FILE             Path to dataset config YAML (default: dataset_identities.yaml)"
    echo "  --skip_hlda                    Skip HLDA fitting"
    echo "  --skip_lda_nmf                 Skip LDA/NMF fitting"
    echo "  --skip_evaluation              Skip model evaluation"
    echo "  --help                         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 pbmc --input_csv data/pbmc/filtered_counts.csv"
    echo "  $0 glioma --train_csv data/glioma/glioma_counts_train.csv --test_csv data/glioma/glioma_counts_test.csv"
}

# Check if we're in the right directory
if [ ! -d "scripts" ]; then
    echo "Error: scripts/ directory not found. Please run this script from the root directory."
    exit 1
fi

# Check if dataset is provided
if [ $# -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    print_usage
    exit 0
fi

DATASET="$1"
shift  # Remove dataset from arguments

# Validate dataset
if [ "$DATASET" != "pbmc" ] && [ "$DATASET" != "glioma" ]; then
    echo "Error: Invalid dataset '$DATASET'. Must be 'pbmc' or 'glioma'."
    print_usage
    exit 1
fi

echo "Running $DATASET pipeline..."
echo "Working directory: $(pwd)"
echo ""

# Set default values based on dataset
if [ "$DATASET" = "pbmc" ]; then
    HELDOUT_COUNTS="1000,1500"
    TOPIC_CONFIGS="7,8,9"
    BASE_OUTPUT_DIR="estimates/pbmc"
    CONFIG_FILE="dataset_identities.yaml"
    
    # Parse PBMC-specific arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --input_csv)
                INPUT_CSV="$2"
                shift 2
                ;;
            --heldout_counts)
                HELDOUT_COUNTS="$2"
                shift 2
                ;;
            --topic_configs)
                TOPIC_CONFIGS="$2"
                shift 2
                ;;
            --base_output_dir)
                BASE_OUTPUT_DIR="$2"
                shift 2
                ;;
            --config_file)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --skip_hlda)
                SKIP_HLDA="--skip_hlda"
                shift
                ;;
            --skip_lda_nmf)
                SKIP_LDA_NMF="--skip_lda_nmf"
                shift
                ;;
            --skip_evaluation)
                SKIP_EVALUATION="--skip_evaluation"
                shift
                ;;
            *)
                echo "Error: Unknown option $1"
                print_usage
                exit 1
                ;;
        esac
    done
    
    # Check required arguments
    if [ -z "$INPUT_CSV" ]; then
        echo "Error: --input_csv is required for PBMC pipeline"
        print_usage
        exit 1
    fi
    
    # Check if input file exists
    if [ ! -f "$INPUT_CSV" ]; then
        echo "Error: Input file not found at $INPUT_CSV"
        exit 1
    fi
    
    echo "PBMC Pipeline Configuration:"
    echo "  Input CSV: $INPUT_CSV"
    echo "  Heldout counts: $HELDOUT_COUNTS"
    echo "  Topic configurations: $TOPIC_CONFIGS"
    echo "  Base output directory: $BASE_OUTPUT_DIR"
    echo "  Config file: $CONFIG_FILE"
    echo ""
    
    # Run PBMC pipeline
    cd scripts/pbmc
    python3 run_pbmc_pipeline.py \
        --input_csv "../../$INPUT_CSV" \
        --heldout_counts "$HELDOUT_COUNTS" \
        --topic_configs "$TOPIC_CONFIGS" \
        --base_output_dir "../../$BASE_OUTPUT_DIR" \
        --config_file "../../$CONFIG_FILE" \
        $SKIP_HLDA $SKIP_LDA_NMF $SKIP_EVALUATION

elif [ "$DATASET" = "glioma" ]; then
    TOPIC_CONFIGS="16"
    BASE_OUTPUT_DIR="estimates/glioma"
    CONFIG_FILE="dataset_identities.yaml"
    
    # Parse glioma-specific arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --train_csv)
                TRAIN_CSV="$2"
                shift 2
                ;;
            --test_csv)
                TEST_CSV="$2"
                shift 2
                ;;
            --topic_configs)
                TOPIC_CONFIGS="$2"
                shift 2
                ;;
            --base_output_dir)
                BASE_OUTPUT_DIR="$2"
                shift 2
                ;;
            --config_file)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --skip_hlda)
                SKIP_HLDA="--skip_hlda"
                shift
                ;;
            --skip_lda_nmf)
                SKIP_LDA_NMF="--skip_lda_nmf"
                shift
                ;;
            --skip_evaluation)
                SKIP_EVALUATION="--skip_evaluation"
                shift
                ;;
            *)
                echo "Error: Unknown option $1"
                print_usage
                exit 1
                ;;
        esac
    done
    
    # Check required arguments
    if [ -z "$TRAIN_CSV" ] || [ -z "$TEST_CSV" ]; then
        echo "Error: --train_csv and --test_csv are required for glioma pipeline"
        print_usage
        exit 1
    fi
    
    # Check if input files exist
    if [ ! -f "$TRAIN_CSV" ]; then
        echo "Error: Train CSV not found at $TRAIN_CSV"
        exit 1
    fi
    
    if [ ! -f "$TEST_CSV" ]; then
        echo "Error: Test CSV not found at $TEST_CSV"
        exit 1
    fi
    
    echo "Glioma Pipeline Configuration:"
    echo "  Train CSV: $TRAIN_CSV"
    echo "  Test CSV: $TEST_CSV"
    echo "  Topic configurations: $TOPIC_CONFIGS"
    echo "  Base output directory: $BASE_OUTPUT_DIR"
    echo "  Config file: $CONFIG_FILE"
    echo ""
    
    # Run glioma pipeline
    cd scripts/glioma
    python3 run_glioma_pipeline.py \
        --train_csv "../../$TRAIN_CSV" \
        --test_csv "../../$TEST_CSV" \
        --topic_configs "$TOPIC_CONFIGS" \
        --base_output_dir "../../$BASE_OUTPUT_DIR" \
        --config_file "../../$CONFIG_FILE" \
        $SKIP_HLDA $SKIP_LDA_NMF $SKIP_EVALUATION
fi

cd ../..

echo ""
echo "Pipeline complete!"
echo "Results saved in: $BASE_OUTPUT_DIR" 