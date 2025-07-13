#!/bin/bash

CONFIG_FILE="dataset_identities.yaml"

echo "=========================================="
echo "Running evaluations across all datasets"
echo "=========================================="

# Function to count identities for a dataset
count_identities() {
    local dataset=$1
    # Find the section for this dataset and count the identity lines
    sed -n "/^${dataset}:/,/^[a-zA-Z]/p" "$CONFIG_FILE" | grep -c "^    - " 2>/dev/null || echo "0"
}

# Function to run evaluations for a dataset
run_evaluations() {
    local base_dir=$1
    local dataset=$2
    local train_csv=$3
    local test_csv=$4
    
    echo "----------------------------------------"
    echo "Processing dataset: $dataset"
    echo "Base directory: $base_dir"
    echo "Train CSV: $train_csv"
    echo "Test CSV: $test_csv"
    echo "----------------------------------------"
    
    # Check if data files exist
    if [ ! -f "$train_csv" ]; then
        echo "WARNING: Train CSV not found: $train_csv"
        return 1
    fi
    if [ ! -f "$test_csv" ]; then
        echo "WARNING: Test CSV not found: $test_csv"
        return 1
    fi
    
    # Count identities for this dataset
    n_identities=$(count_identities "$dataset")
    echo "Dataset $dataset has $n_identities identities"
    
    # Get topic configurations
    topic_configs=()
    for config_dir in $base_dir/*_topic_fit; do
        if [ -d "$config_dir" ]; then
            n_topics=$(basename "$config_dir" | grep -oE '^[0-9]+')
            topic_configs+=("$n_topics")
        fi
    done
    
    if [ ${#topic_configs[@]} -eq 0 ]; then
        echo "WARNING: No topic configurations found in $base_dir"
        return 1
    fi
    
    # Sort topic configurations
    IFS=$'\n' topic_configs_sorted=($(sort -n <<<"${topic_configs[*]}"))
    unset IFS
    
    echo "Found topic configurations: ${topic_configs_sorted[*]}"
    
    # Run evaluate_models.py for each configuration
    for config_dir in $base_dir/*_topic_fit; do
        if [ -d "$config_dir" ]; then
            n_topics=$(basename "$config_dir" | grep -oE '^[0-9]+')
            n_extra_topics=$((n_topics - n_identities))
            
            echo "Running evaluation: $config_dir ($n_extra_topics extra topics)"
            
            python3 scripts/shared/evaluate_models.py \
                --counts_csv "$train_csv" \
                --test_csv "$test_csv" \
                --output_dir "$config_dir" \
                --n_extra_topics "$n_extra_topics" \
                --dataset "$dataset" \
                --config_file "$CONFIG_FILE"
            
            if [ $? -ne 0 ]; then
                echo "ERROR: evaluate_models.py failed for $config_dir"
            else
                echo "SUCCESS: evaluate_models.py completed for $config_dir"
            fi
        fi
    done
    
    # Run analyze_all_fits.py
    # topic_configs_str=$(IFS=,; echo "${topic_configs_sorted[*]}")
    # echo "Running analyze_all_fits.py for $dataset with configurations: $topic_configs_str"
    
    # python3 scripts/shared/analyze_all_fits.py \
    #     --base_dir "$base_dir" \
    #     --topic_configs "$topic_configs_str" \
    #     --config_file "$CONFIG_FILE"
    
    # if [ $? -ne 0 ]; then
    #     echo "ERROR: analyze_all_fits.py failed for $dataset"
    # else
    #     echo "SUCCESS: analyze_all_fits.py completed for $dataset"
    # fi
    
    # echo "Completed processing dataset: $dataset"
    # echo ""
}

# ==========================================
# PBMC Dataset
# ==========================================
echo "Starting PBMC dataset evaluations..."

# PBMC has multiple heldout configurations
for heldout_dir in estimates/pbmc/heldout_*; do
    if [ -d "$heldout_dir" ]; then
        heldout_size=$(basename "$heldout_dir" | grep -oE '[0-9]+')
        echo "Processing PBMC heldout_$heldout_size"
        
        if [ "$heldout_size" == "1500" ]; then
            train_csv="data/pbmc/filtered_counts_train.csv"
            test_csv="data/pbmc/filtered_counts_test.csv"
        else
            train_csv="data/pbmc/filtered_counts.csv"
            test_csv="data/pbmc/filtered_counts_test.csv"
        fi
        
        run_evaluations "$heldout_dir" "pbmc" "$train_csv" "$test_csv"
    fi
done

# ==========================================
# Glioma Dataset
# ==========================================
echo "Starting Glioma dataset evaluations..."

run_evaluations "estimates/glioma" "glioma" \
    "data/glioma/glioma_counts_train.csv" \
    "data/glioma/glioma_counts_test.csv"

# ==========================================
# Cancer Combined Dataset
# ==========================================
echo "Starting Cancer Combined dataset evaluations..."

run_evaluations "estimates/cancer/combined" "cancer_combined" \
    "data/cancer/cancer_counts_train_combined.csv" \
    "data/cancer/cancer_counts_test_combined.csv"

# ==========================================
# Cancer Disease-Specific Dataset
# ==========================================
echo "Starting Cancer Disease-Specific dataset evaluations..."

run_evaluations "estimates/cancer/disease_specific" "cancer_disease_specific" \
    "data/cancer/cancer_counts_train_disease_specific.csv" \
    "data/cancer/cancer_counts_test_disease_specific.csv"

# ==========================================
# Summary
# ==========================================
echo "=========================================="
echo "All dataset evaluations completed!"
echo "=========================================="
echo ""
echo "Processed datasets:"
echo "- PBMC (multiple heldout configurations)"
echo "- Glioma"
echo "- Cancer Combined"
echo "- Cancer Disease-Specific"
echo ""
echo "Generated outputs:"
echo "- Model evaluation plots and metrics"
echo "- SSE analysis and heatmaps"
echo "- UMAP visualizations"
echo "- Cosine similarity matrices"
echo "- Theta heatmaps"
echo "- Cross-model comparisons"