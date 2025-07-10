#!/bin/bash

CONFIG_FILE="dataset_identities.yaml"

# PBMC
DATASET="pbmc"
for heldout_dir in estimates/pbmc/heldout_*; do
  train_csv="$heldout_dir/filtered_counts_train.csv"
  test_csv="$heldout_dir/filtered_counts_test.csv"
  for config_dir in "$heldout_dir"/*_topic_fit; do
    if [ -d "$config_dir" ]; then
      n_topics=$(basename "$config_dir" | grep -oE '^[0-9]+')
      n_identities=$(awk -v ds="$DATASET" '
        $0 ~ "^"ds":" {in_ds=1}
        in_ds && $0 ~ "^  identities:" {in_id=1; next}
        in_ds && $0 ~ "^[^ ]" && !($0 ~ "^"ds":") {in_ds=0; in_id=0}
        in_id && $0 ~ "^    - " {count++}
        in_id && $0 !~ "^    - " && $0 !~ "^$" {exit}
        END {print count+0}
      ' "$CONFIG_FILE")
      n_extra_topics=$((n_topics - n_identities))
      echo "DEBUG: n_topics: $n_topics, n_identities: $n_identities, n_extra_topics: $n_extra_topics"
      echo "Running: $config_dir ($n_extra_topics extra topics)"
      python3 scripts/shared/evaluate_models.py \
        --counts_csv "$train_csv" \
        --test_csv "$test_csv" \
        --output_dir "$config_dir" \
        --n_extra_topics "$n_extra_topics" \
        --dataset "$DATASET" \
        --config_file "$CONFIG_FILE"
    fi
  done
done

# GLIOMA
DATASET="glioma"
for config_dir in estimates/glioma/*_topic_fit; do
  if [ -d "$config_dir" ]; then
    train_csv="$config_dir/filtered_counts_train.csv"
    test_csv="$config_dir/filtered_counts_test.csv"
    n_topics=$(basename "$config_dir" | grep -oE '^[0-9]+')
    n_identities=$(awk -v ds="$DATASET" '
      $0 ~ "^"ds":" {in_ds=1}
      in_ds && $0 ~ "^  identities:" {in_id=1; next}
      in_ds && $0 ~ "^[^ ]" && !($0 ~ "^"ds":") {in_ds=0; in_id=0}
      in_id && $0 ~ "^    - " {count++}
      in_id && $0 !~ "^    - " && $0 !~ "^$" {exit}
      END {print count+0}
    ' "$CONFIG_FILE")
    n_extra_topics=$((n_topics - n_identities))
    echo "DEBUG: n_topics: $n_topics, n_identities: $n_identities, n_extra_topics: $n_extra_topics"
    echo "Running: $config_dir ($n_extra_topics extra topics)"
    python3 scripts/shared/evaluate_models.py \
      --counts_csv "$train_csv" \
      --test_csv "$test_csv" \
      --output_dir "$config_dir" \
      --n_extra_topics "$n_extra_topics" \
      --dataset "$DATASET" \
      --config_file "$CONFIG_FILE"
  fi
done

# ANALYZE ALL FITS
echo "Running analyze_all_fits.py for PBMC..."
python3 scripts/shared/analyze_all_fits.py --base_dir estimates/pbmc --config_file "$CONFIG_FILE"

echo "Running analyze_all_fits.py for GLIOMA..."
python3 scripts/shared/analyze_all_fits.py --base_dir estimates/glioma --config_file "$CONFIG_FILE" 