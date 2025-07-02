# HLDA Scripts - Reorganized Structure

This directory contains scripts for running Hierarchical Latent Dirichlet Allocation (HLDA), LDA, and NMF models on single-cell RNA-seq data, now organized by dataset and functionality.

## New Directory Structure

```
scripts/
├── shared/                    # Core modeling scripts (reusable)
│   ├── fit_hlda.py           # HLDA Gibbs sampling implementation
│   ├── fit_lda_nmf.py        # LDA and NMF fitting
│   ├── evaluate_models.py    # Model evaluation and comparison
│   └── analyze_all_fits.py   # Cross-configuration analysis
├── pbmc/                      # PBMC-specific scripts
│   ├── preprocess_pbmc.py    # PBMC data preprocessing
│   ├── run_pbmc_pipeline.py  # PBMC pipeline with heldout splits
│   └── check_library_sizes.py # Utility for checking library sizes
├── glioma/                    # Glioma-specific scripts
│   ├── preprocess_glioma.py  # Glioma data preprocessing
│   ├── run_glioma_pipeline.py # Glioma pipeline with fixed splits
│   └── inspect_glioma.py     # Utility for inspecting glioma data
└── run_pipelines.sh          # Single entry point for all pipelines
```

## Quick Start

### Single Entry Point (Recommended)

Use the unified entry point script from the **root directory**:

```bash
# Run PBMC pipeline
./scripts/run_pipelines.sh pbmc --input_csv data/pbmc/filtered_counts.csv

# Run glioma pipeline
./scripts/run_pipelines.sh glioma --train_csv data/glioma/glioma_counts_train.csv --test_csv data/glioma/glioma_counts_test.csv

# Get help
./scripts/run_pipelines.sh --help
```

### PBMC Pipeline

The PBMC pipeline creates train/test splits with different heldout cell counts and runs models across multiple topic configurations.

```bash
# Basic usage
./scripts/run_pipelines.sh pbmc --input_csv data/pbmc/filtered_counts.csv

# Custom heldout counts and topic configurations
./scripts/run_pipelines.sh pbmc \
    --input_csv data/pbmc/filtered_counts.csv \
    --heldout_counts "500,1000,1500" \
    --topic_configs "7,8,9,10" \
    --base_output_dir estimates/pbmc_custom
```

### Glioma Pipeline

The glioma pipeline uses pre-existing train/test splits and focuses on topic configuration variation.

```bash
# Basic usage
./scripts/run_pipelines.sh glioma \
    --train_csv data/glioma/glioma_counts_train.csv \
    --test_csv data/glioma/glioma_counts_test.csv

# Custom topic configurations
./scripts/run_pipelines.sh glioma \
    --train_csv data/glioma/glioma_counts_train.csv \
    --test_csv data/glioma/glioma_counts_test.csv \
    --topic_configs "13,14,15,16,17" \
    --base_output_dir estimates/glioma_custom
```

## Individual Script Usage

### Data Preprocessing

#### PBMC Data
```bash
cd scripts/pbmc
python3 preprocess_pbmc.py \
    --data_dir "../../data/pbmc/raw_10x_data" \
    --output_dir "../../data/pbmc" \
    --dataset "pbmc" \
    --config_file "../../dataset_identities.yaml"
```

#### Glioma Data
```bash
cd scripts/glioma
python3 preprocess_glioma.py \
    --dataset "glioma" \
    --config_file "../../dataset_identities.yaml"
```

### Utility Scripts

#### Check Library Sizes
```bash
cd scripts/pbmc
python3 check_library_sizes.py --csv_path "../../data/pbmc/filtered_counts.csv"
```

#### Inspect Glioma Data
```bash
cd scripts/glioma
python3 inspect_glioma.py
```

### Direct Pipeline Execution

#### PBMC Pipeline
```bash
cd scripts/pbmc
python3 run_pbmc_pipeline.py \
    --input_csv "../../data/pbmc/filtered_counts.csv" \
    --heldout_counts "1000,1500" \
    --topic_configs "7,8,9" \
    --base_output_dir "../../estimates/pbmc"
```

#### Glioma Pipeline
```bash
cd scripts/glioma
python3 run_glioma_pipeline.py \
    --train_csv "../../data/glioma/glioma_counts_train.csv" \
    --test_csv "../../data/glioma/glioma_counts_test.csv" \
    --topic_configs "13,14,15,16" \
    --base_output_dir "../../estimates/glioma"
```

## Output Structure

### PBMC Pipeline Output
```
estimates/pbmc/
├── heldout_1000/
│   ├── filtered_counts_train.csv
│   ├── filtered_counts_test.csv
│   ├── 7_topic_fit/
│   │   ├── HLDA/
│   │   ├── LDA/
│   │   ├── NMF/
│   │   └── plots/
│   ├── 8_topic_fit/
│   └── 9_topic_fit/
└── heldout_1500/
    └── ...
```

### Glioma Pipeline Output
```
estimates/glioma/
├── 13_topic_fit/
│   ├── HLDA/
│   ├── LDA/
│   ├── NMF/
│   └── plots/
├── 14_topic_fit/
├── 15_topic_fit/
├── 16_topic_fit/
└── model_comparison/
    ├── train_loglikelihood_matrix.csv
    ├── test_loglikelihood_matrix.csv
    ├── loglikelihood_comparison.png
    ├── overfitting_analysis.csv
    └── comprehensive_sse_heatmap.png
```

## Configuration

Both pipelines use a YAML configuration file (`dataset_identities.yaml`) to define cell type identities for each dataset:

```yaml
pbmc:
  identities:
    - "T cells"
    - "CD19+ B"
    - "CD56+ NK"
    - "CD34+"
    - "Dendritic"
    - "CD14+ Monocyte"

glioma:
  identities:
    - "Oligodendrocyte"
    - "Astrocyte"
    - "Microglia"
    - "Endothelial"
    - "Neuron"
    - "OPC"
    - "T cell"
    - "B cell"
    - "Macrophage"
    - "NK cell"
    - "Neutrophil"
    - "Cancer"
```

## Key Improvements

1. **Organized by Dataset**: Clear separation between PBMC and glioma workflows
2. **Shared Core Code**: Reusable modeling scripts in `shared/` directory
3. **Single Entry Point**: One script to run any pipeline
4. **Eliminated Redundancy**: Removed duplicate pipeline scripts
5. **Better Documentation**: Clear usage examples and structure

## Troubleshooting

- **"scripts/ directory not found"**: Make sure you're running from the root directory
- **Import errors**: The shared scripts use relative imports that should work automatically
- **File not found errors**: Check that your data files exist in the expected locations
- **Permission errors**: Make sure `run_pipelines.sh` is executable: `chmod +x scripts/run_pipelines.sh` 