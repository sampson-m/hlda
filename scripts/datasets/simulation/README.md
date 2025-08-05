# Unified Simulation System

This directory contains the new unified simulation system that replaces the scattered shell scripts with a clean, maintainable Python interface.

## Overview

The new system provides:
- **Single configuration source**: All simulation parameters in one place
- **Unified interface**: Same commands for all datasets
- **Type safety**: Configuration validation and error checking
- **Easy extensibility**: Simple to add new datasets
- **Better documentation**: Built-in help and examples

## Quick Start

### List Available Datasets
```bash
python3 scripts/datasets/simulation/simulate.py list
```

### Generate Simulation Data
```bash
# Generate all DE means for a dataset
python3 scripts/datasets/simulation/simulate.py generate AB_V1

# Generate specific DE means
python3 scripts/datasets/simulation/simulate.py generate ABCD_V1V2 --de-means 0.1 0.2 0.3

# Force overwrite existing data
python3 scripts/datasets/simulation/simulate.py generate ABCD_V1V2_new --force
```

### Run Model Fitting Pipeline
```bash
# Run pipeline for all DE means
python3 scripts/datasets/simulation/simulate.py pipeline ABCD_V1V2

# Run pipeline for specific DE means
python3 scripts/datasets/simulation/simulate.py pipeline ABCD_V1V2 --de-means DE_mean_0.1 DE_mean_0.2
```

## Available Datasets

### AB_V1
- **Identities**: A, B
- **Activity topics**: V1
- **DE means**: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
- **Total topics**: 3

### ABCD_V1V2
- **Identities**: A, B, C, D
- **Activity topics**: V1, V2
- **DE means**: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5
- **Total topics**: 6

### ABCD_V1V2_new
- **Identities**: A, B, C, D
- **Activity topics**: V1, V2
- **DE means**: 0.1, 0.2, 0.3, 0.4, 0.5
- **Total topics**: 6
- **Special features**: Different Dirichlet prior [8, 2, 2], more DE genes (400)

## System Architecture

### `config.py`
Central configuration system defining all simulation parameters:
- `SimulationConfig`: Parameters for data generation
- `PipelineConfig`: Parameters for model fitting
- Validation functions to ensure consistency

### `generate_data.py`
Unified data generation script:
- Uses configuration from `config.py`
- Calls the underlying `simulate_counts.py`
- Provides consistent interface across datasets

### `run_pipelines.py`
Unified pipeline runner:
- Fits HLDA, LDA, and NMF models
- Evaluates model performance
- Analyzes results
- Restores cell names for simulation data

### `simulate.py`
Simple wrapper providing easy access to all functionality:
- Command-line interface with subcommands
- Built-in help and examples
- Error handling and validation

## Adding New Datasets

To add a new dataset:

1. **Add simulation configuration** in `config.py`:
```python
"NEW_DATASET": SimulationConfig(
    name="NEW_DATASET",
    identities=["A", "B", "C"],
    activity_topics=["V1"],
    dirichlet_params=[8, 8, 8],
    activity_fraction=0.3,
    cells_per_identity=3000,
    n_genes=2000,
    n_de_genes=250,
    de_sigma=0.4,
    de_means=[0.1, 0.2, 0.3, 0.4, 0.5]
)
```

2. **Add pipeline configuration** in `config.py`:
```python
"NEW_DATASET": PipelineConfig(
    dataset_name="NEW_DATASET",
    data_root="data/NEW_DATASET",
    estimates_root="estimates/NEW_DATASET",
    topic_config="4_topic_fit",
    n_identity_topics=3,
    n_extra_topics=1,
    n_total_topics=4
)
```

3. **Validate configuration**:
```bash
python3 scripts/datasets/simulation/config.py
```

4. **Use the new dataset**:
```bash
python3 scripts/datasets/simulation/simulate.py generate NEW_DATASET
python3 scripts/datasets/simulation/simulate.py pipeline NEW_DATASET
```

## Configuration Details

### Simulation Parameters
- `identities`: List of identity topic names
- `activity_topics`: List of activity topic names
- `dirichlet_params`: Dirichlet prior parameters for topic proportions
- `activity_fraction`: Fraction of cells with activity topics
- `cells_per_identity`: Number of cells per identity topic
- `n_genes`: Total number of genes
- `n_de_genes`: Number of differentially expressed genes
- `de_sigma`: Standard deviation of DE effect sizes
- `de_means`: List of DE mean values to simulate

### Pipeline Parameters
- `data_root`: Directory containing simulation data
- `estimates_root`: Directory for model estimates
- `topic_config`: Name of topic configuration
- `n_identity_topics`: Number of identity topics
- `n_extra_topics`: Number of extra topics for HLDA
- `n_total_topics`: Total number of topics for LDA/NMF

## Error Handling

The system provides comprehensive error handling:
- **Configuration validation**: Checks for consistency between simulation and pipeline configs
- **File existence checks**: Verifies data files exist before processing
- **Subprocess error handling**: Captures and reports errors from underlying scripts
- **Graceful degradation**: Continues processing other DE means if one fails

## Migration from Old System

The old scattered shell scripts have been moved to `scripts/legacy/simulation/`. See `scripts/legacy/README.md` for migration details.

## Troubleshooting

### Common Issues

1. **Configuration errors**: Run `python3 scripts/datasets/simulation/config.py` to validate
2. **Missing data**: Ensure simulation data exists before running pipeline
3. **Permission errors**: Check file permissions for output directories
4. **Memory issues**: Reduce number of genes or cells for large datasets

### Getting Help

```bash
# General help
python3 scripts/datasets/simulation/simulate.py --help

# Command-specific help
python3 scripts/datasets/simulation/simulate.py generate --help
python3 scripts/datasets/simulation/simulate.py pipeline --help

# List available datasets
python3 scripts/datasets/simulation/simulate.py list
``` 