# Scripts Directory - Organized Structure

This directory contains all scripts organized by functionality and purpose.

## 📁 Directory Structure

### `core/` - Core Functions and Utilities
- **`evaluation.py`** - Core evaluation functions (topic matching, parameter comparison)
- **`noise_analysis.py`** - Noise analysis and signal-to-noise ratio calculations

### `datasets/` - Dataset-Specific Scripts
- **`simulation/`** - Simulation dataset scripts
  - `config.py` - Unified configuration system
  - `generate_data.py` - Data generation
  - `run_pipelines.py` - Pipeline execution
  - `simulate.py` - Main interface
  - `simulate_counts.py` - Count simulation
  - `simulation_train_test_split.py` - Train/test splitting
- **`cancer/`** - Cancer dataset scripts
- **`glioma/`** - Glioma dataset scripts  
- **`pbmc/`** - PBMC dataset scripts

### `analysis/` - Analysis and Evaluation Scripts
- **`organize_noise_analysis.py`** - Organize noise analysis outputs
- **`cleanup_estimates.py`** - Clean up estimates directory
- **`run_all_evaluations.sh`** - Run evaluations across all datasets
- **`run_pipelines.sh`** - Run pipelines for real datasets
- **`run_visualizations_only.py`** - Generate visualizations
- **`combined_evaluation.py`** - Combined evaluation functions

### `experiments/` - Experimental and Research Scripts
- **`simulation/`** - Simulation experiments
  - `test_hlda_sampling_methods.py` - HLDA sampling experiments

### `shared/` - Shared Model Fitting Scripts (Keep for Dependencies)
- **`fit_hlda.py`** - HLDA model fitting (numba dependencies)
- **`fit_lda_nmf.py`** - LDA and NMF model fitting
- **`evaluate_models.py`** - Model evaluation
- **`analyze_all_fits.py`** - Analysis of all model fits
- **`noise.py`** - Noise analysis utilities
- **`test_signal_noise.py`** - Signal/noise testing

### `legacy/` - Old Scripts (Deprecated)
- **`simulation/`** - Old simulation shell scripts
  - `generate_counts_*.sh` - Old data generation scripts
  - `run_simulation_pipelines*.sh` - Old pipeline scripts

### `simulation/` - **REMOVED** (Moved to legacy)
- All simulation scripts moved to `scripts/legacy/simulation/`
- New unified system in `scripts/datasets/simulation/` replaces these

## 🚀 Quick Start

### Simulation Data Generation
```bash
# List available datasets
python3 scripts/datasets/simulation/simulate.py list

# Generate data for a dataset
python3 scripts/datasets/simulation/simulate.py generate AB_V1

# Run pipeline for a dataset
python3 scripts/datasets/simulation/simulate.py pipeline ABCD_V1V2
```

### Real Dataset Analysis
```bash
# Run PBMC pipeline
bash scripts/analysis/run_pipelines.sh pbmc --input_csv data/pbmc/filtered_counts.csv

# Run glioma pipeline
bash scripts/analysis/run_pipelines.sh glioma --train_csv data/glioma/train.csv --test_csv data/glioma/test.csv
```

### Analysis and Cleanup
```bash
# Organize noise analysis files
python3 scripts/analysis/organize_noise_analysis.py

# Clean up estimates directory
python3 scripts/analysis/cleanup_estimates.py --analyze

# Find duplicate files
python3 scripts/analysis/cleanup_estimates.py --find-duplicates
```

## 📊 Key Improvements

### ✅ **Completed**
1. **Unified Simulation System** - Single interface for all simulation operations
2. **Organized Dataset Scripts** - Each dataset type has its own directory
3. **Core Module Extraction** - Common functions moved to core modules
4. **Noise Analysis Organization** - All noise analysis files properly organized
5. **Legacy Cleanup** - Old scripts moved to legacy directory

### 🔄 **In Progress**
1. **Estimates Directory Cleanup** - 23GB of data needs organization
2. **Core Module Consolidation** - Some functions still need to be moved
3. **Documentation Updates** - Need to update all import paths

### 📋 **Next Steps**
1. **Complete Core Extraction** - Move remaining pure functions to core
2. **Estimates Cleanup** - Remove duplicates and organize files
3. **Update Import Paths** - Fix all broken imports after reorganization
4. **Create Migration Guide** - Help users transition to new structure

## 🔧 Dependencies

### Core Dependencies (Keep in shared/)
- **`fit_hlda.py`** - Requires numba for performance
- **`fit_lda_nmf.py`** - Model-specific fitting logic
- **`evaluate_models.py`** - Complex evaluation pipeline

### Pure Functions (Moved to core/)
- **`analyze_all_fits.py`** - Pure analysis functions
- **`noise.py`** - Pure noise analysis utilities
- **`test_signal_noise.py`** - Pure testing functions

## 📈 Benefits

1. **Better Organization** - Clear separation of concerns
2. **Easier Maintenance** - Related functions grouped together
3. **Improved Discoverability** - Easy to find relevant scripts
4. **Reduced Duplication** - Common functions in core modules
5. **Cleaner Interface** - Simple commands for common tasks

## 🚨 Migration Notes

### Breaking Changes
- Old shell scripts moved to `legacy/simulation/`
- Some import paths may need updating
- Dataset scripts moved to `datasets/[dataset_type]/`

### Backward Compatibility
- Original functionality preserved
- Old scripts still available in legacy directory
- New unified interface provides same capabilities

## 📝 Contributing

When adding new scripts:
1. **Core functions** → `core/`
2. **Dataset-specific** → `datasets/[dataset_type]/`
3. **Analysis scripts** → `analysis/`
4. **Experiments** → `experiments/`
5. **Model fitting** → `shared/` (if has dependencies)

Follow the existing naming conventions and add appropriate documentation. 