# Legacy Scripts

This directory contains old scripts that have been replaced by the new unified system.

## What's Here

### simulation/
- `generate_counts_*.sh` - Old shell scripts for generating simulation data
- `run_simulation_pipelines*.sh` - Old shell scripts for running pipelines

## Why They Were Moved

These scripts had several issues:
1. **Scattered and duplicated**: Multiple similar scripts with confusing names
2. **Hard to maintain**: Configuration scattered across multiple files
3. **Inconsistent naming**: "_new" suffixes and unclear naming conventions
4. **No validation**: No checks for configuration consistency

## What Replaced Them

The new unified system in `scripts/datasets/simulation/` provides:

### `config.py`
- Single source of truth for all simulation configurations
- Type-safe configuration with validation
- Easy to add new datasets

### `generate_data.py`
- Unified data generation for all datasets
- Command-line interface with help
- Better error handling and reporting

### `run_pipelines.py`
- Unified pipeline runner for all datasets
- Consistent interface across datasets
- Better logging and error handling

### `simulate.py`
- Simple wrapper providing easy access to all functionality
- Clean command-line interface
- Comprehensive help and examples

## Migration Guide

### Old Way (Deprecated)
```bash
# Generate data
bash scripts/simulation/generate_counts_ABCD_V1V2_new.sh

# Run pipeline
bash scripts/simulation/run_simulation_pipelines_ABCD_V1V2_new.sh
```

### New Way (Recommended)
```bash
# List available datasets
python3 scripts/datasets/simulation/simulate.py list

# Generate data
python3 scripts/datasets/simulation/simulate.py generate ABCD_V1V2_new

# Run pipeline
python3 scripts/datasets/simulation/simulate.py pipeline ABCD_V1V2_new

# Generate specific DE means
python3 scripts/datasets/simulation/simulate.py generate ABCD_V1V2 --de-means 0.1 0.2 0.3
```

## Benefits of New System

1. **Maintainability**: Single configuration file instead of scattered scripts
2. **Consistency**: Same interface for all datasets
3. **Validation**: Automatic checks for configuration consistency
4. **Extensibility**: Easy to add new datasets
5. **Documentation**: Built-in help and examples
6. **Error Handling**: Better error messages and recovery

## When to Remove

These legacy scripts can be removed once:
1. All existing workflows have been migrated to the new system
2. The new system has been thoroughly tested
3. Team members are comfortable with the new interface 