#!/usr/bin/env python3
"""
PBMC Pipeline: Run HLDA, LDA, and NMF models with different held-out cell counts.
Creates new train/test splits and runs the full pipeline for each configuration.
"""

import argparse
import subprocess
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
import random
from sklearn.model_selection import train_test_split
import yaml

# Import default parameters from shared fit_hlda
sys.path.append(str(Path(__file__).parent.parent / "shared"))
from fit_hlda import get_default_parameters

# Get default HLDA parameters
HLDA_PARAMS = get_default_parameters()

# Override with actual parameters used
HLDA_PARAMS.update({
    'n_loops': 15000,
    'burn_in': 5000,
    'thin': 40
})

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        print(f"✓ Completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"✗ Failed after {elapsed:.1f}s (error code: {e.returncode})")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def create_train_test_split(counts_df, heldout_cells, random_state=42):
    """
    Create train/test split with specified number of held-out cells using stratified sampling by cell type.
    The heldout_cells parameter represents the maximum number of cells to hold out PER cell type.
    Each cell type can contribute up to heldout_cells (or 20% of that cell type, whichever is smaller).
    
    Args:
        counts_df: Full count matrix DataFrame
        heldout_cells: Maximum number of cells to hold out PER cell type
        random_state: Random seed for reproducibility
    
    Returns:
        train_df, test_df: Train and test DataFrames
    """
    print(f"Creating stratified train/test split with up to {heldout_cells} held-out cells PER cell type...")
    print(f"  20% limit computed within each cell type individually")
    
    # Extract cell types from index (assuming format like "T cells_1", "CD19+ B_2", etc.)
    cell_types = []
    for idx in counts_df.index:
        # Split by underscore and take the first part as cell type
        cell_type = idx.split('_')[0]
        cell_types.append(cell_type)
    
    # Create a Series with cell types for stratification
    cell_type_series = pd.Series(cell_types, index=counts_df.index)
    
    print(f"  Cell types found: {sorted(cell_type_series.unique())}")
    print(f"  Cell type distribution:")
    for cell_type in sorted(cell_type_series.unique()):
        count = (cell_type_series == cell_type).sum()
        percentage = count / len(counts_df) * 100
        print(f"    {cell_type}: {count} cells ({percentage:.1f}%)")
    
    # Calculate held-out cells per cell type
    # For each cell type, we'll hold out the smaller of:
    # 1. The requested heldout_cells per cell type
    # 2. 20% of that cell type's total cells
    cell_type_counts = cell_type_series.value_counts()
    
    # Calculate 20% limits per cell type
    twenty_percent_limits = {}
    for cell_type, count in cell_type_counts.items():
        twenty_percent_limit = int(count * 0.2)
        twenty_percent_limits[cell_type] = twenty_percent_limit
    
    # Determine actual held-out cells per cell type
    actual_heldout_per_celltype = {}
    for cell_type in cell_type_counts.index:
        requested_per_celltype = heldout_cells
        twenty_percent = twenty_percent_limits[cell_type]
        actual_heldout = min(requested_per_celltype, twenty_percent)
        actual_heldout_per_celltype[cell_type] = actual_heldout
    
    # Calculate total actual held-out cells
    total_actual_heldout = sum(actual_heldout_per_celltype.values())
    
    print(f"  Held-out cell calculation per cell type:")
    for cell_type in sorted(cell_type_counts.index):
        total_count = cell_type_counts[cell_type]
        requested = heldout_cells
        twenty_percent = twenty_percent_limits[cell_type]
        actual = actual_heldout_per_celltype[cell_type]
        print(f"    {cell_type}: {total_count} total, {requested} requested, {twenty_percent} (20%), → {actual} held-out")
    
    print(f"  Maximum requested held-out per cell type: {heldout_cells}")
    print(f"  Total actual held-out across all cell types: {total_actual_heldout}")
    
    # Perform stratified split within each cell type
    train_dfs = []
    test_dfs = []
    
    for cell_type in sorted(cell_type_counts.index):
        # Get data for this cell type
        cell_type_mask = cell_type_series == cell_type
        cell_type_df = counts_df[cell_type_mask]
        
        # Calculate test size for this cell type
        test_size = actual_heldout_per_celltype[cell_type]
        test_size_fraction = test_size / len(cell_type_df)
        
        if test_size > 0:
            # Split this cell type
            cell_type_train, cell_type_test = train_test_split(
                cell_type_df,
                test_size=test_size_fraction,
                random_state=random_state,
                shuffle=True
            )
            train_dfs.append(cell_type_train)
            test_dfs.append(cell_type_test)
        else:
            # No cells to hold out for this cell type
            train_dfs.append(cell_type_df)
    
    # Combine all train and test DataFrames
    train_df = pd.concat(train_dfs, axis=0)
    test_df = pd.concat(test_dfs, axis=0)
    
    print(f"  Train set: {len(train_df)} cells")
    print(f"  Test set: {len(test_df)} cells")
    
    # Verify stratification worked correctly
    print(f"  Cell type distribution in train set:")
    train_cell_types = pd.Series([idx.split('_')[0] for idx in train_df.index])
    for cell_type in sorted(train_cell_types.unique()):
        count = (train_cell_types == cell_type).sum()
        percentage = count / len(train_df) * 100
        print(f"    {cell_type}: {count} cells ({percentage:.1f}%)")
    
    print(f"  Cell type distribution in test set:")
    test_cell_types = pd.Series([idx.split('_')[0] for idx in test_df.index])
    for cell_type in sorted(test_cell_types.unique()):
        count = (test_cell_types == cell_type).sum()
        percentage = count / len(test_df) * 100
        print(f"    {cell_type}: {count} cells ({percentage:.1f}%)")
    
    return train_df, test_df

def run_topic_configuration(train_csv, test_csv, output_dir, n_extra_topics, identity_topics, 
                          dataset, config_file, skip_hlda=False, skip_lda_nmf=False, skip_evaluation=False):
    """
    Run the full pipeline for a specific topic configuration.
    
    Args:
        train_csv: Path to training data CSV
        test_csv: Path to test data CSV
        output_dir: Output directory for this configuration
        n_extra_topics: Number of extra activity topics
        identity_topics: Comma-separated list of identity topics
        skip_*: Flags to skip specific steps
    
    Returns:
        success_count, total_commands: Count of successful commands
    """
    print(f"\n{'='*60}")
    print(f"Running {n_extra_topics + len(identity_topics.split(','))}-topic configuration")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    # Calculate total topics
    n_identity = len(identity_topics.split(','))
    total_topics = n_identity + n_extra_topics
    
    print(f"Configuration: {n_identity} identity + {n_extra_topics} activity = {total_topics} total topics")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    total_commands = 0
    
    # Get path to shared scripts
    shared_dir = Path(__file__).parent.parent / "shared"
    
    # 1. Run HLDA
    if not skip_hlda:
        total_commands += 1
        hlda_cmd = [
            sys.executable, str(shared_dir / "fit_hlda.py"),
            "--counts_csv", str(train_csv),
            "--n_extra_topics", str(n_extra_topics),
            "--output_dir", str(output_dir / "HLDA"),
            "--n_loops", str(HLDA_PARAMS['n_loops']),
            "--burn_in", str(HLDA_PARAMS['burn_in']), 
            "--thin", str(HLDA_PARAMS['thin']),
            "--dataset", dataset,
            "--config_file", config_file
        ]
        if run_command(hlda_cmd, "HLDA Gibbs Sampling"):
            success_count += 1
    
    # 2. Run LDA and NMF
    if not skip_lda_nmf:
        total_commands += 1
        lda_nmf_cmd = [
            sys.executable, str(shared_dir / "fit_lda_nmf.py"),
            "--counts_csv", str(train_csv),
            "--n_topics", str(total_topics),
            "--output_dir", str(output_dir),
            "--model", "both",
            "--max_iter", "500"
        ]
        if run_command(lda_nmf_cmd, "LDA and NMF Fitting"):
            success_count += 1
    
    # 3. Run evaluation
    if not skip_evaluation:
        total_commands += 1
        eval_cmd = [
            sys.executable, str(shared_dir / "evaluate_models.py"),
            "--counts_csv", str(train_csv),
            "--test_csv", str(test_csv),
            "--output_dir", str(output_dir),
            "--n_extra_topics", str(n_extra_topics),
            "--dataset", dataset,
            "--config_file", config_file
        ]
        if run_command(eval_cmd, "Model Evaluation"):
            success_count += 1
    
    return success_count, total_commands

def main():
    parser = argparse.ArgumentParser(description="PBMC pipeline for running models with different held-out cell counts.")
    parser.add_argument("--input_csv", type=str, required=True,
                       help="Path to full count matrix CSV (will be split into train/test)")
    parser.add_argument("--base_output_dir", type=str, default="estimates/pbmc",
                       help="Base directory for outputs (default: estimates/pbmc)")
    parser.add_argument("--heldout_counts", type=str, default="1000,1500",
                       help="Comma-separated list of held-out cell counts (default: 1000,1500)")
    parser.add_argument("--topic_configs", type=str, default="7,8,9",
                       help="Comma-separated list of total topic counts (default: 7,8,9)")
    parser.add_argument("--dataset", type=str, default="pbmc",
                       help="Dataset name (default: pbmc)")
    parser.add_argument("--config_file", type=str, default="../dataset_identities.yaml",
                       help="Path to dataset identity config YAML file")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--skip_hlda", action="store_true",
                       help="Skip HLDA fitting (useful for testing)")
    parser.add_argument("--skip_lda_nmf", action="store_true",
                       help="Skip LDA/NMF fitting (useful for testing)")
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Skip model evaluation (useful for testing)")
    parser.add_argument("--skip_data_splitting", action="store_true",
                       help="Skip data splitting (use existing train/test files)")
    parser.add_argument("--existing_train_csv", type=str,
                       help="Path to existing train CSV (if skipping data splitting)")
    parser.add_argument("--existing_test_csv", type=str,
                       help="Path to existing test CSV (if skipping data splitting)")
    
    args = parser.parse_args()
    
    # Load identity topics from config file
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    if args.dataset not in config:
        raise ValueError(f"Dataset '{args.dataset}' not found in config file {args.config_file}")
    identity_topics = ','.join(config[args.dataset]['identities'])
    
    # Parse arguments
    heldout_counts = [int(x.strip()) for x in args.heldout_counts.split(',')]
    topic_configs = [int(x.strip()) for x in args.topic_configs.split(',')]
    base_output_dir = Path(args.base_output_dir)
    
    print(f"PBMC Pipeline Configuration:")
    print(f"  Input CSV: {args.input_csv}")
    print(f"  Base output directory: {base_output_dir}")
    print(f"  Held-out cell counts: {heldout_counts}")
    print(f"  Topic configurations: {topic_configs}")
    print(f"  Identity topics: {identity_topics}")
    print(f"  Random state: {args.random_state}")
    
    # Set random seed for reproducibility
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    
    # Load the full count matrix
    if not args.skip_data_splitting:
        print(f"\nLoading count matrix from {args.input_csv}...")
        counts_df = pd.read_csv(args.input_csv, index_col=0)
        print(f"Loaded {len(counts_df)} cells with {len(counts_df.columns)} genes")
    
    # Run pipeline for each held-out cell count
    total_success = 0
    total_commands = 0
    
    for heldout_cells in heldout_counts:
        print(f"\n{'='*80}")
        print(f"PROCESSING HELD-OUT COUNT: {heldout_cells}")
        print(f"{'='*80}")
        
        # Create data splits
        if args.skip_data_splitting:
            if not args.existing_train_csv or not args.existing_test_csv:
                print("Error: Must provide existing_train_csv and existing_test_csv when skipping data splitting")
                sys.exit(1)
            train_csv = Path(args.existing_train_csv)
            test_csv = Path(args.existing_test_csv)
            print(f"Using existing splits:")
            print(f"  Train: {train_csv}")
            print(f"  Test: {test_csv}")
        else:
            # Create new train/test split
            train_df, test_df = create_train_test_split(counts_df, heldout_cells, args.random_state)
            
            # Save splits
            heldout_dir = base_output_dir / f"heldout_{heldout_cells}"
            heldout_dir.mkdir(parents=True, exist_ok=True)
            
            train_csv = heldout_dir / "filtered_counts_train.csv"
            test_csv = heldout_dir / "filtered_counts_test.csv"
            
            print(f"Saving train/test splits...")
            train_df.to_csv(train_csv)
            test_df.to_csv(test_csv)
            print(f"  Train saved: {train_csv}")
            print(f"  Test saved: {test_csv}")
        
        # Run each topic configuration
        for total_topics in topic_configs:
            n_extra_topics = total_topics - len(identity_topics.split(','))
            if n_extra_topics < 0:
                print(f"Warning: Skipping {total_topics}-topic config (not enough topics for {len(identity_topics.split(','))} identities)")
                continue
            
            # Create output directory for this configuration
            config_output_dir = base_output_dir / f"heldout_{heldout_cells}" / f"{total_topics}_topic_fit"
            
            # Run the configuration
            success_count, commands_count = run_topic_configuration(
                train_csv=train_csv,
                test_csv=test_csv,
                output_dir=config_output_dir,
                n_extra_topics=n_extra_topics,
                identity_topics=identity_topics,
                dataset=args.dataset,
                config_file=args.config_file,
                skip_hlda=args.skip_hlda,
                skip_lda_nmf=args.skip_lda_nmf,
                skip_evaluation=args.skip_evaluation
            )
            total_success += success_count
            total_commands += commands_count
        
        # After all topic configs for this heldout count, run analyze_all_fits
        heldout_dir = base_output_dir / f"heldout_{heldout_cells}"
        shared_dir = Path(__file__).parent.parent / "shared"
        analyze_cmd = [
            sys.executable, str(shared_dir / "analyze_all_fits.py"), 
            "--base_dir", str(heldout_dir),
            "--topic_configs", args.topic_configs
        ]
        run_command(analyze_cmd, f"Analyze all fits for heldout_{heldout_cells}")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"PBMC PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"Total commands run: {total_commands}")
    print(f"Successful commands: {total_success}")
    print(f"Success rate: {total_success/total_commands*100:.1f}%")
    
    if total_success == total_commands:
        print(f"✓ All configurations completed successfully!")
    else:
        print(f"✗ Some configurations failed.")
        sys.exit(1)
    
    # Print output directory structure
    print(f"\nOutput directory structure:")
    for heldout_cells in heldout_counts:
        heldout_dir = base_output_dir / f"heldout_{heldout_cells}"
        print(f"  {heldout_dir}/")
        for total_topics in topic_configs:
            n_extra_topics = total_topics - len(identity_topics.split(','))
            if n_extra_topics >= 0:
                config_dir = heldout_dir / f"{total_topics}_topic_fit"
                print(f"    {total_topics}_topic_fit/")
                print(f"      ├── HLDA/")
                print(f"      ├── LDA/")
                print(f"      ├── NMF/")
                print(f"      └── plots/")

if __name__ == "__main__":
    main() 