#!/usr/bin/env python3
"""
Glioma Pipeline: Run HLDA, LDA, and NMF models with different topic configurations.
Uses pre-existing train/test splits and focuses only on topic variation.
"""

import argparse
import subprocess
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
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
    parser = argparse.ArgumentParser(description="Glioma pipeline for running models with different topic configurations using pre-existing train/test splits.")
    parser.add_argument("--train_csv", type=str, required=True,
                       help="Path to training count matrix CSV")
    parser.add_argument("--test_csv", type=str, required=True,
                       help="Path to test count matrix CSV")
    parser.add_argument("--base_output_dir", type=str, default="estimates/glioma",
                       help="Base directory for outputs (default: estimates/glioma)")
    parser.add_argument("--topic_configs", type=str, default="13,14,15,16",
                       help="Comma-separated list of total topic counts (default: 13,14,15,16)")
    parser.add_argument("--dataset", type=str, default="glioma",
                       help="Dataset name (default: glioma)")
    parser.add_argument("--config_file", type=str, default="../dataset_identities.yaml",
                       help="Path to dataset identity config YAML file")
    parser.add_argument("--skip_hlda", action="store_true",
                       help="Skip HLDA fitting (useful for testing)")
    parser.add_argument("--skip_lda_nmf", action="store_true",
                       help="Skip LDA/NMF fitting (useful for testing)")
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Skip model evaluation (useful for testing)")
    
    args = parser.parse_args()
    
    # Load identity topics from config file
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    if args.dataset not in config:
        raise ValueError(f"Dataset '{args.dataset}' not found in config file {args.config_file}")
    identity_topics = ','.join(config[args.dataset]['identities'])
    
    # Parse arguments
    topic_configs = [int(x.strip()) for x in args.topic_configs.split(',')]
    base_output_dir = Path(args.base_output_dir)
    
    print(f"Glioma Pipeline Configuration:")
    print(f"  Train CSV: {args.train_csv}")
    print(f"  Test CSV: {args.test_csv}")
    print(f"  Base output directory: {base_output_dir}")
    print(f"  Topic configurations: {topic_configs}")
    print(f"  Identity topics: {identity_topics}")
    
    # Check if train/test files exist
    train_csv = Path(args.train_csv)
    test_csv = Path(args.test_csv)
    
    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    
    print(f"  Train set: {train_csv}")
    print(f"  Test set: {test_csv}")
    
    # Run pipeline for each topic configuration
    total_success = 0
    total_commands = 0
    
    for total_topics in topic_configs:
        n_extra_topics = total_topics - len(identity_topics.split(','))
        if n_extra_topics < 0:
            print(f"Warning: Skipping {total_topics}-topic config (not enough topics for {len(identity_topics.split(','))} identities)")
            continue
        
        # Create output directory for this configuration
        config_output_dir = base_output_dir / f"{total_topics}_topic_fit"
        
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
    
    # Run analyze_all_fits on all topic configurations
    print(f"\n{'='*80}")
    print(f"Running comprehensive analysis across all topic configurations")
    print(f"{'='*80}")
    
    shared_dir = Path(__file__).parent.parent / "shared"
    analyze_cmd = [
        sys.executable, str(shared_dir / "analyze_all_fits.py"), 
        "--base_dir", str(base_output_dir),
        "--topic_configs", args.topic_configs,
        "--output_dir", str(base_output_dir / "model_comparison")
    ]
    run_command(analyze_cmd, f"Comprehensive analysis across all topic configurations")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"GLIOMA PIPELINE COMPLETE")
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
    print(f"  {base_output_dir}/")
    for total_topics in topic_configs:
        n_extra_topics = total_topics - len(identity_topics.split(','))
        if n_extra_topics >= 0:
            config_dir = base_output_dir / f"{total_topics}_topic_fit"
            print(f"    {total_topics}_topic_fit/")
            print(f"      ├── HLDA/")
            print(f"      ├── LDA/")
            print(f"      ├── NMF/")
            print(f"      └── plots/")
    print(f"    model_comparison/")
    print(f"      ├── train_loglikelihood_matrix.csv")
    print(f"      ├── test_loglikelihood_matrix.csv")
    print(f"      ├── loglikelihood_comparison.png")
    print(f"      ├── overfitting_analysis.csv")
    print(f"      └── comprehensive_sse_heatmap.png")

if __name__ == "__main__":
    main() 