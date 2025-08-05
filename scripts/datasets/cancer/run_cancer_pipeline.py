#!/usr/bin/env python3
"""
Cancer Pipeline: Run HLDA, LDA, and NMF models with different topic configurations.
Compares disease-specific vs combined cell type labeling schemes.
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

# Import preprocessing function
from preprocess_cancer import preprocess_cancer_data

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

def run_labeling_scheme(labeling_scheme, base_output_dir, activity_topic_counts, config_file, skip_hlda, skip_lda_nmf, skip_evaluation):
    """Run the full pipeline for a specific labeling scheme."""
    
    print(f"\n{'='*80}")
    print(f"RUNNING {labeling_scheme.upper()} LABELING SCHEME")
    print(f"{'='*80}")
    
    # Check if preprocessed files already exist
    expected_train_csv = f"data/cancer/cancer_counts_train_{labeling_scheme}.csv"
    expected_test_csv = f"data/cancer/cancer_counts_test_{labeling_scheme}.csv"
    
    if Path(expected_train_csv).exists() and Path(expected_test_csv).exists():
        print(f"Found existing preprocessed files for {labeling_scheme} scheme:")
        print(f"  Train: {expected_train_csv}")
        print(f"  Test: {expected_test_csv}")
        train_csv = expected_train_csv
        test_csv = expected_test_csv
        
        # Read unique labels from train file
        import pandas as pd
        train_df = pd.read_csv(train_csv, index_col=0)
        unique_labels = train_df.index.unique()
        print(f"  Unique labels: {list(unique_labels)}")
    else:
        print(f"Preprocessed files not found for {labeling_scheme} scheme. Running preprocessing...")
        train_csv, test_csv, unique_labels = preprocess_cancer_data(labeling_scheme)
    
    # Dataset name for this scheme
    dataset_name = f"cancer_{labeling_scheme}"
    
    # Load identity topics from config file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    if dataset_name not in config:
        raise ValueError(f"Dataset '{dataset_name}' not found in config file {config_file}")
    identity_topics = ','.join(config[dataset_name]['identities'])
    n_identity_topics = len(config[dataset_name]['identities'])
    
    # Create output directory for this scheme
    scheme_output_dir = base_output_dir / labeling_scheme
    
    print(f"  Dataset: {dataset_name}")
    print(f"  Train CSV: {train_csv}")
    print(f"  Test CSV: {test_csv}")
    print(f"  Unique labels: {list(unique_labels)}")
    print(f"  Identity topics ({n_identity_topics}): {identity_topics}")
    print(f"  Output directory: {scheme_output_dir}")
    
    # Calculate total topics for each activity topic count
    total_topic_configs = []
    for n_activity in activity_topic_counts:
        total_topics = n_identity_topics + n_activity
        total_topic_configs.append(total_topics)
        print(f"  Activity topics {n_activity} -> Total topics {total_topics}")
    
    # Run pipeline for each topic configuration
    total_success = 0
    total_commands = 0
    
    for i, total_topics in enumerate(total_topic_configs):
        n_extra_topics = activity_topic_counts[i]
        
        print(f"\nRunning configuration: {n_identity_topics} identity + {n_extra_topics} activity = {total_topics} total topics")
        
        # Create output directory for this configuration
        config_output_dir = scheme_output_dir / f"{total_topics}_topic_fit"
        
        # Run the configuration
        success_count, commands_count = run_topic_configuration(
            train_csv=train_csv,
            test_csv=test_csv,
            output_dir=config_output_dir,
            n_extra_topics=n_extra_topics,
            identity_topics=identity_topics,
            dataset=dataset_name,
            config_file=config_file,
            skip_hlda=skip_hlda,
            skip_lda_nmf=skip_lda_nmf,
            skip_evaluation=skip_evaluation
        )
        total_success += success_count
        total_commands += commands_count
    
    # Run analyze_all_fits on all topic configurations for this scheme
    print(f"\n{'='*60}")
    print(f"Running comprehensive analysis for {labeling_scheme} scheme")
    print(f"{'='*60}")
    
    shared_dir = Path(__file__).parent.parent / "shared"
    analyze_cmd = [
        sys.executable, str(shared_dir / "analyze_all_fits.py"), 
        "--base_dir", str(scheme_output_dir),
        "--topic_configs", ",".join(map(str, total_topic_configs)),
        "--output_dir", str(scheme_output_dir / "model_comparison")
    ]
    run_command(analyze_cmd, f"Comprehensive analysis for {labeling_scheme} scheme")
    
    return total_success, total_commands, scheme_output_dir

def main():
    parser = argparse.ArgumentParser(description="Cancer pipeline for running models with different topic configurations and comparing labeling schemes.")
    parser.add_argument("--base_output_dir", type=str, default="estimates/cancer",
                       help="Base directory for outputs (default: estimates/cancer)")
    parser.add_argument("--activity_topics", type=str, default="1,2,3,4",
                       help="Comma-separated list of activity topic counts (default: 1,2,3,4)")
    parser.add_argument("--config_file", type=str, default="dataset_identities.yaml",
                       help="Path to dataset identity config YAML file")
    parser.add_argument("--skip_hlda", action="store_true",
                       help="Skip HLDA fitting (useful for testing)")
    parser.add_argument("--skip_lda_nmf", action="store_true",
                       help="Skip LDA/NMF fitting (useful for testing)")
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Skip model evaluation (useful for testing)")
    parser.add_argument("--skip_comparison", action="store_true",
                       help="Skip cross-scheme comparison (useful for testing)")
    
    args = parser.parse_args()
    
    # Parse arguments
    activity_topic_counts = [int(x.strip()) for x in args.activity_topics.split(',')]
    base_output_dir = Path(args.base_output_dir)
    
    print(f"Cancer Pipeline Configuration:")
    print(f"  Base output directory: {base_output_dir}")
    print(f"  Activity topic counts: {activity_topic_counts}")
    print(f"  Config file: {args.config_file}")
    
    # Run both labeling schemes
    total_success = 0
    total_commands = 0
    scheme_dirs = {}
    
    # for scheme in ["combined", "disease_specific"]:
    for scheme in ["disease_specific"]:

        success_count, commands_count, scheme_dir = run_labeling_scheme(
            labeling_scheme=scheme,
            base_output_dir=base_output_dir,
            activity_topic_counts=activity_topic_counts,
            config_file=args.config_file,
            skip_hlda=args.skip_hlda,
            skip_lda_nmf=args.skip_lda_nmf,
            skip_evaluation=args.skip_evaluation
        )
        total_success += success_count
        total_commands += commands_count
        scheme_dirs[scheme] = scheme_dir
    
    # Run cross-scheme comparison
    if not args.skip_comparison:
        print(f"\n{'='*80}")
        print(f"RUNNING CROSS-SCHEME COMPARISON")
        print(f"{'='*80}")
        
        # Create comparison script
        comparison_script = Path(__file__).parent / "compare_schemes.py"
        if comparison_script.exists():
            # Calculate topic configs for comparison (will be different for each scheme)
            combined_topics = [9 + n for n in activity_topic_counts]  # 9 identity + activity
            disease_topics = [18 + n for n in activity_topic_counts]  # 18 identity + activity
            
            comparison_cmd = [
                sys.executable, str(comparison_script),
                "--combined_dir", str(scheme_dirs["combined"]),
                "--disease_specific_dir", str(scheme_dirs["disease_specific"]),
                "--output_dir", str(base_output_dir / "scheme_comparison"),
                "--topic_configs", ",".join(map(str, combined_topics))
            ]
            run_command(comparison_cmd, "Cross-scheme comparison")
        else:
            print("Warning: Cross-scheme comparison script not found, skipping...")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"CANCER PIPELINE COMPLETE")
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
    for scheme in ["combined", "disease_specific"]:
        n_identity = 9 if scheme == "combined" else 18
        print(f"    {scheme}/ ({n_identity} identity topics)")
        for n_activity in activity_topic_counts:
            total_topics = n_identity + n_activity
            print(f"      {total_topics}_topic_fit/ ({n_identity} identity + {n_activity} activity)")
            print(f"        ├── HLDA/")
            print(f"        ├── LDA/")
            print(f"        ├── NMF/")
            print(f"        └── plots/")
        print(f"      model_comparison/")
    print(f"    scheme_comparison/")
    print(f"      ├── cosine_similarity_matrix.csv")
    print(f"      ├── topic_comparison_plots.png")
    print(f"      └── differential_expression_comparison.csv")

if __name__ == "__main__":
    main() 