#!/usr/bin/env python3
"""
Unified pipeline runner for simulation datasets.

This script replaces the scattered shell scripts with a clean, maintainable
Python interface for running model fitting pipelines.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Add the parent directory to the path to import the config
sys.path.append(str(Path(__file__).parent))
from config import get_pipeline_config, list_available_datasets


def run_pipeline(dataset_name: str, de_means: Optional[List[str]] = None, 
                config_file: str = "dataset_identities.yaml"):
    """
    Run the complete pipeline for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset to process
        de_means: Specific DE means to process (if None, use all available)
        config_file: Path to the configuration file
    """
    config = get_pipeline_config(dataset_name)
    
    print(f"Running pipeline for {dataset_name}")
    print(f"  Data root: {config.data_root}")
    print(f"  Estimates root: {config.estimates_root}")
    print(f"  Topic config: {config.topic_config}")
    print(f"  Total topics: {config.n_total_topics}")
    print()
    
    # Determine which DE means to process
    if de_means is None:
        if config.de_means is not None:
            de_means = config.de_means
        else:
            # Find all available DE_mean directories
            data_root = Path(config.data_root)
            if not data_root.exists():
                print(f"✗ Data directory {data_root} does not exist!")
                return False
            
            de_means = [d.name for d in data_root.iterdir() 
                       if d.is_dir() and d.name.startswith("DE_mean_")]
            de_means.sort()
    
    print(f"Processing DE means: {de_means}")
    print()
    
    # Process each DE mean
    for de_mean in de_means:
        print(f"=== Processing {de_mean} ===")
        
        simdir = Path(config.data_root) / de_mean
        if not simdir.exists():
            print(f"  ⚠ Directory {simdir} does not exist, skipping...")
            continue
        
        # Define paths
        train_csv = simdir / "filtered_counts_train.csv"
        test_csv = simdir / "filtered_counts_test.csv"
        outdir = Path(config.estimates_root) / de_mean / config.topic_config
        
        if not train_csv.exists():
            print(f"  ✗ Training data {train_csv} not found!")
            continue
        
        # Create output directory
        outdir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Clean up any accidental header rows
        print("  Cleaning up CSV files...")
        for csv_file in [train_csv, test_csv]:
            if csv_file.exists():
                try:
                    cmd = [
                        "python3", "-c",
                        f"import pandas as pd; "
                        f"f='{csv_file}'; "
                        f"df=pd.read_csv(f, index_col=0); "
                        f"df = df[df.index != 'cell']; "
                        f"df.index = pd.Series(df.index).str.split('_').str[0]; "
                        f"df.to_csv(f)"
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    print(f"    ⚠ Warning: Could not clean {csv_file}: {e}")
        
        # Step 2: Fit HLDA
        print("  Fitting HLDA...")
        hlda_cmd = [
            "python3", "scripts/shared/fit_hlda.py",
            "--counts_csv", str(train_csv),
            "--n_extra_topics", str(config.n_extra_topics),
            "--output_dir", str(outdir / "HLDA"),
            "--dataset", config.dataset_name,
            "--config_file", config_file,
            "--n_loops", "10000",
            "--burn_in", "4000",
            "--thin", "30"
        ]
        
        try:
            subprocess.run(hlda_cmd, check=True)
            print("    ✓ HLDA fitting complete")
        except subprocess.CalledProcessError as e:
            print(f"    ✗ HLDA fitting failed: {e}")
            continue
        
        # Step 3: Fit LDA and NMF
        print("  Fitting LDA and NMF...")
        lda_nmf_cmd = [
            "python3", "scripts/shared/fit_lda_nmf.py",
            "--counts_csv", str(train_csv),
            "--n_topics", str(config.n_total_topics),
            "--output_dir", str(outdir),
            "--model", "both",
            "--max_iter", "1000"
        ]
        
        try:
            subprocess.run(lda_nmf_cmd, check=True)
            print("    ✓ LDA/NMF fitting complete")
        except subprocess.CalledProcessError as e:
            print(f"    ✗ LDA/NMF fitting failed: {e}")
            continue
        
        # Step 4: Evaluate models
        print("  Evaluating models...")
        eval_cmd = [
            "python3", "scripts/shared/evaluate_models.py",
            "--counts_csv", str(train_csv),
            "--test_csv", str(test_csv),
            "--output_dir", str(outdir),
            "--n_extra_topics", str(config.n_extra_topics),
            "--dataset", config.dataset_name,
            "--config_file", config_file
        ]
        
        try:
            subprocess.run(eval_cmd, check=True)
            print("    ✓ Model evaluation complete")
        except subprocess.CalledProcessError as e:
            print(f"    ✗ Model evaluation failed: {e}")
            continue
        
        # Step 5: Analyze all fits
        print("  Analyzing fits...")
        analyze_cmd = [
            "python3", "scripts/shared/analyze_all_fits.py",
            "--base_dir", str(Path(config.estimates_root) / de_mean),
            "--topic_configs", str(config.n_total_topics),
            "--config_file", config_file
        ]
        
        try:
            subprocess.run(analyze_cmd, check=True)
            print("    ✓ Fit analysis complete")
        except subprocess.CalledProcessError as e:
            print(f"    ✗ Fit analysis failed: {e}")
            continue
        
        # Step 6: Restore cell names (for simulation data)
        print("  Restoring cell names...")
        for model in ["HLDA", "LDA", "NMF"]:
            est_theta = outdir / model / f"{model}_theta.csv"
            train_cells = simdir / "train_cells.csv"
            
            if est_theta.exists() and train_cells.exists():
                try:
                    restore_cmd = [
                        "python3", "-c",
                        f"import pandas as pd; "
                        f"train_theta = pd.read_csv('{est_theta}', index_col=0); "
                        f"train_cells = pd.read_csv('{train_cells}')['cell']; "
                        f"if len(train_theta) == len(train_cells): "
                        f"    train_theta.index = train_cells; "
                        f"    train_theta.to_csv('{est_theta}'); "
                        f"    print('✓ Restored cell names for {model}')"
                    ]
                    subprocess.run(restore_cmd, check=True)
                except subprocess.CalledProcessError:
                    print(f"    ⚠ Could not restore cell names for {model}")
        
        print(f"  ✓ Completed {de_mean}")
        print()
    
    print(f"=== Pipeline complete for {dataset_name} ===")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run model fitting pipelines for simulation datasets")
    parser.add_argument("dataset", nargs="?", help="Dataset name to process")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--de-means", nargs="+", help="Specific DE means to process")
    parser.add_argument("--config-file", default="dataset_identities.yaml", help="Configuration file path")
    parser.add_argument("--all", action="store_true", help="Process all datasets")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available datasets:")
        for name in list_available_datasets():
            config = get_pipeline_config(name)
            print(f"  {name}:")
            print(f"    Data root: {config.data_root}")
            print(f"    Estimates root: {config.estimates_root}")
            print(f"    Total topics: {config.n_total_topics}")
            if config.de_means:
                print(f"    DE means: {config.de_means}")
            print()
        return
    
    if args.all:
        # Process all datasets
        success = True
        for dataset_name in list_available_datasets():
            print(f"\n{'='*60}")
            print(f"PROCESSING {dataset_name}")
            print(f"{'='*60}")
            if not run_pipeline(dataset_name, args.de_means, args.config_file):
                success = False
        if success:
            print(f"\n{'='*60}")
            print("ALL PIPELINES COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print("SOME PIPELINES FAILED!")
            print(f"{'='*60}")
            sys.exit(1)
        return
    
    if not args.dataset:
        parser.error("Please specify a dataset name or use --list to see available datasets")
    
    # Process single dataset
    if not run_pipeline(args.dataset, args.de_means, args.config_file):
        sys.exit(1)


if __name__ == "__main__":
    main() 