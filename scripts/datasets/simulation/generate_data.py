#!/usr/bin/env python3
"""
Unified data generation script for simulation datasets.

This script replaces the scattered shell scripts with a clean, maintainable
Python interface for generating simulation data.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

# Add the parent directory to the path to import the config
sys.path.append(str(Path(__file__).parent))
from config import get_simulation_config, list_available_datasets


def generate_dataset(dataset_name: str, de_means: List[float] = None, force: bool = False):
    """
    Generate simulation data for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset to generate
        de_means: Specific DE means to generate (if None, use all from config)
        force: Whether to overwrite existing data
    """
    config = get_simulation_config(dataset_name)
    
    # Use specified DE means or all from config
    if de_means is None:
        de_means = config.de_means
    
    print(f"Generating {dataset_name} simulation data")
    print(f"  Identities: {config.identities}")
    print(f"  Activity topics: {config.activity_topics}")
    print(f"  DE means: {de_means}")
    print(f"  Output directory: data/{dataset_name}")
    print()
    
    # Create output directory
    output_dir = Path(f"data/{dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data for each DE mean
    for de_mean in de_means:
        print(f"=== Generating DE_mean_{de_mean} ===")
        
        # Create output path
        out_path = output_dir / f"DE_mean_{de_mean}"
        if out_path.exists() and not force:
            print(f"  ⚠ Directory {out_path} already exists. Use --force to overwrite.")
            continue
        
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Build command for simulate_counts.py
        cmd = [
            "python3", "scripts/simulation/simulate_counts.py",
            "--identities", str(config.identities).replace("'", '"'),
            "--activity-topics", str(config.activity_topics).replace("'", '"'),
            "--dirichlet-params", str(config.dirichlet_params).replace("'", '"'),
            "--activity-frac", str(config.activity_fraction),
            "--cells-per-identity", str(config.cells_per_identity),
            "--n-genes", str(config.n_genes),
            "--n-de", str(config.n_de_genes),
            "--de-mean", str(de_mean),
            "--de-sigma", str(config.de_sigma),
            "--out", str(out_path),
            "--seed", str(config.seed)
        ]
        
        # Run the simulation
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  ✓ Generated DE_mean_{de_mean}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Error generating DE_mean_{de_mean}: {e}")
            print(f"  Command: {' '.join(cmd)}")
            print(f"  Error: {e.stderr}")
            return False
    
    print(f"\n=== Data generation complete! ===")
    print(f"Generated data in: {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate simulation data for specified datasets")
    parser.add_argument("dataset", nargs="?", help="Dataset name to generate")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--de-means", nargs="+", type=float, help="Specific DE means to generate")
    parser.add_argument("--force", action="store_true", help="Overwrite existing data")
    parser.add_argument("--all", action="store_true", help="Generate all datasets")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available datasets:")
        for name in list_available_datasets():
            config = get_simulation_config(name)
            print(f"  {name}:")
            print(f"    Identities: {config.identities}")
            print(f"    Activity topics: {config.activity_topics}")
            print(f"    DE means: {config.de_means}")
            print()
        return
    
    if args.all:
        # Generate all datasets
        success = True
        for dataset_name in list_available_datasets():
            print(f"\n{'='*60}")
            print(f"GENERATING {dataset_name}")
            print(f"{'='*60}")
            if not generate_dataset(dataset_name, args.de_means, args.force):
                success = False
        if success:
            print(f"\n{'='*60}")
            print("ALL DATASETS GENERATED SUCCESSFULLY!")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print("SOME DATASETS FAILED TO GENERATE!")
            print(f"{'='*60}")
            sys.exit(1)
        return
    
    if not args.dataset:
        parser.error("Please specify a dataset name or use --list to see available datasets")
    
    # Generate single dataset
    if not generate_dataset(args.dataset, args.de_means, args.force):
        sys.exit(1)


if __name__ == "__main__":
    main() 