#!/usr/bin/env python3
"""
Unified simulation interface.

This script provides a simple interface to the new unified simulation system,
replacing the scattered shell scripts with a clean, maintainable Python interface.

Usage:
    python3 scripts/datasets/simulation/simulate.py generate AB_V1
    python3 scripts/datasets/simulation/simulate.py pipeline ABCD_V1V2
    python3 scripts/datasets/simulation/simulate.py --help
"""

import argparse
import sys
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))
from generate_data import generate_dataset
from run_pipelines import run_pipeline
from config import list_available_datasets, get_simulation_config, get_pipeline_config


def main():
    parser = argparse.ArgumentParser(
        description="Unified simulation interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate data for AB_V1 dataset
  python3 scripts/datasets/simulation/simulate.py generate AB_V1
  
  # Generate data for specific DE means
  python3 scripts/datasets/simulation/simulate.py generate ABCD_V1V2 --de-means 0.1 0.2 0.3
  
  # Run pipeline for ABCD_V1V2 dataset
  python3 scripts/datasets/simulation/simulate.py pipeline ABCD_V1V2
  
  # Run pipeline with noise analysis
  python3 scripts/datasets/simulation/simulate.py pipeline ABCD_V1V2 --noise-analysis
  
  # List available datasets
  python3 scripts/datasets/simulation/simulate.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available datasets')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate simulation data')
    generate_parser.add_argument('dataset', help='Dataset name to generate')
    generate_parser.add_argument('--de-means', nargs='+', type=float, help='Specific DE means to generate')
    generate_parser.add_argument('--force', action='store_true', help='Overwrite existing data')
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run model fitting pipeline')
    pipeline_parser.add_argument('dataset', help='Dataset name to process')
    pipeline_parser.add_argument('--de-means', nargs='+', help='Specific DE means to process')
    pipeline_parser.add_argument('--config-file', default='dataset_identities.yaml', help='Configuration file path')
    pipeline_parser.add_argument('--noise-analysis', action='store_true', help='Run noise analysis after pipeline')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        print("Available datasets:")
        print()
        for name in list_available_datasets():
            sim_config = get_simulation_config(name)
            pipe_config = get_pipeline_config(name)
            print(f"  {name}:")
            print(f"    Identities: {sim_config.identities}")
            print(f"    Activity topics: {sim_config.activity_topics}")
            print(f"    DE means: {sim_config.de_means}")
            print(f"    Total topics: {pipe_config.n_total_topics}")
            print()
    
    elif args.command == 'generate':
        if not generate_dataset(args.dataset, args.de_means, args.force):
            sys.exit(1)
    
    elif args.command == 'pipeline':
        if not run_pipeline(args.dataset, args.de_means, args.config_file):
            sys.exit(1)
        
        # Run noise analysis if requested
        if args.noise_analysis:
            print("\n" + "="*60)
            print("RUNNING NOISE ANALYSIS")
            print("="*60)
            
            # Import and run noise analysis
            sys.path.append(str(Path(__file__).parent.parent.parent))
            from core.noise_analysis import run_noise_analysis, create_summary_plot
            
            results = run_noise_analysis()
            if results:
                summary_path = Path("estimates") / "noise_analysis_summary.png"
                create_summary_plot(results, summary_path)
                print(f"✓ Noise analysis completed! Summary saved to: {summary_path}")
            else:
                print("✗ No noise analysis results generated")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main() 