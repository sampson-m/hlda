#!/usr/bin/env python3
"""
Combined script to run HLDA, LDA, and NMF models in sequence.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time

# Import default parameters from fit_hlda
from fit_hlda import get_default_parameters

# Get default HLDA parameters
HLDA_PARAMS = get_default_parameters()

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    
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

def main():
    parser = argparse.ArgumentParser(description="Run HLDA, LDA, and NMF models in sequence.")
    parser.add_argument("--counts_csv", type=str, required=True, 
                       help="Path to filtered count matrix CSV")
    parser.add_argument("--test_csv", type=str, required=True,
                       help="Path to test count matrix CSV")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save model outputs")
    parser.add_argument("--n_extra_topics", type=int, default=2,
                       help="Number of extra activity topics (default: 2)")
    parser.add_argument("--identity_topics", type=str, required=True,
                       help="Comma-separated list of identity topic names (e.g. 'T cells,CD19+ B,CD56+ NK')")
    parser.add_argument("--skip_hlda", action="store_true",
                       help="Skip HLDA fitting (useful for testing)")
    parser.add_argument("--skip_lda_nmf", action="store_true",
                       help="Skip LDA/NMF fitting (useful for testing)")
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Skip model evaluation (useful for testing)")
    
    args = parser.parse_args()
    
    # Calculate total topics
    identity_topics = args.identity_topics.split(',')
    n_identity = len(identity_topics)
    n_extra = args.n_extra_topics
    total_topics = n_identity + n_extra
    
    print(f"Configuration: {n_identity} identity + {n_extra} activity = {total_topics} total topics")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    total_commands = 0
    
    # 1. Run HLDA
    if not args.skip_hlda:
        total_commands += 1
        hlda_cmd = [
            sys.executable, "fit_hlda.py",
            "--counts_csv", args.counts_csv,
            "--n_extra_topics", str(args.n_extra_topics),
            "--output_dir", str(output_dir / "HLDA"),
            "--n_loops", str(HLDA_PARAMS['n_loops']),
            "--burn_in", str(HLDA_PARAMS['burn_in']), 
            "--thin", str(HLDA_PARAMS['thin'])
        ]
        if run_command(hlda_cmd, "HLDA Gibbs Sampling"):
            success_count += 1
    
    # 2. Run LDA and NMF
    if not args.skip_lda_nmf:
        total_commands += 1
        lda_nmf_cmd = [
            sys.executable, "fit_lda_nmf.py",
            "--counts_csv", args.counts_csv,
            "--n_topics", str(total_topics),
            "--output_dir", str(output_dir),
            "--model", "both",
            "--max_iter", "500"
        ]
        if run_command(lda_nmf_cmd, "LDA and NMF Fitting"):
            success_count += 1
    
    # 3. Run evaluation
    if not args.skip_evaluation:
        total_commands += 1
        eval_cmd = [
            sys.executable, "evaluate_models.py",
            "--counts_csv", args.counts_csv,
            "--test_csv", args.test_csv,
            "--output_dir", str(output_dir),
            "--identity_topics", args.identity_topics,
            "--n_extra_topics", str(args.n_extra_topics)
        ]
        if run_command(eval_cmd, "Model Evaluation"):
            success_count += 1
    
    # Summary
    print(f"\nSummary: {success_count}/{total_commands} commands completed")
    
    if success_count == total_commands:
        print(f"✓ All models completed successfully!")
    else:
        print(f"✗ Some commands failed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 