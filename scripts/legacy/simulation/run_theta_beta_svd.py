#!/usr/bin/env python3
"""
run_theta_beta_svd.py

Script to compute SVD of θβ^T, extract singular values, and multiply by √L.
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import the evaluation functions
sys.path.append(str(Path(__file__).parent.parent))

from simulation.simulation_evaluation_functions import compute_theta_beta_svd_analysis

def main():
    """Run θβ^T SVD analysis on all simulation data."""
    
    print("=" * 60)
    print("SVD ANALYSIS: θβ^T")
    print("=" * 60)
    
    # Set paths
    data_root = Path("data")
    
    print(f"Data root: {data_root}")
    
    # Run the analysis
    compute_theta_beta_svd_analysis(data_root=str(data_root))
    
    print(f"\n✓ SVD analysis completed!")

if __name__ == "__main__":
    main() 