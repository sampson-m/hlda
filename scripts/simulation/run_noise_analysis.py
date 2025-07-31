#!/usr/bin/env python3
"""
run_noise_analysis.py

Script to run the noise analysis based on the LaTeX derivation.
This implements the orthogonalization analysis for RNA-seq count data to determine
signal-to-noise ratios for identity vs activity topic detection.
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import the evaluation functions
sys.path.append(str(Path(__file__).parent.parent))

from simulation.simulation_evaluation_functions import compute_noise_analysis

def main():
    """Run noise analysis on all simulation data."""
    
    print("=" * 60)
    print("NOISE ANALYSIS: Identity vs Activity Topic Detection")
    print("=" * 60)
    print("Based on orthogonalization analysis from LaTeX derivation")
    print("=" * 60)
    
    # Set paths
    data_root = Path("data")
    output_dir = Path("estimates")
    
    print(f"Data root: {data_root}")
    print(f"Output directory: {output_dir}")
    
    # Run the noise analysis
    results = compute_noise_analysis(
        data_root=str(data_root),
        output_dir=str(output_dir)
    )
    
    if results:
        print(f"\n✓ Noise analysis completed successfully!")
        print(f"  Processed {len(results)} datasets")
        print(f"  Results saved to respective estimate directories")
        
        # Print summary of key findings
        print(f"\nKey findings:")
        for key, data in results.items():
            de_mean = data['de_mean']
            dataset_name = data['dataset_name']
            
            # Get activity SNR values
            activity_snrs = []
            for activity_idx, result in data['results'].items():
                activity_snrs.append(result['activity_snr'])
            
            if activity_snrs:
                avg_activity_snr = sum(activity_snrs) / len(activity_snrs)
                print(f"  {dataset_name} DE_mean_{de_mean}: avg activity SNR = {avg_activity_snr:.4f}")
        
        print(f"\nCheck the following files for detailed results:")
        print(f"  - Individual plots: estimates/*/DE_mean_*/noise_analysis/noise_analysis_plot.png")
        print(f"  - Combined comparison: estimates/noise_analysis_comparison.png")
        print(f"  - Results data: estimates/noise_analysis_comparison.csv")
        
    else:
        print(f"\n✗ No results generated. Check if data files exist.")

if __name__ == "__main__":
    main() 