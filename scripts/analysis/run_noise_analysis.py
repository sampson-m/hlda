#!/usr/bin/env python3
"""
Simple interface to run noise analysis using the new core framework.

This script provides a clean interface to run signal-to-noise ratio analysis
on simulation data without creating excessive files.
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import core modules
sys.path.append(str(Path(__file__).parent.parent))

from core.noise_analysis import run_noise_analysis
import numpy as np


def main():
    """Run clean noise analysis on all simulation data."""
    
    print("=" * 60)
    print("CLEAN NOISE ANALYSIS: Signal-to-Noise Ratio Analysis")
    print("=" * 60)
    print("Using new integrated framework")
    print("=" * 60)
    
    # Set paths
    data_root = Path("data")
    output_root = Path("estimates")
    
    print(f"Data root: {data_root}")
    print(f"Output directory: {output_root}")
    
    # Run the noise analysis
    results = run_noise_analysis(
        data_root=str(data_root),
        output_root=str(output_root)
    )
    
    if results:
        print(f"\n✓ Noise analysis completed successfully!")
        print(f"  Processed {len(results)} datasets")
        
        # All plots are now created within the noise_analysis directory
        print(f"  All results organized in: {output_root}/noise_analysis/")
        
        # Print key findings
        print(f"\nKey findings:")
        for key, data in results.items():
            dataset_name = data['dataset_name']
            de_mean = data['de_mean']
            avg_activity_snr = np.mean([r['activity_snr'] for r in data['results']['results'].values()])
            print(f"  {dataset_name} DE_mean_{de_mean}: avg activity SNR = {avg_activity_snr:.4f}")
        
        print(f"\nCheck the following files for detailed results:")
        print(f"  - All results: estimates/noise_analysis/")
        print(f"  - Individual plots: estimates/noise_analysis/individual_plots/")
        print(f"  - Summary data: estimates/noise_analysis/summary_data/")
        print(f"  - Model recovery: estimates/noise_analysis/model_recovery/")
        
    else:
        print(f"\n✗ No results generated. Check if data files exist.")


if __name__ == "__main__":
    main() 