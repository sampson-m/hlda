#!/usr/bin/env python3
"""
Organize noise analysis output files into a proper directory structure.

This script consolidates all the scattered noise analysis files into a clean,
organized structure for easier access and management.
"""

import shutil
import sys
from pathlib import Path
from datetime import datetime


def organize_noise_analysis_files():
    """
    Organize all noise analysis files into a proper directory structure.
    """
    estimates_dir = Path("estimates")
    if not estimates_dir.exists():
        print("✗ Estimates directory not found!")
        return False
    
    # Create organized directory structure
    organized_dir = Path("estimates/noise_analysis")
    organized_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (organized_dir / "individual_plots").mkdir(exist_ok=True)
    (organized_dir / "topic_specific_plots").mkdir(exist_ok=True)
    (organized_dir / "model_recovery_plots").mkdir(exist_ok=True)
    (organized_dir / "comprehensive_plots").mkdir(exist_ok=True)
    (organized_dir / "data").mkdir(exist_ok=True)
    
    print("Organizing noise analysis files...")
    print(f"Target directory: {organized_dir}")
    print()
    
    # Track what we've organized
    organized_files = []
    skipped_files = []
    
    # Process each dataset directory
    for dataset_dir in estimates_dir.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name.startswith('_'):
            continue
        
        dataset_name = dataset_dir.name
        print(f"Processing {dataset_name}...")
        
        # Create dataset-specific subdirectories
        dataset_organized = organized_dir / dataset_name
        dataset_organized.mkdir(exist_ok=True)
        
        # 1. Move individual DE_mean noise analysis files
        for de_dir in dataset_dir.iterdir():
            if de_dir.is_dir() and de_dir.name.startswith("DE_mean_"):
                noise_dir = de_dir / "noise_analysis"
                if noise_dir.exists():
                    # Create organized structure for this DE mean
                    de_organized = dataset_organized / de_dir.name
                    de_organized.mkdir(exist_ok=True)
                    
                    # Move files
                    for file in noise_dir.iterdir():
                        if file.is_file():
                            dest = de_organized / file.name
                            shutil.copy2(file, dest)
                            organized_files.append(str(dest))
                            print(f"  ✓ Moved {file.name} to {dest}")
        
        # 2. Move dataset-level noise analysis files
        noise_files = [
            "comprehensive_noise_analysis.png",
            "noise_analysis_comparison.csv",
            "noise_analysis_comparison.png",
            "activity_snr_data.csv",
            "identity_snr_data.csv",
            "topic_snr_scatter_plots.png"
        ]
        
        for filename in noise_files:
            source = dataset_dir / filename
            if source.exists():
                dest = dataset_organized / filename
                shutil.copy2(source, dest)
                organized_files.append(str(dest))
                print(f"  ✓ Moved {filename} to {dest}")
        
        # 3. Move topic-specific recovery plots
        topic_files = [
            "topic_specific_theta_recovery.png",
            "topic_specific_beta_recovery.png",
            "topic_specific_recovery_combined.png"
        ]
        
        for filename in topic_files:
            source = dataset_dir / filename
            if source.exists():
                dest = organized_dir / "topic_specific_plots" / f"{dataset_name}_{filename}"
                shutil.copy2(source, dest)
                organized_files.append(str(dest))
                print(f"  ✓ Moved {filename} to topic_specific_plots/")
        
        # 4. Move model recovery plots
        recovery_files = [
            "model_recovery_analysis.png",
            "model_recovery_analysis_residuals.png",
            "model_recovery_analysis.csv",
            "model_recovery_with_residuals.csv"
        ]
        
        for filename in recovery_files:
            source = dataset_dir / filename
            if source.exists():
                dest = organized_dir / "model_recovery_plots" / f"{dataset_name}_{filename}"
                shutil.copy2(source, dest)
                organized_files.append(str(dest))
                print(f"  ✓ Moved {filename} to model_recovery_plots/")
        
        # 5. Move comprehensive plots
        comprehensive_files = [
            "comprehensive_noise_analysis.png"
        ]
        
        for filename in comprehensive_files:
            source = dataset_dir / filename
            if source.exists():
                dest = organized_dir / "comprehensive_plots" / f"{dataset_name}_{filename}"
                shutil.copy2(source, dest)
                organized_files.append(str(dest))
                print(f"  ✓ Moved {filename} to comprehensive_plots/")
        
        # 6. Move data files
        data_files = [
            "noise_analysis_comparison.csv",
            "activity_snr_data.csv",
            "identity_snr_data.csv",
            "model_recovery_analysis.csv",
            "model_recovery_with_residuals.csv"
        ]
        
        for filename in data_files:
            source = dataset_dir / filename
            if source.exists():
                dest = organized_dir / "data" / f"{dataset_name}_{filename}"
                shutil.copy2(source, dest)
                organized_files.append(str(dest))
                print(f"  ✓ Moved {filename} to data/")
    
    # Create summary files
    create_summary_files(organized_dir, organized_files)
    
    print(f"\n✓ Organization complete!")
    print(f"  Organized {len(organized_files)} files")
    print(f"  Organized directory: {organized_dir}")
    
    return True


def create_summary_files(organized_dir, organized_files):
    """Create summary files for the organized structure."""
    
    # Create README
    readme_content = f"""# Organized Noise Analysis Files

This directory contains all noise analysis files organized by type and dataset.

## Directory Structure

- `individual_plots/` - Individual DE_mean noise analysis plots
- `topic_specific_plots/` - Topic-specific recovery plots
- `model_recovery_plots/` - Model recovery analysis plots
- `comprehensive_plots/` - Comprehensive noise analysis plots
- `data/` - CSV data files
- `[dataset_name]/` - Dataset-specific organized files

## Files Organized

Total files: {len(organized_files)}

Organized on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Usage

### View Topic-Specific Recovery Plots
```bash
# Open topic-specific plots directory
open estimates/noise_analysis/topic_specific_plots/
```

### View Model Recovery Analysis
```bash
# Open model recovery plots directory
open estimates/noise_analysis/model_recovery_plots/
```

### Access Data Files
```bash
# View CSV data files
ls estimates/noise_analysis/data/
```

## File Naming Convention

- Individual plots: `[dataset]_[de_mean]_[plot_type].png`
- Topic-specific plots: `[dataset]_topic_specific_[type].png`
- Model recovery plots: `[dataset]_model_recovery_[type].png`
- Data files: `[dataset]_[data_type].csv`
"""
    
    with open(organized_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    # Create file list
    with open(organized_dir / "files_organized.txt", "w") as f:
        f.write(f"Files organized on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total files: {len(organized_files)}\n\n")
        for file in sorted(organized_files):
            f.write(f"{file}\n")


def main():
    """Main function to organize noise analysis files."""
    print("=" * 60)
    print("ORGANIZING NOISE ANALYSIS FILES")
    print("=" * 60)
    
    if organize_noise_analysis_files():
        print("\n✓ Successfully organized noise analysis files!")
        print("\nYou can now find all organized files in: estimates/noise_analysis/")
    else:
        print("\n✗ Failed to organize noise analysis files!")
        sys.exit(1)


if __name__ == "__main__":
    main() 