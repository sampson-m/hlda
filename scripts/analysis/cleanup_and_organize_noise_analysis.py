#!/usr/bin/env python3
"""
Comprehensive cleanup and organization of noise analysis files.

This script:
1. Moves all noise analysis files to estimates/noise_analysis/
2. Removes scattered files from individual dataset folders
3. Creates clean summary outputs with SNR and residual values
4. Organizes files in a logical structure
"""

import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


def find_all_noise_analysis_files(estimates_dir: Path) -> List[Path]:
    """Find all noise analysis related files in estimates directory."""
    noise_files = []
    
    # Find files with noise analysis patterns
    patterns = [
        "*noise*",
        "*snr*", 
        "*signal*",
        "*eigenvalue*",
        "*recovery*",
        "*residual*"
    ]
    
    for pattern in patterns:
        noise_files.extend(estimates_dir.rglob(pattern))
    
    # Remove duplicates and directories
    noise_files = list(set([f for f in noise_files if f.is_file()]))
    
    return noise_files


def organize_noise_analysis_files(estimates_dir: Path, target_dir: Path):
    """Organize all noise analysis files into a clean structure."""
    
    print("=" * 60)
    print("CLEANING UP AND ORGANIZING NOISE ANALYSIS FILES")
    print("=" * 60)
    
    # Create target directory structure
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Subdirectories for organization
    (target_dir / "individual_plots").mkdir(exist_ok=True)
    (target_dir / "summary_data").mkdir(exist_ok=True)
    (target_dir / "model_recovery").mkdir(exist_ok=True)
    
    # Find all noise analysis files
    noise_files = find_all_noise_analysis_files(estimates_dir)
    
    print(f"Found {len(noise_files)} noise analysis files")
    
    # Organize files by type
    moved_files = []
    removed_files = []
    
    for file_path in noise_files:
        try:
            # Skip if it's already in the target directory
            if target_dir in file_path.parents:
                continue
                
            file_name = file_path.name
            
            # Determine target location based on file type
            if file_name.endswith('.png'):
                if 'summary' in file_name.lower() or 'combined' in file_name.lower():
                    target_path = target_dir / file_name
                elif 'recovery' in file_name.lower() or 'residual' in file_name.lower():
                    target_path = target_dir / "model_recovery" / file_name
                else:
                    target_path = target_dir / "individual_plots" / file_name
            elif file_name.endswith('.csv'):
                if 'summary' in file_name.lower() or 'comparison' in file_name.lower():
                    target_path = target_dir / "summary_data" / file_name
                else:
                    target_path = target_dir / "summary_data" / file_name
            else:
                target_path = target_dir / file_name
            
            # Move file
            shutil.move(str(file_path), str(target_path))
            moved_files.append((file_path, target_path))
            
        except Exception as e:
            print(f"Error moving {file_path}: {e}")
            continue
    
    print(f"Moved {len(moved_files)} files to {target_dir}")
    
    return moved_files


def create_comprehensive_summary(estimates_dir: Path, target_dir: Path):
    """Create comprehensive summary with SNR and residual values by topic."""
    
    print("\nCreating comprehensive summary...")
    
    # Collect all SNR data
    snr_data = []
    residual_data = []
    
    # Find all SNR summary files
    snr_files = list(estimates_dir.rglob("*snr_summary.csv"))
    
    for snr_file in snr_files:
        try:
            df = pd.read_csv(snr_file)
            snr_data.append(df)
        except Exception as e:
            print(f"Error reading {snr_file}: {e}")
    
    # Find all residual data files
    residual_files = list(estimates_dir.rglob("*residual*.csv"))
    
    for residual_file in residual_files:
        try:
            df = pd.read_csv(residual_file)
            if 'median' in df.columns or 'residual' in df.columns:
                residual_data.append(df)
        except Exception as e:
            print(f"Error reading {residual_file}: {e}")
    
    # Combine SNR data
    combined_snr = pd.DataFrame()
    if snr_data:
        combined_snr = pd.concat(snr_data, ignore_index=True)
        combined_snr.to_csv(target_dir / "summary_data" / "combined_snr_data.csv", index=False)
        
        # Create SNR summary plot
        create_snr_summary_plot(combined_snr, target_dir / "snr_summary_plot.png")
    
    # Combine residual data
    combined_residual = pd.DataFrame()
    if residual_data:
        combined_residual = pd.concat(residual_data, ignore_index=True)
        combined_residual.to_csv(target_dir / "summary_data" / "combined_residual_data.csv", index=False)
        
        # Create residual summary plot
        create_residual_summary_plot(combined_residual, target_dir / "residual_summary_plot.png")
    
    # Create comprehensive summary
    create_comprehensive_summary_plot(combined_snr, combined_residual, target_dir / "comprehensive_summary.png")


def create_snr_summary_plot(snr_data: pd.DataFrame, output_path: Path):
    """Create summary plot of SNR values by topic and dataset."""
    
    plt.figure(figsize=(15, 10))
    
    # Create subplots for different views
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Activity SNR by DE mean
    for dataset in snr_data['Activity_Topic'].unique():
        dataset_data = snr_data[snr_data['Activity_Topic'] == dataset]
        ax1.plot(dataset_data['DE_Mean'], dataset_data['Activity_SNR'], 
                'o-', label=dataset, markersize=6, linewidth=2)
    
    ax1.set_xlabel('DE Mean (Signal Strength)', fontsize=12)
    ax1.set_ylabel('Activity SNR', fontsize=12)
    ax1.set_title('Activity Signal-to-Noise Ratio vs Signal Strength', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Identity SNR by topic
    identity_data = snr_data.groupby('Identity_Topic')['Identity_SNR'].mean().reset_index()
    bars = ax2.bar(identity_data['Identity_Topic'], identity_data['Identity_SNR'], 
                   color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Identity Topic', fontsize=12)
    ax2.set_ylabel('Average Identity SNR', fontsize=12)
    ax2.set_title('Average Identity SNR by Topic', fontsize=14)
    
    # Add value labels
    for bar, snr in zip(bars, identity_data['Identity_SNR']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{snr:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Activity SNR by topic
    activity_data = snr_data.groupby('Activity_Topic')['Activity_SNR'].mean().reset_index()
    bars = ax3.bar(activity_data['Activity_Topic'], activity_data['Activity_SNR'], 
                   color='skyblue', alpha=0.7)
    ax3.set_xlabel('Activity Topic', fontsize=12)
    ax3.set_ylabel('Average Activity SNR', fontsize=12)
    ax3.set_title('Average Activity SNR by Topic', fontsize=14)
    
    # Add value labels
    for bar, snr in zip(bars, activity_data['Activity_SNR']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{snr:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 4. SNR distribution
    ax4.hist(snr_data['Activity_SNR'], bins=20, alpha=0.7, color='skyblue', label='Activity SNR')
    ax4.hist(snr_data['Identity_SNR'], bins=20, alpha=0.7, color='lightcoral', label='Identity SNR')
    ax4.set_xlabel('Signal-to-Noise Ratio', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('SNR Distribution', fontsize=14)
    ax4.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ SNR summary plot saved to: {output_path}")


def create_residual_summary_plot(residual_data: pd.DataFrame, output_path: Path):
    """Create summary plot of residual values by topic and model."""
    
    if residual_data.empty:
        print("No residual data found")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Median residuals by model type
    if 'Model' in residual_data.columns and 'Median_Residual' in residual_data.columns:
        model_data = residual_data.groupby('Model')['Median_Residual'].mean().reset_index()
        bars = ax1.bar(model_data['Model'], model_data['Median_Residual'], 
                       color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
        ax1.set_xlabel('Model Type', fontsize=12)
        ax1.set_ylabel('Median Residual (%)', fontsize=12)
        ax1.set_title('Median Residuals by Model Type', fontsize=14)
        
        # Add value labels
        for bar, residual in zip(bars, model_data['Median_Residual']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{residual:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 2. Residuals by topic
    if 'Topic' in residual_data.columns:
        topic_data = residual_data.groupby('Topic')['Median_Residual'].mean().reset_index()
        bars = ax2.bar(topic_data['Topic'], topic_data['Median_Residual'], 
                       color='lightgreen', alpha=0.7)
        ax2.set_xlabel('Topic', fontsize=12)
        ax2.set_ylabel('Median Residual (%)', fontsize=12)
        ax2.set_title('Median Residuals by Topic', fontsize=14)
        ax2.tick_params(axis='x', rotation=45)
    
    # 3. Residuals vs DE mean
    if 'DE_Mean' in residual_data.columns:
        for model in residual_data['Model'].unique():
            model_data = residual_data[residual_data['Model'] == model]
            ax3.plot(model_data['DE_Mean'], model_data['Median_Residual'], 
                    'o-', label=model, markersize=6, linewidth=2)
        
        ax3.set_xlabel('DE Mean (Signal Strength)', fontsize=12)
        ax3.set_ylabel('Median Residual (%)', fontsize=12)
        ax3.set_title('Median Residuals vs Signal Strength', fontsize=14)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
    
    # 4. Residual distribution
    if 'Median_Residual' in residual_data.columns:
        ax4.hist(residual_data['Median_Residual'], bins=20, alpha=0.7, color='lightblue')
        ax4.set_xlabel('Median Residual (%)', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Residual Distribution', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Residual summary plot saved to: {output_path}")


def create_comprehensive_summary_plot(snr_data: pd.DataFrame, residual_data: pd.DataFrame, output_path: Path):
    """Create comprehensive summary plot combining SNR and residual information."""
    
    plt.figure(figsize=(20, 12))
    
    # Create a comprehensive layout
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Activity SNR trends (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    for dataset in snr_data['Activity_Topic'].unique():
        dataset_data = snr_data[snr_data['Activity_Topic'] == dataset]
        ax1.plot(dataset_data['DE_Mean'], dataset_data['Activity_SNR'], 
                'o-', label=dataset, markersize=6, linewidth=2)
    ax1.set_xlabel('DE Mean (Signal Strength)', fontsize=12)
    ax1.set_ylabel('Activity SNR', fontsize=12)
    ax1.set_title('Activity SNR vs Signal Strength', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Identity SNR by topic (top right, spans 2 columns)
    ax2 = fig.add_subplot(gs[0, 2:])
    identity_data = snr_data.groupby('Identity_Topic')['Identity_SNR'].mean().reset_index()
    bars = ax2.bar(identity_data['Identity_Topic'], identity_data['Identity_SNR'], 
                   color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Identity Topic', fontsize=12)
    ax2.set_ylabel('Average Identity SNR', fontsize=12)
    ax2.set_title('Average Identity SNR by Topic', fontsize=14)
    
    # 3. Model performance comparison (middle row, spans all columns)
    ax3 = fig.add_subplot(gs[1, :])
    if not residual_data.empty and 'Model' in residual_data.columns:
        model_data = residual_data.groupby('Model')['Median_Residual'].mean().reset_index()
        bars = ax3.bar(model_data['Model'], model_data['Median_Residual'], 
                       color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
        ax3.set_xlabel('Model Type', fontsize=12)
        ax3.set_ylabel('Median Residual (%)', fontsize=12)
        ax3.set_title('Model Performance Comparison', fontsize=14)
        
        # Add value labels
        for bar, residual in zip(bars, model_data['Median_Residual']):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{residual:.1f}%', ha='center', va='bottom', fontsize=12)
    
    # 4. Topic-specific performance (bottom row, 4 columns)
    if not residual_data.empty and 'Topic' in residual_data.columns:
        topics = residual_data['Topic'].unique()[:4]  # Take first 4 topics
        for i, topic in enumerate(topics):
            ax = fig.add_subplot(gs[2, i])
            topic_data = residual_data[residual_data['Topic'] == topic]
            
            if 'Model' in topic_data.columns and 'Median_Residual' in topic_data.columns:
                model_perf = topic_data.groupby('Model')['Median_Residual'].mean()
                bars = ax.bar(model_perf.index, model_perf.values, 
                             color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
                ax.set_title(f'Topic {topic}', fontsize=12)
                ax.set_ylabel('Median Residual (%)', fontsize=10)
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, residual in zip(bars, model_perf.values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{residual:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Comprehensive Noise Analysis Summary', fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comprehensive summary plot saved to: {output_path}")


def cleanup_individual_folders(estimates_dir: Path, target_dir: Path):
    """Remove noise analysis files from individual dataset folders."""
    
    print("\nCleaning up individual dataset folders...")
    
    # Find all noise analysis subdirectories in dataset folders
    noise_dirs = []
    for dataset_dir in estimates_dir.iterdir():
        if dataset_dir.is_dir() and not dataset_dir.name.startswith('_'):
            for de_dir in dataset_dir.iterdir():
                if de_dir.is_dir() and de_dir.name.startswith('DE_mean_'):
                    noise_dir = de_dir / "noise_analysis"
                    if noise_dir.exists():
                        noise_dirs.append(noise_dir)
    
    # Remove noise analysis directories
    removed_count = 0
    for noise_dir in noise_dirs:
        try:
            shutil.rmtree(noise_dir)
            removed_count += 1
        except Exception as e:
            print(f"Error removing {noise_dir}: {e}")
    
    print(f"Removed {removed_count} noise analysis directories from individual folders")


def main():
    """Main function to clean up and organize noise analysis files."""
    
    estimates_dir = Path("estimates")
    target_dir = estimates_dir / "noise_analysis"
    
    if not estimates_dir.exists():
        print("✗ Estimates directory not found!")
        return
    
    # Step 1: Organize files
    moved_files = organize_noise_analysis_files(estimates_dir, target_dir)
    
    # Step 2: Create comprehensive summary
    create_comprehensive_summary(estimates_dir, target_dir)
    
    # Step 3: Clean up individual folders
    cleanup_individual_folders(estimates_dir, target_dir)
    
    # Step 4: Create README
    create_readme(target_dir)
    
    print("\n" + "=" * 60)
    print("✓ CLEANUP AND ORGANIZATION COMPLETED!")
    print("=" * 60)
    print(f"All noise analysis files organized in: {target_dir}")
    print("Individual dataset folders cleaned up")
    print("Comprehensive summary plots created")


def create_readme(target_dir: Path):
    """Create a README file explaining the organized structure."""
    
    readme_content = """# Noise Analysis Results

This directory contains all noise analysis results organized in a clean structure.

## Directory Structure

- `individual_plots/` - Individual SNR analysis plots for each dataset/DE mean
- `summary_data/` - CSV files with combined SNR and residual data
- `model_recovery/` - Model recovery and residual analysis plots
- `snr_summary_plot.png` - Summary of signal-to-noise ratios by topic
- `residual_summary_plot.png` - Summary of model recovery residuals
- `comprehensive_summary.png` - Comprehensive overview combining all metrics

## Key Metrics

### Signal-to-Noise Ratios (SNR)
- **Activity SNR**: Measures the strength of activity topics relative to noise
- **Identity SNR**: Measures the strength of identity topics relative to noise
- Higher SNR values indicate better signal detection

### Model Recovery Residuals
- **Median Residual**: Percentage error in parameter recovery
- Lower values indicate better model performance
- Values < 5% indicate excellent recovery
- Values > 20% may indicate poor recovery

## Interpretation

1. **SNR Analysis**: Shows how well different topics can be detected at various signal strengths
2. **Residual Analysis**: Shows how well different models recover true parameters
3. **Topic Performance**: Identifies which topics are easier or harder to recover
4. **Model Comparison**: Compares HLDA, LDA, and NMF performance

## Files

- `combined_snr_data.csv` - All SNR measurements
- `combined_residual_data.csv` - All residual measurements
- Summary plots show trends and comparisons across datasets
"""
    
    with open(target_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"✓ README created: {target_dir}/README.md")


if __name__ == "__main__":
    main() 