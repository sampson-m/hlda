#!/usr/bin/env python3
"""
Core noise analysis functions for signal-to-noise ratio analysis.

This module implements the orthogonalization analysis for RNA-seq count data
to determine signal-to-noise ratios for identity vs activity topic detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def compute_activity_snr(data_path: Path, output_path: Path) -> Dict:
    """
    Compute activity signal-to-noise ratio for a single dataset.
    
    Parameters:
    -----------
    data_path : Path
        Path to the dataset directory containing simulation files
    output_path : Path
        Path to save results and plots
        
    Returns:
    --------
    Dict containing analysis results
    """
    # Load required files
    counts_file = data_path / "counts.csv"
    if not counts_file.exists():
        counts_file = data_path / "filtered_counts_train.csv"
    
    theta_file = data_path / "theta.csv"
    gene_means_file = data_path / "gene_means.csv"
    
    if not all(f.exists() for f in [counts_file, theta_file, gene_means_file]):
        raise FileNotFoundError(f"Missing required files in {data_path}")
    
    # Load data
    X = pd.read_csv(counts_file, index_col=0)
    theta_true = pd.read_csv(theta_file, index_col=0)
    beta_true = pd.read_csv(gene_means_file, index_col=0)
    
    # Align data
    common_genes = X.columns.intersection(beta_true.index)
    common_cells = X.index.intersection(theta_true.index)
    
    if len(common_genes) == 0 or len(common_cells) == 0:
        raise ValueError("No common genes or cells found")
    
    X = X.loc[common_cells, common_genes]
    theta_true = theta_true.loc[common_cells]
    beta_true = beta_true.loc[common_genes]
    
    # Convert to numpy arrays
    X_np = X.values.astype(float)
    theta_np = theta_true.values.astype(float)
    beta_np = beta_true.values.astype(float)
    L = 1500.0  # Constant library size
    
    # Normalize to probabilities
    theta_np = theta_np / theta_np.sum(axis=1, keepdims=True)
    beta_np = beta_np / beta_np.sum(axis=0, keepdims=True)
    
    # Identify topic types
    topic_names = theta_true.columns.tolist()
    identity_topics = [i for i, name in enumerate(topic_names) if name in ['A', 'B', 'C', 'D']]
    activity_topics = [i for i, name in enumerate(topic_names) if name in ['VAR', 'V1', 'V2']]
    
    if not identity_topics or not activity_topics:
        raise ValueError("Could not identify identity/activity topics")
    
    # Compute noise term (Poisson noise)
    theta_beta = theta_np @ beta_np.T
    L_theta_beta = L * theta_beta
    noise_term = (X_np - L_theta_beta) / np.sqrt(L)
    
    # Compute noise eigenvalues
    U, s, Vt = np.linalg.svd(noise_term, full_matrices=False)
    noise_eigenvalues = s**2
    max_noise_eigenvalue = np.max(noise_eigenvalues)
    
    # Analyze each activity topic
    results = {}
    for activity_idx in activity_topics:
        beta_activity = beta_np[:, activity_idx]
        beta_identity_matrix = beta_np[:, identity_topics]
        
        # Project activity topic onto identity topics
        rho_coeffs = np.linalg.lstsq(beta_identity_matrix, beta_activity, rcond=None)[0]
        beta_activity_projection = beta_identity_matrix @ rho_coeffs
        beta_activity_orthogonal = beta_activity - beta_activity_projection
        
        # Compute rho_a (coefficient of orthogonal component)
        rho_a = np.linalg.norm(beta_activity_orthogonal)
        beta_tilde_a = beta_activity_orthogonal / rho_a if rho_a > 0 else beta_activity_orthogonal
        
        # Compute signals
        theta_activity = theta_np[:, activity_idx]
        theta_activity_norm = np.linalg.norm(theta_activity)
        beta_tilde_norm = np.linalg.norm(beta_tilde_a)
        
        activity_signal = np.sqrt(L) * rho_a * theta_activity_norm * beta_tilde_norm
        
        # Compute identity signals
        identity_signals = []
        for i, identity_idx in enumerate(identity_topics):
            theta_identity = theta_np[:, identity_idx]
            beta_identity = beta_np[:, identity_idx]
            
            theta_combined = theta_identity + theta_activity * rho_coeffs[i]
            theta_combined_norm = np.linalg.norm(theta_combined)
            beta_identity_norm = np.linalg.norm(beta_identity)
            
            identity_signal = np.sqrt(L) * theta_combined_norm * beta_identity_norm
            identity_signals.append(identity_signal)
        
        # Compute signal-to-noise ratios
        activity_snr = activity_signal / max_noise_eigenvalue
        identity_snrs = [signal / max_noise_eigenvalue for signal in identity_signals]
        
        results[activity_idx] = {
            'activity_signal': activity_signal,
            'activity_snr': activity_snr,
            'identity_signals': identity_signals,
            'identity_snrs': identity_snrs,
            'rho_a': rho_a,
            'rho_coeffs': rho_coeffs,
            'max_noise_eigenvalue': max_noise_eigenvalue,
            'topic_name': topic_names[activity_idx]
        }
    
    return {
        'topic_names': topic_names,
        'identity_topics': identity_topics,
        'activity_topics': activity_topics,
        'results': results,
        'noise_eigenvalues': noise_eigenvalues,
        'X_shape': X_np.shape
    }


def create_snr_plot(results: Dict, output_path: Path, dataset_name: str, de_mean: float):
    """
    Create a clean signal-to-noise ratio plot.
    
    Parameters:
    -----------
    results : Dict
        Results from compute_activity_snr
    output_path : Path
        Path to save the plot
    dataset_name : str
        Name of the dataset
    de_mean : float
        DE mean value
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Activity SNR plot
    activity_snrs = []
    activity_names = []
    for activity_idx, result in results['results'].items():
        activity_snrs.append(result['activity_snr'])
        activity_names.append(result['topic_name'])
    
    bars1 = ax1.bar(activity_names, activity_snrs, color='skyblue', alpha=0.7)
    ax1.set_title(f'Activity Topic SNR\n{dataset_name} DE_mean_{de_mean}', fontsize=12)
    ax1.set_ylabel('Signal-to-Noise Ratio')
    ax1.set_ylim(0, max(activity_snrs) * 1.1)
    
    # Add value labels on bars
    for bar, snr in zip(bars1, activity_snrs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{snr:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Identity SNR plot
    identity_names = [results['topic_names'][i] for i in results['identity_topics']]
    identity_snrs = []
    for result in results['results'].values():
        identity_snrs.extend(result['identity_snrs'])
    
    # Average identity SNRs across activity topics
    avg_identity_snrs = []
    for i in range(len(identity_names)):
        snrs = [result['identity_snrs'][i] for result in results['results'].values()]
        avg_identity_snrs.append(np.mean(snrs))
    
    bars2 = ax2.bar(identity_names, avg_identity_snrs, color='lightcoral', alpha=0.7)
    ax2.set_title(f'Identity Topic SNR (Average)\n{dataset_name} DE_mean_{de_mean}', fontsize=12)
    ax2.set_ylabel('Signal-to-Noise Ratio')
    ax2.set_ylim(0, max(avg_identity_snrs) * 1.1)
    
    # Add value labels on bars
    for bar, snr in zip(bars2, avg_identity_snrs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{snr:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_noise_analysis(data_root: str = "data", output_root: str = "estimates") -> Dict:
    """
    Run noise analysis on all simulation datasets.
    
    Parameters:
    -----------
    data_root : str
        Root directory containing simulation data
    output_root : str
        Root directory for output
        
    Returns:
    --------
    Dict containing all results
    """
    data_path = Path(data_root)
    output_path = Path(output_root)
    
    # Create organized noise analysis directory structure
    noise_analysis_dir = output_path / "noise_analysis"
    noise_analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (noise_analysis_dir / "individual_plots").mkdir(exist_ok=True)
    (noise_analysis_dir / "summary_data").mkdir(exist_ok=True)
    (noise_analysis_dir / "model_recovery").mkdir(exist_ok=True)
    
    # Find all DE_mean directories
    de_mean_dirs = []
    for dataset_dir in data_path.iterdir():
        if dataset_dir.is_dir() and not dataset_dir.name.startswith('_'):
            for de_dir in dataset_dir.iterdir():
                if de_dir.is_dir() and de_dir.name.startswith('DE_mean_'):
                    de_mean_dirs.append((dataset_dir.name, de_dir))
    
    print(f"Found {len(de_mean_dirs)} DE_mean directories")
    
    all_results = {}
    all_snr_data = []
    
    for dataset_name, de_dir in de_mean_dirs:
        de_mean = de_dir.name
        de_mean_value = float(de_mean.replace('DE_mean_', ''))
        
        print(f"\nProcessing {dataset_name}/{de_mean}")
        
        try:
            # Run analysis
            results = compute_activity_snr(de_dir, noise_analysis_dir)
            
            # Create individual plot
            plot_name = f"{dataset_name}_{de_mean}_snr_analysis.png"
            plot_path = noise_analysis_dir / "individual_plots" / plot_name
            create_snr_plot(results, plot_path, dataset_name, de_mean_value)
            
            # Collect summary data
            for activity_idx, result in results['results'].items():
                activity_name = result['topic_name']
                for i, identity_idx in enumerate(results['identity_topics']):
                    identity_name = results['topic_names'][identity_idx]
                    all_snr_data.append({
                        'Dataset': dataset_name,
                        'Activity_Topic': activity_name,
                        'Identity_Topic': identity_name,
                        'DE_Mean': de_mean_value,
                        'Activity_SNR': result['activity_snr'],
                        'Identity_SNR': result['identity_snrs'][i],
                        'Rho_a': result['rho_a']
                    })
            
            # Store results
            key = f"{dataset_name}_{de_mean}"
            all_results[key] = {
                'dataset_name': dataset_name,
                'de_mean': de_mean_value,
                'results': results
            }
            
            print(f"  ✓ Completed - Activity SNR: {np.mean([r['activity_snr'] for r in results['results'].values()]):.4f}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    # Save combined SNR data
    if all_snr_data:
        combined_snr_df = pd.DataFrame(all_snr_data)
        combined_snr_df.to_csv(noise_analysis_dir / "summary_data" / "combined_snr_data.csv", index=False)
        
        # Create summary plots
        create_snr_summary_plot(combined_snr_df, noise_analysis_dir / "snr_summary_plot.png")
        create_comprehensive_summary_plot(combined_snr_df, noise_analysis_dir / "comprehensive_summary.png")
        
        # Create SNR vs residual scatter plots
        print("\nCreating SNR vs residual scatter plots...")
        create_snr_vs_residual_scatter_plots(
            noise_analysis_dir / "summary_data" / "combined_snr_data.csv",
            Path("estimates"),
            noise_analysis_dir / "model_recovery"
        )
    
    # Create README
    create_readme(noise_analysis_dir)
    
    print(f"\n✓ Noise analysis completed!")
    print(f"  Processed {len(all_results)} datasets")
    print(f"  Results organized in: {noise_analysis_dir}")
    
    return all_results


def create_summary_plot(all_results: Dict, output_path: Path):
    """
    Create a summary plot showing SNR trends across DE means.
    
    Parameters:
    -----------
    all_results : Dict
        Results from run_noise_analysis
    output_path : Path
        Path to save the summary plot
    """
    # Organize data by dataset
    datasets = {}
    for key, data in all_results.items():
        dataset_name = data['dataset_name']
        if dataset_name not in datasets:
            datasets[dataset_name] = []
        
        de_mean = data['de_mean']
        avg_activity_snr = np.mean([r['activity_snr'] for r in data['results']['results'].values()])
        datasets[dataset_name].append((de_mean, avg_activity_snr))
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (dataset_name, points) in enumerate(datasets.items()):
        points.sort(key=lambda x: x[0])  # Sort by DE mean
        de_means, snrs = zip(*points)
        
        plt.plot(de_means, snrs, 'o-', label=dataset_name, 
                color=colors[i % len(colors)], linewidth=2, markersize=6)
    
    plt.xlabel('DE Mean (Signal Strength)', fontsize=12)
    plt.ylabel('Average Activity SNR', fontsize=12)
    plt.title('Signal-to-Noise Ratio vs Signal Strength', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Run noise analysis
    results = run_noise_analysis()
    
    # Create summary plot
    summary_path = Path("estimates") / "noise_analysis_summary.png"
    create_summary_plot(results, summary_path)
    
    print(f"\nSummary plot saved to: {summary_path}")


def create_snr_summary_plot(snr_data: pd.DataFrame, output_path: Path):
    """Create summary plot of SNR values by topic and dataset."""
    
    # Create subplots for different views
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Activity SNR by DE mean
    for dataset in snr_data['Dataset'].unique():
        dataset_data = snr_data[snr_data['Dataset'] == dataset]
        # Average activity SNR per DE mean for this dataset
        avg_snr = dataset_data.groupby('DE_Mean')['Activity_SNR'].mean().reset_index()
        ax1.plot(avg_snr['DE_Mean'], avg_snr['Activity_SNR'], 
                'o-', label=dataset, markersize=6, linewidth=2)
    
    ax1.set_xlabel('DE Mean (Signal Strength)', fontsize=12)
    ax1.set_ylabel('Average Activity SNR', fontsize=12)
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


def create_comprehensive_summary_plot(snr_data: pd.DataFrame, output_path: Path):
    """Create comprehensive summary plot combining SNR information."""
    
    # Create a comprehensive layout
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Activity SNR trends (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    for dataset in snr_data['Dataset'].unique():
        dataset_data = snr_data[snr_data['Dataset'] == dataset]
        # Average activity SNR per DE mean for this dataset
        avg_snr = dataset_data.groupby('DE_Mean')['Activity_SNR'].mean().reset_index()
        ax1.plot(avg_snr['DE_Mean'], avg_snr['Activity_SNR'], 
                'o-', label=dataset, markersize=6, linewidth=2)
    ax1.set_xlabel('DE Mean (Signal Strength)', fontsize=12)
    ax1.set_ylabel('Average Activity SNR', fontsize=12)
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
    
    # 3. Activity SNR by topic (middle row, spans all columns)
    ax3 = fig.add_subplot(gs[1, :])
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
                f'{snr:.3f}', ha='center', va='bottom', fontsize=12)
    
    # 4. SNR distribution (bottom row, spans all columns)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.hist(snr_data['Activity_SNR'], bins=20, alpha=0.7, color='skyblue', label='Activity SNR')
    ax4.hist(snr_data['Identity_SNR'], bins=20, alpha=0.7, color='lightcoral', label='Identity SNR')
    ax4.set_xlabel('Signal-to-Noise Ratio', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('SNR Distribution', fontsize=14)
    ax4.legend(fontsize=10)
    
    plt.suptitle('Comprehensive Noise Analysis Summary', fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comprehensive summary plot saved to: {output_path}")


def create_readme(target_dir: Path):
    """Create a README file explaining the organized structure."""
    
    readme_content = """# Noise Analysis Results

This directory contains all noise analysis results organized in a clean structure.

## Directory Structure

- `individual_plots/` - Individual SNR analysis plots for each dataset/DE mean
- `summary_data/` - CSV files with combined SNR data
- `model_recovery/` - Model recovery and residual analysis plots (when available)
- `snr_summary_plot.png` - Summary of signal-to-noise ratios by topic
- `comprehensive_summary.png` - Comprehensive overview combining all metrics

## Key Metrics

### Signal-to-Noise Ratios (SNR)
- **Activity SNR**: Measures the strength of activity topics relative to noise
- **Identity SNR**: Measures the strength of identity topics relative to noise
- Higher SNR values indicate better signal detection

## Interpretation

1. **SNR Analysis**: Shows how well different topics can be detected at various signal strengths
2. **Topic Performance**: Identifies which topics are easier or harder to recover
3. **Signal Strength**: Shows how SNR varies with DE mean (signal strength)

## Files

- `combined_snr_data.csv` - All SNR measurements across all datasets
- Summary plots show trends and comparisons across datasets
"""
    
    with open(target_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"✓ README created: {target_dir}/README.md") 


def create_residual_histograms(histogram_df: pd.DataFrame, output_dir: Path):
    """
    Create overlapping histograms of theta residuals for different model types.
    """
    # Filter for activity-specific data only
    activity_data = histogram_df[histogram_df['Topic'].str.contains('_activity', na=False)]
    
    if len(activity_data) == 0:
        print("No activity-specific data found for histograms")
        return
    
    # Get unique topics (without _activity suffix)
    topics = sorted(activity_data['Topic'].str.replace('_activity', '').unique())
    
    # Set up colors for model types
    model_colors = {'HLDA': 'blue', 'LDA': 'red', 'NMF': 'green'}
    
    # Create histograms for each topic
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, topic in enumerate(topics):
        if i >= len(axes):
            break
            
        ax = axes[i]
        topic_data = activity_data[activity_data['Topic'] == f"{topic}_activity"]
        
        if len(topic_data) == 0:
            continue
        
        # Create overlapping histograms for each model type
        for model_type in ['HLDA', 'LDA', 'NMF']:
            model_data = topic_data[topic_data['Model_Type'] == model_type]
            if len(model_data) > 0:
                # Use individual residuals for histogram
                residuals = model_data['Theta_Residual'].values
                if len(residuals) > 0:
                    # Create histogram with transparency
                    ax.hist(residuals, bins=20, alpha=0.6, color=model_colors[model_type], 
                           label=model_type, density=True, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Theta Residual')
        ax.set_ylabel('Density')
        ax.set_title(f'Topic {topic} Residual Distribution - Activity Cells Only')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Remove scientific notation
        ax.ticklabel_format(style='plain', axis='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / "theta_residual_histograms.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create separate histograms by DE mean for better signal level comparison
    de_means = sorted(activity_data['DE_Mean'].unique())
    if len(de_means) > 1:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, de_mean in enumerate(de_means):
            if i >= len(axes):
                break
                
            ax = axes[i]
            de_data = activity_data[activity_data['DE_Mean'] == de_mean]
            
            if len(de_data) == 0:
                continue
            
            # Create overlapping histograms for each model type
            for model_type in ['HLDA', 'LDA', 'NMF']:
                model_data = de_data[de_data['Model_Type'] == model_type]
                if len(model_data) > 0:
                    residuals = model_data['Theta_Residual'].values
                    if len(residuals) > 0:
                        ax.hist(residuals, bins=20, alpha=0.6, color=model_colors[model_type], 
                               label=model_type, density=True, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Theta Residual')
            ax.set_ylabel('Density')
            ax.set_title(f'DE Mean {de_mean} Residual Distribution - Activity Cells Only')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax.ticklabel_format(style='plain', axis='both')
        
        plt.tight_layout()
        plt.savefig(output_dir / "theta_residual_histograms_by_de_mean.png", dpi=300, bbox_inches='tight')
        plt.close()


def create_activity_only_analysis(snr_data_path: Path, estimates_root: Path, output_dir: Path):
    """
    Create scatter plots and histograms for activity-only analysis (only cells that sample from activity topics).
    """
    from scipy.optimize import linear_sum_assignment
    
    print("Creating activity-only analysis...")
    print("Creating activity-only analysis...")
    
    # Load SNR data
    snr_df = pd.read_csv(snr_data_path)
    
    # Get all unique topics from SNR data
    all_topics = set()
    all_topics.update(snr_df['Activity_Topic'].unique())
    all_topics.update(snr_df['Identity_Topic'].unique())
    
    print(f"Found topics: {sorted(all_topics)}")
    
    # Collect all data for plotting (activity cells only)
    plot_data = []
    histogram_data = []
    
    # Process each dataset and DE mean - only simulation data
    simulation_datasets = ['AB_V1', 'ABCD_V1V2', 'ABCD_V1V2_new']
    
    for dataset_name in snr_df['Dataset'].unique():
        if dataset_name not in simulation_datasets:
            continue
            
        dataset_snr = snr_df[snr_df['Dataset'] == dataset_name]
        
        for de_mean in dataset_snr['DE_Mean'].unique():
            de_mean_snr = dataset_snr[dataset_snr['DE_Mean'] == de_mean]
            
            # Find the estimates directory - look for any topic_fit directory
            de_mean_dir = estimates_root / dataset_name / f"DE_mean_{de_mean}"
            if not de_mean_dir.exists():
                print(f"  Skipping {dataset_name}/DE_mean_{de_mean} - no estimates directory found")
                continue
            
            # Find topic_fit directories
            topic_fit_dirs = list(de_mean_dir.glob("*topic_fit"))
            if not topic_fit_dirs:
                print(f"  Skipping {dataset_name}/DE_mean_{de_mean} - no topic_fit directories found")
                continue
            
            # Use the first topic_fit directory found
            estimates_dir = topic_fit_dirs[0]
            print(f"  Using {estimates_dir.name} for {dataset_name}/DE_mean_{de_mean}")
            
            # Load true parameters from data directory
            data_dir = Path("data") / dataset_name / f"DE_mean_{de_mean}"
            true_theta_path = data_dir / "theta.csv"
            train_cells_path = data_dir / "train_cells.csv"
            true_beta_path = data_dir / "gene_means.csv"
            
            if not (true_theta_path.exists() and train_cells_path.exists() and true_beta_path.exists()):
                print(f"  Skipping {dataset_name}/DE_mean_{de_mean} - no true parameters found")
                continue
            
            true_theta = pd.read_csv(true_theta_path, index_col=0)
            train_cells = pd.read_csv(train_cells_path)
            true_beta = pd.read_csv(true_beta_path, index_col=0)
            
            # Create train_theta by subsetting true_theta with train_cells
            train_theta = true_theta.loc[train_cells['cell']]
            
            # Process each model type
            for model_type in ['HLDA', 'LDA', 'NMF']:
                model_dir = estimates_dir / model_type
                if not model_dir.exists():
                    continue
                
                # Load estimated parameters
                est_theta_path = model_dir / f"{model_type}_theta.csv"
                est_beta_path = model_dir / f"{model_type}_beta.csv"
                
                if not (est_theta_path.exists() and est_beta_path.exists()):
                    continue
                
                est_theta = pd.read_csv(est_theta_path, index_col=0)
                est_beta = pd.read_csv(est_beta_path, index_col=0)
                
                # Match topics using Hungarian algorithm
                try:
                    # Use train_theta for matching (test indices)
                    train_theta_subset = train_theta.loc[est_theta.index]
                    
                    # Filter to only include activity cells (cells that sample from activity topics)
                    # Find cells that are NOT pure identity cells (theta != 1.0 for identity topics)
                    identity_topics = [topic for topic in ['V1', 'V2'] if topic in train_theta_subset.columns]
                    
                    if len(identity_topics) > 0:
                        activity_cells = ~(train_theta_subset[identity_topics] == 1.0).any(axis=1)
                        if np.sum(activity_cells) == 0:
                            continue  # No activity cells, skip this dataset/model
                        
                        # Subset to only activity cells
                        train_theta_subset = train_theta_subset.loc[activity_cells]
                        est_theta = est_theta.loc[activity_cells]
                    else:
                        # No identity topics found, skip this dataset
                        continue
                    
                    # Compute correlation matrix between train theta and estimated topics
                    correlation_matrix = np.zeros((len(train_theta_subset.columns), len(est_theta.columns)))
                    
                    for i, train_topic in enumerate(train_theta_subset.columns):
                        for j, est_topic in enumerate(est_theta.columns):
                            train_vals = train_theta_subset.iloc[:, i].values
                            est_vals = est_theta.iloc[:, j].values
                            # Compute correlation
                            correlation_matrix[i, j] = np.corrcoef(train_vals, est_vals)[0, 1]
                    
                    # Use Hungarian algorithm to find optimal matching
                    train_indices, est_indices = linear_sum_assignment(-correlation_matrix)
                    
                    # Compute residuals for each matched topic
                    for train_idx, est_idx in zip(train_indices, est_indices):
                        if train_idx < len(train_theta_subset.columns) and est_idx < len(est_theta.columns):
                            true_topic_name = train_theta_subset.columns[train_idx]
                            est_topic_name = est_theta.columns[est_idx]
                            
                            # Get SNR for this topic
                            topic_snr_data = de_mean_snr[
                                (de_mean_snr['Activity_Topic'] == true_topic_name) | 
                                (de_mean_snr['Identity_Topic'] == true_topic_name)
                            ]
                            
                            if len(topic_snr_data) == 0:
                                continue
                            
                            # Use activity SNR if available, otherwise identity SNR
                            if len(topic_snr_data[topic_snr_data['Activity_Topic'] == true_topic_name]) > 0:
                                snr_value = topic_snr_data[topic_snr_data['Activity_Topic'] == true_topic_name]['Activity_SNR'].iloc[0]
                            else:
                                snr_value = topic_snr_data[topic_snr_data['Identity_Topic'] == true_topic_name]['Identity_SNR'].iloc[0]
                            
                            # Compute RMSE for theta (using train_theta for test cells)
                            train_theta_topic = train_theta_subset.iloc[:, train_idx].values
                            est_theta_topic = est_theta.iloc[:, est_idx].values
                            
                            # For HLDA, only consider cells that can actually have this topic
                            if model_type == 'HLDA':
                                # Find cells where the estimated topic has non-zero probability
                                valid_cells = est_theta_topic > 1e-10
                                if np.sum(valid_cells) > 0:
                                    train_theta_topic = train_theta_topic[valid_cells]
                                    est_theta_topic = est_theta_topic[valid_cells]
                                else:
                                    # Skip this topic if no cells can have it
                                    continue
                            
                            # Compute 90th percentile absolute residuals for scatter plots
                            theta_absolute_residuals = np.abs(train_theta_topic - est_theta_topic)
                            theta_90th_percentile = np.percentile(theta_absolute_residuals, 90)
                            median_theta_residual = theta_90th_percentile
                            
                            # Initialize beta residual
                            median_beta_residual = np.nan
                            
                            # Compute median percent residuals for beta (for scatter plots only)
                            if train_idx < len(true_beta.columns) and est_idx < len(est_beta.columns):
                                # Get common genes between true and estimated beta
                                common_genes = true_beta.index.intersection(est_beta.index)
                                if len(common_genes) > 0:
                                    true_beta_topic = true_beta.loc[common_genes, true_topic_name].values
                                    est_beta_topic = est_beta.loc[common_genes, est_topic_name].values
                                    
                                    # Normalize to probabilities
                                    true_beta_topic_norm = true_beta_topic / np.sum(true_beta_topic)
                                    est_beta_topic_norm = est_beta_topic / np.sum(est_beta_topic)
                                    
                                    # Compute percent residuals
                                    epsilon = 1e-10
                                    beta_percent_residuals = np.abs(true_beta_topic_norm - est_beta_topic_norm) / (true_beta_topic_norm + epsilon) * 100
                                    median_beta_residual = np.median(beta_percent_residuals)
                                else:
                                    median_beta_residual = np.nan
                            else:
                                median_beta_residual = np.nan
                            
                            # Store 90th percentile for scatter plots
                            plot_data.append({
                                'Topic': true_topic_name,
                                'Model_Type': model_type,
                                'Dataset': dataset_name,
                                'Topic_Config': estimates_dir.name,
                                'DE_Mean': de_mean,
                                'SNR': snr_value,
                                'Theta_RMSE': median_theta_residual,
                                'Beta_Residual_Pct': median_beta_residual
                            })
                            
                            # Store individual residuals for histogram plotting
                            for residual in theta_absolute_residuals:
                                histogram_data.append({
                                    'Topic': true_topic_name,
                                    'Model_Type': model_type,
                                    'Dataset': dataset_name,
                                    'Topic_Config': estimates_dir.name,
                                    'DE_Mean': de_mean,
                                    'SNR': snr_value,
                                    'Theta_Residual': residual
                                })
                            
                except Exception as e:
                    print(f"  Error processing {dataset_name}/DE_mean_{de_mean}/{model_type}: {e}")
                    continue
    
    if not plot_data:
        print("No data collected for activity-only analysis")
        return
    
    # Convert to DataFrame
    plot_df = pd.DataFrame(plot_data)
    print(f"Collected {len(plot_df)} data points for activity-only analysis")
    
    # Save the data
    plot_df.to_csv(output_dir / "activity_only_snr_vs_residual_data.csv", index=False)
    
    # Create scatter plots for each topic
    topics = sorted(plot_df['Topic'].unique())
    
    # Set up colors for model types and shapes for topic configurations
    model_colors = {'HLDA': 'blue', 'LDA': 'red', 'NMF': 'green'}
    topic_config_shapes = {
        '6_topic_fit': 'o',      # Circle
        '3_topic_fit': 's',      # Square
        '3_topic_fit_IMPROVED': '^'  # Triangle
    }
    
    # Create comprehensive theta residual scatter plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, topic in enumerate(topics):
        if i >= len(axes):
            break
            
        ax = axes[i]
        topic_data = plot_df[plot_df['Topic'] == topic]
        
        # Create separate legends for model types and topic configurations
        model_legend_handles = []
        model_legend_labels = []
        config_legend_handles = []
        config_legend_labels = []
        
        # Track what we've already added to avoid duplicates
        added_models = set()
        added_configs = set()
        
        for model_type in ['HLDA', 'LDA', 'NMF']:
            model_dataset_data = topic_data[topic_data['Model_Type'] == model_type]
            
            if len(model_dataset_data) > 0:
                for topic_config in model_dataset_data['Topic_Config'].unique():
                    config_data = model_dataset_data[model_dataset_data['Topic_Config'] == topic_config]
                    
                    if len(config_data) > 0:
                        # Get shape for this topic configuration
                        shape = topic_config_shapes.get(topic_config, 'o')
                        
                        # Create scatter plot
                        scatter = ax.scatter(config_data['SNR'], config_data['Theta_RMSE'], 
                                           c=model_colors[model_type], marker=shape, s=100, alpha=0.7)
                        
                        # Add to legend if not already added
                        if model_type not in added_models:
                            model_legend_handles.append(scatter)
                            model_legend_labels.append(model_type)
                            added_models.add(model_type)
                        
                        if topic_config not in added_configs:
                            # Create a dummy scatter for legend
                            dummy_scatter = ax.scatter([], [], c='gray', marker=shape, s=100, alpha=0.7)
                            config_legend_handles.append(dummy_scatter)
                            config_legend_labels.append(topic_config)
                            added_configs.add(topic_config)
        
        ax.set_xlabel('Signal-to-Noise Ratio')
        ax.set_ylabel('90th Percentile Theta Residual')
        ax.set_title(f'Topic {topic} - Activity Cells Only')
        ax.grid(True, alpha=0.3)
        
        # Remove scientific notation
        ax.ticklabel_format(style='plain', axis='both')
        
        # Add legends
        if model_legend_handles:
            ax.legend(model_legend_handles, model_legend_labels, loc='upper left')
        if config_legend_handles:
            ax.legend(config_legend_handles, config_legend_labels, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / "activity_only_snr_vs_theta_residual_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create beta residual scatter plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, topic in enumerate(topics):
        if i >= len(axes):
            break
            
        ax = axes[i]
        topic_data = plot_df[plot_df['Topic'] == topic]
        
        # Create separate legends for model types and topic configurations
        model_legend_handles = []
        model_legend_labels = []
        config_legend_handles = []
        config_legend_labels = []
        
        # Track what we've already added to avoid duplicates
        added_models = set()
        added_configs = set()
        
        for model_type in ['HLDA', 'LDA', 'NMF']:
            model_dataset_data = topic_data[topic_data['Model_Type'] == model_type]
            
            if len(model_dataset_data) > 0:
                for topic_config in model_dataset_data['Topic_Config'].unique():
                    config_data = model_dataset_data[model_dataset_data['Topic_Config'] == topic_config]
                    
                    if len(config_data) > 0:
                        # Get shape for this topic configuration
                        shape = topic_config_shapes.get(topic_config, 'o')
                        
                        # Create scatter plot
                        scatter = ax.scatter(config_data['SNR'], config_data['Beta_Residual_Pct'], 
                                           c=model_colors[model_type], marker=shape, s=100, alpha=0.7)
                        
                        # Add to legend if not already added
                        if model_type not in added_models:
                            model_legend_handles.append(scatter)
                            model_legend_labels.append(model_type)
                            added_models.add(model_type)
                        
                        if topic_config not in added_configs:
                            # Create a dummy scatter for legend
                            dummy_scatter = ax.scatter([], [], c='gray', marker=shape, s=100, alpha=0.7)
                            config_legend_handles.append(dummy_scatter)
                            config_legend_labels.append(topic_config)
                            added_configs.add(topic_config)
        
        ax.set_xlabel('Signal-to-Noise Ratio')
        ax.set_ylabel('Median Beta Residual (%)')
        ax.set_title(f'Topic {topic} - Activity Cells Only')
        ax.grid(True, alpha=0.3)
        
        # Remove scientific notation
        ax.ticklabel_format(style='plain', axis='both')
        
        # Add legends
        if model_legend_handles:
            ax.legend(model_legend_handles, model_legend_labels, loc='upper left')
        if config_legend_handles:
            ax.legend(config_legend_handles, config_legend_labels, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / "activity_only_snr_vs_beta_residual_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create overlapping histograms of residuals
    histogram_df = pd.DataFrame(histogram_data)
    create_activity_only_histograms(histogram_df, output_dir)
    
    print(f"✓ Activity-only analysis completed! Results saved to: {output_dir}")


def create_activity_only_histograms(histogram_df: pd.DataFrame, output_dir: Path):
    """
    Create overlapping histograms of theta residuals for activity-only analysis.
    """
    if len(histogram_df) == 0:
        print("No data found for activity-only histograms")
        return
    
    # Get unique topics
    topics = sorted(histogram_df['Topic'].unique())
    
    # Set up colors for model types
    model_colors = {'HLDA': 'blue', 'LDA': 'red', 'NMF': 'green'}
    
    # Create histograms for each topic
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, topic in enumerate(topics):
        if i >= len(axes):
            break
            
        ax = axes[i]
        topic_data = histogram_df[histogram_df['Topic'] == topic]
        
        if len(topic_data) == 0:
            continue
        
        # Create overlapping histograms for each model type
        for model_type in ['HLDA', 'LDA', 'NMF']:
            model_data = topic_data[topic_data['Model_Type'] == model_type]
            if len(model_data) > 0:
                # Use individual residuals for histogram
                residuals = model_data['Theta_Residual'].values
                if len(residuals) > 0:
                    # Create histogram with transparency
                    ax.hist(residuals, bins=20, alpha=0.6, color=model_colors[model_type], 
                           label=model_type, density=True, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Theta Residual')
        ax.set_ylabel('Density')
        ax.set_title(f'Topic {topic} Residual Distribution - Activity Cells Only')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Remove scientific notation
        ax.ticklabel_format(style='plain', axis='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / "activity_only_theta_residual_histograms.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create separate histograms by DE mean for better signal level comparison
    de_means = sorted(histogram_df['DE_Mean'].unique())
    if len(de_means) > 1:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, de_mean in enumerate(de_means):
            if i >= len(axes):
                break
                
            ax = axes[i]
            de_data = histogram_df[histogram_df['DE_Mean'] == de_mean]
            
            if len(de_data) == 0:
                continue
            
            # Create overlapping histograms for each model type
            for model_type in ['HLDA', 'LDA', 'NMF']:
                model_data = de_data[de_data['Model_Type'] == model_type]
                if len(model_data) > 0:
                    residuals = model_data['Theta_Residual'].values
                    if len(residuals) > 0:
                        ax.hist(residuals, bins=20, alpha=0.6, color=model_colors[model_type], 
                               label=model_type, density=True, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Theta Residual')
            ax.set_ylabel('Density')
            ax.set_title(f'DE Mean {de_mean} Residual Distribution - Activity Cells Only')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax.ticklabel_format(style='plain', axis='both')
        
        plt.tight_layout()
        plt.savefig(output_dir / "activity_only_theta_residual_histograms_by_de_mean.png", dpi=300, bbox_inches='tight')
        plt.close()


def create_snr_vs_residual_scatter_plots(snr_data_path: Path, estimates_root: Path, output_dir: Path):
    """
    Create scatter plots showing SNR vs median residual for each topic.
    
    Parameters:
    -----------
    snr_data_path : Path
        Path to the combined SNR data CSV
    estimates_root : Path
        Root directory containing model estimates
    output_dir : Path
        Directory to save the scatter plots
    """
    import pandas as pd
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    # correlation_matrix will be computed manually
    
    # Load SNR data
    snr_df = pd.read_csv(snr_data_path)
    print(f"Loaded SNR data with {len(snr_df)} data points")
    
    # Get unique topics
    all_topics = set()
    all_topics.update(snr_df['Activity_Topic'].unique())
    all_topics.update(snr_df['Identity_Topic'].unique())
    
    print(f"Found topics: {sorted(all_topics)}")
    
    # Collect all data for plotting
    plot_data = []
    histogram_data = []
    
    # Process each dataset and DE mean - only simulation data
    simulation_datasets = ['AB_V1', 'ABCD_V1V2', 'ABCD_V1V2_new']
    
    for dataset_name in snr_df['Dataset'].unique():
        if dataset_name not in simulation_datasets:
            continue
            
        dataset_snr = snr_df[snr_df['Dataset'] == dataset_name]
        
        for de_mean in dataset_snr['DE_Mean'].unique():
            de_mean_snr = dataset_snr[dataset_snr['DE_Mean'] == de_mean]
            
            # Find the estimates directory - look for any topic_fit directory
            de_mean_dir = estimates_root / dataset_name / f"DE_mean_{de_mean}"
            if not de_mean_dir.exists():
                print(f"  Skipping {dataset_name}/DE_mean_{de_mean} - no estimates directory found")
                continue
            
            # Find topic_fit directories
            topic_fit_dirs = list(de_mean_dir.glob("*topic_fit"))
            if not topic_fit_dirs:
                print(f"  Skipping {dataset_name}/DE_mean_{de_mean} - no topic_fit directories found")
                continue
            
            # Use the first topic_fit directory found
            estimates_dir = topic_fit_dirs[0]
            print(f"  Using {estimates_dir.name} for {dataset_name}/DE_mean_{de_mean}")
            
            # Load true parameters from data directory
            data_dir = Path("data") / dataset_name / f"DE_mean_{de_mean}"
            true_theta_path = data_dir / "theta.csv"
            train_cells_path = data_dir / "train_cells.csv"
            true_beta_path = data_dir / "gene_means.csv"
            
            if not (true_theta_path.exists() and train_cells_path.exists() and true_beta_path.exists()):
                print(f"  Skipping {dataset_name}/DE_mean_{de_mean} - no true parameters found")
                continue
            
            true_theta = pd.read_csv(true_theta_path, index_col=0)
            train_cells = pd.read_csv(train_cells_path)
            true_beta = pd.read_csv(true_beta_path, index_col=0)
            
            # Create train_theta by subsetting true_theta with train_cells
            train_theta = true_theta.loc[train_cells['cell']]
            
            # Process each model type
            for model_type in ['HLDA', 'LDA', 'NMF']:
                model_dir = estimates_dir / model_type
                if not model_dir.exists():
                    continue
                
                # Load estimated parameters
                est_theta_path = model_dir / f"{model_type}_theta.csv"
                est_beta_path = model_dir / f"{model_type}_beta.csv"
                
                if not (est_theta_path.exists() and est_beta_path.exists()):
                    continue
                
                est_theta = pd.read_csv(est_theta_path, index_col=0)
                est_beta = pd.read_csv(est_beta_path, index_col=0)
                
                # Match topics using Hungarian algorithm
                try:
                    # Use train_theta for matching (test indices)
                    train_theta_subset = train_theta.loc[est_theta.index]
                    
                    # Compute correlation matrix between train theta and estimated topics
                    correlation_matrix = np.zeros((len(train_theta_subset.columns), len(est_theta.columns)))
                    
                    for i, train_topic in enumerate(train_theta_subset.columns):
                        for j, est_topic in enumerate(est_theta.columns):
                            train_vals = train_theta_subset.iloc[:, i].values
                            est_vals = est_theta.iloc[:, j].values
                            # Compute correlation
                            correlation_matrix[i, j] = np.corrcoef(train_vals, est_vals)[0, 1]
                    
                    # Use Hungarian algorithm to find optimal matching
                    train_indices, est_indices = linear_sum_assignment(-correlation_matrix)
                    
                    # Compute residuals for each matched topic
                    for train_idx, est_idx in zip(train_indices, est_indices):
                        if train_idx < len(train_theta_subset.columns) and est_idx < len(est_theta.columns):
                            true_topic_name = train_theta_subset.columns[train_idx]
                            est_topic_name = est_theta.columns[est_idx]
                            
                            # Get SNR for this topic
                            topic_snr_data = de_mean_snr[
                                (de_mean_snr['Activity_Topic'] == true_topic_name) | 
                                (de_mean_snr['Identity_Topic'] == true_topic_name)
                            ]
                            
                            if len(topic_snr_data) == 0:
                                continue
                            
                            # Use activity SNR if available, otherwise identity SNR
                            if len(topic_snr_data[topic_snr_data['Activity_Topic'] == true_topic_name]) > 0:
                                snr_value = topic_snr_data[topic_snr_data['Activity_Topic'] == true_topic_name]['Activity_SNR'].iloc[0]
                            else:
                                snr_value = topic_snr_data[topic_snr_data['Identity_Topic'] == true_topic_name]['Identity_SNR'].iloc[0]
                            
                            # Compute RMSE for theta (using train_theta for test cells)
                            train_theta_topic = train_theta_subset.iloc[:, train_idx].values
                            est_theta_topic = est_theta.iloc[:, est_idx].values
                            
                            # For HLDA, only consider cells that can actually have this topic
                            if model_type == 'HLDA':
                                # Find cells where the estimated topic has non-zero probability
                                valid_cells = est_theta_topic > 1e-10
                                if np.sum(valid_cells) > 0:
                                    train_theta_topic = train_theta_topic[valid_cells]
                                    est_theta_topic = est_theta_topic[valid_cells]
                                else:
                                    # Skip this topic if no cells can have it
                                    continue
                            
                            # Compute 90th percentile absolute residuals for scatter plots
                            theta_absolute_residuals = np.abs(train_theta_topic - est_theta_topic)
                            theta_90th_percentile = np.percentile(theta_absolute_residuals, 90)
                            median_theta_residual = theta_90th_percentile
                            
                            # Initialize beta residual
                            median_beta_residual = np.nan
                            
                            # Compute median percent residuals for beta
                            if train_idx < len(true_beta.columns) and est_idx < len(est_beta.columns):
                                # Get common genes between true and estimated beta
                                common_genes = true_beta.index.intersection(est_beta.index)
                                if len(common_genes) > 0:
                                    true_beta_topic = true_beta.loc[common_genes, true_topic_name].values
                                    est_beta_topic = est_beta.loc[common_genes, est_topic_name].values
                                    
                                    # Normalize to probabilities
                                    true_beta_topic_norm = true_beta_topic / np.sum(true_beta_topic)
                                    est_beta_topic_norm = est_beta_topic / np.sum(est_beta_topic)
                                    
                                    # Compute percent residuals
                                    epsilon = 1e-10
                                    beta_percent_residuals = np.abs(true_beta_topic_norm - est_beta_topic_norm) / (true_beta_topic_norm + epsilon) * 100
                                    median_beta_residual = np.median(beta_percent_residuals)
                                else:
                                    median_beta_residual = np.nan
                            else:
                                median_beta_residual = np.nan
                            
                            # Store 90th percentile for scatter plots
                            plot_data.append({
                                'Topic': true_topic_name,
                                'Model_Type': model_type,
                                'Dataset': dataset_name,
                                'Topic_Config': estimates_dir.name,
                                'DE_Mean': de_mean,
                                'SNR': snr_value,
                                'Theta_RMSE': median_theta_residual,
                                'Beta_Residual_Pct': median_beta_residual
                            })
                            
                            # Store individual residuals for histogram plotting
                            for residual in theta_absolute_residuals:
                                histogram_data.append({
                                    'Topic': true_topic_name,
                                    'Model_Type': model_type,
                                    'Dataset': dataset_name,
                                    'Topic_Config': estimates_dir.name,
                                    'DE_Mean': de_mean,
                                    'SNR': snr_value,
                                    'Theta_Residual': residual
                                })
                            
                            # Store data for plotting
                            plot_data.append({
                                'Topic': true_topic_name,
                                'Model_Type': model_type,
                                'Dataset': dataset_name,
                                'Topic_Config': estimates_dir.name,
                                'DE_Mean': de_mean,
                                'SNR': snr_value,
                                'Theta_RMSE': median_theta_residual,
                                'Beta_Residual_Pct': median_beta_residual
                            })
                            
                            # For all topics, also compute residuals only for cells that have activity topics
                            # (These are the cells you care about - activity cells)
                            if true_topic_name in ['A', 'B', 'C', 'D', 'V1', 'V2']:
                                # Filter out cells that have theta = 1.0 for identity topics (V1, V2)
                                # What remains are activity cells
                                identity_topics = [topic for topic in ['V1', 'V2'] if topic in train_theta_subset.columns]
                                
                                if len(identity_topics) > 0:
                                    # Find cells that are NOT pure identity cells (theta != 1.0 for identity topics)
                                    activity_cells = ~(train_theta_subset[identity_topics] == 1.0).any(axis=1)
                                    
                                    if np.sum(activity_cells) > 0:
                                        # Get residuals only for cells with activity topics
                                        train_theta_activity = train_theta_subset.loc[activity_cells].iloc[:, train_idx].values
                                        est_theta_activity = est_theta.loc[activity_cells].iloc[:, est_idx].values
                                        
                                        # For HLDA, only consider cells that can actually have this topic
                                        if model_type == 'HLDA':
                                            valid_cells = est_theta_activity > 1e-10
                                            if np.sum(valid_cells) > 0:
                                                train_theta_activity = train_theta_activity[valid_cells]
                                                est_theta_activity = est_theta_activity[valid_cells]
                                            else:
                                                continue
                                        
                                        # Compute 90th percentile absolute residuals for activity cells only
                                        theta_activity_residuals = np.abs(train_theta_activity - est_theta_activity)
                                        theta_activity_90th_percentile = np.percentile(theta_activity_residuals, 90)
                                        
                                        # Store activity-specific data for scatter plots
                                        plot_data.append({
                                            'Topic': f"{true_topic_name}_activity",
                                            'Model_Type': model_type,
                                            'Dataset': dataset_name,
                                            'Topic_Config': estimates_dir.name,
                                            'DE_Mean': de_mean,
                                            'SNR': snr_value,
                                            'Theta_RMSE': theta_activity_90th_percentile,
                                            'Beta_Residual_Pct': median_beta_residual
                                        })
                                        
                                        # Store individual residuals for histogram plotting
                                        for residual in theta_activity_residuals:
                                            histogram_data.append({
                                                'Topic': f"{true_topic_name}_activity",
                                                'Model_Type': model_type,
                                                'Dataset': dataset_name,
                                                'Topic_Config': estimates_dir.name,
                                                'DE_Mean': de_mean,
                                                'SNR': snr_value,
                                                'Theta_Residual': residual
                                            })
                            
                except Exception as e:
                    print(f"  Error processing {dataset_name}/DE_mean_{de_mean}/{model_type}: {e}")
                    continue
    
    if not plot_data:
        print("No data collected for plotting")
        return
    
    # Convert to DataFrame
    plot_df = pd.DataFrame(plot_data)
    print(f"Collected {len(plot_df)} data points for plotting")
    
    # Save the data
    plot_df.to_csv(output_dir / "snr_vs_residual_data.csv", index=False)
    
    # Create scatter plots for each topic
    topics = sorted(plot_df['Topic'].unique())
    
    # Set up colors for model types and shapes for topic configurations
    model_colors = {'HLDA': 'blue', 'LDA': 'red', 'NMF': 'green'}
    topic_config_shapes = {
        '6_topic_fit': 'o',      # Circle
        '3_topic_fit': 's',      # Square
        '3_topic_fit_IMPROVED': '^'  # Triangle
    }
    
    # Create comprehensive theta residual scatter plots (all topics including activity-specific)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Get all topics including activity-specific ones
    all_topics = sorted(plot_df['Topic'].unique())
    
    for i, topic in enumerate(all_topics):
        if i >= len(axes):
            break
            
        ax = axes[i]
        topic_data = plot_df[plot_df['Topic'] == topic]
        
        # Create separate legends for model types and topic configurations
        model_legend_handles = []
        model_legend_labels = []
        config_legend_handles = []
        config_legend_labels = []
        
        # Track what we've already added to avoid duplicates
        added_models = set()
        added_configs = set()
        
        for model_type in ['HLDA', 'LDA', 'NMF']:
            for topic_config in sorted(topic_data['Topic_Config'].unique()):
                model_config_data = topic_data[
                    (topic_data['Model_Type'] == model_type) & 
                    (topic_data['Topic_Config'] == topic_config)
                ]
                if len(model_config_data) > 0:
                    shape = topic_config_shapes.get(topic_config, 'o')
                    color = model_colors[model_type]
                    scatter = ax.scatter(model_config_data['SNR'], model_config_data['Theta_RMSE'], 
                                       c=color, marker=shape, alpha=0.7, s=50)
                    
                    # Add to model legend (only once per model type)
                    if model_type not in added_models:
                        model_legend_handles.append(scatter)
                        model_legend_labels.append(model_type)
                        added_models.add(model_type)
                    
                    # Add to config legend (only once per config)
                    if topic_config not in added_configs:
                        config_legend_handles.append(scatter)
                        config_legend_labels.append(topic_config)
                        added_configs.add(topic_config)
        
        ax.set_xlabel('Signal-to-Noise Ratio')
        
        # Set appropriate labels based on topic type
        if topic.endswith('_activity'):
            base_topic = topic.replace('_activity', '')
            ax.set_ylabel('90th Percentile Theta Residual (Activity Cells Only)')
            ax.set_title(f'Topic {base_topic} - Activity Cells Only')
        else:
            ax.set_ylabel('90th Percentile Theta Residual (All Cells)')
            ax.set_title(f'Topic {topic} - All Cells')
        
        # Create two separate legends
        if model_legend_handles and config_legend_handles:
            # First legend for model types (top right)
            legend1 = ax.legend(model_legend_handles[:len(model_legend_labels)], model_legend_labels, 
                               loc='upper right', title='Model Type')
            ax.add_artist(legend1)
            
            # Second legend for topic configurations (top left)
            legend2 = ax.legend(config_legend_handles[:len(config_legend_labels)], config_legend_labels, 
                               loc='upper left', title='Topic Config')
        
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='plain', axis='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / "snr_vs_theta_residual_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    

    
    # Create beta residual plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, topic in enumerate(topics):
        if i >= len(axes):
            break
            
        ax = axes[i]
        topic_data = plot_df[plot_df['Topic'] == topic]
        
        # Create separate legends for model types and topic configurations
        model_legend_handles = []
        model_legend_labels = []
        config_legend_handles = []
        config_legend_labels = []
        
        # Track what we've already added to avoid duplicates
        added_models = set()
        added_configs = set()
        
        for model_type in ['HLDA', 'LDA', 'NMF']:
            for topic_config in sorted(topic_data['Topic_Config'].unique()):
                model_config_data = topic_data[
                    (topic_data['Model_Type'] == model_type) & 
                    (topic_data['Topic_Config'] == topic_config)
                ]
                if len(model_config_data) > 0:
                    shape = topic_config_shapes.get(topic_config, 'o')
                    color = model_colors[model_type]
                    scatter = ax.scatter(model_config_data['SNR'], model_config_data['Beta_Residual_Pct'], 
                                       c=color, marker=shape, alpha=0.7, s=50)
                    
                    # Add to model legend (only once per model type)
                    if model_type not in added_models:
                        model_legend_handles.append(scatter)
                        model_legend_labels.append(model_type)
                        added_models.add(model_type)
                    
                    # Add to config legend (only once per config)
                    if topic_config not in added_configs:
                        config_legend_handles.append(scatter)
                        config_legend_labels.append(topic_config)
                        added_configs.add(topic_config)
        
        ax.set_xlabel('Signal-to-Noise Ratio')
        ax.set_ylabel('Median Beta Residual (%)')
        ax.set_title(f'Topic {topic}')
        
        # Create two separate legends
        if model_legend_handles and config_legend_handles:
            # First legend for model types (top right)
            legend1 = ax.legend(model_legend_handles[:len(model_legend_labels)], model_legend_labels, 
                               loc='upper right', title='Model Type')
            ax.add_artist(legend1)
            
            # Second legend for topic configurations (top left)
            legend2 = ax.legend(config_legend_handles[:len(config_legend_labels)], config_legend_labels, 
                               loc='upper left', title='Topic Config')
        
        ax.grid(True, alpha=0.3)
        
        # Remove scientific notation (only if using ScalarFormatter)
        try:
            ax.ticklabel_format(style='plain', axis='both')
        except AttributeError:
            pass  # Skip if not using ScalarFormatter
    
    plt.tight_layout()
    plt.savefig(output_dir / "snr_vs_beta_residual_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Now run the comprehensive analysis plots
    create_comprehensive_analysis_plots(snr_data_path, estimates_root, output_dir) 


def create_comprehensive_analysis_plots(snr_data_path: Path, estimates_root: Path, output_dir: Path):
    """
    Create comprehensive analysis plots as specified by the user.
    
    Creates:
    1. Scatter plots of 90th percentile residuals vs DE mean (faceted by topic)
    2. Histogram plots by DE mean (one page per DE mean, faceted by topic)
    3. Scatter plots of 90th percentile residuals vs SNR (faceted by topic)
    4. Same plots but filtered for activity cells only (V1 > 0 OR V2 > 0)
    
    All plots are saved as multi-page PDFs.
    """
    from scipy.optimize import linear_sum_assignment
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    print("Creating comprehensive analysis plots...")
    
    # Load SNR data
    snr_df = pd.read_csv(snr_data_path)
    
    # Get all unique topics
    all_topics = set()
    all_topics.update(snr_df['Activity_Topic'].unique())
    all_topics.update(snr_df['Identity_Topic'].unique())
    all_topics = sorted(list(all_topics))
    
    print(f"Found topics: {all_topics}")
    
    # Process each dataset and DE mean - only simulation data
    simulation_datasets = ['AB_V1', 'ABCD_V1V2', 'ABCD_V1V2_new']
    
    # Collect data for both analyses
    all_cells_data = []
    activity_cells_data = []
    
    for dataset_name in snr_df['Dataset'].unique():
        if dataset_name not in simulation_datasets:
            continue
            
        dataset_snr = snr_df[snr_df['Dataset'] == dataset_name]
        
        for de_mean in dataset_snr['DE_Mean'].unique():
            de_mean_snr = dataset_snr[dataset_snr['DE_Mean'] == de_mean]
            
            # Find the estimates directory
            de_mean_dir = estimates_root / dataset_name / f"DE_mean_{de_mean}"
            if not de_mean_dir.exists():
                continue
            
            # Find topic_fit directories
            topic_fit_dirs = list(de_mean_dir.glob("*topic_fit"))
            if not topic_fit_dirs:
                continue
            
            # Use the first topic_fit directory found
            estimates_dir = topic_fit_dirs[0]
            
            # Load true parameters
            data_dir = Path("data") / dataset_name / f"DE_mean_{de_mean}"
            true_theta_path = data_dir / "theta.csv"
            train_cells_path = data_dir / "train_cells.csv"
            true_beta_path = data_dir / "gene_means.csv"
            
            if not (true_theta_path.exists() and train_cells_path.exists() and true_beta_path.exists()):
                continue
            
            true_theta = pd.read_csv(true_theta_path, index_col=0)
            train_cells = pd.read_csv(train_cells_path)
            true_beta = pd.read_csv(true_beta_path, index_col=0)
            
            # Create train_theta by subsetting true_theta with train_cells
            train_theta = true_theta.loc[train_cells['cell']]
            
            # Process each model type
            for model_type in ['HLDA', 'LDA', 'NMF']:
                model_dir = estimates_dir / model_type
                if not model_dir.exists():
                    continue
                
                # Load estimated parameters
                est_theta_path = model_dir / f"{model_type}_theta.csv"
                est_beta_path = model_dir / f"{model_type}_beta.csv"
                
                if not (est_theta_path.exists() and est_beta_path.exists()):
                    continue
                
                est_theta = pd.read_csv(est_theta_path, index_col=0)
                est_beta = pd.read_csv(est_beta_path, index_col=0)
                
                try:
                    # Use train_theta for matching (test indices)
                    train_theta_subset = train_theta.loc[est_theta.index]
                    
                    # Compute correlation matrix between train theta and estimated topics
                    correlation_matrix = np.zeros((len(train_theta_subset.columns), len(est_theta.columns)))
                    
                    for i, train_topic in enumerate(train_theta_subset.columns):
                        for j, est_topic in enumerate(est_theta.columns):
                            train_vals = train_theta_subset.iloc[:, i].values
                            est_vals = est_theta.iloc[:, j].values
                            correlation_matrix[i, j] = np.corrcoef(train_vals, est_vals)[0, 1]
                    
                    # Use Hungarian algorithm to find optimal matching
                    train_indices, est_indices = linear_sum_assignment(-correlation_matrix)
                    
                    # Compute residuals for each matched topic
                    for train_idx, est_idx in zip(train_indices, est_indices):
                        if train_idx < len(train_theta_subset.columns) and est_idx < len(est_theta.columns):
                            true_topic_name = train_theta_subset.columns[train_idx]
                            est_topic_name = est_theta.columns[est_idx]
                            
                            # Get SNR for this topic
                            topic_snr_data = de_mean_snr[
                                (de_mean_snr['Activity_Topic'] == true_topic_name) | 
                                (de_mean_snr['Identity_Topic'] == true_topic_name)
                            ]
                            
                            if len(topic_snr_data) == 0:
                                continue
                            
                            # Use activity SNR if available, otherwise identity SNR
                            if len(topic_snr_data[topic_snr_data['Activity_Topic'] == true_topic_name]) > 0:
                                snr_value = topic_snr_data[topic_snr_data['Activity_Topic'] == true_topic_name]['Activity_SNR'].iloc[0]
                            else:
                                snr_value = topic_snr_data[topic_snr_data['Identity_Topic'] == true_topic_name]['Identity_SNR'].iloc[0]
                            
                            # Compute theta residuals for all cells
                            train_theta_topic = train_theta_subset.iloc[:, train_idx].values
                            est_theta_topic = est_theta.iloc[:, est_idx].values
                            
                            # For HLDA, only consider cells that can actually have this topic
                            if model_type == 'HLDA':
                                valid_cells = est_theta_topic > 1e-10
                                if np.sum(valid_cells) > 0:
                                    train_theta_topic = train_theta_topic[valid_cells]
                                    est_theta_topic = est_theta_topic[valid_cells]
                                else:
                                    continue
                            
                            # Compute 90th percentile absolute residuals
                            theta_absolute_residuals = np.abs(train_theta_topic - est_theta_topic)
                            theta_90th_percentile = np.percentile(theta_absolute_residuals, 90)
                            
                            # Compute beta residuals
                            median_beta_residual = np.nan
                            if train_idx < len(true_beta.columns) and est_idx < len(est_beta.columns):
                                common_genes = true_beta.index.intersection(est_beta.index)
                                if len(common_genes) > 0:
                                    true_beta_topic = true_beta.loc[common_genes, true_topic_name].values
                                    est_beta_topic = est_beta.loc[common_genes, est_topic_name].values
                                    
                                    # Normalize to probabilities
                                    true_beta_topic_norm = true_beta_topic / np.sum(true_beta_topic)
                                    est_beta_topic_norm = est_beta_topic / np.sum(est_beta_topic)
                                    
                                    # Compute percent residuals
                                    epsilon = 1e-10
                                    beta_percent_residuals = np.abs(true_beta_topic_norm - est_beta_topic_norm) / (true_beta_topic_norm + epsilon) * 100
                                    median_beta_residual = np.median(beta_percent_residuals)
                            
                            # Store data for all cells analysis
                            all_cells_data.append({
                                'Topic': true_topic_name,
                                'Model_Type': model_type,
                                'Dataset': dataset_name,
                                'Topic_Config': estimates_dir.name,
                                'DE_Mean': de_mean,
                                'SNR': snr_value,
                                'Theta_90th_Percentile': theta_90th_percentile,
                                'Beta_Median_Percent': median_beta_residual,
                                'Dataset_DE_Mean': f"{dataset_name}/DE_mean_{de_mean}",
                                'Theta_Residuals': theta_absolute_residuals.tolist()
                            })
                            
                            # Filter for activity cells (V1 > 0 OR V2 > 0)
                            identity_topics = [topic for topic in ['V1', 'V2'] if topic in train_theta_subset.columns]
                            if len(identity_topics) > 0:
                                activity_cells = (train_theta_subset[identity_topics] > 0).any(axis=1)
                                if np.sum(activity_cells) > 0:
                                    # Get residuals only for activity cells
                                    train_theta_activity = train_theta_subset.loc[activity_cells].iloc[:, train_idx].values
                                    est_theta_activity = est_theta.loc[activity_cells].iloc[:, est_idx].values
                                    
                                    # For HLDA, only consider cells that can actually have this topic
                                    if model_type == 'HLDA':
                                        valid_cells = est_theta_activity > 1e-10
                                        if np.sum(valid_cells) > 0:
                                            train_theta_activity = train_theta_activity[valid_cells]
                                            est_theta_activity = est_theta_activity[valid_cells]
                                        else:
                                            continue
                                    
                                    # Compute 90th percentile absolute residuals for activity cells
                                    theta_activity_residuals = np.abs(train_theta_activity - est_theta_activity)
                                    theta_activity_90th_percentile = np.percentile(theta_activity_residuals, 90)
                                    
                                    # Store data for activity cells analysis
                                    activity_cells_data.append({
                                        'Topic': true_topic_name,
                                        'Model_Type': model_type,
                                        'Dataset': dataset_name,
                                        'Topic_Config': estimates_dir.name,
                                        'DE_Mean': de_mean,
                                        'SNR': snr_value,
                                        'Theta_90th_Percentile': theta_activity_90th_percentile,
                                        'Beta_Median_Percent': median_beta_residual,
                                        'Dataset_DE_Mean': f"{dataset_name}/DE_mean_{de_mean}",
                                        'Theta_Residuals': theta_activity_residuals.tolist()
                                    })
                            
                except Exception as e:
                    print(f"  Error processing {dataset_name}/DE_mean_{de_mean}/{model_type}: {e}")
                    continue
    
    # Convert to DataFrames
    all_cells_df = pd.DataFrame(all_cells_data)
    activity_cells_df = pd.DataFrame(activity_cells_data)
    
    print(f"Collected {len(all_cells_df)} data points for all cells analysis")
    print(f"Collected {len(activity_cells_df)} data points for activity cells analysis")
    
    # Create plots for all cells analysis
    if len(all_cells_df) > 0:
        create_analysis_plots(all_cells_df, output_dir, "all_cells")
    
    # Create plots for activity cells analysis
    if len(activity_cells_df) > 0:
        create_analysis_plots(activity_cells_df, output_dir, "activity_cells")
    
    print("✓ Comprehensive analysis plots completed!")


def create_analysis_plots(data_df: pd.DataFrame, output_dir: Path, analysis_type: str):
    """
    Create all the plots for a given analysis type (all_cells or activity_cells).
    """
    from matplotlib.backends.backend_pdf import PdfPages
    
    # Set up colors and shapes
    model_colors = {'HLDA': 'blue', 'LDA': 'red', 'NMF': 'green'}
    topic_config_shapes = {
        '6_topic_fit': 'o',      # Circle
        '3_topic_fit': 's',      # Square
        '3_topic_fit_IMPROVED': '^'  # Triangle
    }
    
    # Get unique topics
    topics = sorted(data_df['Topic'].unique())
    
    # 1. Scatter plots: DE Mean vs 90th Percentile Residual (faceted by topic)
    create_de_mean_scatter_plots(data_df, output_dir, analysis_type, model_colors, topic_config_shapes)
    
    # 2. Histogram plots: by DE Mean (one page per DE mean, faceted by topic)
    create_de_mean_histogram_plots(data_df, output_dir, analysis_type, model_colors)
    
    # 3. Scatter plots: SNR vs 90th Percentile Residual (faceted by topic)
    create_snr_scatter_plots(data_df, output_dir, analysis_type, model_colors, topic_config_shapes)


def create_de_mean_scatter_plots(data_df: pd.DataFrame, output_dir: Path, analysis_type: str, model_colors: dict, topic_config_shapes: dict):
    """Create scatter plots of DE Mean vs 90th Percentile Residual, faceted by topic."""
    from matplotlib.backends.backend_pdf import PdfPages
    
    topics = sorted(data_df['Topic'].unique())
    
    # Theta residuals
    with PdfPages(output_dir / f"{analysis_type}_de_mean_vs_theta_residuals.pdf") as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Collect all legend handles and labels
        all_handles = []
        all_labels = []
        
        for i, topic in enumerate(topics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            topic_data = data_df[data_df['Topic'] == topic]
            
            # Create scatter plot
            for model_type in ['HLDA', 'LDA', 'NMF']:
                for topic_config in topic_data['Topic_Config'].unique():
                    model_config_data = topic_data[
                        (topic_data['Model_Type'] == model_type) & 
                        (topic_data['Topic_Config'] == topic_config)
                    ]
                    
                    if len(model_config_data) > 0:
                        shape = topic_config_shapes.get(topic_config, 'o')
                        color = model_colors[model_type]
                        scatter = ax.scatter(model_config_data['DE_Mean'], model_config_data['Theta_90th_Percentile'], 
                                           c=color, marker=shape, alpha=0.7, s=50, label=f"{model_type}-{topic_config}")
                        
                        # Collect legend handles and labels (only once)
                        if f"{model_type}-{topic_config}" not in all_labels:
                            all_handles.append(scatter)
                            all_labels.append(f"{model_type}-{topic_config}")
            
            ax.set_xlabel('DE Mean')
            ax.set_ylabel('90th Percentile Theta Residual')
            ax.set_title(f'Topic {topic}')
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(style='plain', axis='both')
        
        # Add single legend to the figure
        fig.legend(all_handles, all_labels, loc='center right', bbox_to_anchor=(0.98, 0.5))
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    # Beta residuals
    with PdfPages(output_dir / f"{analysis_type}_de_mean_vs_beta_residuals.pdf") as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Collect all legend handles and labels
        all_handles = []
        all_labels = []
        
        for i, topic in enumerate(topics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            topic_data = data_df[data_df['Topic'] == topic]
            
            # Create scatter plot
            for model_type in ['HLDA', 'LDA', 'NMF']:
                for topic_config in topic_data['Topic_Config'].unique():
                    model_config_data = topic_data[
                        (topic_data['Model_Type'] == model_type) & 
                        (topic_data['Topic_Config'] == topic_config)
                    ]
                    
                    if len(model_config_data) > 0:
                        shape = topic_config_shapes.get(topic_config, 'o')
                        color = model_colors[model_type]
                        scatter = ax.scatter(model_config_data['DE_Mean'], model_config_data['Beta_Median_Percent'], 
                                           c=color, marker=shape, alpha=0.7, s=50, label=f"{model_type}-{topic_config}")
                        
                        # Collect legend handles and labels (only once)
                        if f"{model_type}-{topic_config}" not in all_labels:
                            all_handles.append(scatter)
                            all_labels.append(f"{model_type}-{topic_config}")
            
            ax.set_xlabel('DE Mean')
            ax.set_ylabel('Median Beta Residual (%)')
            ax.set_title(f'Topic {topic}')
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(style='plain', axis='both')
        
        # Add single legend to the figure
        fig.legend(all_handles, all_labels, loc='center right', bbox_to_anchor=(0.98, 0.5))
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()


def create_de_mean_histogram_plots(data_df: pd.DataFrame, output_dir: Path, analysis_type: str, model_colors: dict):
    """Create histogram plots by DE Mean, one page per DE mean, faceted by topic."""
    from matplotlib.backends.backend_pdf import PdfPages
    
    topics = sorted(data_df['Topic'].unique())
    dataset_de_means = sorted(data_df['Dataset_DE_Mean'].unique())
    
    with PdfPages(output_dir / f"{analysis_type}_de_mean_histograms.pdf") as pdf:
        for dataset_de_mean in dataset_de_means:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            de_data = data_df[data_df['Dataset_DE_Mean'] == dataset_de_mean]
            
            # Collect legend handles and labels
            all_handles = []
            all_labels = []
            
            for i, topic in enumerate(topics):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                topic_data = de_data[de_data['Topic'] == topic]
                
                # Create overlapping histograms for each model type
                for model_type in ['HLDA', 'LDA', 'NMF']:
                    model_data = topic_data[topic_data['Model_Type'] == model_type]
                    
                    if len(model_data) > 0:
                        # Flatten all residuals for this model/topic combination
                        all_residuals = []
                        for residuals_list in model_data['Theta_Residuals']:
                            all_residuals.extend(residuals_list)
                        
                        if len(all_residuals) > 0:
                            hist = ax.hist(all_residuals, bins=20, alpha=0.6, color=model_colors[model_type], 
                                         label=model_type, density=True, edgecolor='black', linewidth=0.5)
                            
                            # Collect legend handles and labels (only once)
                            if model_type not in all_labels:
                                all_handles.append(hist[2][0])  # Get the patch object
                                all_labels.append(model_type)
                
                ax.set_xlabel('Theta Residual')
                ax.set_ylabel('Density')
                ax.set_title(f'Topic {topic}')
                ax.grid(True, alpha=0.3)
                ax.ticklabel_format(style='plain', axis='both')
            
            # Add single legend to the figure
            fig.legend(all_handles, all_labels, loc='center right', bbox_to_anchor=(0.98, 0.5))
            plt.suptitle(f'DE Mean: {dataset_de_mean}', fontsize=16)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()


def create_snr_scatter_plots(data_df: pd.DataFrame, output_dir: Path, analysis_type: str, model_colors: dict, topic_config_shapes: dict):
    """Create scatter plots of SNR vs 90th Percentile Residual, faceted by topic."""
    from matplotlib.backends.backend_pdf import PdfPages
    
    topics = sorted(data_df['Topic'].unique())
    
    # Theta residuals
    with PdfPages(output_dir / f"{analysis_type}_snr_vs_theta_residuals.pdf") as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Collect all legend handles and labels
        all_handles = []
        all_labels = []
        
        for i, topic in enumerate(topics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            topic_data = data_df[data_df['Topic'] == topic]
            
            # Create scatter plot
            for model_type in ['HLDA', 'LDA', 'NMF']:
                for topic_config in topic_data['Topic_Config'].unique():
                    model_config_data = topic_data[
                        (topic_data['Model_Type'] == model_type) & 
                        (topic_data['Topic_Config'] == topic_config)
                    ]
                    
                    if len(model_config_data) > 0:
                        shape = topic_config_shapes.get(topic_config, 'o')
                        color = model_colors[model_type]
                        scatter = ax.scatter(model_config_data['SNR'], model_config_data['Theta_90th_Percentile'], 
                                           c=color, marker=shape, alpha=0.7, s=50, label=f"{model_type}-{topic_config}")
                        
                        # Collect legend handles and labels (only once)
                        if f"{model_type}-{topic_config}" not in all_labels:
                            all_handles.append(scatter)
                            all_labels.append(f"{model_type}-{topic_config}")
            
            ax.set_xlabel('Signal-to-Noise Ratio')
            ax.set_ylabel('90th Percentile Theta Residual')
            ax.set_title(f'Topic {topic}')
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(style='plain', axis='both')
        
        # Add single legend to the figure
        fig.legend(all_handles, all_labels, loc='center right', bbox_to_anchor=(0.98, 0.5))
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    # Beta residuals
    with PdfPages(output_dir / f"{analysis_type}_snr_vs_beta_residuals.pdf") as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Collect all legend handles and labels
        all_handles = []
        all_labels = []
        
        for i, topic in enumerate(topics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            topic_data = data_df[data_df['Topic'] == topic]
            
            # Create scatter plot
            for model_type in ['HLDA', 'LDA', 'NMF']:
                for topic_config in topic_data['Topic_Config'].unique():
                    model_config_data = topic_data[
                        (topic_data['Model_Type'] == model_type) & 
                        (topic_data['Topic_Config'] == topic_config)
                    ]
                    
                    if len(model_config_data) > 0:
                        shape = topic_config_shapes.get(topic_config, 'o')
                        color = model_colors[model_type]
                        scatter = ax.scatter(model_config_data['SNR'], model_config_data['Beta_Median_Percent'], 
                                           c=color, marker=shape, alpha=0.7, s=50, label=f"{model_type}-{topic_config}")
                        
                        # Collect legend handles and labels (only once)
                        if f"{model_type}-{topic_config}" not in all_labels:
                            all_handles.append(scatter)
                            all_labels.append(f"{model_type}-{topic_config}")
            
            ax.set_xlabel('Signal-to-Noise Ratio')
            ax.set_ylabel('Median Beta Residual (%)')
            ax.set_title(f'Topic {topic}')
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(style='plain', axis='both')
        
        # Add single legend to the figure
        fig.legend(all_handles, all_labels, loc='center right', bbox_to_anchor=(0.98, 0.5))
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close() 