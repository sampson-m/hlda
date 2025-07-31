#!/usr/bin/env python3
"""
simulation_evaluation_functions.py

Functions for evaluating simulation results, including comparing true vs estimated thetas.
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import scipy.linalg
from scipy.optimize import linear_sum_assignment
import glob
import os

def match_topics_by_beta_correlation(true_beta, est_beta):
    """
    Match estimated topics to true topics using beta correlation.
    Returns mapping from estimated topic indices to true topic indices.
    """
    # Convert to numpy arrays
    if isinstance(true_beta, pd.DataFrame):
        true_beta_vals = true_beta.values
    else:
        true_beta_vals = true_beta
        
    if isinstance(est_beta, pd.DataFrame):
        est_beta_vals = est_beta.values  
    else:
        est_beta_vals = est_beta
    
    # Compute correlation matrix between all pairs of topics
    n_true_topics = true_beta_vals.shape[1]
    n_est_topics = est_beta_vals.shape[1]
    
    # Correlation matrix: est_topics x true_topics
    corr_matrix = np.zeros((n_est_topics, n_true_topics))
    
    for i in range(n_est_topics):
        for j in range(n_true_topics):
            corr_matrix[i, j] = np.corrcoef(est_beta_vals[:, i], true_beta_vals[:, j])[0, 1]
    
    # Use Hungarian algorithm to find optimal matching
    est_indices, true_indices = linear_sum_assignment(-corr_matrix)
    
    # Create mapping dictionary
    topic_mapping = {}
    for est_idx, true_idx in zip(est_indices, true_indices):
        topic_mapping[est_idx] = true_idx
    
    return topic_mapping, corr_matrix

def compare_theta_true_vs_estimated(true_theta_path, estimated_theta_path, output_path=None, method_name="model", 
                                   true_beta_path=None, estimated_beta_path=None):
    """
    Compare true vs estimated theta matrices with scatter plots for each topic.
    Args:
        true_theta_path: Path to the true theta.csv (from simulation)
        estimated_theta_path: Path to the estimated theta file (from model)
        output_path: Path to save the plot (if None, will not save)
        method_name: Name of the estimation method (for plot title)
        true_beta_path: Path to true beta for topic matching (optional)
        estimated_beta_path: Path to estimated beta for topic matching (optional)
    """
    # Load true and estimated thetas
    true_theta = pd.read_csv(true_theta_path, index_col=0)
    est_theta = pd.read_csv(estimated_theta_path, index_col=0)

    # Print first few index values for debugging
    print(f"      [DEBUG] First 5 indices of true_theta: {list(true_theta.index[:5])}")
    print(f"      [DEBUG] First 5 indices of est_theta: {list(est_theta.index[:5])}")

    # Perform topic matching for LDA/NMF models
    if method_name in ["LDA", "NMF"] and true_beta_path and estimated_beta_path:
        true_beta = pd.read_csv(true_beta_path, index_col=0)
        est_beta = pd.read_csv(estimated_beta_path, index_col=0)
        
        # Scale true beta to probabilities
        true_beta_scaled = true_beta.div(true_beta.sum(axis=0), axis=1)
        
        # Match topics
        topic_mapping, _ = match_topics_by_beta_correlation(true_beta_scaled, est_beta)
        
        # Reorder estimated theta columns according to matching
        est_theta_matched = est_theta.copy()
        for est_idx, true_idx in topic_mapping.items():
            if est_idx < est_theta.shape[1] and true_idx < true_theta.shape[1]:
                est_theta_matched.iloc[:, true_idx] = est_theta.iloc[:, est_idx]
        
        est_theta = est_theta_matched

    # Align cells (intersection only)
    common_cells = true_theta.index.intersection(est_theta.index)
    if len(common_cells) == 0:
        raise ValueError("No overlapping cells between true and estimated theta files.")
    true_theta = true_theta.loc[common_cells]
    est_theta = est_theta.loc[common_cells]

    # Align topics (columns)
    min_topics = min(true_theta.shape[1], est_theta.shape[1])
    true_theta = true_theta.iloc[:, :min_topics]
    est_theta = est_theta.iloc[:, :min_topics]

    # Create scatter plots for each topic
    fig, axes = plt.subplots(1, min_topics, figsize=(5*min_topics, 4))
    if min_topics == 1:
        axes = [axes]
    
    for i in range(min_topics):
        ax = axes[i]
        true_vals = true_theta.iloc[:, i].values
        est_vals = est_theta.iloc[:, i].values
        
        # Scatter plot
        ax.scatter(true_vals, est_vals, alpha=0.6, s=20)
        
        # Add diagonal line
        min_val = min(true_vals.min(), est_vals.min())
        max_val = max(true_vals.max(), est_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        topic_label = true_theta.columns[i] if i < len(true_theta.columns) else f"Topic {i+1}"
        ax.set_title(f"{topic_label}")
        ax.set_xlabel(f"True Theta: {topic_label}")
        ax.set_ylabel(f"Estimated Theta: {topic_label}")
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"Theta Scatter Plots: True vs Estimated ({method_name})")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved theta scatter plot to {output_path}")
    else:
        plt.show()
    plt.close()
    
    return fig

def compare_beta_true_vs_estimated(true_beta_path, estimated_beta_path, output_path=None, method_name="model"):
    """
    Compare true vs estimated beta matrices with scatter plots for each topic.
    Args:
        true_beta_path: Path to the true beta/gene_means.csv (from simulation)
        estimated_beta_path: Path to the estimated beta file (from model)
        output_path: Path to save the plot (if None, will not save)
        method_name: Name of the estimation method (for plot title)
    """
    # Load true and estimated betas
    true_beta = pd.read_csv(true_beta_path, index_col=0)
    est_beta = pd.read_csv(estimated_beta_path, index_col=0)
    
    # Scale true beta to be probabilities (normalize by row sums)
    true_beta_scaled = true_beta.div(true_beta.sum(axis=0), axis=1)

    # Perform topic matching for LDA/NMF models
    if method_name in ["LDA", "NMF"]:
        topic_mapping, _ = match_topics_by_beta_correlation(true_beta_scaled, est_beta)
        
        # Reorder estimated beta columns according to matching
        est_beta_matched = est_beta.copy()
        for est_idx, true_idx in topic_mapping.items():
            if est_idx < est_beta.shape[1] and true_idx < true_beta_scaled.shape[1]:
                est_beta_matched.iloc[:, true_idx] = est_beta.iloc[:, est_idx]
        
        est_beta = est_beta_matched

    # Align genes (intersection only)
    common_genes = true_beta_scaled.index.intersection(est_beta.index)
    if len(common_genes) == 0:
        raise ValueError("No overlapping genes between true and estimated beta files.")
    true_beta_scaled = true_beta_scaled.loc[common_genes]
    est_beta = est_beta.loc[common_genes]

    # Align topics (columns)
    min_topics = min(true_beta_scaled.shape[1], est_beta.shape[1])
    true_beta_scaled = true_beta_scaled.iloc[:, :min_topics]
    est_beta = est_beta.iloc[:, :min_topics]

    # Create scatter plots for each topic
    fig, axes = plt.subplots(1, min_topics, figsize=(5*min_topics, 4))
    if min_topics == 1:
        axes = [axes]
    
    for i in range(min_topics):
        ax = axes[i]
        true_vals = true_beta_scaled.iloc[:, i].values
        est_vals = est_beta.iloc[:, i].values
        
        # Scatter plot (subsample if too many points)
        if len(true_vals) > 1000:
            idx = np.random.choice(len(true_vals), 1000, replace=False)
            true_vals = true_vals[idx]
            est_vals = est_vals[idx]
        
        ax.scatter(true_vals, est_vals, alpha=0.6, s=20)
        
        # Add diagonal line
        min_val = min(true_vals.min(), est_vals.min())
        max_val = max(true_vals.max(), est_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        topic_label = true_beta_scaled.columns[i] if i < len(true_beta_scaled.columns) else f"Topic {i+1}"
        ax.set_title(f"{topic_label}")
        ax.set_xlabel("True Beta (Scaled)")
        ax.set_ylabel("Estimated Beta")
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"Beta Scatter Plots: True vs Estimated ({method_name})")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved beta scatter plot to {output_path}")
    else:
        plt.show()
    plt.close()
    
    return fig

def combine_theta_scatter_plots(true_theta_path, estimated_theta_paths, method_names, output_path=None,
                               true_beta_path=None, estimated_beta_paths=None):
    """
    Combine theta scatter plots from multiple models into a single stacked image.
    """
    n_models = len(estimated_theta_paths)
    if n_models != len(method_names):
        raise ValueError("Number of estimated theta paths must match number of method names")
    
    # Load true theta to get number of topics for layout
    true_theta = pd.read_csv(true_theta_path, index_col=0)
    n_topics = true_theta.shape[1]
    
    # Create combined figure
    fig, axes = plt.subplots(n_models, n_topics, figsize=(5*n_topics, 4*n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)
    if n_topics == 1:
        axes = axes.reshape(-1, 1)
    
    # Populate combined figure
    for model_idx, (est_theta_path, method_name) in enumerate(zip(estimated_theta_paths, method_names)):
        # Load data
        true_theta_model = pd.read_csv(true_theta_path, index_col=0)
        est_theta = pd.read_csv(est_theta_path, index_col=0)
        
        # Perform topic matching for LDA/NMF
        if method_name in ["LDA", "NMF"] and true_beta_path and estimated_beta_paths:
            est_beta_path = estimated_beta_paths[model_idx]
            true_beta = pd.read_csv(true_beta_path, index_col=0)
            est_beta = pd.read_csv(est_beta_path, index_col=0)
            
            # Scale true beta to probabilities
            true_beta_scaled = true_beta.div(true_beta.sum(axis=0), axis=1)
            
            # Match topics
            topic_mapping, _ = match_topics_by_beta_correlation(true_beta_scaled, est_beta)
            
            # Reorder estimated theta columns
            est_theta_matched = est_theta.copy()
            for est_idx, true_idx in topic_mapping.items():
                if est_idx < est_theta.shape[1] and true_idx < true_theta_model.shape[1]:
                    est_theta_matched.iloc[:, true_idx] = est_theta.iloc[:, est_idx]
            
            est_theta = est_theta_matched
        
        # Align cells
        common_cells = true_theta_model.index.intersection(est_theta.index)
        true_theta_model = true_theta_model.loc[common_cells]
        est_theta = est_theta.loc[common_cells]
        
        # Align topics
        min_topics = min(true_theta_model.shape[1], est_theta.shape[1])
        true_theta_model = true_theta_model.iloc[:, :min_topics]
        est_theta = est_theta.iloc[:, :min_topics]
        
        # Create scatter plots for each topic
        for topic_idx in range(min_topics):
            ax = axes[model_idx, topic_idx]
            true_vals = true_theta_model.iloc[:, topic_idx].values
            est_vals = est_theta.iloc[:, topic_idx].values
            
            # Scatter plot
            ax.scatter(true_vals, est_vals, alpha=0.6, s=20)
            
            # Add diagonal line
            min_val = min(true_vals.min(), est_vals.min())
            max_val = max(true_vals.max(), est_vals.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            topic_label = true_theta_model.columns[topic_idx] if topic_idx < len(true_theta_model.columns) else f"Topic {topic_idx+1}"
            ax.set_title(f"{method_name} - {topic_label}")
            ax.set_xlabel(f"True Theta: {topic_label}")
            ax.set_ylabel(f"Estimated Theta: {topic_label}")
            ax.grid(True, alpha=0.3)
    
    plt.suptitle("Combined Theta Scatter Plots: True vs Estimated")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined theta scatter plot to {output_path}")
    else:
        plt.show()
    plt.close()
    
    return fig

def combine_beta_scatter_plots(true_beta_path, estimated_beta_paths, method_names, output_path=None):
    """
    Combine beta scatter plots from multiple models into a single stacked image.
    """
    n_models = len(estimated_beta_paths)
    if n_models != len(method_names):
        raise ValueError("Number of estimated beta paths must match number of method names")
    
    # Load true beta to get number of topics for layout
    true_beta = pd.read_csv(true_beta_path, index_col=0)
    n_topics = true_beta.shape[1]
    
    # Create combined figure
    fig, axes = plt.subplots(n_models, n_topics, figsize=(5*n_topics, 4*n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)
    if n_topics == 1:
        axes = axes.reshape(-1, 1)
    
    # Populate combined figure
    for model_idx, (est_beta_path, method_name) in enumerate(zip(estimated_beta_paths, method_names)):
        # Load data
        true_beta_model = pd.read_csv(true_beta_path, index_col=0)
        est_beta = pd.read_csv(est_beta_path, index_col=0)
        
        # Scale true beta to probabilities
        true_beta_scaled = true_beta_model.div(true_beta_model.sum(axis=0), axis=1)
        
        # Perform topic matching for LDA/NMF
        topic_mapping = {}
        if method_name in ["LDA", "NMF"]:
            topic_mapping, _ = match_topics_by_beta_correlation(true_beta_scaled, est_beta)
        else:
            # For HLDA, assume topics are already aligned
            topic_mapping = {i: i for i in range(min(n_topics, est_beta.shape[1]))}
        
        # Create scatter plots for each topic
        for topic_idx in range(n_topics):
            ax = axes[model_idx, topic_idx]
            
            if topic_idx in topic_mapping:
                matched_topic = topic_mapping[topic_idx]
                if matched_topic < est_beta.shape[1]:
                    true_vals = true_beta_scaled.iloc[:, topic_idx].values
                    est_vals = est_beta.iloc[:, matched_topic].values
                    
                    # Create scatter plot
                    ax.scatter(true_vals, est_vals, alpha=0.6, s=1)
                    
                    # Add diagonal line
                    min_val = min(true_vals.min(), est_vals.min())
                    max_val = max(true_vals.max(), est_vals.max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=1)
                    
                    # Calculate correlation
                    corr = np.corrcoef(true_vals, est_vals)[0, 1]
                    
                    # Labels and title
                    ax.set_xlabel(f'True β (Topic {topic_idx})')
                    ax.set_ylabel(f'Est β (Topic {matched_topic})')
                    ax.set_title(f'{method_name} - Topic {topic_idx}\nr={corr:.3f}')
                else:
                    ax.text(0.5, 0.5, f'No match for\nTopic {topic_idx}', 
                           transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'{method_name} - Topic {topic_idx}')
            else:
                ax.text(0.5, 0.5, f'No match for\nTopic {topic_idx}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{method_name} - Topic {topic_idx}')
            
            # Set equal aspect ratio
            ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined beta scatter plot to {output_path}")
    else:
        plt.show()
    plt.close()
    
    return fig


def create_stacked_comparison_plots(results_dir, output_dir, true_theta=None, true_beta=None, dataset_name=""):
    """
    Create stacked comparison plots for all sampling methods.
    
    Parameters:
    -----------
    results_dir : str
        Directory containing results from test_hlda_sampling_methods.py
    output_dir : str
        Directory to save comparison plots
    true_theta : str, optional
        Path to true theta.csv file
    true_beta : str, optional
        Path to true beta.csv file
    dataset_name : str
        Name of the dataset for plot titles
    """
    
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define sampling methods to compare
    methods = ['standard', 'annealing', 'blocked', 'tempered']
    method_colors = {
        'standard': '#1f77b4',
        'annealing': '#ff7f0e', 
        'blocked': '#2ca02c',
        'tempered': '#d62728'
    }
    
    # Load results for each method
    results = {}
    for method in methods:
        method_dir = results_path / method
        if method_dir.exists():
            beta_file = method_dir / f"{method}_beta.csv"
            theta_file = method_dir / f"{method}_theta.csv"
            
            if beta_file.exists() and theta_file.exists():
                results[method] = {
                    'beta': pd.read_csv(beta_file, index_col=0),
                    'theta': pd.read_csv(theta_file, index_col=0),
                    'method_dir': method_dir
                }
                print(f"✓ Loaded {method} results")
            else:
                print(f"✗ Missing files for {method}")
        else:
            print(f"✗ Method directory not found: {method}")
    
    if not results:
        print("No results found!")
        return
    
    # Create stacked comparison plots
    print(f"\nCreating comparison plots...")
    
    # 1. Beta comparison (gene-topic distributions)
    if true_beta and Path(true_beta).exists():
        print("  Creating beta comparison plot...")
        create_beta_comparison_plot(results, true_beta, output_path, method_colors, dataset_name)
    
    # 2. Theta comparison (cell-topic distributions)  
    if true_theta and Path(true_theta).exists():
        print("  Creating theta comparison plot...")
        create_theta_comparison_plot(results, true_theta, output_path, method_colors, dataset_name)
    
    # 3. Summary statistics table
    print("  Creating summary statistics...")
    create_summary_table(results, output_path, dataset_name)
    
    print(f"\n✓ Comparison plots saved to: {output_path}")


def create_beta_comparison_plot(results, true_beta_path, output_path, method_colors, dataset_name):
    """Create stacked beta comparison plot using existing functions."""
    
    # Prepare paths and method names for the existing combine function
    estimated_beta_paths = []
    method_names = []
    
    for method in ['standard', 'annealing', 'blocked', 'tempered']:
        if method in results:
            beta_file = results[method]['method_dir'] / f"{method}_beta.csv"
            if beta_file.exists():
                estimated_beta_paths.append(str(beta_file))
                method_names.append(method.upper())
    
    if estimated_beta_paths:
        # Use the existing combine function
        combine_beta_scatter_plots(
            true_beta_path=true_beta_path,
            estimated_beta_paths=estimated_beta_paths,
            method_names=method_names,
            output_path=str(output_path / 'beta_comparison_stacked.png')
        )
        print(f"  ✓ Beta comparison plot created using existing function")


def create_theta_comparison_plot(results, true_theta_path, output_path, method_colors, dataset_name):
    """Create stacked theta comparison plot using existing functions."""
    
    # First, fix cell indices in the theta files to match the true theta format
    # Look for train_cells.csv in the same directory as true_theta_path
    true_theta_dir = Path(true_theta_path).parent
    train_cells_file = true_theta_dir / "train_cells.csv"
    
    if train_cells_file.exists():
        print(f"  Fixing cell indices using {train_cells_file}")
        train_cells = pd.read_csv(train_cells_file)['cell']
        
        for method in ['standard', 'annealing', 'blocked', 'tempered']:
            if method in results:
                theta_file = results[method]['method_dir'] / f"{method}_theta.csv"
                if theta_file.exists():
                    train_theta = pd.read_csv(theta_file, index_col=0)
                    if len(train_theta) == len(train_cells):
                        train_theta.index = train_cells
                        train_theta.to_csv(theta_file)
                        print(f"    ✓ Fixed cell indices for {method}")
                    else:
                        print(f"    ⚠ Length mismatch for {method}: {len(train_theta)} vs {len(train_cells)}")
    
    # Prepare paths and method names for the existing combine function
    estimated_theta_paths = []
    method_names = []
    
    for method in ['standard', 'annealing', 'blocked', 'tempered']:
        if method in results:
            theta_file = results[method]['method_dir'] / f"{method}_theta.csv"
            if theta_file.exists():
                estimated_theta_paths.append(str(theta_file))
                method_names.append(method.upper())
    
    if estimated_theta_paths:
        try:
            # Use the existing combine function
            combine_theta_scatter_plots(
                true_theta_path=true_theta_path,
                estimated_theta_paths=estimated_theta_paths,
                method_names=method_names,
                output_path=str(output_path / 'theta_comparison_stacked.png')
            )
            print(f"  ✓ Theta comparison plot created using existing function")
        except Exception as e:
            print(f"  ⚠ Warning: Could not create theta comparison plot: {e}")
            print(f"    This may be due to cell index mismatches between true and estimated theta files")
            print(f"    Skipping theta comparison plot")


def create_summary_table(results, output_path, dataset_name):
    """Create summary statistics table."""
    
    summary_data = []
    
    for method, data in results.items():
        beta = data['beta']
        theta = data['theta']
        
        # Calculate summary statistics
        beta_stats = {
            'Method': method.upper(),
            'Beta_Mean': beta.values.mean(),
            'Beta_Std': beta.values.std(),
            'Beta_Min': beta.values.min(),
            'Beta_Max': beta.values.max(),
            'Theta_Mean': theta.values.mean(),
            'Theta_Std': theta.values.std(),
            'Theta_Min': theta.values.min(),
            'Theta_Max': theta.values.max(),
        }
        summary_data.append(beta_stats)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path / 'summary_statistics.csv', index=False)
    
    # Create a nice formatted table
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Format the data for display
    display_df = summary_df.copy()
    for col in display_df.columns:
        if col != 'Method' and col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}')
    
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns, 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title(f'Summary Statistics - {dataset_name}', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path / 'summary_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()


def compute_residual_eigenvalues_and_plot(data_root="data", samples_root="samples", output_dir="eigenvalue_analysis"):
    """
    Compute X - Lθβ for all simulation data by DE mean, calculate eigenvalues, and plot them.
    
    Parameters:
    -----------
    data_root : str
        Root directory containing simulation data
    samples_root : str
        Root directory containing fitted model samples
    output_dir : str
        Directory to save eigenvalue plots
    """
    data_path = Path(data_root)
    samples_path = Path(samples_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all DE_mean directories
    de_mean_dirs = []
    for dataset_dir in data_path.iterdir():
        if dataset_dir.is_dir() and not dataset_dir.name.startswith('_'):
            for de_dir in dataset_dir.iterdir():
                if de_dir.is_dir() and de_dir.name.startswith('DE_mean_'):
                    de_mean_dirs.append((dataset_dir.name, de_dir))
    
    print(f"Found {len(de_mean_dirs)} DE_mean directories")
    
    # Process each DE_mean directory
    all_eigenvalues = {}
    
    for dataset_name, de_dir in de_mean_dirs:
        de_mean = de_dir.name
        print(f"\nProcessing {dataset_name}/{de_mean}")
        
        # Load true data
        counts_file = de_dir / "filtered_counts_train.csv"
        theta_file = de_dir / "theta.csv"
        gene_means_file = de_dir / "gene_means.csv"
        library_sizes_file = de_dir / "library_sizes.csv"
        
        if not all(f.exists() for f in [counts_file, theta_file, gene_means_file, library_sizes_file]):
            print(f"  Skipping {de_mean} - missing required files")
            continue
        
        # Load data
        X = pd.read_csv(counts_file, index_col=0)  # Count matrix
        theta_true = pd.read_csv(theta_file, index_col=0)  # True theta
        beta_true = pd.read_csv(gene_means_file, index_col=0)  # True beta (gene means)
        
        # Use theta directly - it should contain all cells
        # Align genes between X and beta
        common_genes = X.columns.intersection(beta_true.index)
        
        if len(common_genes) == 0:
            print(f"  Skipping {de_mean} - no common genes")
            continue
        
        X = X.loc[:, common_genes]
        beta_true = beta_true.loc[common_genes]
        
        # Align cells between X and theta (use intersection)
        common_cells = X.index.intersection(theta_true.index)
        if len(common_cells) == 0:
            print(f"  Skipping {de_mean} - no common cells")
            continue
            
        X = X.loc[common_cells, :]
        theta_true = theta_true.loc[common_cells, :]
        
        # Convert to numpy arrays
        X_np = X.values.astype(float)
        theta_np = theta_true.values.astype(float)
        beta_np = beta_true.values.astype(float)
        L_np = np.full(X_np.shape[0], 1500.0)  # Constant library size of 1500
        
        # Normalize theta to be probabilities (rows sum to 1)
        theta_np = theta_np / theta_np.sum(axis=1, keepdims=True)
        
        # Normalize gene means (beta) to be probabilities (columns sum to 1)
        beta_np = beta_np / beta_np.sum(axis=0, keepdims=True)
        
        # Compute Lθβ
        # L is a vector of library sizes, theta is cells x topics, beta is genes x topics
        # We want: L * (theta @ beta.T) where L is broadcasted to match the result
        theta_beta = theta_np @ beta_np.T  # cells x genes
        L_theta_beta = np.outer(L_np, np.ones(X_np.shape[1])) * theta_beta
        
        # Compute residual: X - Lθβ
        residual = X_np - L_theta_beta
        
        # Compute eigenvalues of residual matrix
        try:
            # Use SVD to get eigenvalues (more stable for non-square matrices)
            U, s, Vt = np.linalg.svd(residual, full_matrices=False)
            eigenvalues = s**2  # Convert singular values to eigenvalues
            
            # Store results
            key = f"{dataset_name}_{de_mean}"
            all_eigenvalues[key] = {
                'eigenvalues': eigenvalues,
                'residual_norm': np.linalg.norm(residual, 'fro'),
                'X_norm': np.linalg.norm(X_np, 'fro'),
                'relative_residual': np.linalg.norm(residual, 'fro') / np.linalg.norm(X_np, 'fro'),
                'shape': X_np.shape
            }
            
            print(f"  ✓ Computed eigenvalues for {X_np.shape[0]} cells × {X_np.shape[1]} genes")
            print(f"    Relative residual norm: {all_eigenvalues[key]['relative_residual']:.4f}")
            
        except Exception as e:
            print(f"  ✗ Error computing eigenvalues: {e}")
            continue
    
    # Create plots
    if not all_eigenvalues:
        print("No eigenvalue data to plot")
        return
    
    # 1. Individual eigenvalue plots for each dataset
    print("\nCreating individual eigenvalue plots...")
    for key, data in all_eigenvalues.items():
        eigenvalues = data['eigenvalues']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Eigenvalue spectrum
        ax1.semilogy(range(1, len(eigenvalues) + 1), eigenvalues, 'b-', linewidth=2)
        ax1.set_xlabel('Eigenvalue Index')
        ax1.set_ylabel('Eigenvalue Magnitude (log scale)')
        ax1.set_title(f'Eigenvalue Spectrum - {key}')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative explained variance
        cumulative_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        ax2.plot(range(1, len(eigenvalues) + 1), cumulative_var, 'r-', linewidth=2)
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title(f'Cumulative Explained Variance - {key}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / f'eigenvalues_{key}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Combined eigenvalue comparison plot
    print("Creating combined eigenvalue comparison plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: All eigenvalue spectra
    for key, data in all_eigenvalues.items():
        eigenvalues = data['eigenvalues']
        # Normalize eigenvalues for comparison
        normalized_eigenvalues = eigenvalues / np.sum(eigenvalues)
        ax1.semilogy(range(1, len(eigenvalues) + 1), normalized_eigenvalues, 
                    label=key, alpha=0.7, linewidth=1.5)
    
    ax1.set_xlabel('Eigenvalue Index')
    ax1.set_ylabel('Normalized Eigenvalue Magnitude (log scale)')
    ax1.set_title('Eigenvalue Spectra Comparison')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Relative residual norms
    de_means = []
    residual_norms = []
    for key, data in all_eigenvalues.items():
        de_mean = key.split('_')[-1]  # Extract DE_mean value
        de_means.append(float(de_mean.replace('DE_mean_', '')))
        residual_norms.append(data['relative_residual'])
    
    # Sort by DE_mean
    sorted_indices = np.argsort(de_means)
    de_means = [de_means[i] for i in sorted_indices]
    residual_norms = [residual_norms[i] for i in sorted_indices]
    
    ax2.plot(de_means, residual_norms, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('DE Mean')
    ax2.set_ylabel('Relative Residual Norm ||X - Lθβ|| / ||X||')
    ax2.set_title('Residual Norm vs DE Mean')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path / 'eigenvalue_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Summary statistics
    print("Creating summary statistics...")
    summary_data = []
    for key, data in all_eigenvalues.items():
        de_mean = key.split('_')[-1]
        summary_data.append({
            'Dataset': key,
            'DE_Mean': float(de_mean.replace('DE_mean_', '')),
            'Relative_Residual_Norm': data['relative_residual'],
            'Residual_Norm': data['residual_norm'],
            'X_Norm': data['X_norm'],
            'Shape': f"{data['shape'][0]}×{data['shape'][1]}",
            'Max_Eigenvalue': np.max(data['eigenvalues']),
            'Min_Eigenvalue': np.min(data['eigenvalues']),
            'Eigenvalue_Range': np.max(data['eigenvalues']) / np.min(data['eigenvalues'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('DE_Mean')
    summary_df.to_csv(output_path / 'eigenvalue_summary.csv', index=False)
    
    # Create summary table plot
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Format data for display
    display_df = summary_df.copy()
    for col in ['Relative_Residual_Norm', 'Residual_Norm', 'X_Norm', 'Max_Eigenvalue', 'Min_Eigenvalue', 'Eigenvalue_Range']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}')
    
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns, 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    plt.title('Eigenvalue Analysis Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_path / 'eigenvalue_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Eigenvalue analysis completed!")
    print(f"  Results saved to: {output_path}")
    print(f"  Processed {len(all_eigenvalues)} datasets")
    
    return all_eigenvalues


def compute_eigenvalue_histograms(data_root="data", output_dir="eigenvalue_histograms"):
    """
    Compute X - Lθβ for all simulation data by DE mean, calculate eigenvalues, and create histograms.
    
    Parameters:
    -----------
    data_root : str
        Root directory containing simulation data
    output_dir : str
        Directory to save eigenvalue histograms
    """
    data_path = Path(data_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all DE_mean directories
    de_mean_dirs = []
    for dataset_dir in data_path.iterdir():
        if dataset_dir.is_dir() and not dataset_dir.name.startswith('_'):
            for de_dir in dataset_dir.iterdir():
                if de_dir.is_dir() and de_dir.name.startswith('DE_mean_'):
                    de_mean_dirs.append((dataset_dir.name, de_dir))
    
    print(f"Found {len(de_mean_dirs)} DE_mean directories")
    
    # Process each DE_mean directory
    all_eigenvalues = {}
    
    for dataset_name, de_dir in de_mean_dirs:
        de_mean = de_dir.name
        print(f"\nProcessing {dataset_name}/{de_mean}")
        
        # Load true data
        counts_file = de_dir / "filtered_counts_train.csv"
        theta_file = de_dir / "theta.csv"
        gene_means_file = de_dir / "gene_means.csv"
        
        if not all(f.exists() for f in [counts_file, theta_file, gene_means_file]):
            print(f"  Skipping {de_mean} - missing required files")
            continue
        
        # Load data
        X = pd.read_csv(counts_file, index_col=0)  # Count matrix
        theta_true = pd.read_csv(theta_file, index_col=0)  # True theta
        beta_true = pd.read_csv(gene_means_file, index_col=0)  # True beta (gene means)
        
        # Align genes between X and beta
        common_genes = X.columns.intersection(beta_true.index)
        
        if len(common_genes) == 0:
            print(f"  Skipping {de_mean} - no common genes")
            continue
        
        X = X.loc[:, common_genes]
        beta_true = beta_true.loc[common_genes]
        
        # Align cells between X and theta (use intersection)
        common_cells = X.index.intersection(theta_true.index)
        if len(common_cells) == 0:
            print(f"  Skipping {de_mean} - no common cells")
            continue
            
        X = X.loc[common_cells, :]
        theta_true = theta_true.loc[common_cells, :]
        
        # Convert to numpy arrays
        X_np = X.values.astype(float)
        theta_np = theta_true.values.astype(float)
        beta_np = beta_true.values.astype(float)
        L_np = np.full(X_np.shape[0], 1500.0)  # Constant library size of 1500
        
        # Normalize theta to be probabilities (rows sum to 1)
        theta_np = theta_np / theta_np.sum(axis=1, keepdims=True)
        
        # Normalize gene means (beta) to be probabilities (columns sum to 1)
        beta_np = beta_np / beta_np.sum(axis=0, keepdims=True)
        
        # Compute Lθβ
        # L is a vector of library sizes, theta is cells x topics, beta is genes x topics
        # We want: L * (theta @ beta.T) where L is broadcasted to match the result
        theta_beta = theta_np @ beta_np.T  # cells x genes
        L_theta_beta = np.outer(L_np, np.ones(X_np.shape[1])) * theta_beta
        
        # Compute residual: X - Lθβ
        residual = (X_np - L_theta_beta) / np.sqrt(1500)
        
        # Compute eigenvalues of residual matrix
        try:
            # Use SVD to get eigenvalues (more stable for non-square matrices)
            U, s, Vt = np.linalg.svd(residual, full_matrices=False)
            eigenvalues = s**2  # Convert singular values to eigenvalues
            
            # Store results
            key = f"{dataset_name}_{de_mean}"
            all_eigenvalues[key] = {
                'eigenvalues': eigenvalues,
                'residual_norm': np.linalg.norm(residual, 'fro'),
                'X_norm': np.linalg.norm(X_np, 'fro'),
                'relative_residual': np.linalg.norm(residual, 'fro') / np.linalg.norm(X_np, 'fro'),
                'shape': X_np.shape
            }
            
            print(f"  ✓ Computed eigenvalues for {X_np.shape[0]} cells × {X_np.shape[1]} genes")
            print(f"    Relative residual norm: {all_eigenvalues[key]['relative_residual']:.4f}")
            
        except Exception as e:
            print(f"  ✗ Error computing eigenvalues: {e}")
            continue
    
    # Create histograms
    if not all_eigenvalues:
        print("No eigenvalue data to plot")
        return
    
    # 1. Individual histograms for each dataset
    print("\nCreating individual eigenvalue histograms...")
    for key, data in all_eigenvalues.items():
        eigenvalues = data['eigenvalues']
        
        plt.figure(figsize=(10, 6))
        
        # Create histogram with log scale
        plt.hist(eigenvalues, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
        plt.ylabel('Frequency')
        plt.title(f'Eigenvalue Distribution - {key}\nRelative Residual: {data["relative_residual"]:.4f}')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / f'eigenvalue_histogram_{key}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Combined histogram comparison
    print("Creating combined histogram comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: All histograms overlaid
    for key, data in all_eigenvalues.items():
        eigenvalues = data['eigenvalues']
        de_mean = float(key.split('_')[-1].replace('DE_mean_', ''))
        ax1.hist(eigenvalues, bins=30, alpha=0.3, label=f'DE={de_mean}', density=True)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Eigenvalue Magnitude (log scale)')
    ax1.set_ylabel('Density')
    ax1.set_title('Eigenvalue Distributions (All DE Means)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residual norms vs DE mean
    de_means = []
    residual_norms = []
    for key, data in all_eigenvalues.items():
        de_mean = float(key.split('_')[-1].replace('DE_mean_', ''))
        de_means.append(de_mean)
        residual_norms.append(data['relative_residual'])
    
    # Sort by DE_mean
    sorted_indices = np.argsort(de_means)
    de_means = [de_means[i] for i in sorted_indices]
    residual_norms = [residual_norms[i] for i in sorted_indices]
    
    ax2.plot(de_means, residual_norms, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('DE Mean')
    ax2.set_ylabel('Relative Residual Norm ||X - Lθβ|| / ||X||')
    ax2.set_title('Residual Norm vs DE Mean')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path / 'eigenvalue_histogram_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Summary statistics
    print("Creating summary statistics...")
    summary_data = []
    for key, data in all_eigenvalues.items():
        de_mean = key.split('_')[-1]
        summary_data.append({
            'Dataset': key,
            'DE_Mean': float(de_mean.replace('DE_mean_', '')),
            'Relative_Residual_Norm': data['relative_residual'],
            'Residual_Norm': data['residual_norm'],
            'X_Norm': data['X_norm'],
            'Shape': f"{data['shape'][0]}×{data['shape'][1]}",
            'Max_Eigenvalue': np.max(data['eigenvalues']),
            'Min_Eigenvalue': np.min(data['eigenvalues']),
            'Mean_Eigenvalue': np.mean(data['eigenvalues']),
            'Median_Eigenvalue': np.median(data['eigenvalues']),
            'Eigenvalue_Range': np.max(data['eigenvalues']) / np.min(data['eigenvalues'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('DE_Mean')
    summary_df.to_csv(output_path / 'eigenvalue_histogram_summary.csv', index=False)
    
    print(f"\n✓ Eigenvalue histogram analysis completed!")
    print(f"  Results saved to: {output_path}")
    print(f"  Processed {len(all_eigenvalues)} datasets")
    
    return all_eigenvalues


def compute_theta_beta_svd_analysis(data_root="data"):
    """
    Compute SVD of θβ^T, extract singular values, and multiply by √L.
    
    Parameters:
    -----------
    data_root : str
        Root directory containing simulation data
    """
    data_path = Path(data_root)
    
    # Find all DE_mean directories
    de_mean_dirs = []
    for dataset_dir in data_path.iterdir():
        if dataset_dir.is_dir() and not dataset_dir.name.startswith('_'):
            for de_dir in dataset_dir.iterdir():
                if de_dir.is_dir() and de_dir.name.startswith('DE_mean_'):
                    de_mean_dirs.append((dataset_dir.name, de_dir))
    
    print(f"Found {len(de_mean_dirs)} DE_mean directories")
    
    # Process each DE_mean directory
    for dataset_name, de_dir in de_mean_dirs:
        de_mean = de_dir.name
        print(f"\nProcessing {dataset_name}/{de_mean}")
        
        # Load true data
        theta_file = de_dir / "theta.csv"
        gene_means_file = de_dir / "gene_means.csv"
        
        if not all(f.exists() for f in [theta_file, gene_means_file]):
            print(f"  Skipping {de_mean} - missing required files")
            continue
        
        # Load data
        theta_true = pd.read_csv(theta_file, index_col=0)  # True theta
        beta_true = pd.read_csv(gene_means_file, index_col=0)  # True beta (gene means)
        
        # Convert to numpy arrays
        theta_np = theta_true.values.astype(float)
        beta_np = beta_true.values.astype(float)
        
        # Normalize theta to be probabilities (rows sum to 1)
        theta_np = theta_np / theta_np.sum(axis=1, keepdims=True)
        
        # Normalize gene means (beta) to be probabilities (columns sum to 1)
        beta_np = beta_np / beta_np.sum(axis=0, keepdims=True)
        
        # Compute θβ^T
        theta_beta_T = theta_np @ beta_np.T
        
        # Compute SVD of θβ^T
        try:
            U, s, Vt = np.linalg.svd(theta_beta_T, full_matrices=False)
            
            # Multiply singular values by √L where L = 1500
            L = 1500.0
            sqrt_L = np.sqrt(L)
            s_scaled = s * sqrt_L
            
            print(f"  ✓ Computed SVD for {theta_beta_T.shape[0]} cells × {theta_beta_T.shape[1]} genes")
            print(f"    Number of singular values: {len(s)}")
            print(f"    √L = {sqrt_L:.2f}")
            print(f"    Singular values (s_i): {s}")
            print(f"    Scaled singular values (s_i * √L): {s_scaled}")
            print(f"    Sum of singular values: {np.sum(s):.4f}")
            print(f"    Sum of scaled singular values: {np.sum(s_scaled):.4f}")
            
        except Exception as e:
            print(f"  ✗ Error computing SVD: {e}")
            continue


def compute_noise_analysis(data_root="data", output_dir="estimates"):
    """
    Implement the noise analysis from the LaTeX derivation.
    
    This function computes:
    1. The orthogonal component of activity topics from identity topics
    2. Signal-to-noise ratios for identity and activity topics
    3. Determines thresholds where signal is too low for accurate model fitting
    
    Parameters:
    -----------
    data_root : str
        Root directory containing simulation data
    output_dir : str
        Root directory for estimates (will create noise_analysis subdirectories)
    """
    data_path = Path(data_root)
    estimates_path = Path(output_dir)
    
    # Find all DE_mean directories
    de_mean_dirs = []
    for dataset_dir in data_path.iterdir():
        if dataset_dir.is_dir() and not dataset_dir.name.startswith('_'):
            for de_dir in dataset_dir.iterdir():
                if de_dir.is_dir() and de_dir.name.startswith('DE_mean_'):
                    de_mean_dirs.append((dataset_dir.name, de_dir))
    
    print(f"Found {len(de_mean_dirs)} DE_mean directories")
    
    # Process each DE_mean directory
    all_results = {}
    
    for dataset_name, de_dir in de_mean_dirs:
        de_mean = de_dir.name
        print(f"\nProcessing {dataset_name}/{de_mean}")
        
        # Load true data - handle both AB_V1 and ABCD_V1V2 formats
        counts_file = de_dir / "counts.csv"  # AB_V1 format
        if not counts_file.exists():
            counts_file = de_dir / "filtered_counts_train.csv"  # ABCD_V1V2 format
        
        theta_file = de_dir / "theta.csv"
        gene_means_file = de_dir / "gene_means.csv"
        library_sizes_file = de_dir / "library_sizes.csv"
        
        if not all(f.exists() for f in [counts_file, theta_file, gene_means_file, library_sizes_file]):
            print(f"  Skipping {de_mean} - missing required files")
            continue
        
        # Load data
        X = pd.read_csv(counts_file, index_col=0)  # Count matrix
        theta_true = pd.read_csv(theta_file, index_col=0)  # True theta
        beta_true = pd.read_csv(gene_means_file, index_col=0)  # True beta (gene means)
        
        # Align genes between X and beta
        common_genes = X.columns.intersection(beta_true.index)
        if len(common_genes) == 0:
            print(f"  Skipping {de_mean} - no common genes")
            continue
        
        X = X.loc[:, common_genes]
        beta_true = beta_true.loc[common_genes]
        
        # Align cells between X and theta
        common_cells = X.index.intersection(theta_true.index)
        if len(common_cells) == 0:
            print(f"  Skipping {de_mean} - no common cells")
            continue
            
        X = X.loc[common_cells, :]
        theta_true = theta_true.loc[common_cells, :]
        
        # Convert to numpy arrays
        X_np = X.values.astype(float)
        theta_np = theta_true.values.astype(float)
        beta_np = beta_true.values.astype(float)
        L = 1500.0  # Constant library size
        
        # Normalize theta to be probabilities (rows sum to 1)
        theta_np = theta_np / theta_np.sum(axis=1, keepdims=True)
        
        # Normalize gene means (beta) to be probabilities (columns sum to 1)
        beta_np = beta_np / beta_np.sum(axis=0, keepdims=True)
        
        # Identify identity topics (A, B) and activity topic (VAR/V1)
        # AB_V1 format: [A, B, V1], ABCD_V1V2 format: [A, B, C, D, V1, V2]
        topic_names = theta_true.columns.tolist()
        identity_topics = [i for i, name in enumerate(topic_names) if name in ['A', 'B', 'C', 'D']]
        activity_topics = [i for i, name in enumerate(topic_names) if name in ['VAR', 'V1', 'V2']]
        
        if not identity_topics or not activity_topics:
            print(f"  Skipping {de_mean} - could not identify identity/activity topics")
            continue
        
        print(f"    Identity topics: {[topic_names[i] for i in identity_topics]}")
        print(f"    Activity topics: {[topic_names[i] for i in activity_topics]}")
        
        # Step 1: Compute noise term (Poisson noise)
        # X = Lθβ + (Poisson(Lθβ) - Lθβ)
        # Noise term = (Poisson(Lθβ) - Lθβ) / √L
        theta_beta = theta_np @ beta_np.T  # cells x genes
        L_theta_beta = L * theta_beta
        noise_term = (X_np - L_theta_beta) / np.sqrt(L)
        
        # Compute noise eigenvalues via PCA
        try:
            U, s, Vt = np.linalg.svd(noise_term, full_matrices=False)
            noise_eigenvalues = s**2
            max_noise_eigenvalue = np.max(noise_eigenvalues)
        except Exception as e:
            print(f"  ✗ Error computing noise eigenvalues: {e}")
            continue
        
        # Step 2: Decompose activity topic into identity components + orthogonal component
        results = {}
        
        for activity_idx in activity_topics:
            beta_activity = beta_np[:, activity_idx]  # Activity topic gene profile
            
            # Create matrix of identity topic profiles
            beta_identity_matrix = beta_np[:, identity_topics]  # genes x num_identity_topics
            
            # Project activity topic onto identity topics
            # beta_activity = sum(rho_i * beta_identity_i) + rho_a * beta_activity_orthogonal
            try:
                # Solve: beta_identity_matrix @ rho = beta_activity
                rho_coeffs = np.linalg.lstsq(beta_identity_matrix, beta_activity, rcond=None)[0]
                
                # Compute the projection onto identity topics
                beta_activity_projection = beta_identity_matrix @ rho_coeffs
                
                # Compute the orthogonal component
                beta_activity_orthogonal = beta_activity - beta_activity_projection
                
                # Compute rho_a (coefficient of orthogonal component)
                rho_a = np.linalg.norm(beta_activity_orthogonal)
                
                # Create unit vector β̃_a (orthogonal to identity topics)
                if rho_a > 0:
                    beta_tilde_a = beta_activity_orthogonal / rho_a
                else:
                    beta_tilde_a = beta_activity_orthogonal
                
                print(f"    Activity topic {topic_names[activity_idx]}:")
                print(f"      rho_a (orthogonal magnitude): {rho_a:.6f}")
                print(f"      Projection coefficients: {dict(zip([topic_names[i] for i in identity_topics], rho_coeffs))}")
                
            except Exception as e:
                print(f"  ✗ Error in orthogonalization: {e}")
                continue
            
            # Step 3: Compute signal estimates
            # Activity signal ≈ √L * rho_a * ||θ_a|| * ||β̃_a^T||
            # where ρ_a is the coefficient and ||β̃_a^T|| is the norm of the unit vector
            theta_activity = theta_np[:, activity_idx]
            theta_activity_norm = np.linalg.norm(theta_activity)
            beta_tilde_norm = np.linalg.norm(beta_tilde_a)
            
            activity_signal = np.sqrt(L) * rho_a * theta_activity_norm * beta_tilde_norm
            
            # Identity signals ≈ √L * ||θ_i + θ_a * rho_i|| * ||β_i||
            identity_signals = []
            for i, identity_idx in enumerate(identity_topics):
                theta_identity = theta_np[:, identity_idx]
                beta_identity = beta_np[:, identity_idx]
                
                # θ_i + θ_a * rho_i
                theta_combined = theta_identity + theta_activity * rho_coeffs[i]
                theta_combined_norm = np.linalg.norm(theta_combined)
                beta_identity_norm = np.linalg.norm(beta_identity)
                
                identity_signal = np.sqrt(L) * theta_combined_norm * beta_identity_norm
                identity_signals.append(identity_signal)
                
                print(f"      Identity topic {topic_names[identity_idx]} signal: {identity_signal:.6f}")
            
            # Step 4: Compute signal-to-noise ratios
            activity_snr = activity_signal / max_noise_eigenvalue
            identity_snrs = [signal / max_noise_eigenvalue for signal in identity_signals]
            
            print(f"      Activity signal-to-noise ratio: {activity_snr:.6f}")
            print(f"      Identity signal-to-noise ratios: {[f'{snr:.6f}' for snr in identity_snrs]}")
            
            # Store results
            results[activity_idx] = {
                'activity_signal': activity_signal,
                'activity_snr': activity_snr,
                'identity_signals': identity_signals,
                'identity_snrs': identity_snrs,
                'rho_a': rho_a,
                'rho_coeffs': rho_coeffs,
                'max_noise_eigenvalue': max_noise_eigenvalue,
                'noise_eigenvalues': noise_eigenvalues,
                'beta_activity_orthogonal': beta_activity_orthogonal,
                'beta_activity_projection': beta_activity_projection
            }
        
        # Store all results for this dataset
        key = f"{dataset_name}_{de_mean}"
        all_results[key] = {
            'dataset_name': dataset_name,
            'de_mean': float(de_mean.replace('DE_mean_', '')),
            'topic_names': topic_names,
            'identity_topics': identity_topics,
            'activity_topics': activity_topics,
            'results': results,
            'X_shape': X_np.shape,
            'noise_eigenvalues': noise_eigenvalues
        }
        
        print(f"  ✓ Completed noise analysis")
    
    # Create output directory structure and save results
    print("\nSaving results...")
    for key, data in all_results.items():
        dataset_name = data['dataset_name']
        de_mean = data['de_mean']
        
        # Create output directory in estimates structure
        output_subdir = estimates_path / dataset_name / f"DE_mean_{de_mean}" / "noise_analysis"
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_summary = []
        for activity_idx, result in data['results'].items():
            activity_name = data['topic_names'][activity_idx]
            
            for i, identity_idx in enumerate(data['identity_topics']):
                identity_name = data['topic_names'][identity_idx]
                
                results_summary.append({
                    'Activity_Topic': activity_name,
                    'Identity_Topic': identity_name,
                    'DE_Mean': de_mean,
                    'Activity_Signal': result['activity_signal'],
                    'Activity_SNR': result['activity_snr'],
                    'Identity_Signal': result['identity_signals'][i],
                    'Identity_SNR': result['identity_snrs'][i],
                    'Rho_a': result['rho_a'],
                    'Rho_coeff': result['rho_coeffs'][i],
                    'Max_Noise_Eigenvalue': result['max_noise_eigenvalue']
                })
        
        # Save summary CSV
        summary_df = pd.DataFrame(results_summary)
        summary_df.to_csv(output_subdir / "noise_analysis_results.csv", index=False)
        
        # Create visualization
        create_noise_analysis_plot(data, output_subdir)
    
    # Create separate combined PDFs for AB and ABCD datasets
    create_eigenvalue_histogram_pdfs(all_results, estimates_path)
    
    # Create model recovery analysis with improved visualization
    create_model_recovery_analysis(all_results, estimates_path, data_root)
    
    # Create comprehensive plots for each dataset separately
    create_comprehensive_plots_by_dataset(all_results, estimates_path, data_root)
    
    # Create combined comparison plot with improved styling
    create_combined_noise_analysis_plot(all_results, estimates_path)
    
    # Create topic-specific SNR scatter plots
    create_topic_snr_scatter_plots(all_results, estimates_path)
    
    print(f"\n✓ Noise analysis completed!")
    print(f"  Processed {len(all_results)} datasets")
    print(f"  Results saved to respective estimate directories")
    
    return all_results


def create_noise_analysis_plot(data, output_dir):
    """
    Create noise analysis visualization for a single dataset with improved styling.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Set up improved color scheme
    activity_color = '#d62728'  # Red for activity
    identity_color = '#2ca02c'  # Green for identity
    noise_color = '#ff7f0e'     # Orange for noise
    projection_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    
    # Plot 1: Activity Signal-to-noise ratios
    ax1 = axes[0, 0]
    activity_snrs = []
    activity_names = []
    
    for activity_idx, result in data['results'].items():
        activity_name = data['topic_names'][activity_idx]
        activity_snrs.append(result['activity_snr'])
        activity_names.append(activity_name)
    
    if len(activity_snrs) > 0:
        bars1 = ax1.bar(range(len(activity_snrs)), activity_snrs, alpha=0.8, color=activity_color, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Activity Topics', fontsize=12)
        ax1.set_ylabel('Signal-to-Noise Ratio', fontsize=12)
        ax1.set_title(f'Activity Topic Signal-to-Noise Ratios (DE_mean = {data["de_mean"]})', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(activity_snrs)))
        ax1.set_xticklabels(activity_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        
        # Add value labels on bars
        for i, (bar, snr) in enumerate(zip(bars1, activity_snrs)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{snr:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'No activity topics found', 
                transform=ax1.transAxes, ha='center', va='center', fontsize=12)
        ax1.set_title(f'Activity Signal-to-Noise Ratios (DE_mean = {data["de_mean"]})', fontsize=14, fontweight='bold')
    
    # Plot 2: Identity Signal-to-noise ratios
    ax2 = axes[0, 1]
    identity_snrs = []
    identity_names = []
    
    # Collect identity SNRs from the first activity topic result
    if data['results']:
        first_result = next(iter(data['results'].values()))
        for i, identity_idx in enumerate(data['identity_topics']):
            identity_name = data['topic_names'][identity_idx]
            identity_snrs.append(first_result['identity_snrs'][i])
            identity_names.append(identity_name)
    
    if len(identity_snrs) > 0:
        bars2 = ax2.bar(range(len(identity_snrs)), identity_snrs, alpha=0.8, color=identity_color, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Identity Topics', fontsize=12)
        ax2.set_ylabel('Signal-to-Noise Ratio', fontsize=12)
        ax2.set_title(f'Identity Topic Signal-to-Noise Ratios (DE_mean = {data["de_mean"]})', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(identity_snrs)))
        ax2.set_xticklabels(identity_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        
        # Add value labels on bars
        for i, (bar, snr) in enumerate(zip(bars2, identity_snrs)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{snr:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No identity topics found', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        ax2.set_title(f'Identity Signal-to-Noise Ratios (DE_mean = {data["de_mean"]})', fontsize=14, fontweight='bold')
    
    # Plot 3: Noise eigenvalue histogram
    ax3 = axes[1, 0]
    noise_eigenvalues = data['noise_eigenvalues']
    ax3.hist(noise_eigenvalues, bins=50, alpha=0.7, color=noise_color, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Noise Eigenvalue Magnitude', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Noise Eigenvalue Distribution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=10)
    
    # Add vertical line for max eigenvalue
    max_eigenvalue = np.max(noise_eigenvalues)
    ax3.axvline(max_eigenvalue, color='blue', linestyle='--', linewidth=2, 
                label=f'Max: {max_eigenvalue:.2e}')
    ax3.legend(fontsize=10)
    
    # Plot 4: Orthogonalization coefficients
    ax4 = axes[1, 1]
    x_pos = np.arange(len(data['identity_topics']))
    width = 0.35
    
    for i, (activity_idx, result) in enumerate(data['results'].items()):
        activity_name = data['topic_names'][activity_idx]
        rho_coeffs = result['rho_coeffs']
        identity_names = [data['topic_names'][idx] for idx in data['identity_topics']]
        
        color = projection_colors[i % len(projection_colors)]
        bars = ax4.bar(x_pos + i*width, rho_coeffs, width, alpha=0.8, 
                      label=f'{activity_name} projection', color=color, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar, coeff in zip(bars, rho_coeffs):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{coeff:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax4.set_xlabel('Identity Topics', fontsize=12)
    ax4.set_ylabel('Projection Coefficient (ρ)', fontsize=12)
    ax4.set_title('Activity Topic Projection onto Identity Topics', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos + width/2)
    ax4.set_xticklabels(identity_names, rotation=45, ha='right')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "noise_analysis_plot.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_combined_noise_analysis_plot(all_results, estimates_path):
    """
    Create separate comparison plots for each dataset with improved styling.
    """
    # Group data by dataset
    datasets = {}
    for key, data in all_results.items():
        dataset_name = data['dataset_name']
        if dataset_name not in datasets:
            datasets[dataset_name] = []
        datasets[dataset_name].append(data)
    
    # Process each dataset separately
    for dataset_name, dataset_results in datasets.items():
        print(f"  Creating combined plots for {dataset_name}...")
        
        # Collect data for this dataset
        comparison_data = []
        
        for data in dataset_results:
            de_mean = data['de_mean']
            
            for activity_idx, result in data['results'].items():
                activity_name = data['topic_names'][activity_idx]
                
                # Activity signal-to-noise
                comparison_data.append({
                    'DE_Mean': de_mean,
                    'Signal_Type': 'Activity',
                    'Topic': activity_name,
                    'SNR': result['activity_snr'],
                    'Signal_Magnitude': result['activity_signal'],
                    'Rho_a': result['rho_a']
                })
                
                # Identity signal-to-noise
                for i, identity_idx in enumerate(data['identity_topics']):
                    identity_name = data['topic_names'][identity_idx]
                    comparison_data.append({
                        'DE_Mean': de_mean,
                        'Signal_Type': 'Identity',
                        'Topic': f'{activity_name} vs {identity_name}',
                        'SNR': result['identity_snrs'][i],
                        'Signal_Magnitude': result['identity_signals'][i],
                        'Rho_a': result['rho_a']
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create combined plot for this dataset
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Activity SNR vs DE_mean
        ax1 = axes[0, 0]
        activity_data = comparison_df[comparison_df['Signal_Type'] == 'Activity']
        if not activity_data.empty:
            ax1.scatter(activity_data['DE_Mean'], activity_data['SNR'], 
                       color='#1f77b4', marker='o', s=80, alpha=0.8, zorder=5,
                       edgecolors='black', linewidth=0.5)
        
        ax1.set_xlabel('DE Mean', fontsize=12)
        ax1.set_ylabel('Activity Signal-to-Noise Ratio', fontsize=12)
        ax1.set_title(f'{dataset_name}: Activity Signal-to-Noise Ratio vs DE Mean', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        
        # Plot 2: Identity SNR vs DE_mean
        ax2 = axes[0, 1]
        identity_data = comparison_df[comparison_df['Signal_Type'] == 'Identity']
        if not identity_data.empty:
            ax2.scatter(identity_data['DE_Mean'], identity_data['SNR'], 
                       color='#ff7f0e', marker='s', s=80, alpha=0.8, zorder=5,
                       edgecolors='black', linewidth=0.5)
        
        ax2.set_xlabel('DE Mean', fontsize=12)
        ax2.set_ylabel('Identity Signal-to-Noise Ratio', fontsize=12)
        ax2.set_title(f'{dataset_name}: Identity Signal-to-Noise Ratio vs DE Mean', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        
        # Plot 3: Max noise eigenvalue vs DE_mean
        ax3 = axes[1, 0]
        max_noise_eigenvalues = []
        de_means = []
        for data in dataset_results:
            for activity_idx, result in data['results'].items():
                max_noise_eigenvalues.append(result['max_noise_eigenvalue'])
                de_means.append(data['de_mean'])
                break
        
        if max_noise_eigenvalues:
            ax3.scatter(de_means, max_noise_eigenvalues, 
                       color='#2ca02c', marker='^', s=80, alpha=0.8, zorder=5,
                       edgecolors='black', linewidth=0.5)
        
        ax3.set_xlabel('DE Mean', fontsize=12)
        ax3.set_ylabel('Max Noise Eigenvalue', fontsize=12)
        ax3.set_title(f'{dataset_name}: Max Noise Eigenvalue vs DE Mean', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', which='major', labelsize=10)
        
        # Plot 4: Activity signal vs max noise eigenvalue
        ax4 = axes[1, 1]
        activity_signals = []
        max_noise_eigenvalues = []
        for data in dataset_results:
            for activity_idx, result in data['results'].items():
                activity_signals.append(result['activity_signal'])
                max_noise_eigenvalues.append(result['max_noise_eigenvalue'])
                break
        
        if activity_signals:
            ax4.scatter(max_noise_eigenvalues, activity_signals, 
                       color='#d62728', marker='d', s=80, alpha=0.8, zorder=5,
                       edgecolors='black', linewidth=0.5)
        
        ax4.set_xlabel('Max Noise Eigenvalue', fontsize=12)
        ax4.set_ylabel('Activity Signal', fontsize=12)
        ax4.set_title(f'{dataset_name}: Activity Signal vs Max Noise Eigenvalue', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        
        # Save to dataset-specific directory
        dataset_dir = estimates_path / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        plt.savefig(dataset_dir / "noise_analysis_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save comparison data to dataset-specific directory
        comparison_df.to_csv(dataset_dir / "noise_analysis_comparison.csv", index=False)
        
        print(f"    ✓ {dataset_name} combined plots saved to: {dataset_dir}")


def create_eigenvalue_histogram_pdfs(all_results, estimates_path):
    """
    Create separate PDFs with eigenvalue histograms for AB and ABCD datasets.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    
    # Separate results by dataset type
    ab_results = {k: v for k, v in all_results.items() if v['dataset_name'] == 'AB_V1'}
    abcd_results = {k: v for k, v in all_results.items() if v['dataset_name'] == 'ABCD_V1V2'}
    
    # Create PDF for AB_V1 datasets
    if ab_results:
        ab_pdf_path = estimates_path / "AB_V1" / "noise_eigenvalue_histograms.pdf"
        ab_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        
        with PdfPages(ab_pdf_path) as pdf:
            for key, data in sorted(ab_results.items(), key=lambda x: x[1]['de_mean']):
                fig, ax = plt.subplots(figsize=(10, 6))
                
                noise_eigenvalues = data['noise_eigenvalues']
                ax.hist(noise_eigenvalues, bins=50, alpha=0.7, color='red', edgecolor='black')
                ax.set_xlabel('Noise Eigenvalue Magnitude')
                ax.set_ylabel('Frequency')
                ax.set_title(f'AB_V1 - DE_mean_{data["de_mean"]} - Noise Eigenvalue Distribution')
                ax.grid(True, alpha=0.3)
                
                # Add vertical line for max eigenvalue
                max_eigenvalue = np.max(noise_eigenvalues)
                ax.axvline(max_eigenvalue, color='blue', linestyle='--', linewidth=2, 
                          label=f'Max: {max_eigenvalue:.2e}')
                ax.legend()
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
        
        print(f"  AB_V1 eigenvalue histograms saved to: {ab_pdf_path}")
    
    # Create PDF for ABCD_V1V2 datasets
    if abcd_results:
        abcd_pdf_path = estimates_path / "ABCD_V1V2" / "noise_eigenvalue_histograms.pdf"
        abcd_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        
        with PdfPages(abcd_pdf_path) as pdf:
            for key, data in sorted(abcd_results.items(), key=lambda x: x[1]['de_mean']):
                fig, ax = plt.subplots(figsize=(10, 6))
                
                noise_eigenvalues = data['noise_eigenvalues']
                ax.hist(noise_eigenvalues, bins=50, alpha=0.7, color='red', edgecolor='black')
                ax.set_xlabel('Noise Eigenvalue Magnitude')
                ax.set_ylabel('Frequency')
                ax.set_title(f'ABCD_V1V2 - DE_mean_{data["de_mean"]} - Noise Eigenvalue Distribution')
                ax.grid(True, alpha=0.3)
                
                # Add vertical line for max eigenvalue
                max_eigenvalue = np.max(noise_eigenvalues)
                ax.axvline(max_eigenvalue, color='blue', linestyle='--', linewidth=2, 
                          label=f'Max: {max_eigenvalue:.2e}')
                ax.legend()
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
        
        print(f"  ABCD_V1V2 eigenvalue histograms saved to: {abcd_pdf_path}")


def compute_model_recovery_metrics(estimates_path, data_root):
    """
    Compute model recovery metrics for HLDA, LDA, and NMF models with proper topic matching.
    
    Returns:
    --------
    recovery_data : list
        List of dictionaries containing recovery metrics for each topic-model combination
    """
    recovery_data = []
    
    # Find all DE_mean directories in estimates
    for dataset_dir in estimates_path.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name.startswith('_'):
            continue
            
        dataset_name = dataset_dir.name
        
        for de_dir in dataset_dir.iterdir():
            if not de_dir.is_dir() or not de_dir.name.startswith('DE_mean_'):
                continue
                
            de_mean = float(de_dir.name.replace('DE_mean_', ''))
            print(f"  Processing {dataset_name}/{de_dir.name}")
            
            # Look for model results in different possible locations
            possible_topic_configs = ['6_topic_fit', '3_topic_fit', '3_topic_fit_IMPROVED']
            
            for topic_config in possible_topic_configs:
                topic_dir = de_dir / topic_config
                if not topic_dir.exists():
                    continue
                    
                # Check for each method
                for method in ['HLDA', 'LDA', 'NMF']:
                    method_dir = topic_dir / method
                    if not method_dir.exists():
                        continue
                        
                    # Find theta and beta files
                    theta_files = list(method_dir.glob("*theta*.csv"))
                    beta_files = list(method_dir.glob("*beta*.csv"))
                    
                    if not theta_files or not beta_files:
                        continue
                        
                    est_theta_file = theta_files[0]
                    est_beta_file = beta_files[0]
                    
                    # Get true data paths (in data directory, not estimates)
                    data_dir = Path(data_root) / dataset_name / de_dir.name
                    true_theta_file = data_dir / "theta.csv"
                    true_beta_file = data_dir / "gene_means.csv"
                    
                    if not true_theta_file.exists() or not true_beta_file.exists():
                        continue
                        
                    try:
                        # Load data
                        true_theta = pd.read_csv(true_theta_file, index_col=0)
                        est_theta = pd.read_csv(est_theta_file, index_col=0)
                        true_beta = pd.read_csv(true_beta_file, index_col=0)
                        est_beta = pd.read_csv(est_beta_file, index_col=0)
                        
                        # Load train cells if available
                        train_cells_file = data_dir / "train_cells.csv"
                        
                        if train_cells_file.exists():
                            train_cells_df = pd.read_csv(train_cells_file, index_col=0)
                            train_cells = train_cells_df.index.tolist()
                            
                            # Subset to train cells only
                            train_cells_common = [cell for cell in train_cells 
                                                if cell in true_theta.index and cell in est_theta.index]
                            
                            if len(train_cells_common) > 0:
                                true_theta_subset = true_theta.loc[train_cells_common]
                                est_theta_subset = est_theta.loc[train_cells_common]
                            else:
                                # Use all common cells if train cells not available
                                common_cells = true_theta.index.intersection(est_theta.index)
                                if len(common_cells) == 0:
                                    continue
                                true_theta_subset = true_theta.loc[common_cells]
                                est_theta_subset = est_theta.loc[common_cells]
                        else:
                            # Use all common cells if train cells file not available
                            common_cells = true_theta.index.intersection(est_theta.index)
                            if len(common_cells) == 0:
                                continue
                            true_theta_subset = true_theta.loc[common_cells]
                            est_theta_subset = est_theta.loc[common_cells]
                        
                        # Get topic names from true data
                        true_topic_names = list(true_theta.columns)
                        est_topic_names = list(est_theta.columns)
                        
                        # Match topics using Hungarian algorithm for optimal assignment
                        from scipy.optimize import linear_sum_assignment
                        
                        # Compute correlation matrix between true and estimated topics
                        min_topics = min(len(true_topic_names), len(est_topic_names))
                        correlation_matrix = np.zeros((min_topics, min_topics))
                        
                        for i in range(min_topics):
                            for j in range(min_topics):
                                if i < true_theta_subset.shape[1] and j < est_theta_subset.shape[1]:
                                    corr = np.corrcoef(true_theta_subset.iloc[:, i], est_theta_subset.iloc[:, j])[0, 1]
                                    correlation_matrix[i, j] = corr if not np.isnan(corr) else 0
                        
                        # Use Hungarian algorithm to find optimal topic matching
                        true_indices, est_indices = linear_sum_assignment(-correlation_matrix)  # Negative for maximization
                        
                        # Store individual topic correlations
                        for true_idx, est_idx in zip(true_indices, est_indices):
                            if true_idx < len(true_topic_names) and est_idx < len(est_topic_names):
                                true_topic_name = true_topic_names[true_idx]
                                est_topic_name = est_topic_names[est_idx]
                                theta_corr = correlation_matrix[true_idx, est_idx]
                                
                                # For beta correlations, we need to match genes
                                if true_idx < true_beta.shape[1] and est_idx < est_beta.shape[1]:
                                    # Get common genes
                                    common_genes = [gene for gene in true_beta.index if gene in est_beta.index]
                                    if len(common_genes) > 0:
                                        true_beta_topic = true_beta.loc[common_genes, true_topic_name]
                                        est_beta_topic = est_beta.loc[common_genes, est_topic_name]
                                        beta_corr = np.corrcoef(true_beta_topic, est_beta_topic)[0, 1]
                                        beta_corr = beta_corr if not np.isnan(beta_corr) else 0
                                    else:
                                        beta_corr = 0
                                else:
                                    beta_corr = 0
                                
                                recovery_data.append({
                                    'dataset': dataset_name,
                                    'de_mean': de_mean,
                                    'method': method,
                                    'topic_config': topic_config,
                                    'true_topic': true_topic_name,
                                    'est_topic': est_topic_name,
                                    'theta_corr': theta_corr,
                                    'beta_corr': beta_corr,
                                    'topic_pair': f"{true_topic_name} ↔ {est_topic_name}"
                                })
                        
                        print(f"    ✓ {method}: matched {len(true_indices)} topics")
                        
                    except Exception as e:
                        print(f"    ✗ Error processing {method}: {e}")
                        continue
    
    return recovery_data


def create_model_recovery_analysis(all_results, estimates_path, data_root):
    """
    Create scatter plots showing relationship between signal/noise ratio and model recovery.
    Uses topic names as colors and model types as shapes.
    """
    print("Computing model recovery metrics...")
    recovery_data = compute_model_recovery_metrics(estimates_path, data_root)
    
    if not recovery_data:
        print("  No model recovery data found. Skipping recovery analysis.")
        return
    
    # Create DataFrame
    recovery_df = pd.DataFrame(recovery_data)
    
    # Merge with noise analysis data to get activity SNR
    noise_data = []
    for key, data in all_results.items():
        for activity_idx, result in data['results'].items():
            noise_data.append({
                'dataset': data['dataset_name'],
                'de_mean': data['de_mean'],
                'activity_snr': result['activity_snr']
            })
            break  # Only take first activity topic
    
    noise_df = pd.DataFrame(noise_data)
    
    # Merge recovery data with noise data
    merged_df = recovery_df.merge(noise_df, on=['dataset', 'de_mean'], how='left')
    
    # Set up topic colors and model markers
    topic_colors = {
        'A': '#1f77b4',   # Blue
        'B': '#ff7f0e',   # Orange
        'C': '#2ca02c',   # Green
        'D': '#d62728',   # Red
        'V1': '#9467bd',  # Purple
        'V2': '#8c564b',  # Brown
        'VAR': '#e377c2'  # Pink
    }
    
    model_markers = {
        'HLDA': 'o',  # Circle
        'LDA': 's',   # Square
        'NMF': '^'    # Triangle
    }
    
    # Group data by dataset
    datasets = {}
    for _, row in merged_df.iterrows():
        dataset = row['dataset']
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(row)
    
    # Create separate plots for each dataset
    for dataset_name, dataset_data in datasets.items():
        print(f"  Creating model recovery plots for {dataset_name}...")
        
        dataset_df = pd.DataFrame(dataset_data)
        
        # Create scatter plots with improved styling
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Activity SNR vs Theta Correlation
        ax1 = axes[0, 0]
        for true_topic in dataset_df['true_topic'].unique():
            topic_data = dataset_df[dataset_df['true_topic'] == true_topic]
            color = topic_colors.get(true_topic, '#1f77b4')
            
            for method in topic_data['method'].unique():
                method_data = topic_data[topic_data['method'] == method]
                marker = model_markers.get(method, 'o')
                
                ax1.scatter(method_data['activity_snr'], method_data['theta_corr'], 
                           label=f'{true_topic} ({method})', alpha=0.8, s=80, 
                           color=color, marker=marker, edgecolors='black', linewidth=0.5)
        
        ax1.set_xlabel('Activity Signal-to-Noise Ratio', fontsize=12)
        ax1.set_ylabel('Theta Correlation', fontsize=12)
        ax1.set_title(f'{dataset_name}: Theta Correlation vs Activity SNR', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        
        # Plot 2: Activity SNR vs Beta Correlation
        ax2 = axes[0, 1]
        for true_topic in dataset_df['true_topic'].unique():
            topic_data = dataset_df[dataset_df['true_topic'] == true_topic]
            color = topic_colors.get(true_topic, '#1f77b4')
            
            for method in topic_data['method'].unique():
                method_data = topic_data[topic_data['method'] == method]
                marker = model_markers.get(method, 'o')
                
                ax2.scatter(method_data['activity_snr'], method_data['beta_corr'], 
                           label=f'{true_topic} ({method})', alpha=0.8, s=80,
                           color=color, marker=marker, edgecolors='black', linewidth=0.5)
        
        ax2.set_xlabel('Activity Signal-to-Noise Ratio', fontsize=12)
        ax2.set_ylabel('Beta Correlation', fontsize=12)
        ax2.set_title(f'{dataset_name}: Beta Correlation vs Activity SNR', fontsize=14, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        
        # Plot 3: DE Mean vs Theta Correlation
        ax3 = axes[1, 0]
        for true_topic in dataset_df['true_topic'].unique():
            topic_data = dataset_df[dataset_df['true_topic'] == true_topic]
            color = topic_colors.get(true_topic, '#1f77b4')
            
            for method in topic_data['method'].unique():
                method_data = topic_data[topic_data['method'] == method]
                marker = model_markers.get(method, 'o')
                
                ax3.scatter(method_data['de_mean'], method_data['theta_corr'], 
                           color=color, marker=marker, s=80, alpha=0.8, zorder=5,
                           edgecolors='black', linewidth=0.5)
        
        ax3.set_xlabel('DE Mean', fontsize=12)
        ax3.set_ylabel('Theta Correlation', fontsize=12)
        ax3.set_title(f'{dataset_name}: Theta Correlation vs DE Mean', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', which='major', labelsize=10)
        
        # Plot 4: DE Mean vs Beta Correlation
        ax4 = axes[1, 1]
        for true_topic in dataset_df['true_topic'].unique():
            topic_data = dataset_df[dataset_df['true_topic'] == true_topic]
            color = topic_colors.get(true_topic, '#1f77b4')
            
            for method in topic_data['method'].unique():
                method_data = topic_data[topic_data['method'] == method]
                marker = model_markers.get(method, 'o')
                
                ax4.scatter(method_data['de_mean'], method_data['beta_corr'], 
                           color=color, marker=marker, s=80, alpha=0.8, zorder=5,
                           edgecolors='black', linewidth=0.5)
        
        ax4.set_xlabel('DE Mean', fontsize=12)
        ax4.set_ylabel('Beta Correlation', fontsize=12)
        ax4.set_title(f'{dataset_name}: Beta Correlation vs DE Mean', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        
        # Save to dataset-specific directory
        dataset_dir = estimates_path / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        plt.savefig(dataset_dir / "model_recovery_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save recovery data to dataset-specific directory
        dataset_df.to_csv(dataset_dir / "model_recovery_analysis.csv", index=False)
        
        print(f"    ✓ {dataset_name} model recovery plots saved to: {dataset_dir}")
        print(f"    Recovery data includes {len(dataset_df)} data points")
        
        # Print summary statistics for this dataset
        print(f"    Model Recovery Summary for {dataset_name}:")
        for method in dataset_df['method'].unique():
            method_data = dataset_df[dataset_df['method'] == method]
            avg_theta = method_data['theta_corr'].mean()
            avg_beta = method_data['beta_corr'].mean()
            print(f"      {method}: Avg Theta Corr = {avg_theta:.3f}, Avg Beta Corr = {avg_beta:.3f}")
    
    # Also save the combined data to the main estimates directory
    merged_df.to_csv(estimates_path / "model_recovery_analysis.csv", index=False)
    print(f"  Combined model recovery data saved to: {estimates_path}")


def create_topic_snr_scatter_plots(all_results, estimates_path):
    """
    Create separate scatter plots for each dataset showing SNR vs DE_mean for each topic.
    
    Creates separate plots for each dataset:
    - AB_V1: Activity and Identity topic plots
    - ABCD_V1V2: Activity and Identity topic plots
    """
    # Group data by dataset
    datasets = {}
    for key, data in all_results.items():
        dataset_name = data['dataset_name']
        if dataset_name not in datasets:
            datasets[dataset_name] = []
        datasets[dataset_name].append(data)
    
    # Set up color schemes for different topics
    activity_colors = {
        'V1': '#d62728',  # Red
        'V2': '#ff7f0e',  # Orange
        'VAR': '#9467bd'  # Purple
    }
    
    identity_colors = {
        'A': '#1f77b4',   # Blue
        'B': '#2ca02c',   # Green
        'C': '#ff7f0e',   # Orange
        'D': '#d62728'    # Red
    }
    
    # Process each dataset separately
    for dataset_name, dataset_results in datasets.items():
        print(f"  Creating plots for {dataset_name}...")
        
        # Collect data for this dataset
        activity_data = []
        identity_data = []
        
        for data in dataset_results:
            de_mean = data['de_mean']
            
            # Get activity topic data
            for activity_idx, result in data['results'].items():
                activity_name = data['topic_names'][activity_idx]
                
                activity_data.append({
                    'DE_Mean': de_mean,
                    'Topic': activity_name,
                    'SNR': result['activity_snr'],
                    'Signal_Magnitude': result['activity_signal'],
                    'Rho_a': result['rho_a']
                })
                
                # Get identity topic data for this activity topic
                for i, identity_idx in enumerate(data['identity_topics']):
                    identity_name = data['topic_names'][identity_idx]
                    
                    identity_data.append({
                        'DE_Mean': de_mean,
                        'Topic': f'{activity_name} vs {identity_name}',
                        'SNR': result['identity_snrs'][i],
                        'Signal_Magnitude': result['identity_signals'][i],
                        'Rho_a': result['rho_a'],
                        'Activity_Topic': activity_name,
                        'Identity_Topic': identity_name
                    })
        
        activity_df = pd.DataFrame(activity_data)
        identity_df = pd.DataFrame(identity_data)
        
        # Create the plots for this dataset
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Activity Topics SNR vs DE_mean
        for topic in activity_df['Topic'].unique():
            topic_data = activity_df[activity_df['Topic'] == topic]
            color = activity_colors.get(topic, '#1f77b4')
            
            ax1.scatter(topic_data['DE_Mean'], topic_data['SNR'], 
                       label=topic, 
                       color=color, marker='o', s=100, alpha=0.8,
                       edgecolors='black', linewidth=0.5)
        
        ax1.set_xlabel('DE Mean', fontsize=12)
        ax1.set_ylabel('Activity Signal-to-Noise Ratio', fontsize=12)
        ax1.set_title(f'{dataset_name}: Activity Topics SNR vs DE Mean', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        
        # Plot 2: Identity Topics SNR vs DE_mean
        for activity_topic in identity_df['Activity_Topic'].unique():
            activity_data = identity_df[identity_df['Activity_Topic'] == activity_topic]
            
            for identity_topic in activity_data['Identity_Topic'].unique():
                topic_data = activity_data[activity_data['Identity_Topic'] == identity_topic]
                color = identity_colors.get(identity_topic, '#1f77b4')
                
                ax2.scatter(topic_data['DE_Mean'], topic_data['SNR'], 
                           label=f'{activity_topic} vs {identity_topic}', 
                           color=color, marker='o', s=100, alpha=0.8,
                           edgecolors='black', linewidth=0.5)
        
        ax2.set_xlabel('DE Mean', fontsize=12)
        ax2.set_ylabel('Identity Signal-to-Noise Ratio', fontsize=12)
        ax2.set_title(f'{dataset_name}: Identity Topics SNR vs DE Mean', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        
        # Save to dataset-specific directory
        dataset_dir = estimates_path / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        plt.savefig(dataset_dir / "topic_snr_scatter_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save the data to dataset-specific directory
        activity_df.to_csv(dataset_dir / "activity_snr_data.csv", index=False)
        identity_df.to_csv(dataset_dir / "identity_snr_data.csv", index=False)
        
        print(f"    ✓ {dataset_name} plots saved to: {dataset_dir}")
        print(f"    Activity data points: {len(activity_df)}")
        print(f"    Identity data points: {len(identity_df)}")


def create_comprehensive_plots_by_dataset(all_results, estimates_path, data_root):
    """
    Create comprehensive plots for each dataset separately, including model recovery analysis.
    """
    # Group results by dataset
    datasets = {}
    for key, data in all_results.items():
        dataset_name = data['dataset_name']
        if dataset_name not in datasets:
            datasets[dataset_name] = []
        datasets[dataset_name].append(data)
    
    for dataset_name, dataset_results in datasets.items():
        # Sort by DE_mean
        dataset_results.sort(key=lambda x: x['de_mean'])
        
        # Create comprehensive plot for this dataset
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data for plotting
        de_means = [data['de_mean'] for data in dataset_results]
        
        # Collect all topic data
        all_snrs = []
        all_de_means = []
        all_topic_names = []
        all_rho_values = []
        all_max_eigenvalues = []
        
        # Collect model recovery data
        model_recovery_data = []
        
        for data in dataset_results:
            de_mean = data['de_mean']
            
            # Get all topic data
            for activity_idx, result in data['results'].items():
                activity_name = data['topic_names'][activity_idx]
                
                # Activity topic
                all_snrs.append(result['activity_snr'])
                all_de_means.append(de_mean)
                all_topic_names.append(activity_name)
                all_rho_values.append(result['rho_a'])
                all_max_eigenvalues.append(result['max_noise_eigenvalue'])
                
                # Identity topics
                for i, identity_idx in enumerate(data['identity_topics']):
                    identity_name = data['topic_names'][identity_idx]
                    all_snrs.append(result['identity_snrs'][i])
                    all_de_means.append(de_mean)
                    all_topic_names.append(identity_name)
                    all_rho_values.append(result['rho_coeffs'][i])
                    all_max_eigenvalues.append(result['max_noise_eigenvalue'])  # Same noise for all topics
            
            # Get model recovery data for this DE_mean
            est_dir = estimates_path / dataset_name / f"DE_mean_{de_mean}"
            for method in ['hlda', 'lda', 'nmf']:
                possible_dirs = [
                    est_dir / method,
                    est_dir / "3_topic_fit" / method.upper(),
                    est_dir / "3_topic_fit_IMPROVED" / method.upper()
                ]
                
                method_dir = None
                for possible_dir in possible_dirs:
                    if possible_dir.exists():
                        method_dir = possible_dir
                        break
                
                if method_dir is not None:
                    try:
                        theta_files = list(method_dir.glob("*theta*.csv"))
                        beta_files = list(method_dir.glob("*beta*.csv"))
                        
                        if theta_files and beta_files:
                            est_theta_file = theta_files[0]
                            est_beta_file = beta_files[0]
                            true_theta_file = est_dir / "theta.csv"
                            true_beta_file = est_dir / "gene_means.csv"
                            train_cells_file = Path(data_root) / dataset_name / f"DE_mean_{de_mean}" / "train_cells.csv"
                            
                            if all([true_theta_file.exists(), true_beta_file.exists(), train_cells_file.exists()]):
                                # Load data
                                true_theta = pd.read_csv(true_theta_file, index_col=0)
                                est_theta = pd.read_csv(est_theta_file, index_col=0)
                                true_beta = pd.read_csv(true_beta_file, index_col=0)
                                est_beta = pd.read_csv(est_beta_file, index_col=0)
                                train_cells_df = pd.read_csv(train_cells_file, index_col=0)
                                train_cells = train_cells_df.index.tolist()
                                
                                # Subset to train cells
                                train_cells_common = [cell for cell in train_cells if cell in true_theta.index and cell in est_theta.index]
                                
                                if len(train_cells_common) > 0:
                                    true_theta_train = true_theta.loc[train_cells_common]
                                    est_theta_train = est_theta.loc[train_cells_common]
                                    
                                    # Align topics
                                    min_topics = min(true_theta_train.shape[1], est_theta_train.shape[1])
                                    true_theta_train = true_theta_train.iloc[:, :min_topics]
                                    est_theta_train = est_theta_train.iloc[:, :min_topics]
                                    
                                    # Compute correlations for each topic
                                    for i in range(min_topics):
                                        corr = np.corrcoef(true_theta_train.iloc[:, i], est_theta_train.iloc[:, i])[0, 1]
                                        if not np.isnan(corr):
                                            # Get the corresponding SNR for this topic
                                            topic_name = true_theta_train.columns[i] if i < len(true_theta_train.columns) else f"Topic_{i+1}"
                                            
                                            # Find matching SNR data
                                            matching_snr = None
                                            for j, snr in enumerate(all_snrs):
                                                if all_topic_names[j] == topic_name and all_de_means[j] == de_mean:
                                                    matching_snr = snr
                                                    break
                                            
                                            if matching_snr is not None:
                                                model_recovery_data.append({
                                                    'method': method.upper(),
                                                    'topic': topic_name,
                                                    'snr': matching_snr,
                                                    'correlation': corr,
                                                    'de_mean': de_mean
                                                })
                    except Exception as e:
                        continue
        
        # Create color mapping for topics
        unique_topics = list(set(all_topic_names))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_topics)))
        topic_colors = dict(zip(unique_topics, colors))
        
        # Plot 1: DE Mean vs SNR ratio (all topics, colored by topic)
        ax1 = axes[0, 0]
        for topic in unique_topics:
            topic_indices = [i for i, name in enumerate(all_topic_names) if name == topic]
            topic_de_means = [all_de_means[i] for i in topic_indices]
            topic_snrs = [all_snrs[i] for i in topic_indices]
            ax1.scatter(topic_de_means, topic_snrs, alpha=0.7, s=100, 
                       color=topic_colors[topic], label=topic)
        
        ax1.set_xlabel('DE Mean')
        ax1.set_ylabel('Signal-to-Noise Ratio')
        ax1.set_title(f'{dataset_name}: SNR vs DE Mean (All Topics)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: SNR ratio vs correlation coefficients (model recovery)
        ax2 = axes[0, 1]
        if model_recovery_data:
            recovery_df = pd.DataFrame(model_recovery_data)
            for method in recovery_df['method'].unique():
                method_data = recovery_df[recovery_df['method'] == method]
                ax2.scatter(method_data['snr'], method_data['correlation'], 
                           alpha=0.7, s=100, label=method)
            
            ax2.set_xlabel('Signal-to-Noise Ratio')
            ax2.set_ylabel('Theta Correlation Coefficient')
            ax2.set_title(f'{dataset_name}: Model Recovery vs SNR')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No model recovery data available', 
                    transform=ax2.transAxes, ha='center', va='center')
            ax2.set_title(f'{dataset_name}: Model Recovery vs SNR')
        
        # Plot 3: DE Mean vs max eigenvalues (all topics, colored by topic)
        ax3 = axes[1, 0]
        for topic in unique_topics:
            topic_indices = [i for i, name in enumerate(all_topic_names) if name == topic]
            topic_de_means = [all_de_means[i] for i in topic_indices]
            topic_eigenvalues = [all_max_eigenvalues[i] for i in topic_indices]
            ax3.scatter(topic_de_means, topic_eigenvalues, alpha=0.7, s=100, 
                       color=topic_colors[topic], label=topic)
        
        ax3.set_xlabel('DE Mean')
        ax3.set_ylabel('Max Noise Eigenvalue')
        ax3.set_title(f'{dataset_name}: Max Noise Eigenvalue vs DE Mean')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: DE Mean vs rho coefficients (all topics, colored by topic)
        ax4 = axes[1, 1]
        for topic in unique_topics:
            topic_indices = [i for i, name in enumerate(all_topic_names) if name == topic]
            topic_de_means = [all_de_means[i] for i in topic_indices]
            topic_rhos = [all_rho_values[i] for i in topic_indices]
            ax4.scatter(topic_de_means, topic_rhos, alpha=0.7, s=100, 
                       color=topic_colors[topic], label=topic)
        
        ax4.set_xlabel('DE Mean')
        ax4.set_ylabel('ρ Coefficient')
        ax4.set_title(f'{dataset_name}: ρ Coefficient vs DE Mean')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to dataset-specific directory
        dataset_dir = estimates_path / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(dataset_dir / "comprehensive_noise_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Comprehensive plot saved for {dataset_name}: {dataset_dir}/comprehensive_noise_analysis.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare true vs estimated thetas and plot correlation heatmap.")
    parser.add_argument("--true_theta", required=True, help="Path to true theta.csv from simulation")
    parser.add_argument("--estimated_theta", required=True, help="Path to estimated theta file")
    parser.add_argument("--output", help="Path to save the plot (optional)")
    parser.add_argument("--method", default="model", help="Name of the estimation method (for plot title)")
    parser.add_argument("--true_beta", help="Path to true beta file for topic matching (optional)")
    parser.add_argument("--estimated_beta", help="Path to estimated beta file for topic matching (optional)")
    args = parser.parse_args()
    
    compare_theta_true_vs_estimated(
        args.true_theta, args.estimated_theta, args.output, args.method,
        args.true_beta, args.estimated_beta
    )