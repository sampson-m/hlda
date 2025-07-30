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