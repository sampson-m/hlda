#!/usr/bin/env python3
"""
Core evaluation functions for comparing true vs estimated model parameters.

This module provides functions for:
- Topic matching between true and estimated parameters
- Computing correlation and residual metrics
- Creating comparison visualizations
- Model recovery analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
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

    # Create scatter plots for each topic
    n_topics = min(true_theta.shape[1], est_theta.shape[1])
    
    if n_topics <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    else:
        n_cols = 3
        n_rows = (n_topics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
    
    for i in range(n_topics):
        if i < n_topics:
            ax = axes[i]
            
            # Get topic data
            true_topic = true_theta.iloc[:, i]
            est_topic = est_theta.iloc[:, i]
            
            # Create scatter plot
            ax.scatter(true_topic, est_topic, alpha=0.6, s=20)
            
            # Add diagonal line
            min_val = min(true_topic.min(), est_topic.min())
            max_val = max(true_topic.max(), est_topic.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            # Calculate correlation
            corr = np.corrcoef(true_topic, est_topic)[0, 1]
            
            # Labels and title
            ax.set_xlabel('True Theta')
            ax.set_ylabel('Estimated Theta')
            ax.set_title(f'Topic {i+1} (r={corr:.3f})')
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_topics, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'{method_name}: True vs Estimated Theta', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"      ✓ Theta comparison plot saved to: {output_path}")
    
    plt.show()


def compare_beta_true_vs_estimated(true_beta_path, estimated_beta_path, output_path=None, method_name="model"):
    """
    Compare true vs estimated beta matrices with scatter plots for each topic.
    
    Args:
        true_beta_path: Path to the true beta file (from simulation)
        estimated_beta_path: Path to the estimated beta file (from model)
        output_path: Path to save the plot (if None, will not save)
        method_name: Name of the estimation method (for plot title)
    """
    # Load true and estimated betas
    true_beta = pd.read_csv(true_beta_path, index_col=0)
    est_beta = pd.read_csv(estimated_beta_path, index_col=0)
    
    # Align genes (intersection only)
    common_genes = true_beta.index.intersection(est_beta.index)
    if len(common_genes) == 0:
        raise ValueError("No overlapping genes between true and estimated beta files.")
    true_beta = true_beta.loc[common_genes]
    est_beta = est_beta.loc[common_genes]
    
    # Align topics (intersection only)
    common_topics = true_beta.columns.intersection(est_beta.columns)
    if len(common_topics) == 0:
        raise ValueError("No overlapping topics between true and estimated beta files.")
    true_beta = true_beta.loc[:, common_topics]
    est_beta = est_beta.loc[:, common_topics]
    
    # Create scatter plots for each topic
    n_topics = len(common_topics)
    
    if n_topics <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    else:
        n_cols = 3
        n_rows = (n_topics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
    
    for i, topic in enumerate(common_topics):
        if i < len(axes):
            ax = axes[i]
            
            # Get topic data
            true_topic = true_beta[topic]
            est_topic = est_beta[topic]
            
            # Create scatter plot
            ax.scatter(true_topic, est_topic, alpha=0.6, s=20)
            
            # Add diagonal line
            min_val = min(true_topic.min(), est_topic.min())
            max_val = max(true_topic.max(), est_topic.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            # Calculate correlation
            corr = np.corrcoef(true_topic, est_topic)[0, 1]
            
            # Labels and title
            ax.set_xlabel('True Beta')
            ax.set_ylabel('Estimated Beta')
            ax.set_title(f'{topic} (r={corr:.3f})')
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_topics, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'{method_name}: True vs Estimated Beta', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"      ✓ Beta comparison plot saved to: {output_path}")
    
    plt.show()


def compute_model_recovery_metrics(estimates_path, data_root):
    """
    Compute model recovery metrics for HLDA, LDA, and NMF models using median percent residuals.
    
    Returns:
    --------
    recovery_data : list
        List of dictionaries containing recovery metrics for each topic-model combination
    """
    recovery_data = []
    
    # Find all DE_mean directories in estimates
    for dataset_dir in Path(estimates_path).iterdir():
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
                        min_topics = min(len(true_topic_names), len(est_topic_names))
                        correlation_matrix = np.zeros((min_topics, min_topics))
                        
                        for i in range(min_topics):
                            for j in range(min_topics):
                                if i < true_theta_subset.shape[1] and j < est_theta_subset.shape[1]:
                                    corr = np.corrcoef(true_theta_subset.iloc[:, i], est_theta_subset.iloc[:, j])[0, 1]
                                    correlation_matrix[i, j] = corr if not np.isnan(corr) else 0
                        
                        # Use Hungarian algorithm to find optimal topic matching
                        true_indices, est_indices = linear_sum_assignment(-correlation_matrix)  # Negative for maximization
                        
                        # Store individual topic correlations and residuals
                        for true_idx, est_idx in zip(true_indices, est_indices):
                            if true_idx < len(true_topic_names) and est_idx < len(est_topic_names):
                                true_topic_name = true_topic_names[true_idx]
                                est_topic_name = est_topic_names[est_idx]
                                theta_corr = correlation_matrix[true_idx, est_idx]
                                
                                # Compute median percent residuals for theta
                                true_theta_topic = true_theta_subset.iloc[:, true_idx].values
                                est_theta_topic = est_theta_subset.iloc[:, est_idx].values
                                
                                # Normalize to probabilities for fair comparison
                                true_theta_topic_norm = true_theta_topic / np.sum(true_theta_topic)
                                est_theta_topic_norm = est_theta_topic / np.sum(est_theta_topic)
                                
                                # Compute percent residuals: |true - est| / true * 100
                                # Add small epsilon to avoid division by zero
                                epsilon = 1e-10
                                theta_percent_residuals = np.abs(true_theta_topic_norm - est_theta_topic_norm) / (true_theta_topic_norm + epsilon) * 100
                                median_theta_residual = np.median(theta_percent_residuals)
                                
                                # For beta correlations and residuals, we need to match genes
                                if true_idx < true_beta.shape[1] and est_idx < est_beta.shape[1]:
                                    # Get common genes
                                    common_genes = [gene for gene in true_beta.index if gene in est_beta.index]
                                    if len(common_genes) > 0:
                                        true_beta_topic = true_beta.loc[common_genes, true_topic_name]
                                        est_beta_topic = est_beta.loc[common_genes, est_topic_name]
                                        beta_corr = np.corrcoef(true_beta_topic, est_beta_topic)[0, 1]
                                        beta_corr = beta_corr if not np.isnan(beta_corr) else 0
                                        
                                        # Compute median percent residuals for beta
                                        true_beta_topic_norm = true_beta_topic.values / np.sum(true_beta_topic.values)
                                        est_beta_topic_norm = est_beta_topic.values / np.sum(est_beta_topic.values)
                                        
                                        # Compute percent residuals: |true - est| / true * 100
                                        # Add small epsilon to avoid division by zero
                                        epsilon = 1e-10
                                        beta_percent_residuals = np.abs(true_beta_topic_norm - est_beta_topic_norm) / (true_beta_topic_norm + epsilon) * 100
                                        median_beta_residual = np.median(beta_percent_residuals)
                                    else:
                                        beta_corr = 0
                                        median_beta_residual = np.nan
                                else:
                                    beta_corr = 0
                                    median_beta_residual = np.nan
                                
                                recovery_data.append({
                                    'dataset': dataset_name,
                                    'de_mean': de_mean,
                                    'method': method,
                                    'topic_config': topic_config,
                                    'true_topic': true_topic_name,
                                    'est_topic': est_topic_name,
                                    'theta_corr': theta_corr,
                                    'beta_corr': beta_corr,
                                    'median_theta_residual_pct': median_theta_residual,
                                    'median_beta_residual_pct': median_beta_residual,
                                    'topic_pair': f"{true_topic_name} ↔ {est_topic_name}"
                                })
                        
                        print(f"    ✓ {method}: matched {len(true_indices)} topics")
                        
                    except Exception as e:
                        print(f"    ✗ Error processing {method}: {e}")
                        continue
    
    return recovery_data 