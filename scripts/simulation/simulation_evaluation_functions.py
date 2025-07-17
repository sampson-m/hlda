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
        
        # Calculate correlation
        corr = np.corrcoef(true_vals, est_vals)[0, 1]
        topic_label = true_theta.columns[i] if i < len(true_theta.columns) else f"Topic {i+1}"
        ax.set_title(f"{topic_label}\nCorr: {corr:.3f}")
        ax.set_xlabel("True Theta")
        ax.set_ylabel("Estimated Theta")
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
        
        # Calculate correlation
        corr = np.corrcoef(true_vals, est_vals)[0, 1]
        topic_label = true_beta_scaled.columns[i] if i < len(true_beta_scaled.columns) else f"Topic {i+1}"
        ax.set_title(f"{topic_label}\nCorr: {corr:.3f}")
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
            
            # Calculate correlation
            corr = np.corrcoef(true_vals, est_vals)[0, 1]
            topic_label = true_theta_model.columns[topic_idx] if topic_idx < len(true_theta_model.columns) else f"Topic {topic_idx+1}"
            ax.set_title(f"{method_name} - {topic_label}\nCorr: {corr:.3f}")
            ax.set_xlabel("True Theta")
            ax.set_ylabel("Estimated Theta")
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

def compute_signal_noise_metrics(X, theta, beta, L, n_identity_topics, output_path=None):
    """
    Compute signal and noise metrics based on the orthogonalization analysis.
    """
    # Convert to numpy arrays if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(theta, pd.DataFrame):
        theta = theta.values
    if isinstance(beta, pd.DataFrame):
        beta = beta.values
    
    # Ensure correct shapes
    assert X.shape[0] == theta.shape[0], "X and theta must have same number of cells"
    assert X.shape[1] == beta.shape[0], "X and beta must have same number of genes"
    assert theta.shape[1] == beta.shape[1], "theta columns must match beta columns"
    
    # Compute expected counts: L * theta * beta^T (since beta is genes×topics)
    expected_counts = L * theta @ beta.T
    
    # Compute normalization term: sqrt(L * theta * beta)
    normalization = np.sqrt(expected_counts)
    
    # Avoid division by zero
    normalization = np.maximum(normalization, 1e-10)
    
    # Normalized count matrix
    X_norm = X / normalization
    
    # Expected normalized counts
    expected_norm = expected_counts / normalization
    
    # Poisson noise term: (X - expected) / sqrt(expected)
    poisson_noise = (X - expected_counts) / normalization
    
    # QR decomposition of beta matrix (beta is genes×topics)
    Q, R = scipy.linalg.qr(beta)  # QR of beta so Q has shape (genes x topics)
    
    # Transform theta: theta_tilde = theta * R^T
    theta_tilde = theta @ R.T
    
    # Orthogonal beta: beta_tilde = Q^T (topics x genes)
    beta_tilde = Q.T
    
    # Separate identity and activity topics
    identity_indices = list(range(n_identity_topics))
    activity_indices = list(range(n_identity_topics, theta.shape[1]))
    
    if len(activity_indices) == 0:
        # No activity topics - all signal is identity signal
        results = {
            'noise': np.linalg.norm(X_norm - expected_norm + poisson_noise, 'fro'),
            'identity_signal': np.linalg.norm(X_norm - expected_norm, 'fro'),
            'activity_signal': 0.0
        }
    else:
        # Decompose activity topic(s) - for simplicity, handle first activity topic
        activity_idx = activity_indices[0]
        beta_A = beta_tilde[activity_idx, :]  # Activity topic (genes,)
        
        # Project activity topic onto identity topics
        beta_identity = beta_tilde[identity_indices, :]  # Identity topics (n_identity x genes)
        
        # Compute projections: s_i = <beta_i, beta_A>
        projections = beta_identity @ beta_A  # (n_identity,)
        
        # Compute parallel component: sum_i s_i * beta_i
        parallel_component = projections @ beta_identity  # (genes,)
        
        # Compute perpendicular component: beta_A - parallel_component
        perpendicular_component = beta_A - parallel_component  # (genes,)
        
        # Compute signal components
        # Identity signal: theta_i * beta_i + theta_A * parallel_component
        identity_signal_matrix = (theta_tilde[:, identity_indices] @ beta_identity + 
                                np.outer(theta_tilde[:, activity_idx], parallel_component))
        
        # Activity signal: theta_A * perpendicular_component
        activity_signal_matrix = np.outer(theta_tilde[:, activity_idx], perpendicular_component)
        
        # Normalize by L and normalization term
        identity_signal_norm = L * identity_signal_matrix / normalization
        activity_signal_norm = L * activity_signal_matrix / normalization
        
        # Compute metrics (Frobenius norms)
        results = {
            'noise': np.linalg.norm(X_norm - expected_norm + poisson_noise, 'fro'),
            'identity_signal': np.linalg.norm(X_norm - identity_signal_norm, 'fro'),
            'activity_signal': np.linalg.norm(X_norm - activity_signal_norm, 'fro')
        }
    
    # Save results if output path provided
    if output_path:
        results_df = pd.DataFrame([results])
        results_df.to_csv(output_path, index=False)
        print(f"Saved signal/noise metrics to {output_path}")
    
    return results

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