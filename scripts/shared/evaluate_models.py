#!/usr/bin/env python3
"""
Evaluation script for comparing HLDA, LDA, and NMF models.
"""

import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
from sklearn.decomposition import PCA
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
import cvxpy as cp
from scipy.optimize import linear_sum_assignment
import warnings
import re
import yaml
import matplotlib.patches as mpatches
import gc
from itertools import combinations
import scanpy as sc
import anndata as ad
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
warnings.filterwarnings('ignore')

# Import default parameters from fit_hlda
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from fit_hlda import get_default_parameters

# Get default HLDA parameters
HLDA_PARAMS = get_default_parameters()

# HLDA_PARAMS will be updated with CLI arguments in main()

# --- Utility functions ---
def extract_cell_identity(cell_id: str) -> str:
    """
    Extract cell identity from cell name.
    
    For simulation data, cell names are in the format "identity_cellnumber"
    (e.g., "A_1", "B_2", "C_3").
    
    Args:
        cell_id: Cell identifier string
        
    Returns:
        Cell identity (e.g., "A", "B", "C")
    """
    if '_' in cell_id:
        return cell_id.split('_')[0]
    else:
        # If no underscore, assume the whole string is the identity
        return cell_id

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def plot_geweke_histograms(sample_root: Path, keys: list[str], out_dir: Path, n_save, n_cells, n_genes, n_topics):
    out_dir = ensure_dir(out_dir)
    def load_chain(sample_root: Path, key: str, n_save, n_cells, n_genes, n_topics) -> np.ndarray:
        if key == 'A':
            return np.memmap(
                filename=str(sample_root / "A_chain.memmap"),
                mode="r",
                dtype=np.int32,
                shape=(n_save, n_genes, n_topics)
            )
        elif key == 'D':
            return np.memmap(
                filename=str(sample_root / "D_chain.memmap"),
                mode="r",
                dtype=np.int32,
                shape=(n_save, n_cells, n_topics)
            )
        else:
            raise ValueError(f"No memmap defined for key '{key}'")
    def geweke_z(chain: np.ndarray, first: float = 0.1, last: float = 0.5) -> float:
        n = len(chain)
        n1, n2 = max(1, int(first * n)), max(1, int(last * n))
        a, b = chain[:n1], chain[-n2:]
        v1, v2 = a.var(ddof=1), b.var(ddof=1)
        if v1 == 0 or v2 == 0:
            return np.nan
        return (a.mean() - b.mean()) / np.sqrt(v1 / n1 + v2 / n2)
    for key in keys:
        try:
            chain = load_chain(sample_root, key, n_save, n_cells, n_genes, n_topics)
        except ValueError as e:
            continue
        z_vals = []
        for idx in np.ndindex(chain.shape[1:]):
            ts = chain[(Ellipsis,) + idx]
            z = geweke_z(ts)
            if np.isfinite(z):
                z_vals.append(z)
        if not z_vals:
            continue
        plt.figure(figsize=(6, 4))
        plt.hist(z_vals, bins=50, edgecolor="black")
        plt.xlabel("Geweke z-score")
        plt.ylabel("Count")
        plt.title(f"Chain \"{key}\"")
        plt.tight_layout()
        plt.savefig(out_dir / f"geweke_{key}.png", dpi=300)
        plt.close()



def estimate_theta_simplex(X: np.ndarray, B, l1) -> np.ndarray:
    B = np.asarray(B)
    n, k = X.shape[0], B.shape[1]
    Theta = np.empty((n, k))
    for i, x in enumerate(X):
        th = cp.Variable(k, nonneg=True)
        obj = cp.sum_squares(B @ th - x) + l1 * cp.sum(th)
        # Use proper cvxpy constraint
        constraint = cp.sum(th) == 1.0
        prob = cp.Problem(cp.Minimize(obj), [constraint])
        prob.solve(solver=cp.OSQP, eps_abs=1e-6, verbose=False)
        Theta[i] = th.value
    return Theta

def get_or_estimate_test_theta(test_df: pd.DataFrame, beta: pd.DataFrame, model_name: str, output_dir: Path) -> pd.DataFrame:
    """
    Get test theta by either loading from cache or estimating and saving.
    Now always recomputes and overwrites the file for testing purposes.
    Args:
        test_df: Test count matrix
        beta: Beta matrix for the model
        model_name: Name of the model (HLDA, LDA, NMF)
        output_dir: Directory to save/load theta file
    Returns:
        Test theta DataFrame
    """
    # Construct the theta file path - write to outer layer model_comparison directory
    model_comparison_dir = ensure_dir(output_dir / "model_comparison")
    theta_file = model_comparison_dir / f"{model_name}_test_theta_nnls.csv"

    print(f"      🔄 [TESTING] Always recomputing test theta for {model_name}...")
    X_test_prop = test_df.div(test_df.sum(axis=1), axis=0).values
    theta_test = estimate_theta_simplex(X_test_prop, beta.values, l1=0.002)
    theta_test_df = pd.DataFrame(theta_test, index=test_df.index, columns=beta.columns)
    try:
        theta_test_df.to_csv(theta_file)
        print(f"      ✓ Saved test theta for {model_name}: {theta_file}")
    except Exception as e:
        print(f"      ⚠ Could not save test theta: {e}")
    return theta_test_df

def incremental_sse_custom(
    X_df: pd.DataFrame,
    beta_df: pd.DataFrame,
    identities: pd.Series,
    activity_topics: list[str],
    theta_out: Path | None = None,
) -> pd.DataFrame:
    """Incremental SSE with all possible activity topic combinations."""
    X_prop = X_df.div(X_df.sum(axis=1), axis=0).values
    
    # Estimate theta simplex once for all topics
    B_full = beta_df.values
    Theta_full = estimate_theta_simplex(X_prop, B_full, l1=0.002)
    
    # Create DataFrame with full theta for easy column extraction
    theta_full_df = pd.DataFrame(Theta_full, index=X_df.index, columns=beta_df.columns)
    
    results = []
    for ident in sorted(set(identities)):
        mask = identities == ident
        if not mask.any():
            continue
            
        # Always start with identity-only
        topic_steps = [[ident]]
        
        # Generate all possible combinations of activity topics
        
        # Add individual activity topics
        for activity_topic in activity_topics:
            topic_steps.append([ident, activity_topic])
        
        # Add all combinations of 2 or more activity topics
        for r in range(2, len(activity_topics) + 1):
            for combo in combinations(activity_topics, r):
                topic_steps.append([ident] + list(combo))
            
        for topics_now in topic_steps:
            # Only use topics that exist in beta_df
            topics_now = [t for t in topics_now if t in beta_df.columns]
            if len(topics_now) == 0:
                continue
                
            # Extract relevant columns from the full theta and beta
            Theta_subset = theta_full_df.loc[mask, topics_now].values
            B_subset = beta_df[topics_now].values
            
            # Calculate SSE using the subset
            sse = reconstruction_sse(X_prop[mask], Theta_subset, B_subset)
            
            # Create topic combination label
            if len(topics_now) == 1 and topics_now[0] == ident:
                topic_label = f"{ident}_only"
            else:
                topic_label = "+".join(topics_now)
            
            results.append({
                "topics": topic_label,
                "identity": ident,
                "cells": int(mask.sum()),
                "SSE": sse,
            })
    
    # Save the full theta if requested
    if theta_out is not None:
        ensure_dir(theta_out.parent)
        theta_full_df.to_csv(theta_out)
        
    return pd.DataFrame(results)

def extract_top_genes_per_topic(beta_df: pd.DataFrame, n_top_genes: int = 10) -> pd.DataFrame:
    """
    For each topic, select top_n genes with highest specificity:
    specificity = beta[g, t] - mean(beta[g, t'] for t' != t)
    """
    top_genes_dict = {}
    topics = beta_df.columns
    for topic in topics:
        # For each gene, compute specificity for this topic
        this_topic = beta_df[topic]
        other_topics = beta_df.drop(columns=topic)
        specificity = this_topic - other_topics.mean(axis=1)
        top_indices = specificity.nlargest(n_top_genes).index
        top_genes_dict[topic] = list(top_indices)
    # Create DataFrame with topics as columns and gene ranks as rows
    max_genes = max(len(genes) for genes in top_genes_dict.values())
    result_data = {}
    for topic, genes in top_genes_dict.items():
        padded_genes = genes + [''] * (max_genes - len(genes))
        result_data[topic] = padded_genes
    result_df = pd.DataFrame(result_data)
    result_df.index = [f"Gene_{i+1}" for i in range(max_genes)]
    result_df.index.name = "Gene Rank"
    return result_df

def reconstruction_sse(X: np.ndarray, Theta: np.ndarray, Beta) -> float:
    Beta = np.asarray(Beta)
    recon = Theta @ Beta.T
    return np.square(X - recon).sum()


def create_true_beta_from_counts(counts_df: pd.DataFrame, identity_topics: list[str]) -> pd.DataFrame:

    # Convert to proportions (like the SSE function does)
    counts_prop = counts_df.div(counts_df.sum(axis=1), axis=0)
    
    # Group by cell identity and average
    true_beta_dict = {}
    for identity in identity_topics:
        # Find cells that match this identity (exact match or starts with)
        mask = counts_prop.index.str.startswith(identity) | (counts_prop.index == identity)
        if mask.any():
            identity_avg = counts_prop[mask].mean(axis=0)
            true_beta_dict[identity] = identity_avg
    
    true_beta = pd.DataFrame(true_beta_dict)
    return true_beta

def match_topics_to_identities(beta_est: pd.DataFrame, true_beta: pd.DataFrame, 
                              identity_topics: list[str], n_extra_topics: int) -> dict:

    # Compute cosine similarity between estimated topics and true identity topics
    est_topics = beta_est.values.T  # shape: (n_est_topics, n_genes)
    true_topics = true_beta.values.T  # shape: (n_identity_topics, n_genes)
    
    # Normalize for cosine similarity
    est_norms = np.linalg.norm(est_topics, axis=1, keepdims=True)
    true_norms = np.linalg.norm(true_topics, axis=1, keepdims=True)
    
    est_normalized = est_topics / (est_norms + 1e-12)
    true_normalized = true_topics / (true_norms + 1e-12)
    
    # Compute similarity matrix
    similarity = est_normalized @ true_normalized.T  # shape: (n_est_topics, n_identity_topics)
    
    # Use Hungarian algorithm to find optimal matching
    est_indices, true_indices = linear_sum_assignment(-similarity)  # Negative for maximization
    
    # Create mapping
    topic_mapping = {}
    matched_identities = set()
    
    for est_idx, true_idx in zip(est_indices, true_indices):
        est_topic_name = beta_est.columns[est_idx]
        true_topic_name = identity_topics[true_idx]
        topic_mapping[est_topic_name] = true_topic_name
        matched_identities.add(true_topic_name)
    
    # Assign activity topics to unmatched estimated topics
    unmatched_est = [col for col in beta_est.columns if col not in topic_mapping]
    unmatched_identities = [ident for ident in identity_topics if ident not in matched_identities]
    
    # Assign remaining topics as activity topics
    for i, est_topic in enumerate(unmatched_est):
        if i < n_extra_topics:
            activity_name = f"V{i+1}"
            topic_mapping[est_topic] = activity_name
        else:
            # If we have more estimated topics than expected, assign as extra activity topics
            activity_name = f"V{i+1}"
            topic_mapping[est_topic] = activity_name
    
    return topic_mapping

def prepare_model_topics(model_name: str, beta: pd.DataFrame, theta: pd.DataFrame, 
                        counts_df: pd.DataFrame, identity_topics: list[str], 
                        n_extra_topics: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:

    if model_name == "HLDA":
        # HLDA already has meaningful labels, no matching needed
        topic_mapping = {col: col for col in beta.columns}  # Identity mapping
        return beta, theta, topic_mapping
    
    elif model_name in ["LDA", "NMF"]:
        # Create true beta matrix for matching
        true_beta = create_true_beta_from_counts(counts_df, identity_topics)
        topic_mapping = match_topics_to_identities(beta, true_beta, identity_topics, n_extra_topics)
        
        beta_renamed = beta.rename(columns=topic_mapping)
        theta_renamed = theta.rename(columns=topic_mapping)
        
        return beta_renamed, theta_renamed, topic_mapping
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def plot_true_vs_estimated_similarity(true_beta: pd.DataFrame, estimated_beta: pd.DataFrame, 
                                    model_name: str, out_png: Path):

    true_norms = np.linalg.norm(true_beta.values, axis=0, keepdims=True)
    est_norms = np.linalg.norm(estimated_beta.values, axis=0, keepdims=True)
    
    true_normalized = true_beta.values / (true_norms + 1e-12)
    est_normalized = estimated_beta.values / (est_norms + 1e-12)
    
    similarity = true_normalized.T @ est_normalized  # shape: (n_identities, n_topics)
    
    # Order both axes consistently: identity topics first, then activity topics
    def sort_topics(topics):
        """Sort topics with identity topics first, then activity topics (V1, V2, etc.)"""
        identity_topics = [t for t in topics if not t.startswith('V')]
        activity_topics = sorted([t for t in topics if t.startswith('V')])
        return identity_topics + activity_topics
    
    # Get consistent ordering for both axes
    y_axis_ordered = sort_topics(true_beta.columns.tolist())  # True identities (y-axis)
    x_axis_ordered = sort_topics(estimated_beta.columns.tolist())  # Estimated topics (x-axis)
    
    sim_df = pd.DataFrame(
        similarity,
        index=true_beta.columns,  # True identities
        columns=estimated_beta.columns  # Estimated topics
    )
    
    # Reorder the similarity matrix to match consistent ordering
    sim_df_ordered = sim_df.reindex(index=y_axis_ordered, columns=x_axis_ordered, fill_value=0.0)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(sim_df_ordered, annot=True, fmt=".2f", cmap="viridis", 
                cbar_kws={'label': 'Cosine Similarity'})
    plt.title(f"True vs Estimated Similarity ({model_name})\nTrue Identities vs Estimated Topics")
    plt.xlabel("Estimated Topics")
    plt.ylabel("True Cell Type Identities")
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the plot
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save the similarity matrix as CSV
    sim_csv = out_png.with_suffix('.csv')
    sim_df_ordered.to_csv(sim_csv)

def plot_true_self_similarity(true_beta: pd.DataFrame, out_png: Path):
    true_norms = np.linalg.norm(true_beta.values, axis=0, keepdims=True)
    true_normalized = true_beta.values / (true_norms + 1e-12)
    similarity = true_normalized.T @ true_normalized
    
    # Order consistently: identity topics first, then activity topics
    def sort_topics(topics):
        """Sort topics with identity topics first, then activity topics (V1, V2, etc.)"""
        identity_topics = [t for t in topics if not t.startswith('V')]
        activity_topics = sorted([t for t in topics if t.startswith('V')])
        return identity_topics + activity_topics
    
    # Get consistent ordering for both axes
    ordered_topics = sort_topics(true_beta.columns.tolist())
    
    sim_df = pd.DataFrame(
        similarity,
        index=true_beta.columns,
        columns=true_beta.columns
    )
    
    # Reorder the similarity matrix to match consistent ordering
    sim_df_ordered = sim_df.reindex(index=ordered_topics, columns=ordered_topics, fill_value=0.0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_df_ordered, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Cosine Similarity'})
    plt.title("True Identity Self-Similarity\nAveraged Gene Expression Profiles")
    plt.xlabel("True Cell Type Identities")
    plt.ylabel("True Cell Type Identities")
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save the similarity matrix as CSV
    sim_csv = out_png.with_suffix('.csv')
    sim_df_ordered.to_csv(sim_csv)


def plot_combined_theta_usage_umaps(counts_df: pd.DataFrame, cell_identities: pd.Series, 
                                   theta_dfs: dict, activity_topics: list, out_dir: Path, suffix: str = ""):
    """
    Create combined UMAP plots showing theta usage for all models (HLDA, LDA, NMF) in one row.
    
    Args:
        counts_df: Count matrix
        cell_identities: Cell type labels
        theta_dfs: Dictionary with model names as keys and theta DataFrames as values
        activity_topics: List of activity topic names
        out_dir: Output directory
        suffix: Suffix for filename (e.g., "_test")
    """
    try:
        import scanpy as sc
        import anndata as ad
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    
    # Create AnnData object from count matrix
    adata = ad.AnnData(X=counts_df.values.astype(np.float32), 
                       obs=pd.DataFrame(index=counts_df.index),
                       var=pd.DataFrame(index=counts_df.columns))
    
    # Add cell identities
    adata.obs['cell_type'] = cell_identities.values
    
    # Add theta usage for each model
    for model_name, theta in theta_dfs.items():
        activity_cols = [col for col in theta.columns if col in activity_topics]
        if activity_cols:
            # Handle potential duplicate indices
            theta_usage = theta[activity_cols].sum(axis=1)
            if theta_usage.index.has_duplicates:
                # Map by cell type for duplicates using vectorized operations
                usage_by_type = theta_usage.groupby(theta_usage.index).mean()
                cell_types = pd.Series([extract_cell_identity(cell_id) for cell_id in adata.obs.index], 
                                      index=adata.obs.index)
                adata.obs[f'{model_name}_theta_usage'] = cell_types.map(usage_by_type).fillna(0.0).astype(np.float32)
            else:
                # Safe to use direct mapping
                adata.obs[f'{model_name}_theta_usage'] = theta_usage.reindex(adata.obs.index).fillna(0).values.astype(np.float32)
        else:
            adata.obs[f'{model_name}_theta_usage'] = 0
    
    # Preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # PCA and neighbors
    n_pcs = min(50, min(adata.n_vars, adata.n_obs) - 1)
    sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs)
    n_neighbors = min(15, adata.n_obs - 1)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=min(30, n_pcs))
    
    # UMAP
    sc.tl.umap(adata, random_state=42)
    
    # Create combined plot: one row with three models
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    model_order = ['HLDA', 'LDA', 'NMF']
    for i, model in enumerate(model_order):
        if model in theta_dfs:
            usage_col = f'{model}_theta_usage'
            if usage_col in adata.obs:
                sc.pl.umap(adata, color=usage_col, ax=axes[i], show=False, 
                          title=f'{model} Activity Usage{suffix}', size=20, colorbar_loc=None)
                # Add individual colorbar for each subplot
                plt.colorbar(axes[i].collections[0], ax=axes[i], fraction=0.03, pad=0.04)
            else:
                axes[i].text(0.5, 0.5, f'{model}\nNo Data', ha='center', va='center', 
                            transform=axes[i].transAxes, fontsize=14)
                axes[i].set_title(f'{model} Activity Usage{suffix}')
        else:
            axes[i].text(0.5, 0.5, f'{model}\nNot Available', ha='center', va='center', 
                        transform=axes[i].transAxes, fontsize=14)
            axes[i].set_title(f'{model} Activity Usage{suffix}')
    
    # Adjust layout to prevent overlap of labels and colorbars
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)  # Add horizontal spacing for colorbars
    
    ensure_dir(out_dir)
    output_file = out_dir / f"combined_theta_usage_umaps{suffix}.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Memory cleanup
    del adata
    gc.collect()

def plot_umap_scanpy_clustering(counts_df: pd.DataFrame, cell_identities: pd.Series, model_name: str, 
                                theta: pd.DataFrame, activity_topics: list, out_dir: Path):
    """
    Create UMAP plots using actual count matrix with Scanpy clustering.
    Colors by: 1) Activity program usage, 2) Cell type labels, 3) Scanpy clusters
    Memory-optimized version.
    """
    try:
        pass
    except ImportError:
        return
    
    
    # Create AnnData object from count matrix with float32 to save memory
    adata = ad.AnnData(X=counts_df.values.astype(np.float32), 
                       obs=pd.DataFrame(index=counts_df.index),
                       var=pd.DataFrame(index=counts_df.columns))
    
    # Add cell identities to obs
    adata.obs['cell_type'] = cell_identities.values
    
    # Add theta values to obs for activity usage (float32 to save memory)
    for col in theta.columns:
        adata.obs[f'theta_{col}'] = theta[col].values.astype(np.float32)
    
    # Compute activity usage
    activity_cols = [col for col in theta.columns if col in activity_topics]
    if activity_cols:
        adata.obs['activity_usage'] = theta[activity_cols].sum(axis=1).values.astype(np.float32)
    else:
        adata.obs['activity_usage'] = 0
    
    # Preprocessing with memory optimization
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Compute PCA with fewer components to save memory
    n_pcs = min(50, min(adata.n_vars, adata.n_obs) - 1)  # Use fewer PCs for large datasets
    sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs)
    
    # Compute neighbors with fewer PCs to save memory
    n_neighbors = min(10, adata.n_obs - 1)  # Ensure n_neighbors < n_cells
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=min(30, n_pcs))
    
    # Compute UMAP
    sc.tl.umap(adata, random_state=42)
    
    # Clustering
    sc.tl.leiden(adata, resolution=0.5)
    
    # Create plots with smaller figure size to save memory
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. UMAP colored by activity usage
    sc.pl.umap(adata, color='activity_usage', ax=axes[0], show=False, 
                title=f'{model_name}: Activity Usage', size=15)
    
    # 2. UMAP colored by cell type
    sc.pl.umap(adata, color='cell_type', ax=axes[1], show=False, 
                title=f'{model_name}: Cell Type', size=15)
    
    
    # 3. UMAP colored by scanpy clusters
    sc.pl.umap(adata, color='leiden', ax=axes[2], show=False, 
                title=f'{model_name}: Scanpy Clusters', size=15)
    
    plt.tight_layout()
    ensure_dir(out_dir)
    plt.savefig(out_dir / f"{model_name}_umap_scanpy_clustering.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # Save cluster assignments
    cluster_df = pd.DataFrame({
        'cell_id': adata.obs.index,
        'scanpy_cluster': adata.obs['leiden'].values,
        'cell_type': adata.obs['cell_type'].values,
        'activity_usage': adata.obs['activity_usage'].values
    })
    cluster_df.to_csv(out_dir / f"{model_name}_scanpy_clusters.csv", index=False)
    
    # Clean up memory
    del adata
    gc.collect()
    
    return None  # Don't return adata to save memory



def get_num_activity_topics(topic_str, identity):
    # Count how many V's are in the topic string (excluding the identity itself)
    return len(re.findall(r'V\d+', topic_str))

def plot_combined_identity_sse_heatmap(sse_df, output_dir):
    
    
    # Check the format of the SSE data and handle both formats
    sample_topics = sse_df['topics'].iloc[0] if len(sse_df) > 0 else ""
    sample_identity = sse_df['identity'].iloc[0] if len(sse_df) > 0 else ""
    
    
    # Handle two different SSE file formats:
    # Format 1: Combined scheme - topics start with identity (e.g., "B cell_only", "B cell+V1")
    # Format 2: Disease-specific scheme - topics don't start with identity (e.g., "V1", "V2")
    
    if sample_topics.startswith(sample_identity):
        # Format 1: Filter normally
        sse_df = sse_df[sse_df.apply(lambda row: row['topics'].startswith(row['identity']), axis=1)]
    else:
        # Format 2: Keep all rows (topics like V1, V2 are valid)
        pass
    
    if len(sse_df) == 0:
        return
    
    # Create row label for identity and model
    sse_df['identity_model'] = sse_df['identity'] + ' | ' + sse_df['model']
    
    # Extract the activity part from the topic string to use as column names
    def extract_activity_part(topic_str, identity):
        try:
            # Ensure we're working with strings
            topic_str = str(topic_str)
            identity = str(identity)
            
            # Handle two different formats:
            if topic_str.startswith(identity):
                # Format 1: "B cell_only", "B cell+V1" -> "identity_only", "+V1"
                activity_part = topic_str.replace(identity, "").replace("_only", "")
                activity_part = activity_part.strip("+")
                if activity_part == "":
                    return "identity_only"
                else:
                    return "+" + activity_part
            else:
                # Format 2: "V1", "V2", "V1+V2" -> "+V1", "+V2", "+V1+V2"
                if topic_str.startswith("V"):
                    return "+" + topic_str
                else:
                    return "identity_only"
        except Exception as e:
            return "identity_only"
    
    # Apply the function row by row to avoid DataFrame assignment issues
    activity_combos = []
    for idx, row in sse_df.iterrows():
        activity_combo = extract_activity_part(row['topics'], row['identity'])
        activity_combos.append(activity_combo)
    
    sse_df['activity_combo'] = activity_combos
    
    # Sort by identity then model for grouping
    sse_df = sse_df.sort_values(['identity', 'model'])
    
    # Create pivot table with activity combinations as columns
    pivot = sse_df.pivot_table(index='identity_model', columns='activity_combo', values='SSE', aggfunc='first')
    
    # Reorder columns to have a logical progression
    column_order = []
    
    # First add identity_only if it exists
    if 'identity_only' in pivot.columns:
        column_order.append('identity_only')
    
    # Add +V1, +V2, +V3, etc. in order
    for i in range(1, 10):  # Support up to 9 activity topics
        v_col = f'+V{i}'
        if v_col in pivot.columns:
            column_order.append(v_col)
    
    # Add combinations like +V1+V2, +V1+V3, etc.
    for col in sorted(pivot.columns):
        if col not in column_order and col.startswith('+V'):
            column_order.append(col)
    
    # Add any remaining columns that don't match our patterns
    for col in pivot.columns:
        if col not in column_order:
            column_order.append(col)
    
    # Only reorder if we have columns to reorder with
    if column_order:
        pivot = pivot[column_order]
    
    
    # Check if pivot table is empty
    if pivot.empty or pivot.shape[0] == 0 or pivot.shape[1] == 0:
        return
    
    # Plot 1: Raw SSE values
    plt.figure(figsize=(12, max(8, 0.4 * len(pivot))))
    ax = sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis_r', linewidths=0.5, linecolor='white')
    plt.title('SSE by Cell Identity, Model, and Activity Topic Combinations (Raw Values)')
    plt.xlabel('Activity Topic Combinations')
    plt.ylabel('Cell Identity | Model')
    
    # Add red border to the single lowest SSE for each identity
    for identity in sse_df['identity'].unique():
        sub = sse_df[sse_df['identity'] == identity]
        if sub.empty:
            continue
        min_idx = sub['SSE'].idxmin()
        min_activity = sub.loc[min_idx, 'activity_combo']
        min_model = sub.loc[min_idx, 'model']
        row_label = f'{identity} | {min_model}'
        if row_label in pivot.index and min_activity in pivot.columns:
            row_idx = list(pivot.index).index(row_label)
            col_idx = list(pivot.columns).index(min_activity)
            ax.add_patch(Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='red', linewidth=3))
    
    plt.tight_layout()
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'sse_heatmap_combined_raw.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Percentage change from identity-only baseline
    # Calculate percentage change: (SSE_combo - SSE_identity_only) / SSE_identity_only * 100
    pivot_pct = pivot.copy()
    
    if 'identity_only' in pivot_pct.columns:
        # Calculate percentage change for each row
        identity_baseline = pivot_pct['identity_only']
        for col in pivot_pct.columns:
            if col != 'identity_only':
                pivot_pct[col] = ((pivot_pct[col] - identity_baseline) / identity_baseline * 100)
        
        # Remove identity_only column since it's always 0% change
        pivot_pct = pivot_pct.drop(columns=['identity_only'])
        
        # Plot with diverging colormap (green for negative/good, red for positive/bad)
        plt.figure(figsize=(12, max(8, 0.4 * len(pivot_pct))))
        
        # Use RdYlGn_r colormap (green for negative values, red for positive)
        # Center at 0 since that's our baseline
        vmax = max(abs(pivot_pct.min().min()), abs(pivot_pct.max().max()))
        ax = sns.heatmap(pivot_pct, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                        center=0, vmin=-vmax, vmax=vmax,
                        linewidths=0.5, linecolor='white',
                        cbar_kws={'label': 'Percent Change from Identity-only (%)'})
        
        plt.title('SSE Percentage Change from Identity-only Baseline\n(Green = Improvement, Red = Worse)')
        plt.xlabel('Activity Topic Combinations')
        plt.ylabel('Cell Identity | Model')
        
        # Add red border to the single best improvement for each identity
        for identity in sse_df['identity'].unique():
            sub = sse_df[sse_df['identity'] == identity]
            if sub.empty:
                continue
            min_idx = sub['SSE'].idxmin()
            min_activity = sub.loc[min_idx, 'activity_combo']
            min_model = sub.loc[min_idx, 'model']
            row_label = f'{identity} | {min_model}'
            if row_label in pivot_pct.index and min_activity in pivot_pct.columns:
                row_idx = list(pivot_pct.index).index(row_label)
                col_idx = list(pivot_pct.columns).index(min_activity)
                ax.add_patch(Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='red', linewidth=3))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'sse_heatmap_combined_pct_change.png', dpi=300, bbox_inches='tight')
        plt.close()
    
# NOTE: Function no longer used after simplifying theta heatmap section
def create_meta_cells(theta_df, cell_identities, cells_per_group=3, all_topics=None):
    """
    Create meta-cells by averaging groups of cells within each cell type.
    
    Returns:
        tuple: (meta_rows, meta_identities, topics_used)
    """
    if all_topics is None:
        topics_used = list(theta_df.columns)
    else:
        # Ensure consistent topic order across models
        topics_used = [t for t in all_topics if t in theta_df.columns]
    
    meta_rows = []
    meta_identities = []
    
    for ct in sorted(cell_identities.unique()):
        ct_mask = cell_identities == ct
        ct_cells = theta_df.loc[ct_mask]
        for i in range(0, len(ct_cells), cells_per_group):
            chunk = ct_cells.iloc[i:i+cells_per_group]
            if len(chunk) == 0:
                continue
            avg_theta = chunk[topics_used].mean(axis=0)
            meta_rows.append(avg_theta.values)
            meta_identities.append(ct)
    
    return meta_rows, meta_identities, topics_used

def stack_theta_heatmaps(output_dir: Path, model_names: list = None):
    """
    Stack theta heatmaps from all models vertically into a single combined image.
    
    Args:
        output_dir: Base output directory containing model subdirectories
        model_names: List of model names to include (default: ["HLDA", "LDA", "NMF"])
    """
    if model_names is None:
        model_names = ["HLDA", "LDA", "NMF"]
    
    print("  Stacking theta heatmaps...")
    
    # Collect heatmap image paths
    heatmap_paths = []
    valid_models = []
    
    for model in model_names:
        heatmap_path = output_dir / model / "plots" / f"{model}_theta_heatmap.png"
        if heatmap_path.exists():
            heatmap_paths.append(heatmap_path)
            valid_models.append(model)
            print(f"    Found heatmap for {model}")
        else:
            print(f"    Missing heatmap for {model}: {heatmap_path}")
    
    if len(heatmap_paths) == 0:
        print("    No heatmap images found to stack")
        return None
    
    # Load images
    images = []
    for path in heatmap_paths:
        try:
            img = mpimg.imread(path)
            images.append(img)
        except Exception as e:
            print(f"    Error loading {path}: {e}")
            continue
    
    if len(images) == 0:
        print("    No valid images could be loaded")
        return None
    
    # Calculate dimensions
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    max_width = max(widths)
    label_height = 50  # Space for model labels
    total_height = sum(heights) + label_height * len(images)
    
    # Create combined figure
    fig_width = max_width / 100  # Convert pixels to inches (assuming 100 DPI)
    fig_height = total_height / 100
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Stack images vertically
    y_offset = 0
    for i, (img, model_name) in enumerate(zip(images, valid_models)):
        # Calculate position for this image
        img_height = heights[i]
        y_pos = 1.0 - (y_offset + img_height + label_height) / total_height
        height_frac = img_height / total_height
        
        # Add image
        ax = fig.add_axes((0.0, y_pos, 1.0, height_frac))
        ax.imshow(img)
        ax.axis('off')
        
        # Add model label above image
        label_y = 1.0 - (y_offset + label_height/2) / total_height
        fig.text(0.5, label_y, model_name, ha='center', va='center', 
                fontsize=18, weight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        y_offset += img_height + label_height
    
    # Save combined image
    output_file = output_dir / "model_comparison" / "theta_heatmaps_combined.png"
    ensure_dir(output_file.parent)
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    fig.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"    ✓ Combined theta heatmaps saved to: {output_file}")
    return output_file

# Global cache for true beta to avoid recomputation
_TRUE_BETA_CACHE = {}

def compute_true_beta_from_counts(counts_df: pd.DataFrame, cell_identities: pd.Series, 
                                 identity_topics: list) -> pd.DataFrame:
    """
    Compute true beta (cell type average expression) from count matrix.
    This is expensive so we cache the result.
    """
    # Create cache key based on data shapes and identity topics
    cache_key = f"{counts_df.shape}_{len(cell_identities)}_{','.join(sorted(identity_topics))}"
    
    if cache_key in _TRUE_BETA_CACHE:
        print(f"         Using cached true beta")
        return _TRUE_BETA_CACHE[cache_key]
    
    print(f"         Computing true beta from count matrix...")
    
    # Create true beta from count matrix (average expression per cell type)
    counts_prop = counts_df.div(counts_df.sum(axis=1), axis=0)
    true_beta_dict = {}
    
    for identity in identity_topics:
        # Find cells that match this identity
        mask = cell_identities == identity
        if mask.any():
            identity_avg = counts_prop[mask].mean(axis=0)
            true_beta_dict[identity] = identity_avg
    
    if not true_beta_dict:
        return None
    
    true_beta = pd.DataFrame(true_beta_dict)
    
    # Cache the result
    _TRUE_BETA_CACHE[cache_key] = true_beta
    print(f"         Cached true beta for future use")
    
    return true_beta

def create_knn_meta_cells(theta_data: np.ndarray, cells_per_group: int) -> list:
    """
    Create meta-cells using KNN clustering within a cell type.
    
    Args:
        theta_data: Numpy array of theta values for a single cell type
        cells_per_group: Target number of cells per meta-cell
    
    Returns:
        List of averaged theta vectors (meta-cells)
    """
    if len(theta_data) == 0:
        return []
    
    if len(theta_data) <= cells_per_group:
        # If we have fewer cells than group size, just average all
        return [np.mean(theta_data, axis=0)]
    
    # Determine number of meta-cells to create
    n_meta_cells = max(1, len(theta_data) // cells_per_group)
    
    # Use KMeans-like approach: find cluster centers, then average neighborhoods
    from sklearn.cluster import KMeans
    
    try:
        # Cluster cells into groups
        kmeans = KMeans(n_clusters=n_meta_cells, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(theta_data)
        
        # Create meta-cells by averaging within each cluster
        meta_cells = []
        for cluster_id in range(n_meta_cells):
            cluster_mask = cluster_labels == cluster_id
            if cluster_mask.any():
                cluster_cells = theta_data[cluster_mask]
                meta_cell = np.mean(cluster_cells, axis=0)
                meta_cells.append(meta_cell)
        
        return meta_cells
        
    except Exception as e:
        print(f"           KMeans failed ({e}), using simple chunking")
        # Fallback to simple chunking
        meta_cells = []
        for i in range(0, len(theta_data), cells_per_group):
            chunk = theta_data[i:i+cells_per_group]
            if len(chunk) > 0:
                meta_cells.append(np.mean(chunk, axis=0))
        return meta_cells

def create_meta_cells_chunked(theta: pd.DataFrame, cell_identities: pd.Series, 
                              cells_per_group: int, temp_dir: Path, model_name: str = None) -> Path:
    """
    Create meta-cells in chunks and save to disk to reduce memory usage.
    Returns:
        Path to saved meta-cells CSV file
    """
    print(f"         Creating meta-cells in chunks...")
    # Ensure temp directory exists
    temp_dir.mkdir(parents=True, exist_ok=True)
    all_topics = list(theta.columns)
    group_identities = cell_identities
    cell_types_to_process = sorted(pd.Series(group_identities).unique())
    print(f"         Processing {len(cell_types_to_process)} cell types...")
    if model_name:
        meta_cells_file = temp_dir / f"{model_name}_meta_cells.csv"
    else:
        meta_cells_file = temp_dir / "meta_cells_temp.csv"
    # Remove the file if it already exists to avoid column misalignment
    if meta_cells_file.exists():
        meta_cells_file.unlink()
    meta_cells_list = []
    meta_identities_list = []
    wrote_header = False
    for ct_idx, ct in enumerate(cell_types_to_process):
        ct_mask = group_identities == ct
        if not ct_mask.any():
            continue
        ct_cells = theta.loc[ct_mask]
        ct_data = ct_cells.values.astype(np.float32)
        ct_meta_cell_vectors = create_knn_meta_cells(ct_data, cells_per_group)
        for i, meta_cell in enumerate(ct_meta_cell_vectors):
            meta_cells_list.append(meta_cell)
            meta_identities_list.append(ct)
        del ct_cells, ct_data, ct_meta_cell_vectors
        gc.collect()
        if (ct_idx + 1) % 3 == 0 or ct_idx == len(cell_types_to_process) - 1:
            if meta_cells_list:
                meta_cells_df = pd.DataFrame(meta_cells_list, columns=all_topics)
                meta_cells_df['cell_type'] = meta_identities_list
                # Write header only for the first chunk
                if not wrote_header:
                    meta_cells_df.to_csv(meta_cells_file, index=False)
                    wrote_header = True
                else:
                    meta_cells_df.to_csv(meta_cells_file, mode='a', header=False, index=False)
                meta_cells_list = []
                meta_identities_list = []
                del meta_cells_df
                gc.collect()
    return meta_cells_file

def plot_theta_heatmap(theta: pd.DataFrame, cell_identities: pd.Series, model_name: str, out_png: Path, identity_topics: list, cells_per_group: int = 10, use_consistent_ordering: bool = False, reference_ordering: list = None):
    """
    Plot a heatmap of theta (topic usage) with meta-cells (averaged groups of cells) as columns (x-axis), grouped by cell type, and topics as rows (y-axis).
    Meta-cells are computed by splitting each cell type's cells into chunks of cells_per_group and averaging within each chunk.
    Only one x-axis label is shown per cell type, centered in its group. Legend is below the plot. Uses 'viridis' colormap.
    """
    print(f"    Creating theta heatmap for {model_name} with {len(theta)} cells...")

    # Check if theta has 'cell_type' column (meta-cells format) or uses cell_identities
    if 'cell_type' in theta.columns:
        # Meta-cells format: extract cell types and topic data
        cell_types = theta['cell_type'].values
        topic_cols = [col for col in theta.columns if col != 'cell_type']
        theta_data = theta[topic_cols]
        all_topics = topic_cols
    else:
        # Regular format: use provided cell_identities
        cell_types = cell_identities.values
        theta_data = theta
        all_topics = list(theta.columns)
    
    # Reorder topics: identity topics first, then activity topics (V1, V2, etc.) at bottom
    def sort_topics_for_display(topics):
        """Sort topics with identity topics first, then activity topics at bottom"""
        identity_topics_sorted = [t for t in topics if not t.startswith('V')]
        activity_topics_sorted = sorted([t for t in topics if t.startswith('V')])
        return identity_topics_sorted + activity_topics_sorted
    
    # Apply topic ordering
    ordered_topics = sort_topics_for_display(all_topics)
    try:
        # Check if all ordered topics exist in the data
        missing_topics = [t for t in ordered_topics if t not in theta_data.columns]
        if missing_topics:
            print(f"    Warning: Missing topics in data: {missing_topics}")
            # Use only topics that exist in the data
            ordered_topics = [t for t in ordered_topics if t in theta_data.columns]
        
        theta_data = theta_data[ordered_topics]
        all_topics = ordered_topics  # Update for y-axis labels
    except KeyError as e:
        print(f"    Warning: Column reordering failed: {e}")
        # Fall back to original order
        all_topics = list(theta_data.columns)

    # Smart ordering: alternate large and small groups to space out labels
    def get_smart_cell_type_order(cell_types_array):
        """Order cell types alternating large/small groups to prevent label overlap"""
        
        # Count cells per type
        unique_types = np.unique(cell_types_array)
        type_counts = {ct: np.sum(cell_types_array == ct) for ct in unique_types}
        
        # Define dataset-specific optimal orderings based on cell counts
        # These orderings alternate large/small groups to space labels
        
        # PBMC dataset ordering (distribute small types between large buffers)
        # T cells: 48,657 (71%) | CD56+ NK: 8,776 (13%) | CD19+ B: 5,908 (9%) | CD14+ Monocyte: 2,862 (4%) | Dendritic: 2,099 (3%) | CD34+: 277 (0.4%)
        pbmc_order = ['CD34+', 'T cells', 'Dendritic', 'CD56+ NK', 'CD14+ Monocyte', 'CD19+ B']
        
        # Glioma dataset ordering (start small, distribute small types evenly)
        glioma_order = ['B cell', 'Diff.-like', 'Fibroblast', 'Myeloid', 'Endothelial', 
                       'Stem-like', 'Pericyte', 'Oligodendrocyte', 'T cell', 'Prolif. stem-like', 'Dendritic cell', 'Granulocyte']
        
        # Cancer combined ordering (distribute small types between large buffers)
        # T cell: 8,374 (39%) | malignant cell: 6,478 (30%) | fibroblast: 2,645 (12%) | myeloid cell: 1,896 (9%) | endothelial cell: 1,509 (7%) | B cell: 439 (2%) | monocyte: 94 (0.4%) | others: <100 each
        cancer_combined_order = ['microglial cell', 'T cell', 'plasmacytoid dendritic cell, human', 'malignant cell', 'monocyte', 'fibroblast', 'B cell', 'myeloid cell', 'endothelial cell']
        
        # Cancer disease-specific ordering (breast vs melanoma - just two large groups)
        cancer_disease_order = ['breast_T cell', 'melanoma_microglial cell', 'melanoma_T cell', 'melanoma_B cell', 'breast_fibroblast', 'melanoma_monocyte', 'breast_endothelial cell', 'breast_plasmacytoid dendritic cell', 'breast_myeloid cell', 'breast_malignant cell', 'breast_B cell', 'melanoma_malignant cell']
        
        # Simulation dataset ordering (A and B identity types)
        simulation_order = ['A', 'B']
        
        # Detect which dataset we're working with and use appropriate ordering
        type_names = set(unique_types)
        
        if 'A' in type_names and 'B' in type_names and len(type_names) == 2:
            # Simulation dataset (AB_V1)
            preferred_order = [ct for ct in simulation_order if ct in type_names]
        elif 'T cells' in type_names or 'CD56+ NK' in type_names:
            # PBMC dataset
            preferred_order = [ct for ct in pbmc_order if ct in type_names]
        elif 'Diff.-like' in type_names or 'Stem-like' in type_names:
            # Glioma dataset
            preferred_order = [ct for ct in glioma_order if ct in type_names]
        elif 'breast' in type_names or 'melanoma' in type_names:
            # Cancer disease-specific
            preferred_order = [ct for ct in cancer_disease_order if ct in type_names]
        elif 'malignant cell' in type_names:
            # Cancer combined
            preferred_order = [ct for ct in cancer_combined_order if ct in type_names]
        else:
            # Fallback: sort by count (largest first, then alternate)
            sorted_by_count = sorted(unique_types, key=lambda x: type_counts[x], reverse=True)
            preferred_order = sorted_by_count
        
        # Add any types not in our predefined order
        remaining_types = [ct for ct in unique_types if ct not in preferred_order]
        preferred_order.extend(sorted(remaining_types))
        
        return preferred_order
    
    # Get smart ordering
    unique_types = get_smart_cell_type_order(cell_types)
    
    # Reorder data according to smart ordering (no spacing, just smart order)
    reordered_indices = []
    for ct in unique_types:
        ct_mask = cell_types == ct
        ct_indices = np.where(ct_mask)[0]
        reordered_indices.extend(ct_indices)
    
    # Apply reordering
    reordered_theta_data = theta_data.values[reordered_indices]
    reordered_cell_types = cell_types[reordered_indices]
    
    # Compute x-tick positions and labels: one per cell type group (centered)
    xticks = []
    xticklabels = []
    start = 0
    
    for ct in unique_types:
        ct_count = np.sum(reordered_cell_types == ct)
        if ct_count > 0:
            # Center the tick in the middle of this cell type's columns
            xticks.append(start + ct_count // 2)
            xticklabels.append(ct)
            start += ct_count

    # Figure size based on number of meta-cells and topics
    # Increase width based on number of unique cell types to prevent label overlap
    min_width = max(8, len(unique_types) * 1.5)  # At least 1.5 inches per cell type
    max_width = 20
    fig_width = min(max_width, min_width)
    fig_height = 0.7 * len(ordered_topics) + 3  # Extra space for rotated labels

    # Plot (transpose: topics as rows, meta-cells as columns)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(reordered_theta_data.T, aspect='auto', cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
    
    # X-axis: only one label per cell type, centered
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=12)
    
    # Y-axis: topics (use ordered topics to match reordered data)
    ax.set_yticks(range(len(ordered_topics)))
    ax.set_yticklabels(ordered_topics, fontsize=9)
    
    # Ensure y-axis limits show all topics
    ax.set_ylim(-0.5, len(ordered_topics) - 0.5)

    # Draw colored rectangles for each cell type group on the x-axis
    type_colors = dict(zip(unique_types, sns.color_palette('tab20', n_colors=len(unique_types))))
    start = 0
    for ct in unique_types:
        ct_count = np.sum(reordered_cell_types == ct)
        if ct_count > 0:
            ax.add_patch(mpatches.Rectangle((start-0.5, -0.5), ct_count, len(ordered_topics), color=type_colors[ct], alpha=0.08, linewidth=0))
            start += ct_count
    
    # Colorbar for expression
    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label('Topic usage (proportion)', fontsize=10)
    ax.set_title(f"{model_name} topic usage heatmap (meta-cells, {cells_per_group} cells/group)", fontsize=12)
    
    # Adjust layout with extra bottom margin for rotated labels
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Add extra space at bottom for rotated labels
    
    # Ensure output directory is a 'plots' subfolder
    out_png = Path(out_png)
    if out_png.parent.name != 'plots':
        out_png = out_png.parent / 'plots' / out_png.name
    ensure_dir(out_png.parent)
    print(f"    Saving theta heatmap to: {out_png}")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()

    # Aggressive memory cleanup
    gc.collect()

def plot_cumulative_sse_lineplot(sse_summary_df, identity_topics, activity_topics, output_dir):
    """
    Plot cumulative SSE as activity topics are added to identity topics.
    Shows one line per model, summing SSE across all cell types.
    X-axis: descriptive topic combinations; Y-axis: total SSE
    """
    
    # Filter to only include rows that start with identity (exclude mixed combinations)
    sse_df = sse_summary_df[sse_summary_df.apply(lambda row: row['topics'].startswith(row['identity']), axis=1)].copy()
    
    # Count number of activity topics in each combination
    def count_activity_topics(topic_str, identity):
        """Count number of activity topics in a combination string"""
        if topic_str == f"{identity}_only":
            return 0
        else:
            # Count activity topics
            return len([t for t in activity_topics if t in topic_str])
    
    sse_df['n_activity_topics'] = sse_df.apply(lambda row: count_activity_topics(row['topics'], row['identity']), axis=1)
    
    # Create descriptive labels for combinations
    def create_combo_label(n_activity):
        if n_activity == 0:
            return "Identity only"
        elif n_activity == 1:
            return "Identity + V1"
        elif n_activity == 2:
            return "Identity + V1 + V2"
        elif n_activity == 3:
            return "Identity + V1 + V2 + V3"
        else:
            return f"Identity + V1...V{n_activity}"
    
    # Sum SSE across all cell types for each model and SPECIFIC activity topic combination
    model_totals = []
    for model in sse_df['model'].unique():
        model_df = sse_df[sse_df['model'] == model]
        
        # Create the specific cumulative combinations we want to show
        for n_activity in sorted(model_df['n_activity_topics'].unique()):
            # Define the specific topic combination for this cumulative step
            if n_activity == 0:
                target_pattern = "_only"
            elif n_activity == 1:
                target_pattern = "+V1"
            elif n_activity == 2:
                target_pattern = "+V1+V2"
            elif n_activity == 3:
                target_pattern = "+V1+V2+V3"
            else:
                # For higher numbers, construct the pattern
                v_parts = [f"V{i+1}" for i in range(n_activity)]
                target_pattern = "+" + "+".join(v_parts)
            
            # Find rows that match this specific pattern
            if n_activity == 0:
                subset = model_df[model_df['topics'].str.endswith('_only')]
            else:
                subset = model_df[model_df['topics'].str.endswith(target_pattern)]
            
            if len(subset) > 0:
                total_sse = subset['SSE'].sum()  # Sum across all cell types
                combo_label = create_combo_label(n_activity)
                model_totals.append({
                    'model': model,
                    'n_activity_topics': n_activity,
                    'combo_label': combo_label,
                    'total_sse': total_sse
                })
    
    model_totals_df = pd.DataFrame(model_totals)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Get unique combination labels in order
    combo_labels = sorted(model_totals_df['combo_label'].unique(), 
                         key=lambda x: model_totals_df[model_totals_df['combo_label'] == x]['n_activity_topics'].iloc[0])
    
    # Plot lines for each model
    for model in model_totals_df['model'].unique():
        model_data = model_totals_df[model_totals_df['model'] == model].sort_values('n_activity_topics')
        plt.plot(model_data['combo_label'], model_data['total_sse'], 
                marker='o', label=model, linewidth=2, markersize=6)
    
    plt.xlabel('Topic Combination')
    plt.ylabel('Total SSE (summed across cell types)')
    plt.title('Cumulative SSE as Activity Topics are Added')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(output_dir) / 'cumulative_sse_by_topic.png', dpi=200, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate topic models (HLDA, LDA, NMF) and run metrics/plots.")
    parser.add_argument("--counts_csv", type=str, required=True, help="Path to filtered count matrix CSV")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test count matrix CSV")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory with model outputs and for saving plots/metrics")
    parser.add_argument("--n_extra_topics", type=int, required=True, help="Number of extra topics (e.g. 3 for V1,V2,V3)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (must match a key in the config file)")
    parser.add_argument("--config_file", type=str, default="dataset_identities.yaml", help="Path to dataset identity config YAML file")
    parser.add_argument("--n_loops", type=int, default=15000, help="HLDA number of loops (default: 15000)")
    parser.add_argument("--burn_in", type=int, default=5000, help="HLDA burn-in iterations (default: 5000)")  
    parser.add_argument("--thin", type=int, default=40, help="HLDA thinning interval (default: 40)")
    args = parser.parse_args()

    # Update HLDA parameters with CLI arguments
    HLDA_PARAMS.update({
        'n_loops': args.n_loops,
        'burn_in': args.burn_in,
        'thin': args.thin
    })

    print(f"🚀 Starting model evaluation...")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    print(f"Extra topics: {args.n_extra_topics}")
    print(f"HLDA parameters: {args.n_loops} loops, {args.burn_in} burn-in, {args.thin} thin")
    print()

    output_dir = Path(args.output_dir)

    # Handle glioma dataset file structure
    if args.dataset == "glioma":
        import os
        if not os.path.exists(args.counts_csv):
            args.counts_csv = str(Path("data/glioma/glioma_counts_train.csv"))
        if not os.path.exists(args.test_csv):
            args.test_csv = str(Path("data/glioma/glioma_counts_test.csv"))

    # Load count matrix
    counts_df = pd.read_csv(args.counts_csv, index_col=0)
    
    # Load test count matrix
    test_df = pd.read_csv(args.test_csv, index_col=0)

    # Load model outputs
    model_names = ["HLDA", "LDA", "NMF"]
    model_files = {
        m: {
            "beta": output_dir / m / f"{m.upper()}_beta.csv",
            "theta": output_dir / m / f"{m.upper()}_theta.csv"
        }
        for m in model_names
    }
    models = {}
    for m in model_names:
        if model_files[m]["beta"].exists() and model_files[m]["theta"].exists():
            beta = pd.read_csv(model_files[m]["beta"], index_col=0)
            theta = pd.read_csv(model_files[m]["theta"], index_col=0)
            models[m] = {"beta": beta, "theta": theta}
    
    # Load identity topics from config file
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    if args.dataset not in config:
        raise ValueError(f"Dataset '{args.dataset}' not found in config file {args.config_file}")
    identity_topics = config[args.dataset]['identities']
    
    # Parse identity topics and extra topics from command line arguments
    extra_topics = [f"V{i+1}" for i in range(args.n_extra_topics)]
    
    # Prepare topics for each model (match and label appropriately)
    topic_mappings = {}
    for m, d in models.items():
        beta_renamed, theta_renamed, topic_mapping = prepare_model_topics(
            m, d["beta"], d["theta"], counts_df, identity_topics, args.n_extra_topics
        )
        # Update the model data with renamed matrices
        d["beta"] = beta_renamed
        d["theta"] = theta_renamed
        topic_mappings[m] = topic_mapping

    # --- Cache for test theta DataFrames ---
    test_theta_dfs = {}

    # Helper to get or cache test theta
    def get_test_theta_cached(model_key, test_df, beta, output_dir):
        if model_key in test_theta_dfs:
            return test_theta_dfs[model_key]
        theta = get_or_estimate_test_theta(test_df, beta, model_key, output_dir)
        test_theta_dfs[model_key] = theta
        return theta

   # 4) Geweke histograms for HLDA (A and D chains)
    print("4) Generating Geweke histograms for HLDA...")
    if "HLDA" in models:
       hlda_plots_dir = output_dir / "HLDA" / "plots"
       hlda_plots_dir.mkdir(parents=True, exist_ok=True)
       beta = models["HLDA"]["beta"]
       theta = models["HLDA"]["theta"]
       n_genes, n_topics = beta.shape
       n_cells = theta.shape[0]
       # Calculate n_save based on actual parameters used in HLDA run
       n_loops = HLDA_PARAMS['n_loops']
       burn_in = HLDA_PARAMS['burn_in']
       thin = HLDA_PARAMS['thin']
       n_save = (n_loops - burn_in) // thin
       sample_root = output_dir / "HLDA" / "samples"
       if not sample_root.exists():
           sample_root = output_dir / "HLDA"
       try:
           plot_geweke_histograms(
               sample_root=sample_root,
               keys=["A", "D"],
               out_dir=hlda_plots_dir,
               n_save=n_save,
               n_cells=n_cells,
               n_genes=n_genes,
               n_topics=n_topics
           )
       except Exception as e:
           pass

   # 6) SSE evaluation with custom topic order on held-out test set
    print("6) Running SSE evaluation on test set...")
   
   # Use the provided test_df (no need to create train/test split)
   # Extract full cell type identities from cell names using the global function
    test_identities = pd.Series([extract_cell_identity(i) for i in test_df.index], index=test_df.index)
   
    all_sse_results = []
    for m, d in models.items():
       print(f"   - Running SSE for {m} model...")
       model_plots_dir = output_dir / m / "plots"
       model_plots_dir.mkdir(parents=True, exist_ok=True)
       beta = d["beta"]
       
       # Use cached test theta and write to disk (get_or_estimate_test_theta always writes)
       theta_test = get_test_theta_cached(m, test_df, beta, output_dir)
       # (No need to write again, function already writes)
       
       sse_df = incremental_sse_custom(
           test_df,
           beta,
           test_identities,
           extra_topics,
           theta_out=output_dir / "model_comparison" / f"{m}_test_theta_nnls.csv"
       )
       sse_df.to_csv(model_plots_dir / f"{m}_test_sse.csv", index=False)
       
       # Add model column and collect for summary
       sse_df['model'] = m
       all_sse_results.append(sse_df)
   
   # Combine all SSE results
    if all_sse_results:
       combined_sse_df = pd.concat(all_sse_results, ignore_index=True)
       combined_sse_df.to_csv(output_dir / "sse_summary.csv", index=False)
    else:
       combined_sse_df = pd.DataFrame()

   # 7) Extract top genes per topic for each model
    print("7) Extracting top genes per topic...")
    for m, d in models.items():
       model_plots_dir = output_dir / m / "plots"
       model_plots_dir.mkdir(parents=True, exist_ok=True)
       beta = d["beta"]
       
       top_genes_df = extract_top_genes_per_topic(beta, n_top_genes=10)
       top_genes_df.to_csv(model_plots_dir / f"{m}_top_genes_per_topic.csv")

   # 8) Plot true vs estimated similarity
    print("8) Generating true vs estimated similarity plots...")
    true_beta = create_true_beta_from_counts(counts_df, identity_topics)
    for m, d in models.items():
        model_plots_dir = output_dir / m / "plots"
        model_plots_dir.mkdir(parents=True, exist_ok=True)
        beta = d["beta"]
        out_png = model_plots_dir / f"{m}_true_vs_estimated_similarity.png"
        # Print first few index values for debugging
        print(f"      [DEBUG] First 5 indices of true_beta: {list(true_beta.index[:5])}")
        print(f"      [DEBUG] First 5 indices of estimated_beta: {list(beta.index[:5])}")
        plot_true_vs_estimated_similarity(true_beta, beta, m, out_png)

   # 9) Plot true identity self-similarity
    print("9) Generating true identity self-similarity plot...")
    common_plots_dir = output_dir / "model_comparison"
    common_plots_dir.mkdir(parents=True, exist_ok=True)
    out_png = common_plots_dir / "true_identity_self_similarity.png"
    plot_true_self_similarity(true_beta, out_png)

   # Memory cleanup 
    gc.collect()

   # 10) Plot UMAP with Scanpy clustering (train data)
    print("10) Generating UMAP plots for train data...")
    activity_topics = [f"V{i+1}" for i in range(args.n_extra_topics)]
    model_plots_dir = output_dir / "UMAP"
    model_plots_dir.mkdir(parents=True, exist_ok=True)

    # Compute UMAP once for train data
    cell_identities = pd.Series([extract_cell_identity(i) for i in counts_df.index], index=counts_df.index)
    import scanpy as sc
    import anndata as ad
    adata = ad.AnnData(X=counts_df.values.astype(np.float32),  # Use float32 instead of float64
                      obs=pd.DataFrame(index=counts_df.index),
                      var=pd.DataFrame(index=counts_df.columns))
    adata.obs['cell_type'] = cell_identities.values
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_pcs = min(50, min(adata.n_vars, adata.n_obs) - 1)
    sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs)
    n_neighbors = min(10, adata.n_obs - 1)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=min(30, n_pcs))
    sc.tl.umap(adata, random_state=42)
    sc.tl.leiden(adata, resolution=0.5)
    sc.tl.umap(adata, random_state=42)

    # Add theta usage for each model
    for model_key in ["HLDA", "LDA", "NMF"]:
        if model_key in models:
            theta = models[model_key]["theta"]
            # Only use activity topics that exist in theta
            activity_cols = [col for col in activity_topics if col in theta.columns]
            if activity_cols:
                # Simple sum of activity topics, ensuring index alignment
                theta_usage = theta[activity_cols].sum(axis=1)
                # Ensure the theta_usage has the same index as adata.obs
                if theta_usage.index.equals(adata.obs.index):
                    adata.obs[f'{model_key}_theta_usage'] = theta_usage.values
                else:
                    # Reindex if needed (should be rare for train data)
                    adata.obs[f'{model_key}_theta_usage'] = theta_usage.reindex(adata.obs.index).fillna(0).values
                # Debug output
                print(f"      {model_key} train theta usage: {activity_cols} -> range [{theta_usage.min():.3f}, {theta_usage.max():.3f}], mean {theta_usage.mean():.3f}")
            else:
                adata.obs[f'{model_key}_theta_usage'] = 0
                print(f"      {model_key} train theta usage: no activity topics found")

    # Plot UMAPs
    sc.pl.umap(adata, color='leiden', show=False, title='Scanpy Clusters', size=15, save=None)
    plt.savefig(model_plots_dir / 'umap_leiden.png', dpi=200, bbox_inches='tight')
    plt.close()
    sc.pl.umap(adata, color='cell_type', show=False, title='Cell Type', size=15, save=None)
    plt.savefig(model_plots_dir / 'umap_cell_type.png', dpi=200, bbox_inches='tight')
    plt.close()
    for model_key in ["HLDA", "LDA", "NMF"]:
        if f'{model_key}_theta_usage' in adata.obs:
            sc.pl.umap(adata, color=f'{model_key}_theta_usage', show=False, title=f'{model_key} Activity Usage', size=15, save=None)
            plt.savefig(model_plots_dir / f'umap_{model_key.lower()}_theta_usage.png', dpi=200, bbox_inches='tight')
            plt.close()
    
    # Memory cleanup before combined UMAP
    gc.collect()
    
    print("   - Creating combined theta usage UMAP for train data...")
    # Create combined theta usage UMAP (train data)
    train_theta_dfs = {}
    for model_key in ["HLDA", "LDA", "NMF"]:
        if model_key in models:
            train_theta_dfs[model_key] = models[model_key]["theta"]
    
    if train_theta_dfs:
        plot_combined_theta_usage_umaps(counts_df, cell_identities, train_theta_dfs, 
                                       activity_topics, output_dir / 'model_comparison', suffix="_train")
    
    # Memory cleanup after combined UMAP
    del train_theta_dfs
    del adata  # Clean up main adata object
    gc.collect()

    print("   - Processing test data for UMAP...")
    # Repeat for test data
    test_identities = pd.Series([extract_cell_identity(i) for i in test_df.index], index=test_df.index)
    adata_test = ad.AnnData(X=test_df.values.astype(np.float32),  # Use float32 for memory efficiency
                            obs=pd.DataFrame(index=test_df.index),
                            var=pd.DataFrame(index=test_df.columns))
    adata_test.obs['cell_type'] = test_identities.values
    sc.pp.normalize_total(adata_test, target_sum=1e4)
    sc.pp.log1p(adata_test)
    n_pcs = min(50, min(adata_test.n_vars, adata_test.n_obs) - 1)
    sc.tl.pca(adata_test, svd_solver='arpack', n_comps=n_pcs)
    n_neighbors = min(10, adata_test.n_obs - 1)
    sc.pp.neighbors(adata_test, n_neighbors=n_neighbors, n_pcs=min(30, n_pcs))
    sc.tl.umap(adata_test, random_state=42)
    sc.tl.leiden(adata_test, resolution=0.5)
    for model_key in ["HLDA", "LDA", "NMF"]:
        if model_key in models:
            beta = models[model_key]["beta"]
            # Use cached test theta for consistency
            theta_test = get_test_theta_cached(model_key, test_df, beta, output_dir)
            activity_cols = [col for col in activity_topics if col in theta_test.columns]
            if activity_cols:
                # Simple sum of activity topics, ensuring index alignment
                theta_usage = theta_test[activity_cols].sum(axis=1)
                # Ensure the theta_usage has the same index as adata_test.obs
                if theta_usage.index.equals(adata_test.obs.index):
                    adata_test.obs[f'{model_key}_theta_usage'] = theta_usage.values
                else:
                    # Reindex if needed (should be rare for test data)
                    adata_test.obs[f'{model_key}_theta_usage'] = theta_usage.reindex(adata_test.obs.index).fillna(0).values
                # Debug output
                print(f"      {model_key} test theta usage: {activity_cols} -> range [{theta_usage.min():.3f}, {theta_usage.max():.3f}], mean {theta_usage.mean():.3f}")
            else:
                adata_test.obs[f'{model_key}_theta_usage'] = 0
                print(f"      {model_key} test theta usage: no activity topics found")
    sc.pl.umap(adata_test, color='leiden', show=False, title='Scanpy Clusters', size=15, save=None)
    plt.savefig(model_plots_dir / 'umap_leiden_test.png', dpi=200, bbox_inches='tight')
    plt.close()
    sc.pl.umap(adata_test, color='cell_type', show=False, title='Cell Type', size=15, save=None)
    plt.savefig(model_plots_dir / 'umap_cell_type_test.png', dpi=200, bbox_inches='tight')
    plt.close()
    for model_key in ["HLDA", "LDA", "NMF"]:
        if f'{model_key}_theta_usage' in adata_test.obs:
            sc.pl.umap(adata_test, color=f'{model_key}_theta_usage', show=False, title=f'{model_key} Activity Usage', size=15, save=None)
            plt.savefig(model_plots_dir / f'umap_{model_key.lower()}_theta_usage_test.png', dpi=200, bbox_inches='tight')
            plt.close()
    
    print("   - Creating combined theta usage UMAP for test data...")
    # Create combined theta usage UMAP (test data)
    test_theta_dfs_combined = {}
    for m, d in models.items():
        beta = d["beta"]
        theta_test_df = get_test_theta_cached(m, test_df, beta, output_dir)
        test_theta_dfs_combined[m] = theta_test_df
    
    if test_theta_dfs_combined:
        plot_combined_theta_usage_umaps(test_df, test_identities, test_theta_dfs_combined, 
                                       activity_topics, output_dir / 'model_comparison', suffix="_test")
    
    # Memory cleanup
    del test_theta_dfs_combined
    del adata_test
    gc.collect()

    # 11) Plot UMAP with Scanpy clustering (test data)
    print("11) Generating individual UMAP plots for test data by model...")
    test_identities = pd.Series([extract_cell_identity(i) for i in test_df.index], index=test_df.index)
    
    for m, d in models.items():
        print(f"   - Generating UMAP for {m} test data...")
        model_plots_dir = output_dir / m / "plots"
        model_plots_dir.mkdir(parents=True, exist_ok=True)
        beta = d["beta"]
        
        # Use cached test theta
        theta_test_df = get_test_theta_cached(m, test_df, beta, output_dir)
        
        # New Scanpy UMAP clustering for test data
        plot_umap_scanpy_clustering(test_df, test_identities, f"{m}_test", theta_test_df, activity_topics, model_plots_dir)

    print('generating theta heatmaps')
    
    # Clean up any old meta_cells_temp.csv file from previous runs
    old_meta_file = output_dir / 'meta_cells_temp.csv'
    if old_meta_file.exists():
        try:
            old_meta_file.unlink()
            print("  Cleaned up old meta_cells_temp.csv file")
        except:
            pass
    
    for m, d in models.items():
        print(f"  Creating theta heatmap for {m}...")
        
        model_plots_dir = output_dir / m / "plots"
        model_plots_dir.mkdir(parents=True, exist_ok=True)
        out_png = model_plots_dir / f"{m}_theta_heatmap.png"

        # Extract cell identities from theta index (handle both real data and simulation data)
        cell_identities_from_theta = pd.Series([extract_cell_identity(i) for i in d['theta'].index], index=d['theta'].index)

        # Create meta-cells for this specific model in its own directory
        model_temp_dir = output_dir / m / "temp"
        meta_file = create_meta_cells_chunked(
            theta=d['theta'], 
            cell_identities=cell_identities_from_theta, 
            cells_per_group=3, 
            temp_dir=model_temp_dir,
            model_name=m
        )
        
        # Read the model-specific meta-cells file
        meta_theta = pd.read_csv(meta_file)
        # Use the 'cell_type' column for cell_identities if present
        if 'cell_type' in meta_theta.columns:
            cell_identities = meta_theta['cell_type']
        else:
            cell_identities = None
        plot_theta_heatmap(
            theta=meta_theta, 
            cell_identities=cell_identities, 
            model_name=m, 
            out_png=out_png, 
            identity_topics=identity_topics, 
            cells_per_group=3,  # Match the cells_per_group used in create_meta_cells_chunked
            use_consistent_ordering=False, 
            reference_ordering=None
        )
    
    # Stack all theta heatmaps into a combined image
    print("  Stacking theta heatmaps...")
    stack_theta_heatmaps(output_dir, model_names=list(models.keys()))
    
    # # 13) Plot SSE heatmap
    # print("13) Generating SSE heatmap...")
    # if not combined_sse_df.empty:
    #     plot_combined_identity_sse_heatmap(combined_sse_df, output_dir / 'plots')
    # # 14) Plot cumulative SSE lineplot
    # print("14) Generating cumulative SSE lineplot...")
    # activity_topics = [f"V{i+1}" for i in range(args.n_extra_topics)]
    # plot_cumulative_sse_lineplot(combined_sse_df, identity_topics, activity_topics, output_dir / 'model_comparison')
    
    # # Final memory cleanup
    # gc.collect()
    
    # print("✅ Evaluation completed successfully!")
    # print(f"All outputs saved to: {output_dir}")
    

if __name__ == "__main__":
    main() 


