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
warnings.filterwarnings('ignore')

# Import default parameters from fit_hlda
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from fit_hlda import get_default_parameters

# Get default HLDA parameters
HLDA_PARAMS = get_default_parameters()

# Override with actual parameters used
HLDA_PARAMS.update({
    'n_loops': 15000,
    'burn_in': 5000,
    'thin': 40
})

# --- Utility functions ---
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def compute_pca_projection(counts_df: pd.DataFrame,
                           n_components: int = 6,
                           random_state: int = 0):
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    pca.fit(counts_df.values.astype(np.float32))
    eigvecs = pca.components_
    cell_props = counts_df.div(counts_df.sum(axis=1), axis=0).values.astype(np.float32)
    cell_proj  = cell_props @ eigvecs.T
    return eigvecs, cell_proj

def plot_pca_pair(pc_x: int, pc_y: int,
                  beta_proj_est: np.ndarray, est_names: list, est_label_mask: np.ndarray,
                  beta_proj_true: np.ndarray, true_names: list,
                  cell_proj: np.ndarray, mixture_mask: np.ndarray,
                  model_name: str, out_png: Path, cell_identities: Optional[pd.Series] = None):
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Color cells by cell type if identities are provided
    if cell_identities is not None:
        # Create color mapping for cell types
        unique_cell_types = sorted(cell_identities.unique())
        # Use a color palette that works well for categorical data
        colors = sns.color_palette("husl", n_colors=len(unique_cell_types))
        color_map = dict(zip(unique_cell_types, colors))
        
        # Plot cells colored by cell type
        for cell_type in unique_cell_types:
            mask = cell_identities == cell_type
            if mask.any():
                ax.scatter(cell_proj[mask, pc_x], cell_proj[mask, pc_y],
                          s=12, c=[color_map[cell_type]], alpha=0.6, marker="o", 
                          label=f"{cell_type}")
    else:
        # Fallback to original gray/orange coloring
        ax.scatter(cell_proj[~mixture_mask, pc_x], cell_proj[~mixture_mask, pc_y],
                   s=12, c="lightgray", alpha=0.4, marker="o", label="cells (pure)")
        ax.scatter(cell_proj[mixture_mask, pc_x], cell_proj[mixture_mask, pc_y],
                   s=16, c="tab:orange", alpha=0.8, marker="^", label="cells w/ activity")
    
    # topics layers
    ax.scatter(beta_proj_est[[pc_x, pc_y], :][0], beta_proj_est[[pc_x, pc_y], :][1],
               s=130, marker="*", c="tab:red", label=f"{model_name} β̂")
    for i, t in enumerate(est_names):
        if est_label_mask[i]:
            ax.text(beta_proj_est[pc_x, i], beta_proj_est[pc_y, i], t, fontsize=8, ha="left", va="bottom")
    ax.scatter(beta_proj_true[[pc_x, pc_y], :][0], beta_proj_true[[pc_x, pc_y], :][1],
               s=100, marker="^", c="tab:green", label="expression avg")
    for i, t in enumerate(true_names):
        ax.text(beta_proj_true[pc_x, i], beta_proj_true[pc_y, i], t, fontsize=8, ha="right", va="top")
    ax.set_xlabel(f"PC{pc_x+1}")
    ax.set_ylabel(f"PC{pc_y+1}")
    ax.set_title(f"{model_name}: PC{pc_x+1} vs PC{pc_y+1} (Estimated vs True Cell Identity)")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ensure_dir(out_png.parent)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

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
            print(f"WARNING: {e}  Skipping '{key}'.")
            continue
        z_vals = []
        for idx in np.ndindex(chain.shape[1:]):
            ts = chain[(Ellipsis,) + idx]
            z = geweke_z(ts)
            if np.isfinite(z):
                z_vals.append(z)
        if not z_vals:
            print(f"WARNING: no finite Geweke scores for key '{key}'. Skipping plot.")
            continue
        plt.figure(figsize=(6, 4))
        plt.hist(z_vals, bins=50, edgecolor="black")
        plt.xlabel("Geweke z-score")
        plt.ylabel("Count")
        plt.title(f"Chain \"{key}\"")
        plt.tight_layout()
        plt.savefig(out_dir / f"geweke_{key}.png", dpi=300)
        plt.close()

def structure_plot_py(
    theta: pd.DataFrame,
    identities: pd.Series | list[str],
    out_png: Optional[str | Path] = None,
    topics: Optional[list[str]] = None,
    max_cells: int = 500,
    gap: int = 1,
    colors: Optional[dict] = None,
    random_state: int = 42,
    figsize: tuple = (14, 4),
    title: Optional[str] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
):
    """
    Python version of structure_plot: stacked bar plot of topic proportions, ordered by 1D PCA within group, with group gaps.
    """
    # Ensure unique indices for theta and ids
    theta = theta.copy()
    theta.index = pd.RangeIndex(len(theta))
    if isinstance(identities, list):
        ids = pd.Series(identities, index=theta.index)
    else:
        ids = identities.copy()
        ids.index = theta.index
    ids = ids.astype(str)
    if ids.str.contains("_").any():
        ids = ids.str.split("_").str[0]
    unique_ids = sorted(ids.unique())

    # Topics to plot
    if topics is None:
        topics = list(theta.columns)
    else:
        topics = [t for t in topics if t in theta.columns]
    k = len(topics)

    # Color palette
    if colors is None:
        palette = sns.color_palette("colorblind", n_colors=k)
        colors = {topic: palette[i] for i, topic in enumerate(topics)}

    n0 = len(theta)
    if n0 > max_cells:
        np.random.seed(random_state)
        sampled_idx = np.random.choice(theta.index, max_cells, replace=False)
        theta = theta.loc[sampled_idx]
        ids = ids.loc[sampled_idx]

    # For each group, order by 1D PCA and concatenate
    ordered_indices = []
    gap_counter = 0
    for group in unique_ids:
        mask = ids == group
        group_theta = theta.loc[mask, topics]
        if len(group_theta) > max_cells:
            group_theta = group_theta.sample(n=max_cells, random_state=random_state)
        if len(group_theta) == 0:
            continue
        if len(group_theta) > 1:
            pca = PCA(n_components=1, random_state=random_state)
            y = pca.fit_transform(group_theta.values)
            order = np.argsort(y[:, 0])
            group_idx = group_theta.index[order]
        else:
            group_idx = group_theta.index
        ordered_indices.extend(group_idx)
        # Add gap (row of zeros) between groups except last
        if gap > 0 and group != unique_ids[-1]:
            gap_idx = [f"__gap___{group}_{i}_{gap_counter}" for i in range(gap)]
            gap_counter += 1
            ordered_indices.extend(gap_idx)
    # Build the matrix for plotting
    plot_theta = pd.DataFrame(
        np.zeros((len(ordered_indices), len(topics)), dtype=float),
        index=pd.Index(ordered_indices),
        columns=pd.Index(topics)
    )
    for idx in theta.index:
        if idx in plot_theta.index:
            plot_theta.loc[idx] = theta.loc[idx, topics]
    
    # Use provided fig/ax or create new ones
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    bottom = np.zeros(len(plot_theta))
    for topic in topics:
        values = plot_theta[topic].to_numpy(dtype=float)
        ax.bar(
            range(len(plot_theta)),
            values,
            bottom=bottom,
            color=colors[topic],
            width=1.0,
            label=topic if topic not in ax.get_legend_handles_labels()[1] else "_nolegend_",
            linewidth=0
        )
        bottom = bottom + values
    # Add group labels at the center of each group
    group_centers = []
    group_labels = []
    for group in unique_ids:
        mask = [i for i, idx in enumerate(plot_theta.index) if not str(idx).startswith("__gap__") and ids.get(idx, None) == group]
        if mask:
            center = (mask[0] + mask[-1]) / 2
            group_centers.append(center)
            group_labels.append(group)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, rotation=45, ha='right')
    ax.set_xlim(-0.5, len(plot_theta) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Topic Proportion")
    ax.set_xlabel("Cell Groups")
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10, title="Topics")
    ax.set_title(title if title is not None else "Structure Plot: Topic Membership by Cell")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    
    # Only save if this is a standalone plot (not part of subplot) and out_png is provided
    if (fig is None or ax is None) and out_png is not None:
        plt.tight_layout(rect=(0, 0, 0.85, 1))
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close(fig)



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
        else:
            print(f"[WARNING] No cells found for identity '{identity}'")
    
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
    
    sim_df = pd.DataFrame(
        similarity,
        index=true_beta.columns,  # True identities
        columns=estimated_beta.columns  # Estimated topics
    )
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_df, annot=True, fmt=".3f", cmap="viridis", 
                cbar_kws={'label': 'Cosine Similarity'})
    plt.title(f"True vs Estimated Similarity ({model_name})\nTrue Identities vs Estimated Topics")
    plt.xlabel("Estimated Topics")
    plt.ylabel("True Cell Type Identities")
    plt.tight_layout()
    
    # Save the plot
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save the similarity matrix as CSV
    sim_csv = out_png.with_suffix('.csv')
    sim_df.to_csv(sim_csv)

def plot_true_self_similarity(true_beta: pd.DataFrame, out_png: Path):
    true_norms = np.linalg.norm(true_beta.values, axis=0, keepdims=True)
    true_normalized = true_beta.values / (true_norms + 1e-12)
    similarity = true_normalized.T @ true_normalized
    sim_df = pd.DataFrame(
        similarity,
        index=true_beta.columns,
        columns=true_beta.columns
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_df, annot=True, fmt=".3f", cmap="viridis", cbar_kws={'label': 'Cosine Similarity'})
    plt.title("True Identity Self-Similarity\nAveraged Gene Expression Profiles")
    plt.xlabel("True Cell Type Identities")
    plt.ylabel("True Cell Type Identities")
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()

def compute_loglikelihood(X: np.ndarray, beta: np.ndarray, theta: np.ndarray) -> float:

    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(beta, pd.DataFrame):
        beta = beta.values
    if isinstance(theta, pd.DataFrame):
        theta = theta.values
    
    X_prop = X / (X.sum(axis=1, keepdims=True) + 1e-12)
    recon = theta @ beta.T
    recon = recon / (recon.sum(axis=1, keepdims=True) + 1e-12)
    
    eps = 1e-12
    recon = np.clip(recon, eps, 1 - eps)
    log_likelihood = np.sum(X_prop * np.log(recon))
    
    return log_likelihood

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
        print(f"  Scanpy not available for {model_name}, skipping UMAP plots")
        return
    
    print(f"    Creating UMAP for {model_name} with {len(counts_df)} cells and {len(counts_df.columns)} genes")
    
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
    print(f"    Normalizing data...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Compute PCA with fewer components to save memory
    print(f"    Computing PCA...")
    n_pcs = min(50, min(adata.n_vars, adata.n_obs) - 1)  # Use fewer PCs for large datasets
    sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs)
    
    # Compute neighbors with fewer PCs to save memory
    print(f"    Computing neighbors...")
    n_neighbors = min(10, adata.n_obs - 1)  # Ensure n_neighbors < n_cells
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=min(30, n_pcs))
    
    # Compute UMAP
    print(f"    Computing UMAP...")
    sc.tl.umap(adata, random_state=42)
    
    # Clustering
    print(f"    Computing clustering...")
    sc.tl.leiden(adata, resolution=0.5)
    
    # Create plots with smaller figure size to save memory
    print(f"    Creating plots...")
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
    
    print(f"    Completed UMAP for {model_name}")
    return None  # Don't return adata to save memory



def get_num_activity_topics(topic_str, identity):
    # Count how many V's are in the topic string (excluding the identity itself)
    return len(re.findall(r'V\d+', topic_str))

def plot_combined_identity_sse_heatmap(sse_df, output_dir):
    
    # Only keep rows where the topic string starts with the identity
    sse_df = sse_df[sse_df.apply(lambda row: row['topics'].startswith(row['identity']), axis=1)]
    
    # Create row label for identity and model
    sse_df['identity_model'] = sse_df['identity'] + ' | ' + sse_df['model']
    
    # Extract the activity part from the topic string to use as column names
    def extract_activity_part(topic_str, identity):
        # Remove the identity part and "_only" suffix
        activity_part = topic_str.replace(identity, "").replace("_only", "")
        # Clean up any leading/trailing + signs
        activity_part = activity_part.strip("+")
        if activity_part == "":
            return "identity_only"
        else:
            return "+" + activity_part
    
    sse_df['activity_combo'] = sse_df.apply(lambda row: extract_activity_part(row['topics'], row['identity']), axis=1)
    
    # Sort by identity then model for grouping
    sse_df = sse_df.sort_values(['identity', 'model'])
    
    # Create pivot table with activity combinations as columns
    pivot = sse_df.pivot_table(index='identity_model', columns='activity_combo', values='SSE', aggfunc='first')
    
    # Reorder columns to have a logical progression
    column_order = ['identity_only']
    # Add +V1, +V2, +V3, etc. in order
    for i in range(1, 10):  # Support up to 9 activity topics
        v_col = f'+V{i}'
        if v_col in pivot.columns:
            column_order.append(v_col)
    # Add combinations like +V1+V2, +V1+V3, etc.
    for col in sorted(pivot.columns):
        if col not in column_order and col.startswith('+V'):
            column_order.append(col)
    
    # Reorder the columns
    pivot = pivot[column_order]
    
    # Plot
    plt.figure(figsize=(12, max(8, 0.4 * len(pivot))))
    ax = sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis_r', linewidths=0.5, linecolor='white')
    plt.title('SSE by Cell Identity, Model, and Activity Topic Combinations')
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
    plt.savefig(output_dir / 'sse_heatmap_combined.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_theta_heatmap(theta: pd.DataFrame, cell_identities: pd.Series, model_name: str, out_png: Path, identity_topics: list):
    """
    Plot a heatmap of theta (topic usage) with topics on y-axis and cells on x-axis, grouped by cell type and ordered by max topic usage.
    Memory-optimized version.
    """
    print(f"    Creating theta heatmap for {model_name} with {len(theta)} cells...")
    
    # Assign each cell to its max-usage identity topic (for ordering)
    max_topic = theta[identity_topics].idxmax(axis=1)
    # Create a DataFrame for sorting
    sort_df = pd.DataFrame({
        'cell_type': cell_identities,
        'max_topic': max_topic
    }, index=theta.index)
    # Sort by cell type, then by max topic usage, then by value within topic
    sort_order = []
    for ct in sorted(cell_identities.unique()):
        ct_mask = sort_df['cell_type'] == ct
        ct_cells = sort_df[ct_mask].index
        # Within cell type, order by max topic, then by value in that topic
        for topic in identity_topics:
            topic_cells = sort_df.loc[ct_cells][sort_df.loc[ct_cells, 'max_topic'] == topic].index
            if len(topic_cells) > 0:
                # Order by theta value in that topic
                topic_cells_sorted = theta.loc[topic_cells, topic].sort_values(ascending=False).index
                sort_order.extend(topic_cells_sorted)
    # Reorder theta and cell_identities
    theta_sorted = theta.loc[sort_order]
    cell_identities_sorted = cell_identities.loc[sort_order]
    # Use raw theta values (0-1)
    theta_norm = theta_sorted[identity_topics]
    
    # Limit figure width for large datasets to save memory
    max_width = 20  # Maximum width in inches
    fig_width = min(max_width, max(8, len(theta_norm)//100))
    fig_height = 0.5*len(identity_topics) + 2
    
    # Plot
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(theta_norm.T, aspect='auto', cmap='bwr', vmin=0, vmax=1)
    # Y-axis: topics
    ax.set_yticks(range(len(identity_topics)))
    ax.set_yticklabels(identity_topics, fontsize=10)
    # X-axis: no labels (too many cells), but add cell type color bar
    ax.set_xticks([])
    
    # Cell type color bar - more memory efficient
    unique_types = cell_identities_sorted.unique()
    type_colors = dict(zip(unique_types, sns.color_palette('tab20', n_colors=len(unique_types))))
    
    # Create color bar more efficiently for large datasets
    if len(cell_identities_sorted) > 1000:
        # For large datasets, sample colors to avoid too many rectangles
        sample_step = max(1, len(cell_identities_sorted) // 1000)
        for i in range(0, len(cell_identities_sorted), sample_step):
            color = type_colors[cell_identities_sorted.iloc[i]]
            ax.add_patch(mpatches.Rectangle((i-0.5, -1.5), sample_step, 0.5, color=color, linewidth=0))
    else:
        # For smaller datasets, create individual rectangles
        cell_type_colors = cell_identities_sorted.map(type_colors)
        for i, color in enumerate(cell_type_colors):
            ax.add_patch(mpatches.Rectangle((i-0.5, -1.5), 1, 0.5, color=color, linewidth=0))
    
    # Legend for cell types
    legend_patches = [mpatches.Patch(color=type_colors[ct], label=ct) for ct in unique_types]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9, title='Cell type')
    # Colorbar for expression
    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label('Topic usage (proportion)', fontsize=10)
    ax.set_title(f"{model_name} topic usage heatmap", fontsize=12)
    plt.tight_layout()
    ensure_dir(out_png.parent)
    print(f"    Saving theta heatmap to: {out_png}")
    plt.savefig(out_png, dpi=200, bbox_inches='tight')  # Reduced DPI
    plt.close()
    
    # Memory cleanup
    del theta_sorted, cell_identities_sorted, theta_norm
    gc.collect()

def plot_cumulative_sse_lineplot(sse_summary_df, identity_topics, activity_topics, output_dir):
    """
    Plot cumulative SSE as topics are added one by one. X-axis: topic names; Y-axis: SSE; one line per model.
    """

    # Determine topic order: identities first, then activities
    topic_order = identity_topics + activity_topics
    # For each model, get SSE for cumulative topic sets
    models = sse_summary_df['model'].unique()
    plt.figure(figsize=(max(8, len(topic_order)*1.2), 6))
    for model in models:
        model_df = sse_summary_df[sse_summary_df['model'] == model]
        sse_vals = []
        x_labels = []
        for i in range(1, len(topic_order)+1):
            topics_now = topic_order[:i]
            # Find the row in the SSE summary that matches this combo (order-insensitive)
            # The topic label is either 'identity_only' or '+'.join(topics_now)
            if len(topics_now) == 1:
                topic_label = f"{topics_now[0]}_only"
            else:
                topic_label = '+'.join(topics_now)
            row = model_df[model_df['topics'] == topic_label]
            if not row.empty:
                sse_vals.append(row['SSE'].values[0])
            else:
                sse_vals.append(float('nan'))
            x_labels.append(topics_now[-1])
        plt.plot(x_labels, sse_vals, marker='o', label=model)
    plt.xlabel('Topic added')
    plt.ylabel('Cumulative SSE (test set)')
    plt.title('Cumulative SSE as topics are added')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model')
    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(output_dir) / 'cumulative_sse_by_topic.png', dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate topic models (HLDA, LDA, NMF) and run metrics/plots.")
    parser.add_argument("--counts_csv", type=str, required=True, help="Path to filtered count matrix CSV")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test count matrix CSV")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory with model outputs and for saving plots/metrics")
    parser.add_argument("--n_extra_topics", type=int, required=True, help="Number of extra topics (e.g. 3 for V1,V2,V3)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (must match a key in the config file)")
    parser.add_argument("--config_file", type=str, default="dataset_identities.yaml", help="Path to dataset identity config YAML file")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

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
    print(f"Models loaded: {list(models.keys())}")
    
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
    
    # --- Structure plot code commented out ---
    #    print("Generating structure plots...")
    #    all_topics = set()
    #    for m, d in models.items():
    #        all_topics.update(d["theta"].columns)
    #    all_topics_sorted = sorted(all_topics)
    #    palette = sns.color_palette("colorblind", n_colors=len(all_topics_sorted))
    #    global_colors = {topic: palette[i] for i, topic in enumerate(all_topics_sorted)}
    #    for m, d in models.items():
    #        if m in ["LDA", "NMF"]:
    #            mapping_info = ", ".join([f"{orig}→{mapped}" for orig, mapped in topic_mappings[m].items()])
    #            custom_title = f"Structure Plot: Topic Membership by Cell ({m}) - Matched: {mapping_info}"
    #        else:
    #            custom_title = f"Structure Plot: Topic Membership by Cell ({m})"
    #        model_topics = d["theta"].columns
    #        model_colors = {topic: global_colors[topic] for topic in model_topics if topic in global_colors}
    #        model_plots_dir = output_dir / m / "plots"
    #        model_plots_dir.mkdir(parents=True, exist_ok=True)
    #        individual_out_png = model_plots_dir / f"{m}_structure_plot.png"
    #        structure_plot_py(d["theta"], counts_df.index.to_series(), individual_out_png, 
    #                         title=custom_title, colors=model_colors)
    #    fig, axes = plt.subplots(len(models), 1, figsize=(14, 4 * len(models)))
    #    if len(models) == 1:
    #        axes = [axes]
    #    for i, (m, d) in enumerate(models.items()):
    #        if m in ["LDA", "NMF"]:
    #            mapping_info = ", ".join([f"{orig}→{mapped}" for orig, mapped in topic_mappings[m].items()])
    #            custom_title = f"Structure Plot: Topic Membership by Cell ({m}) - Matched: {mapping_info}"
    #        else:
    #            custom_title = f"Structure Plot: Topic Membership by Cell ({m})"
    #        model_topics = d["theta"].columns
    #        model_colors = {topic: global_colors[topic] for topic in model_topics if topic in global_colors}
    #        structure_plot_py(d["theta"], counts_df.index.to_series(), None, 
    #                         title=custom_title, fig=fig, ax=axes[i], colors=model_colors)
    #    combined_plots_dir = output_dir / "plots"
    #    combined_plots_dir.mkdir(parents=True, exist_ok=True)
    #    plt.tight_layout()
    #    plt.savefig(combined_plots_dir / "combined_structure_plots.png", dpi=300, bbox_inches='tight')
    #    plt.close()

    # 3) Cosine similarity matrix heatmap of estimated beta topics (self-similarity)
    print("Generating cosine similarity heatmaps...")
    for m, d in models.items():
        model_plots_dir = output_dir / m / "plots"
        model_plots_dir.mkdir(parents=True, exist_ok=True)
        beta = d["beta"]
        # Compute cosine similarity between topics (columns)
        topic_matrix = beta.values  # shape: (n_genes, n_topics)
        sim_matrix = cosine_similarity(topic_matrix.T)  # shape: (n_topics, n_topics)
        sim_df = pd.DataFrame(sim_matrix, index=beta.columns, columns=beta.columns)
        plt.figure(figsize=(8, 6))
        sns.heatmap(sim_df, annot=True, fmt=".2f", cmap="viridis")
        plt.title(f"Cosine Similarity Between Topics ({m})")
        plt.tight_layout()
        out_png = model_plots_dir / f"{m}_beta_cosine_similarity.png"
        plt.savefig(out_png, dpi=150)
        plt.close()

    # 4) Geweke histograms for HLDA (A and D chains)
    if "HLDA" in models:
        print("Generating Geweke histograms for HLDA...")
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
        print(f"  Expected n_save: {n_save} (n_loops={n_loops}, burn_in={burn_in}, thin={thin})")
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
            print(f"  [WARN] Could not plot Geweke histograms: {e}")
            print(f"  This might be due to memmap file size mismatch. Check if the actual HLDA parameters match the expected ones.")

    # 5) PCA pair plots for each model
    # (Delete or comment out the section that calls compute_pca_projection, plot_pca_pair, etc.)

    # 6) SSE evaluation with custom topic order on held-out test set
    print("Running SSE evaluation...")
    
    # Use the provided test_df (no need to create train/test split)
    test_identities = pd.Series([i.split("_")[0] for i in test_df.index], index=test_df.index)
    
    all_sse_results = []
    for m, d in models.items():
        model_plots_dir = output_dir / m / "plots"
        model_plots_dir.mkdir(parents=True, exist_ok=True)
        beta = d["beta"]
        
        sse_df = incremental_sse_custom(
            test_df,
            beta,
            test_identities,
            extra_topics,
            theta_out=model_plots_dir / f"{m}_test_theta_nnls.csv"
        )
        sse_df.to_csv(model_plots_dir / f"{m}_test_sse.csv", index=False)
        
        # Add model column and collect for summary
        sse_df['model'] = m
        all_sse_results.append(sse_df)
    
    # Combine all SSE results
    if all_sse_results:
        combined_sse_df = pd.concat(all_sse_results, ignore_index=True)
        combined_sse_df.to_csv(output_dir / "sse_summary.csv", index=False)

    # 7) Extract top genes per topic for each model
    print("Extracting top genes per topic...")
    for m, d in models.items():
        model_plots_dir = output_dir / m / "plots"
        model_plots_dir.mkdir(parents=True, exist_ok=True)
        beta = d["beta"]
        
        top_genes_df = extract_top_genes_per_topic(beta, n_top_genes=10)
        top_genes_df.to_csv(model_plots_dir / f"{m}_top_genes_per_topic.csv")

    # 8) Plot true vs estimated similarity
    print("Computing true vs estimated similarity...")
    true_beta = create_true_beta_from_counts(counts_df, identity_topics)
    for m, d in models.items():
        model_plots_dir = output_dir / m / "plots"
        model_plots_dir.mkdir(parents=True, exist_ok=True)
        beta = d["beta"]
        out_png = model_plots_dir / f"{m}_true_vs_estimated_similarity.png"
        plot_true_vs_estimated_similarity(true_beta, beta, m, out_png)

    # 9) Plot true identity self-similarity
    print("Computing true identity self-similarity...")
    common_plots_dir = output_dir / "plots"
    common_plots_dir.mkdir(parents=True, exist_ok=True)
    out_png = common_plots_dir / "true_identity_self_similarity.png"
    plot_true_self_similarity(true_beta, out_png)

    # 10) Compute log-likelihood on both train and test datasets, and save metrics
    print("Computing log-likelihood on train and test datasets...")
    metrics = []
    for m, d in models.items():
        model_plots_dir = output_dir / m / "plots"
        model_plots_dir.mkdir(parents=True, exist_ok=True)
        beta = d["beta"]
        theta = d["theta"]
        
        # Compute log-likelihood on train data (using learned theta)
        X_train_prop = counts_df.div(counts_df.sum(axis=1), axis=0).values
        train_log_likelihood = compute_loglikelihood(X_train_prop, beta.values, theta.values)
        
        # Compute log-likelihood on test data (estimate theta for test data)
        X_test_prop = test_df.div(test_df.sum(axis=1), axis=0).values
        theta_test = estimate_theta_simplex(X_test_prop, beta.values, l1=0.002)
        test_log_likelihood = compute_loglikelihood(X_test_prop, beta.values, theta_test)
        
        metrics.append({
            "model": m,
            "train_log_likelihood": train_log_likelihood,
            "test_log_likelihood": test_log_likelihood
        })
    # Save metrics summary
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    
    # Memory cleanup after log-likelihood computation
    gc.collect()

    # 11) Plot UMAP with Scanpy clustering (train data)
    print("Generating UMAP with Scanpy clustering for train data...")
    activity_topics = [f"V{i+1}" for i in range(args.n_extra_topics)]
    
    for m, d in models.items():
        model_plots_dir = output_dir / m / "plots"
        model_plots_dir.mkdir(parents=True, exist_ok=True)
        theta = d["theta"]
        cell_identities = pd.Series([i.split("_")[0] for i in counts_df.index], index=counts_df.index)
        
        # New Scanpy UMAP clustering
        plot_umap_scanpy_clustering(counts_df, cell_identities, m, theta, activity_topics, model_plots_dir)
    
    # 12) Plot UMAP with Scanpy clustering (test data)
    print("Generating UMAP with Scanpy clustering for test data...")
    test_identities = pd.Series([i.split("_")[0] for i in test_df.index], index=test_df.index)
    
    for m, d in models.items():
        model_plots_dir = output_dir / m / "plots"
        model_plots_dir.mkdir(parents=True, exist_ok=True)
        beta = d["beta"]
        
        # Estimate theta for test data
        X_test_prop = test_df.div(test_df.sum(axis=1), axis=0).values
        theta_test = estimate_theta_simplex(X_test_prop, beta.values, l1=0.002)
        theta_test_df = pd.DataFrame(theta_test, index=test_df.index, columns=beta.columns)
        
        # New Scanpy UMAP clustering for test data
        plot_umap_scanpy_clustering(test_df, test_identities, f"{m}_test", theta_test_df, activity_topics, model_plots_dir)

    # 14) Plot SSE heatmap
    plot_combined_identity_sse_heatmap(combined_sse_df, output_dir / 'plots')

    # 15) Plot theta heatmap for train and test
    for m, d in models.items():
        print(f"Plotting theta heatmap for {m}...")
        model_plots_dir = output_dir / m / "plots"
        model_plots_dir.mkdir(parents=True, exist_ok=True)
        theta = d["theta"]
        cell_identities = pd.Series([i.split("_")[0] for i in counts_df.index], index=counts_df.index)
        out_png = model_plots_dir / f"{m}_theta_heatmap.png"
        plot_theta_heatmap(theta, cell_identities, m, out_png, identity_topics)
    for m, d in models.items():
        print(f"Plotting theta heatmap for {m} (test)...")
        model_plots_dir = output_dir / m / "plots"
        model_plots_dir.mkdir(parents=True, exist_ok=True)
        beta = d["beta"]
        X_test_prop = test_df.div(test_df.sum(axis=1), axis=0).values
        theta_test = estimate_theta_simplex(X_test_prop, beta.values, l1=0.002)
        theta_test_df = pd.DataFrame(theta_test, index=test_df.index, columns=beta.columns)
        test_identities = pd.Series([i.split("_")[0] for i in test_df.index], index=test_df.index)
        out_png = model_plots_dir / f"{m}_test_theta_heatmap.png"
        plot_theta_heatmap(theta_test_df, test_identities, f"{m} (test)", out_png, identity_topics)

    # 16) Plot cumulative SSE lineplot
    activity_topics = [f"V{i+1}" for i in range(args.n_extra_topics)]
    plot_cumulative_sse_lineplot(combined_sse_df, identity_topics, activity_topics, output_dir / 'plots')
    
    # Final memory cleanup
    gc.collect()
    
    print("Evaluation completed successfully!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 


