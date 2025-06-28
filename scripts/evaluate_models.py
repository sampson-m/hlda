import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
import cvxpy as cp
from scipy.optimize import nnls
from scipy.optimize import linear_sum_assignment
import scanpy as sc
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# --- PBMC Cell Type Definitions (matching fit_hlda.py) ---
PBMC_CELL_TYPES = [
    'T cells',
    'CD19+ B', 
    'CD56+ NK',
    'CD34+',
    'Dendritic',
    'CD14+ Monocyte'
]

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
    out_png: str | Path,
    topics: Optional[list[str]] = None,
    max_cells: int = 500,
    gap: int = 1,
    colors: Optional[dict] = None,
    random_state: int = 42,
    figsize: tuple = (14, 4),
    title: Optional[str] = None,
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
            gap_idx = [f"__gap__{group}_{i}_{gap_counter}" for i in range(gap)]
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
        prob = cp.Problem(cp.Minimize(obj), [cp.sum(th) == 1])
        prob.solve(solver=cp.OSQP, eps_abs=1e-6, verbose=False)
        Theta[i] = th.value
    return Theta

def extract_top_genes_per_topic(beta_df: pd.DataFrame, n_top_genes: int = 10) -> pd.DataFrame:
    """
    Extract the top n genes by probability for each topic in beta.
    
    Args:
        beta_df: DataFrame with genes as rows and topics as columns
        n_top_genes: Number of top genes to extract per topic (default: 10)
    
    Returns:
        DataFrame with topics as columns and top genes as rows
    """
    top_genes_dict = {}
    
    for topic in beta_df.columns:
        # Get the top n genes for this topic
        top_indices = beta_df[topic].nlargest(n_top_genes).index
        top_genes_dict[topic] = list(top_indices)
    
    # Create DataFrame with topics as columns
    max_genes = max(len(genes) for genes in top_genes_dict.values())
    result_data = {}
    
    for topic, genes in top_genes_dict.items():
        # Pad with empty strings if needed
        padded_genes = genes + [''] * (max_genes - len(genes))
        result_data[topic] = padded_genes
    
    result_df = pd.DataFrame.from_dict(result_data, orient='index').T
    
    # Add row labels for gene rank
    result_df.index = [f"Gene_{i+1}" for i in range(len(result_df))]
    
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
    """Incremental SSE with custom topic addition order: [id], [id,V1], [id,V2], [id,V1,V2]."""
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
            
        topic_steps = [[ident]]
        # Add [ident, V1], [ident, V2], [ident, V1, V2] (if present)
        if len(activity_topics) >= 1:
            topic_steps.append([ident, activity_topics[0]])
        if len(activity_topics) >= 2:
            topic_steps.append([ident, activity_topics[1]])
            topic_steps.append([ident, activity_topics[0], activity_topics[1]])
            
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
            
            results.append({
                "topics": "+".join(topics_now),
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
        
        # Match topics to identities
        topic_mapping = match_topics_to_identities(beta, true_beta, identity_topics, n_extra_topics)
        
        # Rename beta and theta columns using the mapping
        beta_renamed = beta.rename(columns=topic_mapping)
        theta_renamed = theta.rename(columns=topic_mapping)
        
        return beta_renamed, theta_renamed, topic_mapping
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def plot_true_vs_estimated_similarity(true_beta: pd.DataFrame, estimated_beta: pd.DataFrame, 
                                    model_name: str, out_png: Path):
    """
    Plot cosine similarity heatmap between true averaged gene expression profiles 
    and estimated beta topics.
    
    Args:
        true_beta: DataFrame with true averaged expression profiles (genes x identities)
        estimated_beta: DataFrame with estimated topic distributions (genes x topics)
        model_name: Name of the model for the plot title
        out_png: Output path for the plot
    """
    # Compute cosine similarity between true and estimated profiles
    # Normalize for cosine similarity
    true_norms = np.linalg.norm(true_beta.values, axis=0, keepdims=True)
    est_norms = np.linalg.norm(estimated_beta.values, axis=0, keepdims=True)
    
    true_normalized = true_beta.values / (true_norms + 1e-12)
    est_normalized = estimated_beta.values / (est_norms + 1e-12)
    
    # Compute similarity matrix: true identities (rows) x estimated topics (columns)
    similarity = true_normalized.T @ est_normalized  # shape: (n_identities, n_topics)
    
    # Create DataFrame for easier plotting
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
    """
    Plot cosine similarity heatmap between true averaged gene expression profiles 
    against themselves (self-similarity matrix).
    
    Args:
        true_beta: DataFrame with true averaged expression profiles (genes x identities)
        out_png: Output path for the plot
    """
    # Compute cosine similarity between true profiles and themselves
    # Normalize for cosine similarity
    true_norms = np.linalg.norm(true_beta.values, axis=0, keepdims=True)
    true_normalized = true_beta.values / (true_norms + 1e-12)
    
    # Compute self-similarity matrix: true identities (rows) x true identities (columns)
    similarity = true_normalized.T @ true_normalized  # shape: (n_identities, n_identities)
    
    # Create DataFrame for easier plotting
    sim_df = pd.DataFrame(
        similarity,
        index=true_beta.columns,  # True identities
        columns=true_beta.columns  # True identities
    )
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_df, annot=True, fmt=".3f", cmap="viridis", 
                cbar_kws={'label': 'Cosine Similarity'})
    plt.title("True Identity Self-Similarity\nAveraged Gene Expression Profiles")
    plt.xlabel("True Cell Type Identities")
    plt.ylabel("True Cell Type Identities")
    plt.tight_layout()
    
    # Save the plot
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save the similarity matrix as CSV
    sim_csv = out_png.with_suffix('.csv')
    sim_df.to_csv(sim_csv)
    
    # Print summary statistics
    print(f"True identity self-similarity summary:")
    print(f"  Mean similarity: {sim_df.values.mean():.3f}")
    print(f"  Min similarity: {sim_df.values.min():.3f}")
    print(f"  Max similarity: {sim_df.values.max():.3f}")

def compute_perplexity_and_loglikelihood(X_test: np.ndarray, beta: np.ndarray, theta: np.ndarray) -> tuple[float, float]:
    """
    Compute perplexity and log-likelihood on test data.
    
    Args:
        X_test: Test count matrix (cells x genes)
        beta: Topic-gene distributions (genes x topics)
        theta: Cell-topic distributions (cells x topics)
    
    Returns:
        tuple: (perplexity, log_likelihood)
    """
    # Convert to proportions
    X_prop = X_test / (X_test.sum(axis=1, keepdims=True) + 1e-12)
    
    # Compute reconstruction: theta @ beta.T
    recon = theta @ beta.T
    
    # Normalize reconstruction
    recon = recon / (recon.sum(axis=1, keepdims=True) + 1e-12)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-12
    recon = np.clip(recon, eps, 1 - eps)
    
    # Compute log-likelihood
    log_likelihood = np.sum(X_prop * np.log(recon))
    
    # Compute perplexity
    n_tokens = X_test.sum()
    perplexity = np.exp(-log_likelihood / n_tokens)
    
    return perplexity, log_likelihood

def plot_umap_theta(theta: pd.DataFrame, cell_identities: pd.Series, model_name: str, out_png: Path):
    """
    Create UMAP visualization of theta matrix colored by cell type.
    
    Args:
        theta: Cell-topic proportions (cells x topics)
        cell_identities: Cell type identities
        model_name: Name of the model
        out_png: Output path for the plot
    """
    # Fit UMAP
    try:
        umap_reducer = UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
        theta_2d = umap_reducer.fit_transform(theta.values)
    except ImportError:
        print(f"  UMAP not available for {model_name}, skipping UMAP plot")
        return
    
    # Create color mapping
    unique_cell_types = sorted(cell_identities.unique())
    colors = sns.color_palette("husl", n_colors=len(unique_cell_types))
    color_map = dict(zip(unique_cell_types, colors))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for cell_type in unique_cell_types:
        mask = cell_identities == cell_type
        if mask.any():
            ax.scatter(theta_2d[mask, 0], theta_2d[mask, 1], 
                      c=[color_map[cell_type]], label=cell_type, alpha=0.7, s=20)
    
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"UMAP of Topic Proportions ({model_name})")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save plot
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compute silhouette score
    try:
        sil_score = silhouette_score(theta_2d, cell_identities)
        print(f"  {model_name} UMAP silhouette score: {sil_score:.3f}")
    except:
        print(f"  {model_name} UMAP silhouette score: N/A (insufficient clusters)")

def run_scanpy_comparison(counts_df: pd.DataFrame, output_dir: Path):
    """
    Run scanpy analysis for comparison with topic models.
    
    Args:
        counts_df: Count matrix with cell identities in index
        output_dir: Output directory
    """
    print("Running scanpy analysis...")
    
    # Create AnnData object
    adata = sc.AnnData(X=counts_df.values)
    adata.var_names = list(counts_df.columns)
    adata.obs_names = list(counts_df.index)
    
    # Add cell type annotations
    cell_identities = pd.Series([i.split("_")[0] for i in counts_df.index], index=counts_df.index)
    adata.obs['cell_type'] = cell_identities.values
    
    # Basic preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Find highly variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    # PCA
    sc.pp.pca(adata, use_highly_variable=True)
    
    # Compute neighbors
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    
    # UMAP
    sc.tl.umap(adata)
    
    # Leiden clustering
    sc.tl.leiden(adata, resolution=0.5)
    
    # Save results
    plots_dir = output_dir / "scanpy"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # UMAP colored by cell type
    sc.pl.umap(adata, color='cell_type', show=False)
    plt.savefig(plots_dir / "scanpy_umap_cell_type.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # UMAP colored by leiden clusters
    sc.pl.umap(adata, color='leiden', show=False)
    plt.savefig(plots_dir / "scanpy_umap_leiden.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compute silhouette score for scanpy UMAP
    try:
        sil_score = silhouette_score(adata.obsm['X_umap'], cell_identities)
        print(f"  Scanpy UMAP silhouette score: {sil_score:.3f}")
    except:
        print(f"  Scanpy UMAP silhouette score: N/A")
    
    # Save scanpy results
    adata.write(plots_dir / "scanpy_results.h5ad")
    
    return adata

def main():
    parser = argparse.ArgumentParser(description="Evaluate topic models (HLDA, LDA, NMF) and run metrics/plots.")
    parser.add_argument("--counts_csv", type=str, required=True, help="Path to filtered count matrix CSV")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test count matrix CSV")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory with model outputs and for saving plots/metrics")
    parser.add_argument("--identity_topics", type=str, required=True, help="Comma-separated list of identity topic names (e.g. 'A,B,C,D')")
    parser.add_argument("--n_extra_topics", type=int, required=True, help="Number of extra topics (e.g. 3 for V1,V2,V3)")
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
    
    # Parse identity topics and extra topics from command line arguments
    identity_topics = args.identity_topics.split(',')
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
    
    # 2) Stacked membership plots for each model
    print("Generating structure plots...")
    for m, d in models.items():
        plots_dir = output_dir / m / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        out_png = plots_dir / f"{m}_structure_plot.png"
        
        # Create a custom title that includes matching information
        if m in ["LDA", "NMF"]:
            mapping_info = ", ".join([f"{orig}→{mapped}" for orig, mapped in topic_mappings[m].items()])
            custom_title = f"Structure Plot: Topic Membership by Cell ({m}) - Matched: {mapping_info}"
        else:
            custom_title = f"Structure Plot: Topic Membership by Cell ({m})"
        
        structure_plot_py(d["theta"], counts_df.index.to_series(), out_png, title=custom_title)

    # 3) Cosine similarity matrix heatmap of estimated beta topics (self-similarity)
    print("Generating cosine similarity heatmaps...")
    for m, d in models.items():
        plots_dir = output_dir / m / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        beta = d["beta"]
        # Compute cosine similarity between topics (columns)
        topic_matrix = beta.values  # shape: (n_genes, n_topics)
        sim_matrix = cosine_similarity(topic_matrix.T)  # shape: (n_topics, n_topics)
        sim_df = pd.DataFrame(sim_matrix, index=beta.columns, columns=beta.columns)
        plt.figure(figsize=(8, 6))
        sns.heatmap(sim_df, annot=True, fmt=".2f", cmap="viridis")
        plt.title(f"Cosine Similarity Between Topics ({m})")
        plt.tight_layout()
        out_png = plots_dir / f"{m}_beta_cosine_similarity.png"
        plt.savefig(out_png, dpi=150)
        plt.close()

    # 4) Geweke histograms for HLDA (A and D chains)
    if "HLDA" in models:
        print("Generating Geweke histograms for HLDA...")
        plots_dir = output_dir / "HLDA" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        beta = models["HLDA"]["beta"]
        theta = models["HLDA"]["theta"]
        n_genes, n_topics = beta.shape
        n_cells = theta.shape[0]
        # Calculate n_save based on new parameters: 10k iterations, 4k burn-in, thin=20
        n_loops, burn_in, thin = 10000, 4000, 20
        n_save = (n_loops - burn_in + 1) // thin
        sample_root = output_dir / "HLDA" / "samples"
        if not sample_root.exists():
            sample_root = output_dir / "HLDA"
        try:
            plot_geweke_histograms(
                sample_root=sample_root,
                keys=["A", "D"],
                out_dir=plots_dir,
                n_save=n_save,
                n_cells=n_cells,
                n_genes=n_genes,
                n_topics=n_topics
            )
        except Exception as e:
            print(f"[WARN] Could not plot Geweke histograms: {e}")

    # 5) PCA pair plots for each model
    print("Generating PCA pair plots...")
    eigvecs, cell_proj = compute_pca_projection(counts_df)
    
    # Create true beta from count matrix averages for comparison
    true_beta = create_true_beta_from_counts(counts_df, identity_topics)
    true_beta_proj = eigvecs @ true_beta.values.astype(np.float32)
    true_names = list(true_beta.columns)
    
    # Extract cell identities from counts_df index
    cell_identities = pd.Series([i.split("_")[0] for i in counts_df.index], index=counts_df.index)
    
    for m, d in models.items():
        plots_dir = output_dir / m / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        beta = d["beta"]
        est_names = list(beta.columns)
        beta_est_proj = eigvecs @ beta.values.astype(np.float32)
        label_mask = np.ones(len(est_names), dtype=bool)
        mixture_mask = np.zeros(cell_proj.shape[0], dtype=bool)
        pc_pairs = [(0, 1), (2, 3), (4, 5)]
        for pcx, pcy in pc_pairs:
            out_png = plots_dir / f"{m}_PC{pcx+1}{pcy+1}.png"
            plot_pca_pair(
                pcx, pcy,
                beta_est_proj, est_names, label_mask,
                true_beta_proj, true_names,
                cell_proj, mixture_mask,
                m, out_png, cell_identities
            )

    # 6) SSE evaluation with custom topic order on held-out test set
    print("Running SSE evaluation...")
    
    # Use the provided test_df (no need to create train/test split)
    test_identities = pd.Series([i.split("_")[0] for i in test_df.index], index=test_df.index)
    
    for m, d in models.items():
        plots_dir = output_dir / m / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        beta = d["beta"]
        
        sse_df = incremental_sse_custom(
            test_df,
            beta,
            test_identities,
            extra_topics,
            theta_out=plots_dir / f"{m}_test_theta_nnls.csv"
        )
        sse_df.to_csv(plots_dir / f"{m}_test_sse.csv", index=False)

    # 7) Extract top genes per topic for each model
    print("Extracting top genes per topic...")
    for m, d in models.items():
        plots_dir = output_dir / m / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        beta = d["beta"]
        
        top_genes_df = extract_top_genes_per_topic(beta, n_top_genes=10)
        top_genes_df.to_csv(plots_dir / f"{m}_top_genes_per_topic.csv")

    # 8) Plot true vs estimated similarity
    print("Computing true vs estimated similarity...")
    for m, d in models.items():
        plots_dir = output_dir / m / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        beta = d["beta"]
        out_png = plots_dir / f"{m}_true_vs_estimated_similarity.png"
        plot_true_vs_estimated_similarity(true_beta, beta, m, out_png)

    # 9) Plot true identity self-similarity
    print("Computing true identity self-similarity...")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_png = plots_dir / "true_identity_self_similarity.png"
    plot_true_self_similarity(true_beta, out_png)
    
    # NOTE: SSE evaluation is currently disabled due to missing dependencies (holdout_test_set, identity_topics, extra_topics)
    
    # Print bash command for eas

    # 10) Compute perplexity and log-likelihood
    print("Computing perplexity and log-likelihood...")
    for m, d in models.items():
        plots_dir = output_dir / m / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        beta = d["beta"]
        theta = d["theta"]
        X_test = test_df.values
        perplexity, log_likelihood = compute_perplexity_and_loglikelihood(X_test, beta, theta)
        print(f"{m} perplexity: {perplexity}")
        print(f"{m} log-likelihood: {log_likelihood}")

    # 11) Plot UMAP of theta
    print("Generating UMAP of theta...")
    for m, d in models.items():
        plots_dir = output_dir / m / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        theta = d["theta"]
        cell_identities = pd.Series([i.split("_")[0] for i in counts_df.index], index=counts_df.index)
        out_png = plots_dir / f"{m}_umap_theta.png"
        plot_umap_theta(theta, cell_identities, m, out_png)

    # 12) Run scanpy comparison
    print("Running scanpy comparison...")
    try:
        run_scanpy_comparison(counts_df, output_dir)
    except Exception as e:
        print(f"  Scanpy comparison failed: {e}")

if __name__ == "__main__":
    main() 

