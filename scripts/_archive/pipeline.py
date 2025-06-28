#!/usr/bin/env python3
"""
hsim_pipeline.py
================
Pipeline to
1. **load** pre‑simulated counts / truth (no longer simulating here)
2. **fit** LDA & NMF on a fixed K
3. **match topics** to reference gene‑means
4. **visualise** β‑vs‑λ, θ histograms, and optionally true‑vs‑est θ

Two operating modes, controlled by **INFER_TRUE_TOPICS** in USER CONFIG:

* **INFER_TRUE_TOPICS = True**
  * We assume the model K equals the *true* number of topics.
  * All discovered topics are matched (A, B, C, D, V1, V2).
  * Scatter plots of *true* vs *estimated* θ are produced.

* **INFER_TRUE_TOPICS = False**
  * We assume K differs from truth (e.g. fewer topics fit).
  * Only **leaf topics** (A, B, C, D) are matched & plotted.
  * θ histograms are still produced, but no scatter to truth.

Edit the USER CONFIG block and run `python hsim_pipeline.py`.
"""
from __future__ import annotations

import ast
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.decomposition import LatentDirichletAllocation, NMF, PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from umap.umap_ import UMAP
import time

# -----------------------------------------------------------------------------
# 📝 USER CONFIG
# -----------------------------------------------------------------------------
 

DATA_ROOT = Path("../data/ABCD_V1V2")
COUNTS_FILE = DATA_ROOT / "counts.csv"
GM_FILE = DATA_ROOT / "gene_means.csv"
THETA_FILE = DATA_ROOT / "theta.csv"
GENE_META_FILE = DATA_ROOT / "gene_metadata.csv"

EST_ROOT = Path("../estimates/ABCD_V1V2/5_topic_fit")
SAMPLE_ROOT = Path("../samples/ABCD_V1V2/5_topic_fit")

TOPICS: List[str] = ["A", "B", "C", "D", "V1"]
LEAF_TOPICS: List[str] = ["A", "B", "C", "D"]
ACTIVITY_TOPS = ["V1", "V2"]
MIXTURE_THRESH = 0.01
K = len(TOPICS)

NUM_LOOPS   = 15000    # total Gibbs sweeps
BURN_IN     = 5000     # burn-in iterations
THIN        =   40     # thinning interval
N_GENES=2000

INFER_TRUE_TOPICS: bool = False

SEED = 0 

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


# -------------- topic matching helpers ---------------------------------------

def match_topics(beta_hat: pd.DataFrame,
                 gene_means: pd.DataFrame,
                 topics_true: list[str] = TOPICS) -> dict:

    common = beta_hat.index.intersection(gene_means.index)
    Bh = beta_hat.values
    Bt = gene_means[topics_true].div(gene_means[topics_true].sum(axis=0), axis=1).values

    cos = (Bt.T @ Bh) / (
        np.linalg.norm(Bt, axis=0)[:, None] *
        np.linalg.norm(Bh, axis=0)[None, :] + 1e-12
    ) 

    rows, cols = linear_sum_assignment(1 - cos)
    return {beta_hat.columns[j]: topics_true[i] for i, j in zip(rows, cols)}


def match_leaf_topics(beta_raw: pd.DataFrame, gene_means: pd.DataFrame, leaf_topics: List[str]):
    return match_sklearn_topics(beta_raw, gene_means, leaf_topics)

def match_topics_sse(beta_hat: pd.DataFrame,
                     gene_means: pd.DataFrame,
                     K_total: int) -> dict[str,str]:

    # 1) Define true topics based on K_total
    if K_total == 5:
        true_topics = ['A','B','C','D','V1']
    elif K_total == 6:
        true_topics = ['A','B','C','D','V1','V2']
    elif K_total == 7:
        true_topics = ['A','B','C','D','V1','V2']
    else:
        raise ValueError(f"Unsupported K_total={K_total}")

    if all(isinstance(c, (int, np.integer)) for c in beta_hat.columns):
        beta_hat = beta_hat.rename(
            columns={c: TOPIC_INT2STR[int(c)] for c in beta_hat.columns}
        )

    Bt = gene_means[true_topics] \
           .div(gene_means[true_topics].sum(axis=0), axis=1) \
           .values  # shape: (n_genes × len(true_topics))
    Bh = beta_hat.values  # shape: (n_genes × n_hat_topics)
    cos = (Bt.T @ Bh) / (
        np.linalg.norm(Bt, axis=0)[:,None] *
        np.linalg.norm(Bh, axis=0)[None,:] + 1e-12
    )

    rows, cols = linear_sum_assignment(1 - cos)
    mapping = { beta_hat.columns[j]: true_topics[i]
                for i,j in zip(rows, cols) }

    if K_total == 7:
        leftover = set(beta_hat.columns) - set(mapping)
        for col in leftover:
            mapping[col] = 'V3'

    return mapping

def compute_pca_projection(counts_df: pd.DataFrame,
                           n_components: int = 6,
                           random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    pca.fit(counts_df.values.astype(np.float32))
    eigvecs = pca.components_                          # shape (2, n_genes)

    cell_props = counts_df.div(counts_df.sum(axis=1), axis=0).values.astype(np.float32)
    cell_proj  = cell_props @ eigvecs.T                # (n_cells × 2)
    return eigvecs, cell_proj


# -------------- plotting helpers ----------------------------

def _identity_from_cell_id(cid: str) -> str:
    """Assumes cell IDs are like 'A_001', returns 'A'."""
    return cid.split("_")[0]

def plot_topics_layer(ax,
                     proj: np.ndarray,
                     names: List[str],
                     label_mask: np.ndarray,
                     marker: str,
                     color: str,
                     label: str,
                     text_align: str):
    ax.scatter(proj[0], proj[1], s=110 if marker == "*" else 90,
               marker=marker, c=color, label=label)
    ha, va = ("left", "bottom") if text_align == "lb" else ("right", "top")
    for i, n in enumerate(names):
        if label_mask[i]:
            ax.text(proj[0, i], proj[1, i], n, fontsize=8, ha=ha, va=va)



def plot_pca(beta_proj_est: np.ndarray,
             est_names: List[str],          # all inferred column names
             est_label_mask: np.ndarray,    # True → label this inferred topic
             beta_proj_true: np.ndarray,
             true_names: List[str],         # all ground-truth topic names
             cell_proj: np.ndarray,
             mixture_mask: np.ndarray,
             model_name: str,
             out_png: Path) -> None:

    fig, ax = plt.subplots(figsize=(8, 7))

    # ----- cells --------------------------------------------------------------
    ax.scatter(cell_proj[~mixture_mask, 0], cell_proj[~mixture_mask, 1],
               s=12, c="lightgray", alpha=0.5, marker="o", label="cells (pure)")
    ax.scatter(cell_proj[mixture_mask, 0], cell_proj[mixture_mask, 1],
               s=18, c="tab:orange", alpha=0.8, marker="^", label="cells w/ activity")

    # ----- estimated topics (β̂) ---------------------------------------------
    plot_topics_layer(ax,
                      proj=beta_proj_est,
                      names=est_names,
                      label_mask=est_label_mask,
                      marker="*",
                      color="tab:red",
                      label=f"{model_name} β̂",
                      text_align="lb")

    # ----- ground-truth topics (β) -------------------------------------------
    true_label_mask = np.ones(len(true_names), dtype=bool)  # label them all
    plot_topics_layer(ax,
                      proj=beta_proj_true,
                      names=true_names,
                      label_mask=true_label_mask,
                      marker="^",
                      color="tab:green",
                      label="true β",
                      text_align="rt")

    # ----- final styling ------------------------------------------------------
    ax.set_xlabel("PC-1"); ax.set_ylabel("PC-2")
    ax.set_title(f"{model_name} – mixture vs. pure")
    ax.legend(frameon=False, fontsize=8, loc="best")

    ensure_dir(out_png.parent)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_pca_by_identity(beta_proj_est: np.ndarray,
                         est_names: List[str],
                         est_label_mask: np.ndarray,
                         beta_proj_true: np.ndarray,
                         true_names: List[str],
                         cell_proj: np.ndarray,
                         cell_ids: List[str],
                         model_name: str,
                         out_png: Path) -> None:

    # ---------------- cell colours -------------------------------
    identities = [_identity_from_cell_id(cid) for cid in cell_ids]
    uniq_ids   = sorted(set(identities))
    palette    = sns.color_palette("husl", len(uniq_ids))
    id2color   = dict(zip(uniq_ids, palette))
    colors     = [id2color[i] for i in identities]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(cell_proj[:, 0], cell_proj[:, 1],
               s=14, c=colors, alpha=0.7, marker="o")

    # ---------------- topics layers ------------------------------
    plot_topics_layer(ax,
                      proj=beta_proj_est,
                      names=est_names,
                      label_mask=est_label_mask,
                      marker="*",
                      color="tab:red",
                      label=f"{model_name} β̂",
                      text_align="lb")

    true_label_mask = np.ones(len(true_names), dtype=bool)  # label all
    plot_topics_layer(ax,
                      proj=beta_proj_true,
                      names=true_names,
                      label_mask=true_label_mask,
                      marker="^",
                      color="tab:green",
                      label="true β",
                      text_align="rt")

    # legend for identities
    for uid in uniq_ids:
        ax.scatter([], [], c=[id2color[uid]], label=uid, s=40)

    _style_and_save(ax, f"{model_name} – cells by identity", out_png)

def plot_pca_pair(pc_x: int, pc_y: int,
                  beta_proj_est: np.ndarray, est_names: List[str], est_label_mask: np.ndarray,
                  beta_proj_true: np.ndarray, true_names: List[str],
                  cell_proj: np.ndarray, mixture_mask: np.ndarray,
                  model_name: str, out_png: Path):
    
    fig, ax = plt.subplots(figsize=(7, 6))

    # cells layer
    ax.scatter(cell_proj[~mixture_mask, pc_x], cell_proj[~mixture_mask, pc_y],
               s=12, c="lightgray", alpha=0.4, marker="o", label="cells (pure)")
    ax.scatter(cell_proj[mixture_mask, pc_x], cell_proj[mixture_mask, pc_y],
               s=16, c="tab:orange", alpha=0.8, marker="^", label="cells w/ activity")

    # topics layers
    plot_topics_layer(ax, beta_proj_est[[pc_x, pc_y], :], est_names, est_label_mask,
                      marker="*", color="tab:red", label=f"{model_name} β̂", text_align="lb")
    true_mask = np.ones(len(true_names), dtype=bool)
    plot_topics_layer(ax, beta_proj_true[[pc_x, pc_y], :], true_names, true_mask,
                      marker="^", color="tab:green", label="true β", text_align="rt")

    _style_and_save(ax,
                    title=f"{model_name}: PC{pc_x+1} vs PC{pc_y+1}",
                    out_png=out_png,
                    xlabel=f"PC{pc_x+1}",
                    ylabel=f"PC{pc_y+1}"
    )


def _add_topic_layers(ax, beta_proj_est, beta_proj_true, topic_names, model_name):
    ax.scatter(beta_proj_est[0], beta_proj_est[1], s=130, marker="*", c="tab:red", label=f"{model_name} β̂")
    for i, t in enumerate(topic_names):
        ax.text(beta_proj_est[0, i], beta_proj_est[1, i], t, fontsize=8, ha="left", va="bottom")
    ax.scatter(beta_proj_true[0], beta_proj_true[1], s=100, marker="^", c="tab:green", label="true β")
    for i, t in enumerate(topic_names):
        ax.text(beta_proj_true[0, i], beta_proj_true[1, i], t, fontsize=8, ha="right", va="top")


def _style_and_save(ax,
                     title: str,
                     out_png: Path,
                     xlabel: Optional[str] = None,
                     ylabel: Optional[str] = None):
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ensure_dir(out_png.parent)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def compare_topic(gm: pd.DataFrame, beta: pd.DataFrame, meta: pd.DataFrame, topic: str) -> pd.DataFrame:
    other = [t for t in gm.columns if t != topic][0]

    df = pd.DataFrame(index=gm.index)
    df[f"gene_mean_{topic}"] = gm[topic]

    # bring in gene‑level metadata
    cols_meta = [c for c in meta.columns if c.startswith("DE_factor_") or c in {"DE", "DE_groups", "DE_group"}]
    df = df.join(meta[cols_meta], how="left")

    # beta estimates & diagnostics
    df = df.loc[df.index.intersection(beta.index)]
    df["beta_prob"] = beta.loc[df.index, topic]
    df[f"fold_vs_{other}"] = beta.loc[df.index, topic] / beta.loc[df.index, other]
    df["lambda_frac"] = df[f"gene_mean_{topic}"] / df[f"gene_mean_{topic}"].sum()

    # DE flag per topic
    if "DE_groups" in df.columns:
        df["is_DE"] = df["DE_groups"].apply(lambda x: topic in x if isinstance(x, (list, tuple)) else topic in ast.literal_eval(x) if pd.notna(x) else False)
    else:  # fall back to old column name
        df["is_DE"] = df["DE_group"].apply(lambda x: topic in x if isinstance(x, (list, tuple)) else topic in ast.literal_eval(x) if pd.notna(x) else False)

    return df

def plot_beta_vs_lambda(df: pd.DataFrame, topic: str, out_png: Path, only_de: bool = False):
    """Scatter β̂ vs true λ fraction for one topic."""
    sub = df[df['is_DE']] if only_de else df
    title = f"{topic} ({'DE only' if only_de else 'all genes'})"
    pal = {True: 'red', False: 'gray'}
    plt.figure(figsize=(5, 5))
    plt.scatter(sub['lambda_frac'], sub['beta_prob'], c=sub['is_DE'].map(pal), alpha=0.6)
    m = min(sub['lambda_frac'].min(), sub['beta_prob'].min())
    M = max(sub['lambda_frac'].max(), sub['beta_prob'].max())
    plt.plot([m, M], [m, M], '--', color='black')
    plt.title(title)
    plt.xlabel("Beta")
    plt.ylabel("Beta hat")
    if not only_de:
        plt.legend(handles=[Patch(color='red', label='DE'), Patch(color='gray', label='non DE')])
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_theta_histograms_by_identity(
    theta: pd.DataFrame,
    identities: pd.Series | list[str],
    out_png: Path,
    bins: int = 60,              # fewer bins → less clutter
    bin_range: tuple = (0, 1),
    zero_thresh: float = 0.025,   # skip θ < threshold
    show_kde: bool = False,
):

    ids = (identities if isinstance(identities, pd.Series)
           else pd.Series(identities, index=theta.index))
    ids = ids.astype(str).str.split("_").str[0]

    valid_groups = ids.dropna().unique()
    if valid_groups.size == 0:
        return

    fig, axes = plt.subplots(1, len(valid_groups),
                             figsize=(4 * len(valid_groups), 4),
                             sharey=True)
    axes = np.atleast_1d(axes)  # ensure iterable even for one group

    for ax, grp in zip(axes, valid_groups):
        mask = ids == grp
        for topic in theta.columns:
            data = theta.loc[mask, topic]
            data = data[data > zero_thresh]          # <-- trim zeros
            if data.empty:
                continue

            sns.histplot(data,
                         bins=bins,
                         binrange=bin_range,
                         stat="density",
                         element="step",
                         alpha=0.5,
                         ax=ax,
                         label=topic)
            if show_kde:
                sns.kdeplot(data, ax=ax, linewidth=1)

        ax.set_title(f"Identity: {grp}")
        ax.set_xlabel("θ")
        if ax is axes[0]:
            ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_theta_true_vs_est_by_identity(true_theta: pd.DataFrame, est_theta: pd.DataFrame, identities: pd.Series, model_name: str, out_png: Path):
    """Scatter true vs estimated θ split by identity."""
    topics = true_theta.columns.tolist()
    ids = identities

    ids = ids.astype(str).str.split("_").str[0]
    groups = ids.unique().tolist()

    fig, axes = plt.subplots(len(groups), len(topics), figsize=(5 * len(topics), 4 * len(groups)), sharex=True, sharey=True)
    for i, grp in enumerate(groups):
        mask = (ids == grp)
        for j, topic in enumerate(topics):
            ax = axes[i, j]
            ax.scatter(true_theta.loc[mask, topic], est_theta.loc[mask, topic], alpha=0.4, s=10)
            ax.plot([0, 1], [0, 1], '--', color='gray')
            if i == 0:
                ax.set_title(topic)
            if i == len(groups) - 1:
                ax.set_xlabel("Theta")
            if j == 0:
                ax.set_ylabel("Theta-hat")
    plt.suptitle(f"{model_name}: true vs est θ by identity", y=1.02)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

def cosine_similarity_matrix(true_beta: pd.DataFrame, beta_hat: pd.DataFrame, eps: float = 1e-12, save_path=None):

    Tb = true_beta.values
    Eb = beta_hat.values

    numer = Tb.T @ Eb
    denom = (np.linalg.norm(Tb, axis=0, keepdims=True).T @
             np.linalg.norm(Eb, axis=0, keepdims=True) + eps)
    cos = numer / denom

    sim_df = pd.DataFrame(cos, index=true_beta.columns, columns=beta_hat.columns)
    sim_df = sim_df.sort_index(axis=0).sort_index(axis=1)
    h, w = sim_df.shape
    plt.figure(figsize=(1.2 * w + 2, 1.2 * h + 2))
    ax = sns.heatmap(sim_df, cmap="viridis", annot=True, fmt=".2f",
                     linewidths=0.4, linecolor="white", square=True,
                     cbar_kws={"label": "cosine similarity"})
    ax.set_xlabel("beta est")
    ax.set_ylabel("beta")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path / 'beta_cosine_sim_est.png', dpi=300)
        plt.close()

    Tb_norms = np.linalg.norm(Tb, axis=0)
    numer_tt = Tb.T @ Tb
    denom_tt = np.outer(Tb_norms, Tb_norms) + eps
    cos_tt = numer_tt / denom_tt

    sim_tt_df = pd.DataFrame(cos_tt,
                             index=true_beta.columns,
                             columns=true_beta.columns)
    sim_tt_df = sim_tt_df.sort_index(axis=0).sort_index(axis=1)

    h_tt, w_tt = sim_tt_df.shape
    plt.figure(figsize=(1.2 * w_tt + 2, 1.2 * h_tt + 2))
    ax_tt = sns.heatmap(sim_tt_df, cmap="viridis", annot=True, fmt=".2f",
                        linewidths=0.4, linecolor="white", square=True,
                        cbar_kws={"label": "cosine similarity"})
    ax_tt.set_xlabel("beta_true")
    ax_tt.set_ylabel("beta_true")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path / 'beta_cosine_sim_true.png', dpi=300)
        plt.close()


# ════════════════ geweke helpers ══════════════════════════════════

def load_chain(sample_root: Path, key: str, n_save, n_cells, n_genes, n_topics) -> np.ndarray:
    """
    Load and return a memmap array for the requested 'key'.
    Supported keys:
      - 'A': loads A_chain.memmap with shape (n_save, n_genes, K)
      - 'D': loads D_chain.memmap with shape (n_save, n_cells, K)

    Replace the `# TODO` lines below with your actual values.
    """
    if key == 'A':
        # Number of thinned samples that were saved
        n_save = n_save 
        n_genes = n_genes
        k = n_topics

        return np.memmap(
            filename=str(sample_root / "A_chain.memmap"),
            mode="r",
            dtype=np.int32,
            shape=(n_save, n_genes, n_topics)
        )

    elif key == 'D':
        n_save = n_save
        n_cells = n_cells
        k = n_topics

        return np.memmap(
            filename=str(sample_root / "D_chain.memmap"),
            mode="r",
            dtype=np.int32,
            shape=(n_save, n_cells, n_topics)
        )

    else:
        raise ValueError(f"No memmap defined for key '{key}'")


def plot_geweke_histograms(sample_root: Path, keys: list[str], out_dir: Path, n_save, n_cells, n_genes, n_topics):
    """
    For each key in `keys` (e.g. ['A','D',…]), load its memmap chain and compute Geweke
    z-scores for every scalar entry across the saved iterations. Then plot and save
    a histogram of those z-scores into `out_dir`.

    This version calls load_chain(...), which now returns a memmap instead of pickled arrays.
    """
    out_dir = ensure_dir(out_dir)

    for key in keys:
        try:
            chain = load_chain(sample_root, key, n_save, n_cells, n_genes, n_topics)  # shape = (n_save, dim1, dim2)
        except ValueError as e:
            print(f"WARNING: {e}  Skipping '{key}'.")
            continue

        # chain.shape = (n_save, …).  We want to compute Geweke z for each "coordinate" in the remaining dims.
        # E.g. for 'A', chain[(Ellipsis,) + (i, j)] is the time-series of counts for gene i, topic j.
        z_vals = []
        for idx in np.ndindex(chain.shape[1:]):
            ts = chain[(Ellipsis,) + idx]  # 1D array of length n_save
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


def geweke_z(chain: np.ndarray, first: float = 0.1, last: float = 0.5) -> float:
    n = len(chain)
    n1, n2 = max(1, int(first * n)), max(1, int(last * n))
    a, b = chain[:n1], chain[-n2:]
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    if v1 == 0 or v2 == 0:
        return np.nan
    return (a.mean() - b.mean()) / np.sqrt(v1 / n1 + v2 / n2)


# ════════════════ fitting helpers ══════════════════════════════════
def run_sklearn_model(counts: pd.DataFrame, model_name: str, out_dir: Path, k: int):
    """
    Fit either an LDA or NMF model with K topics and write β / θ CSVs
    into `out_dir`.
    """
    ensure_dir(out_dir)

    print(f"Running {model_name} model")

    X = counts.values.astype(float)

    t0 = time.time()
    if model_name == "lda":
        model = LatentDirichletAllocation(
            n_components=k,
            random_state=SEED,
            learning_method="batch",
            max_iter=5000,
        )
    elif model_name == "nmf":
        model = NMF(
            n_components=k,
            init="nndsvd",
            random_state=SEED,
            max_iter=5000,
            solver="cd",
##            beta_loss="kullback-leibler",
        )
    else:
        raise ValueError(f"Unknown model '{model_name}'")

    elapsed = time.time() - t0
    est_hours = elapsed / 3600.0

    print(f"Compute {model_name} fit in {elapsed} seconds or {est_hours} hours")


    theta = model.fit_transform(X)                
    theta = theta / theta.sum(axis=1, keepdims=True) 
    beta  = normalize(model.components_, norm="l1", axis=1) 

    # write out
    pd.DataFrame(beta.T, index=counts.columns, columns=range(k)).to_csv(
        out_dir / f"{model_name}_beta.csv"
    )
    pd.DataFrame(theta, index=counts.index, columns=range(k)).to_csv(
        out_dir / f"{model_name}_theta.csv"
    )

def main():
    sns.set_style("whitegrid")

    ensure_dir(EST_ROOT)

    # ---------- load data ----------
    counts_df = pd.read_csv(COUNTS_FILE, index_col=0)
    gene_means = pd.read_csv(GM_FILE, index_col=0)
    gene_meta = pd.read_csv(GENE_META_FILE, index_col=0)
    
    true_theta = pd.read_csv(THETA_FILE, index_col=0)


    # ---------- fit models ----------
    for mod in ("lda", "nmf"):
##    mod="nmf"
        run_sklearn_model(counts_df, mod, EST_ROOT / mod.upper(), k=K)

    # ---------- evaluation loop ----------
    models = [
        ("LDA", EST_ROOT / "LDA"),
        ("NMF", EST_ROOT / "NMF"),
        ("HLDA", EST_ROOT / "HLDA")
    ]

    topics_eval = TOPICS if INFER_TRUE_TOPICS else LEAF_TOPICS
    
    print('fitting pca')
    mixture_mask = true_theta[ACTIVITY_TOPS].sum(axis=1).gt(MIXTURE_THRESH).values
    true_beta = gene_means
    true_beta = true_beta.div(true_beta.sum(axis=0), axis=1)
    eigvecs, cell_proj = compute_pca_projection(counts_df)
    true_beta_proj = eigvecs @ true_beta.values.astype(np.float32)   # (2 × K)

    for model_name, model_dir in models:
        beta_path  = model_dir / f"{model_name.lower()}_beta.csv"
        theta_path = model_dir / f"{model_name.lower()}_theta.csv"
        if not beta_path.exists():
            continue

        beta_raw  = pd.read_csv(beta_path,  index_col=0)
        theta_raw = pd.read_csv(theta_path, index_col=0)

        mapping = (
            match_topics(beta_raw, gene_means, TOPICS)
            if INFER_TRUE_TOPICS
            else match_topics(beta_raw, gene_means, LEAF_TOPICS)
        )
        rename_dict = {str(orig): tgt for orig, tgt in mapping.items()}
        beta_all   = beta_raw.rename(columns=rename_dict)      # full set for plotting
        theta_all  = theta_raw.rename(columns=rename_dict)
        beta  = beta_all[topics_eval]
        theta = theta_all[topics_eval]

        est_names       = list(beta_all.columns)                   
        beta_est_proj   = eigvecs @ beta_all.values.astype(np.float32)
        true_names      = list(true_beta.columns)
        true_beta_proj  = eigvecs @ true_beta.values.astype(np.float32)

        label_mask = np.array([name in LEAF_TOPICS for name in est_names])

        cosine_similarity_matrix(
                                true_beta,
                                beta_all,
                                save_path=model_dir / "plots" / "beta_similarity.png",
                            )

        pc_pairs = [(0, 1), (2, 3), (4, 5)]
        for pcx, pcy in pc_pairs:
            out_png = model_dir / "plots" / f"{model_name}_PC{pcx+1}{pcy+1}.png"
            plot_pca_pair(pcx, pcy,
                          beta_est_proj, est_names, label_mask,
                          true_beta_proj, true_names,
                          cell_proj, mixture_mask,
                          model_name, out_png)

        plot_pca(beta_est_proj, est_names, label_mask,
                 true_beta_proj, true_names,
                 cell_proj, mixture_mask,
                 model_name,
                 model_dir / "plots" / f"{model_name}_PCA_mixture.png")

        plot_pca_by_identity(beta_est_proj, est_names, label_mask,
                     true_beta_proj, true_names,
                     cell_proj, counts_df.index.tolist(),
                     model_name,
                     model_dir / "plots" / f"{model_name}_PCA_byID.png")

        if model_name == "HLDA":
            plot_geweke_histograms(
                sample_root=SAMPLE_ROOT,    
                keys=["A", "D"],          
                out_dir=model_dir / "plots"
            )


        plot_dir = ensure_dir(model_dir / "plots")
        for topic in topics_eval:
            comp = compare_topic(gene_means, beta, gene_meta, topic)
            print('Plotting beta vs beta-hat')
            plot_beta_vs_lambda(comp, topic, plot_dir / f"{model_name}_{topic}_all.png", only_de=False)

        print('Plotting theta hist')
        plot_theta_histograms_by_identity(theta_raw, counts_df.index, plot_dir / f"{model_name}_theta_by_identity.png")


        if INFER_TRUE_TOPICS and true_theta is not None:
            plot_theta_true_vs_est_by_identity(true_theta, theta, counts_df.index.to_series(), model_name, plot_dir / f"{model_name}_theta_true_vs_est_by_id.png")

    print("✅ Pipeline finished. Outputs in", EST_ROOT.resolve())


if __name__ == "__main__":
    main()
