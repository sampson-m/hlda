#!/usr/bin/env python3
"""
hsim_pipeline.py

Unified pipeline:
 1) simulate scRNA-seq counts (A vs. A_var)
 2) fit 2-topic models (LDA, NMF, custom Gibbs HLDA)
 3) export β/θ CSVs
 4) plot β vs λ-fraction, θ histograms, and Geweke diagnostics
"""
import os
from pathlib import Path
from matplotlib.patches import Patch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import LatentDirichletAllocation, NMF
from scipy.stats import pearsonr
import ast
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
from hsim import simulate_counts
from _functions import run_gibbs_pipeline

# --- Configuration ----------------------------------------------------------
TOPICS   = ["A", "B", "C", "D", "V1", "V2"]
LEAF_TOPICS = ["A", "B", "C", "D"]
K = len(TOPICS) 
EST_ROOT = Path("../estimates/ABCD_V1V2/5_topic_fit/")
SAMPLE_ROOT = Path("../samples/ABCD_V1V2/5_topic_fit") 

# --- Utility functions ------------------------------------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def compare_topic(gm, beta, topic):

    other = [t for t in TOPICS if t != topic][0]
    df = gm[[topic, 'DE', 'DE_group']].rename(
        columns={topic: f'gene_mean_{topic}'}
    )
    df = df.loc[df.index.intersection(beta.index)]
    df['beta_prob'] = beta.loc[df.index, topic]
    df[f'fold_vs_{other}'] = beta.loc[df.index, topic] / beta.loc[df.index, other]
    df['lambda_frac'] = df[f'gene_mean_{topic}'] / df[f'gene_mean_{topic}'].sum()
    df['is_DE'] = df['DE_group'].apply(
        lambda x: topic in x
                  if isinstance(x, (list, tuple))
                  else topic in ast.literal_eval(x)
    )
    return df

def plot_beta_vs_lambda(df, topic, out_png: Path, only_de=False):

    if only_de:
        df = df[df['is_DE']]
        title = f"{topic} (DE only)"
    else:
        title = topic

    pal = {True: 'red', False: 'gray'}
    plt.figure(figsize=(5,5))
    plt.scatter(df['lambda_frac'], df['beta_prob'],
                c=df['is_DE'].map(pal), alpha=0.6)
    m, M = df[['lambda_frac','beta_prob']].min().min(), df[['lambda_frac','beta_prob']].max().max()
    plt.plot([m, M], [m, M], '--', color='black')

    plt.title(title)
    plt.xlabel("β")
    plt.ylabel("β-hat")
    if not only_de:
        plt.legend(handles=[
            Patch(color='red', label='DE'),
            Patch(color='gray', label='non-DE')
        ])
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def load_chain(key):
    arrs = []
    for fn in sorted(SAMPLE_ROOT.glob("constants_*.pkl")):
        with open(fn, "rb") as f:
            d = pickle.load(f)
        arrs.append(d[key])
    return np.stack(arrs)

def geweke_z(chain, first=0.1, last=0.5):
    n = len(chain)
    n1 = max(1, int(first * n))
    n2 = max(1, int(last  * n))
    a, b = chain[:n1], chain[-n2:]
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    if v1==0 or v2==0:
        return np.nan
    return (a.mean() - b.mean()) / np.sqrt(v1/n1 + v2/n2)

def plot_geweke_histograms(sample_dir: Path, plot_dir: Path):

    sample_dir = Path(sample_dir)
    plot_dir   = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    A_chain = load_chain("A")
    D_chain = load_chain("D")

    zA = []
    ns, G, K = A_chain.shape
    for g in range(G):
        for k in range(K):
            z = geweke_z(A_chain[:, g, k])
            if np.isfinite(z):
                zA.append(z)

    zD = []
    _, C, _ = D_chain.shape
    for c in range(C):
        for k in range(K):
            z = geweke_z(D_chain[:, c, k])
            if np.isfinite(z):
                zD.append(z)

    for zs, name in [(zA, "A"), (zD, "D")]:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(zs, bins=50, edgecolor='black')
        ax.set_title(f"Geweke z–scores for {name}-chain")
        ax.set_xlabel("z-score")
        ax.set_ylabel("Count")
        fig.savefig(plot_dir / f"geweke_{name}.png", dpi=300)
        plt.close(fig)

def plot_theta_histograms_by_identity(theta: pd.DataFrame,
                                      identities,
                                      out_png: Path,
                                      bins: int = 20):

    if isinstance(identities, pd.Series):
        ids = identities.reindex(theta.index)
    else:
        ids = pd.Series(list(identities), index=theta.index)

    if ids.isna().all():
        print(f"[plot] no matching cells; skipping {out_png}")
        return

    groups = ids.dropna().unique()
    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    # 4) plot
    for ax, grp in zip(axes, groups):
        sel = (ids == grp)
        for topic in theta.columns:
            sns.histplot(
                theta.loc[sel, topic],
                stat='density',
                element='step',
                bins=bins,
                label=topic,
                ax=ax,
                alpha=0.7
            )
        ax.set_title(f"Identity: {grp}")
        ax.set_xlabel("θ")
        if ax is axes[0]:
            ax.set_ylabel("Density")
        ax.legend(title="Topic", loc="upper right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    
def plot_theta_true_vs_est_by_identity(true_theta: pd.DataFrame,
                                       est_theta: pd.DataFrame,
                                       identities: pd.Series,
                                       model_name: str,
                                       out_png: Path):

    topics = true_theta.columns.tolist()
    groups = identities.unique().tolist()  # ['A','B']
    n_groups, n_topics = len(groups), len(topics)

    fig, axes = plt.subplots(n_groups, n_topics,
                             figsize=(5*n_topics, 4*n_groups),
                             sharex=True, sharey=True)

    for i, grp in enumerate(groups):
        mask = (identities == grp)
        for j, topic in enumerate(topics):
            ax = axes[i, j]
            x = true_theta.loc[mask, topic]
            y = est_theta.loc[mask, topic]
            ax.scatter(x, y, alpha=0.4, s=10)
            ax.plot([0,1], [0,1], "--", color="gray")

            if i == 0:
                ax.set_title(topic, fontsize=14)
            if i == n_groups-1:
                ax.set_xlabel(f"True θ_{topic}", fontsize=12)
            if j == 0:
                ax.set_ylabel(f"Est θ_{topic}", fontsize=12)

    plt.suptitle(f"{model_name} — True vs. Estimated θ by identity", y=1.02, fontsize=16)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
# --- Model wrappers --------------------------------------------------------

def run_sklearn_model(counts: pd.DataFrame,
                      gene_means: pd.DataFrame,
                      model_name: str,
                      out_dir: Path):
    ensure_dir(out_dir)
    plot_dir = ensure_dir(out_dir / "plots")

    X = counts.drop(columns=["cell_identity"], errors="ignore").values
    K = len(TOPICS)

    # 1) fit
    if model_name == "lda":
        m = LatentDirichletAllocation(n_components=K, random_state=0)
    elif model_name == "nmf":
        m = NMF(n_components=K,
                init="nndsvd",
                random_state=0,
                max_iter=5000)
    else:
        raise ValueError(f"Unknown model {model_name}")

    W = m.fit_transform(X)    # shape: (n_cells, K)
    H = m.components_         # shape: (K, n_genes)

    genes = counts.columns

    H_norm = H / H.sum(axis=1)[:, None]
    beta_raw = pd.DataFrame(H_norm.T,
                            index=genes,
                            columns=list(range(K)))

    beta_raw.to_csv(out_dir / f"{model_name}_beta.csv")

    theta_raw = pd.DataFrame(W,
                             index=counts.index,
                             columns=list(range(K)))
    theta_norm = theta_raw.div(theta_raw.sum(axis=1), axis=0)
    theta_norm.to_csv(out_dir / f"{model_name}_theta.csv")


def match_sklearn_topics(beta_est: pd.DataFrame,
                         gene_means: pd.DataFrame,
                         topics: list[str]) -> dict[int,str]:
    K = beta_est.shape[1]
    true_frac = {t: gene_means[t] / gene_means[t].sum() for t in topics}
    true_frac_df = pd.DataFrame(true_frac, index=beta_est.index)

    corr = np.zeros((K, K))
    for i in range(K):
        for j, t in enumerate(topics):
            corr[i, j] = np.corrcoef(beta_est.iloc[:, i], true_frac_df.iloc[:, j])[0,1]
    row_ind, col_ind = linear_sum_assignment(-corr)
    mapping = {i: topics[j] for i,j in zip(row_ind, col_ind)}
    return mapping

def match_leaf_topics(beta_raw: pd.DataFrame,
                      gene_means: pd.DataFrame,
                      leaf_topics: list[str]) -> dict[int,str]:
    K = beta_raw.shape[1]
    # compute true gene‐fractions
    true_frac = {t: gene_means[t] / gene_means[t].sum() for t in leaf_topics}
    true_df   = pd.DataFrame(true_frac, index=beta_raw.index)

    # build correlation matrix (raw_i vs leaf_j)
    corr = np.zeros((K, len(leaf_topics)))
    for i in range(K):
        for j, t in enumerate(leaf_topics):
            corr[i,j] = np.corrcoef(beta_raw.iloc[:,i], true_df[t])[0,1]

    # Hungarian: maximize total corr
    row_ind, col_ind = linear_sum_assignment(-corr)
    return {i: leaf_topics[j] for i,j in zip(row_ind, col_ind)}

# --- Projections ------------------------------------------------------------

def normalize_cell_counts(counts_df):
    return counts_df.div(counts_df.sum(axis=1), axis=0)

def fit_topic_embedding(beta_df, method='umap', n_components=5, **kwargs):
    X = beta_df.T.values
    if method == 'pca':
        model = PCA(n_components=n_components, **kwargs)
        Z = model.fit_transform(X)
    elif method == 'tsne':
        model = TSNE(n_components=n_components, **kwargs)
        Z = model.fit_transform(X)
    elif method == 'umap':
        if not HAS_UMAP:
            raise ImportError("UMAP not available")
        model = umap.UMAP(n_components=n_components, **kwargs)
        Z = model.fit_transform(X)
    else:
        raise ValueError(method)
    cols = [f"{method.upper()}{i+1}" for i in range(n_components)]
    return model, pd.DataFrame(Z, index=beta_df.columns, columns=cols)

def project_cells(model, props_df, beta_df):
    genes = beta_df.index
    Xc = props_df[genes].values
    Zc = model.transform(Xc)
    cols = [f"Dim{i+1}" for i in range(Zc.shape[1])]
    return pd.DataFrame(Zc, index=props_df.index, columns=cols)

# --- Main -------------------------------------------------------------------

def main():
    sns.set_style("whitegrid")

    data_root   = Path("../data/ABCD_V1V2")
    counts_file = data_root / "counts.csv"
    gm_file     = data_root / "gene_means.csv"
    theta_file  = data_root / "theta.csv"

    ensure_dir(data_root)

    counts_df          = pd.read_csv(counts_file, index_col=0)
    gene_means_matrix  = pd.read_csv(gm_file, index_col=0)
    true_theta         = pd.read_csv(theta_file, index_col=0)
    true_theta = true_theta[TOPICS]

    for mod in ("lda", "nmf"):
        run_sklearn_model(counts_df, gene_means_matrix,
                          model_name=mod,
                          out_dir=EST_ROOT / mod.upper())

    models = [
        ("LDA", EST_ROOT/"LDA"),
        ("NMF", EST_ROOT/"NMF"),
##        ("HLDA", EST_ROOT/"HLDA")
    ]
    
    props_df = normalize_cell_counts(counts_df)

    for model_name, model_dir in models:
        beta_path  = (model_dir/f"{model_name.lower()}_beta.csv"
                      if model_name!="HLDA"
                      else model_dir/"HLDA_beta.csv")
        theta_path = (model_dir/f"{model_name.lower()}_theta.csv"
                      if model_name!="HLDA"
                      else model_dir/"HLDA_theta.csv")

        # read the raw outputs
        beta_raw  = pd.read_csv(beta_path,  index_col=0)
        theta_raw = pd.read_csv(theta_path, index_col=0)

        if model_name != "HLDA":
            raw_map = match_sklearn_topics(beta_raw, gene_means_matrix, TOPICS)
            str_map = {str(k): v for k, v in raw_map.items()}
            beta  = beta_raw .rename(columns=str_map)[TOPICS]
            theta = theta_raw.rename(columns=str_map)[TOPICS]
        else:
            beta  = beta_raw[TOPICS]
            theta = theta_raw[TOPICS]

        emb, topic2d = fit_topic_embedding(beta, method="pca")

        cell2d = project_cells(emb, props_df, beta)
        ids   = props_df.index.to_series()
        cmap  = {"A":"red","B":"blue"}
        cols  = ids.map(cmap)

        plot_dir = ensure_dir(model_dir/"plots")
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(cell2d.Dim1, cell2d.Dim2,
                   c=cols, s=10, alpha=0.7, label="cells")
        for tp in topic2d.index:
            ax.scatter(topic2d.loc[tp,"PCA1"],
                       topic2d.loc[tp,"PCA2"],
                       s=200, edgecolor="black", label=tp)
        ax.set_title(f"{model_name} 2D projection")
        ax.legend()
        fig.savefig(plot_dir/f"{model_name}_projection.png", dpi=300)
        plt.close(fig)

        if model_name == "HLDA":

            plot_geweke_histograms(
            sample_dir=Path("../samples/AB"),
            plot_dir=Path("../estimates/AB/HLDA/plots")
            )

        # 1) β vs λ_frac for each topic
        for topic in TOPICS:
            comp = compare_topic(gene_means_matrix, beta, topic)
            plot_beta_vs_lambda(comp,
                                topic,
                                model_dir/"plots"/f"{model_name}_{topic}_all.png",
                                only_de=False)
            plot_beta_vs_lambda(comp,
                                topic,
                                model_dir/"plots"/f"{model_name}_{topic}_DE.png",
                                only_de=True)

        plot_theta_histograms_by_identity(theta,
                                          counts_df.index,
                                          model_dir/"plots"/f"{model_name}_theta_by_identity.png")
        
##        plot_theta_true_vs_est_by_identity(
##            true_theta,
##            theta,
##            counts_df.index.to_series(),   # or however you store identity labels
##            model_name,
##            model_dir/"plots"/f"{model_name}_theta_true_vs_est_by_id.png"
##        )

if __name__ == "__main__":
    main()
