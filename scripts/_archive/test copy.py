import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# ———————————————————————————————————————————————
# 1) Load utilities
# ———————————————————————————————————————————————
def load_gene_means(path="../data/hsim/gene_means_matrix.csv"):
    gm = pd.read_csv(path, index_col=0)
    gm.index   = gm.index.str.strip()
    gm.columns = gm.columns.str.strip()
    return gm

def save_nonDE_comparisons_pdf(
    true_exp: pd.DataFrame,
    model_exp: pd.DataFrame,
    lib_df: pd.DataFrame,
    gene_means: pd.DataFrame,
    output_pdf: str = "../estimates/hsim/nonDE_comparisons.pdf",
    max_genes: int = None
):
    """
    For every gene *not* marked DE, plot true vs. model expected counts
    and write each scatter to its own page in a PDF.
    If max_genes is set and there are more non-DE genes, randomly sample that many.
    """
    # 1) Identify non-DE genes
    non_de = gene_means.index[~gene_means["DE"]]
    genes = list(non_de)
    total = len(genes)
    if total == 0:
        print("No non-DE genes found.")
        return

    # 2) possibly subsample
    if max_genes is not None and total > max_genes:
        genes = random.sample(genes, max_genes)
        print(f"Sampling {max_genes} / {total} non-DE genes for plotting.")

    # 3) Open PDF and plot each gene
    with PdfPages(output_pdf) as pdf:
        for gene in genes:
            df = pd.DataFrame({
                "true_count"  : true_exp[gene].to_numpy(),
                "model_count" : model_exp[gene].to_numpy(),
                "identity"    : lib_df["cell_id"].to_numpy()
            })

            fig, ax = plt.subplots(figsize=(6,6))
            sns.scatterplot(
                data=df,
                x="true_count",
                y="model_count",
                hue="identity",
                palette="tab10",
                alpha=0.6,
                ax=ax
            )
            # y=x reference line
            m, M = df[["true_count","model_count"]].min().min(), \
                   df[["true_count","model_count"]].max().max()
            ax.plot([m, M], [m, M], "--", color="gray")

            ax.set_title(f"Gene {gene} (non-DE)")
            ax.set_xlabel("True Expected Count")
            ax.set_ylabel("Estimated Exp Count")
            ax.legend(title="Cell Identity", loc="best")

            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved non-DE comparisons for {len(genes)} genes to {output_pdf}")


def save_DE_group_comparisons_pdf(
    true_exp: pd.DataFrame,
    model_exp: pd.DataFrame,
    lib_df: pd.DataFrame,
    gene_means: pd.DataFrame,
    target: str,
    output_pdf: str
):
    """
    For every gene that is DE in `target` (i.e. target in DE_group split),
    plot true vs. model expected counts and write each scatter to its own
    page in a single PDF.
    """
    # 1) Identify DE genes for this target
    de_mask = gene_means["DE_group"].apply(
        lambda s: isinstance(s, str) and (target in s.split(","))
    )
    de_genes = gene_means.index[de_mask]
    if de_genes.empty:
        print(f"No DE genes found for group {target}.")
        return

    # 2) Open the multi‐page PDF
    with PdfPages(output_pdf) as pdf:
        for gene in de_genes:
            # build a small DataFrame by row‐order
            df = pd.DataFrame({
                "true_count" : true_exp[gene].to_numpy(),
                "model_count": model_exp[gene].to_numpy(),
                "identity"   : lib_df["cell_id"].to_numpy()
            })

            # 3) Make the scatter
            fig, ax = plt.subplots(figsize=(6,6))
            sns.scatterplot(
                data=df,
                x="true_count",
                y="model_count",
                hue="identity",
                palette="tab10",
                alpha=0.6,
                ax=ax
            )
            # dashed y=x
            m, M = df[["true_count","model_count"]].min().min(), \
                   df[["true_count","model_count"]].max().max()
            ax.plot([m, M], [m, M], "--", color="gray")

            ax.set_title(f"Gene {gene} (DE in {target})")
            ax.set_xlabel("True Expected Count")
            ax.set_ylabel("Estimated Exp Count")
            ax.legend(title="Cell Identity", loc="best")

            # 4) Save this page
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved DE‐{target} comparisons for {len(de_genes)} genes to {output_pdf}")

def load_beta_matrix(path="../estimates/hsim/HLDA_beta.csv"):
    beta = pd.read_csv(path, index_col=0)
    # beta.index are genes, beta.columns are topics
    beta.columns = ['root','AB_parent','A','B','C']
    beta.columns = [c.strip() for c in beta.columns]
    beta.index   = beta.index.str.strip()
    return beta

def load_theta_matrix(path="../estimates/hsim/HLDA_theta.csv"):
    # theta: cells × topics
    theta = pd.read_csv(path, index_col=0)
    return theta

def load_library_sizes(path="../estimates/hsim/library_sizes.csv"):
    """
    Expects columns: cell_identity, sampled_library_size, true_library_size
    Rows in the same order as counts_df.
    """
    lib = pd.read_csv(path)
    assert {'cell_id','sampled_library_size','true_library_size'}.issubset(lib.columns)
    return lib

# ———————————————————————————————————————————————
# 2) Compute true expected counts
# ———————————————————————————————————————————————
def compute_true_expected_counts(lib_df, gene_means_df):
    genes      = gene_means_df.index.tolist()
    n_cells    = len(lib_df)
    true_exp   = pd.DataFrame(np.zeros((n_cells,len(genes))), columns=genes)
    for i, row in lib_df.iterrows():
        grp   = row['cell_id']
        mu    = gene_means_df[grp]
        frac  = mu / mu.sum()
        true_exp.iloc[i] = row['true_library_size'] * frac.values
    return true_exp

# ———————————————————————————————————————————————
# 3) Compute model expected counts
# ———————————————————————————————————————————————
def compute_estimated_counts(lib_df, theta_df, beta_df):
    # ensure theta rows align with lib_df rows
    theta_vals = theta_df.values  # (n_cells × K)
    beta_vals  = beta_df.T.values # (K × n_genes)
    P          = theta_vals.dot(beta_vals)  # (n_cells × n_genes)
    model_exp  = pd.DataFrame(P, columns=beta_df.index)
    # scale by each cell's sampled lib
    model_exp = model_exp.mul(lib_df['sampled_library_size'].values, axis=0)
    return model_exp

# ———————————————————————————————————————————————
# 4) Plot for a specific gene, colored by identity
# ———————————————————————————————————————————————
def plot_expected_comparison_for_gene(
    gene, true_exp, model_exp, cell_identity, save_path=None
):
    df = pd.DataFrame({
        'true_count' : true_exp[gene].to_numpy(),
        'model_count': model_exp[gene].to_numpy(),
        'identity'   : cell_identity
    })

    plt.figure(figsize=(6,6))
    ax = sns.scatterplot(
        data=df, x='true_count', y='model_count',
        hue='identity', palette='tab10', alpha=0.6
    )
    m, M = df[['true_count','model_count']].min().min(), df[['true_count','model_count']].max().max()
    ax.plot([m, M], [m, M], '--', color='gray')
    ax.set_xlabel("True Expected Count")
    ax.set_ylabel("Estimated Exp Count")
    ax.set_title(f"Gene {gene}: True vs. Model")
    ax.legend(title="Cell Identity")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def select_high_root_C_cells(theta_df, lib_df, threshold=0.9):
    """
    Return the integer row‐indices of cells that
    are identity 'C' and have theta['Topic_0'] > threshold.
    """
    is_C      = np.array(lib_df['cell_id'] == 'C')
    high_root = np.array(theta_df['Topic_0'] > threshold)
    return list(np.where(is_C & high_root)[0])

def plot_sum_DE_C_low_root_cells(
    counts_df: pd.DataFrame,
    true_exp: pd.DataFrame,
    theta_df: pd.DataFrame,
    lib_df: pd.DataFrame,
    gene_means_df: pd.DataFrame,
    threshold: float = 0.2,
    save_path: str = "../estimates/hsim/plots/sum_DE_C_lowroot_scatter.png"
):
    """
    For C‐cells with theta['Topic_0'] < threshold,
    sum true vs. actual counts over genes DE in C,
    then scatter‐plot those sums in one figure.
    """
    # 1) select low‐root C cells
    is_C      = np.array(lib_df["cell_id"] == "C")
    low_root  = np.array(theta_df["Topic_0"] < threshold)
    idxs      = np.where(is_C & low_root)[0].tolist()
    if not idxs:
        print(f"No C‐cells with Topic_0 < {threshold}")
        return

    # 2) DE‐in‐C genes
    de_mask = gene_means_df["DE_group"].apply(
        lambda s: isinstance(s, str) and ("C" in s.split(","))
    )
    de_genes = gene_means_df.index[de_mask].tolist()
    if not de_genes:
        raise ValueError("No DE‐in‐C genes found.")

    # 3) sums for the selected cells
    sum_true   = true_exp.iloc[idxs][de_genes].sum(axis=1).values
    sum_actual = counts_df.iloc[idxs][de_genes].sum(axis=1).values
    root_prob  = theta_df.iloc[idxs]["Topic_0"].values

    # 4) build DataFrame & plot
    df = pd.DataFrame({
        "sum_true":   sum_true,
        "sum_actual": sum_actual,
        "root_prob":  root_prob
    })

    plt.figure(figsize=(6,6))
    ax = sns.scatterplot(
        data=df,
        x="sum_true",
        y="sum_actual",
        hue="root_prob",
        palette="viridis",
        alpha=0.8
    )
    # 1:1 line
    m, M = df[["sum_true","sum_actual"]].min().min(), df[["sum_true","sum_actual"]].max().max()
    ax.plot([m, M], [m, M], "--", color="gray")

    ax.set_xlabel("Sum True Expected Counts (DE in C)")
    ax.set_ylabel("Sum Actual Counts (DE in C)")
    ax.set_title(f"Low‐root C Cells (Topic_0 < {threshold})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved sum‐scatter for low‐root C cells to {save_path}")


def plot_high_root_C_cells(
    counts_df,      # DataFrame of raw counts, rows in same order as lib_df
    theta_df,       # DataFrame of theta proportions, same row order
    lib_df,         # DataFrame with 'cell_identity', 'sampled_library_size', 'true_library_size'
    gene_means_df,  # DataFrame of gene means + DE metadata
    true_exp,       # DataFrame of true expected counts
    output_pdf="../estimates/hsim/high_root_C_cells.pdf",
    threshold=0.9
):
    """
    For each cell of identity C with theta['Topic_0'] > threshold,
    scatter true_exp vs. actual counts for DE-C genes,
    one page per cell in a PDF.
    """
    # 1) find qualifying cells
    idxs = select_high_root_C_cells(theta_df, lib_df, threshold)
    if not idxs:
        print(f"No C‐cells with Topic_0 > {threshold}")
        return

    # 2) find genes DE in C
    de_genes = gene_means_df.index[
        gene_means_df["DE_group"].apply(
            lambda s: isinstance(s, str) and "C" in s.split(",")
        )
    ].tolist()
    if not de_genes:
        print("No DE‐in‐C genes found.")
        return

    # 3) Plot one page per cell
    with PdfPages(output_pdf) as pdf:
        for i in idxs:
            # pull the root proportion by position
            root_prob   = theta_df.iloc[i]['Topic_0']
            # true expected counts for DE-C genes (by position then by column)
            true_vals   = true_exp.iloc[i][de_genes].astype(float).values
            # observed counts for those genes
            actual_vals = counts_df.iloc[i][de_genes].astype(float).values

            fig, ax = plt.subplots(figsize=(6,6))
            sns.scatterplot(
                x=true_vals,
                y=actual_vals,
                alpha=0.7,
                ax=ax
            )
            # dashed y=x
            m, M = min(true_vals.min(), actual_vals.min()), max(true_vals.max(), actual_vals.max())
            ax.plot([m, M], [m, M], "--", color="gray")

            ax.set_title(f"Cell #{i} (C, root={root_prob:.2f})")
            ax.set_xlabel("True Expected Count (DE in C)")
            ax.set_ylabel("Actual Observed Count")
            plt.tight_layout()

            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved high‐root C cell plots ({len(idxs)} pages) to {output_pdf}")

def plot_sum_DE_C_high_root_cells(
    counts_df: pd.DataFrame,
    true_exp: pd.DataFrame,
    theta_df: pd.DataFrame,
    lib_df: pd.DataFrame,
    gene_means_df: pd.DataFrame,
    threshold: float = 0.9,
    save_path: str = "../estimates/hsim/sum_DE_C_highroot_scatter.png"
):
    """
    For C‐cells with theta['Topic_0'] > threshold, sum true vs actual counts
    over genes DE in C, then scatter‐plot those sums in one figure.
    """
    # 1) select high‐root C cells
    is_C      = np.array(lib_df["cell_id"] == "C")
    high_root = np.array(theta_df["Topic_0"] > threshold)
    idxs      = np.where(is_C & high_root)[0].tolist()
    
    if not idxs:
        print(f"No C‐cells with Topic_0 > {threshold}")
        return
    
    # 2) DE‐in‐C genes
    de_mask = gene_means_df["DE_group"].apply(
        lambda s: isinstance(s, str) and ("C" in s.split(","))
    )
    de_genes = gene_means_df.index[de_mask].tolist()
    if not de_genes:
        raise ValueError("No DE‐in‐C genes found.")
    
    # 3) compute per‐cell sums for the selected cells
    sum_true   = true_exp.iloc[idxs][de_genes].sum(axis=1).values
    sum_actual = counts_df.iloc[idxs][de_genes].sum(axis=1).values
    
    # optional: root proportions for coloring
    root_prob = theta_df.iloc[idxs]["Topic_0"].values
    
    # 4) build plot DataFrame
    df = pd.DataFrame({
        "sum_true":   sum_true,
        "sum_actual": sum_actual,
        "root_prob":  root_prob
    })
    
    # 5) scatter
    plt.figure(figsize=(6,6))
    ax = sns.scatterplot(
        data=df,
        x="sum_true",
        y="sum_actual",
        hue="root_prob",
        palette="viridis",
        alpha=0.8
    )
    # Add the 1:1 reference line
    m, M = df[["sum_true","sum_actual"]].min().min(), df[["sum_true","sum_actual"]].max().max()
    ax.plot([m, M], [m, M], "--", color="gray")

    ax.set_xlabel("Sum True Expected Counts (DE in C)")
    ax.set_ylabel("Sum Actual Counts (DE in C)")
    ax.set_title(f"High‐root C Cells (Topic_0 > {threshold})")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved sum‐scatter for high‐root C cells to {save_path}")

# ———————————————————————————————————————————————
# Example usage
# ———————————————————————————————————————————————
if __name__ == "__main__":
    # Load your inputs
    counts_df  = pd.read_csv("../data/hsim/simulated_counts.csv")
    lib_df     = load_library_sizes("../data/hsim/library_sizes.csv")
    gene_means = load_gene_means("../data/hsim/gene_means_matrix.csv")
    beta_df    = load_beta_matrix("../estimates/hsim/HLDA_beta.csv")
    theta_df   = load_theta_matrix("../estimates/hsim/HLDA_theta.csv")

    # Compute expectations
    true_exp  = compute_true_expected_counts(lib_df, gene_means)
    model_exp = compute_estimated_counts(lib_df, theta_df, beta_df)

    # Identity vector from counts_df (row order matches)
    cell_identity = counts_df["cell_identity"]

    plot_sum_DE_C_high_root_cells(
        counts_df=counts_df,
        true_exp=true_exp,
        theta_df=theta_df,
        lib_df=lib_df,
        gene_means_df=gene_means,
        threshold=0.45,
        save_path="../estimates/hsim/plots/sum_DE_C_highroot_scatter.png"
    )

    plot_sum_DE_C_low_root_cells(
    counts_df=counts_df,
    true_exp=true_exp,
    theta_df=theta_df,
    lib_df=lib_df,
    gene_means_df=gene_means,
    threshold=0.2,
    save_path="../estimates/hsim/plots/sum_DE_C_lowroot_scatter.png"
    )
    plot_sum_DE_C_low_root_cells(
    counts_df=counts_df,
    true_exp=true_exp,
    theta_df=theta_df,
    lib_df=lib_df,
    gene_means_df=gene_means,
    threshold=0.45,
    save_path="../estimates/hsim/plots/sum_DE_C_lowroot_scatter_b.png"
    )

##    plot_expected_comparison_for_gene(
##        gene="Gene_1324",
##        true_exp=true_exp,
##        model_exp=model_exp,
##        cell_identity=cell_identity,
##        save_path="../estimates/hsim/plots/compare_Gene_1324_colored.png"
##    )
##    save_DE_group_comparisons_pdf(
##    true_exp, model_exp, lib_df, gene_means,
##    target="A",
##    output_pdf="../estimates/hsim/plots/DE_A_comparisons.pdf"
##    )
##    save_DE_group_comparisons_pdf(
##    true_exp, model_exp, lib_df, gene_means,
##    target="B",
##    output_pdf="../estimates/hsim/plots/DE_B_comparisons.pdf"
##    )
##    save_DE_group_comparisons_pdf(
##    true_exp, model_exp, lib_df, gene_means,
##    target="C",
##    output_pdf="../estimates/hsim/plots/DE_C_comparisons.pdf"
##    )
##    save_nonDE_comparisons_pdf(
##        true_exp=true_exp,
##        model_exp=model_exp,
##        lib_df=lib_df,
##        gene_means=gene_means,
##        output_pdf="../estimates/hsim/plots/nonDE_comparisons.pdf",
##        max_genes=2000
##    )
