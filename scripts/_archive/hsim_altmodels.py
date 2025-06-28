#!/usr/bin/env python3
# =============================================================
#   hsim_altmodels.py
#   LDA & NMF (2 topics) diagnostics for the A / A_var dataset
# =============================================================
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation, NMF

# -------------------------------------------------------------
#  Paths
# -------------------------------------------------------------
DATA_ROOT = Path("../data/hsim_A_var")
EST_ROOT  = Path("../estimates/hsim_A_var")
COUNTS_FN = DATA_ROOT / "simulated_counts.csv"        # cells × genes (raw counts)
GM_FN     = DATA_ROOT / "gene_means_matrix.csv"
THETA_TRUE_FN = DATA_ROOT / "theta.csv"     # single column = θ_A

# -------------------------------------------------------------
#  Load helpers
# -------------------------------------------------------------
def load_counts():
    return pd.read_csv(COUNTS_FN, index_col=0)

def load_gene_means():
    gm = pd.read_csv(GM_FN, index_col=0)
    gm.index = gm.index.str.strip()
    return gm

def load_theta_true():
    th = pd.read_csv(THETA_TRUE_FN, index_col=False)
    if th.columns[0].lower().startswith("unnamed"):
        th = th.drop(th.columns[0], axis=1)
    th.columns = ["A"]
    th["A_var"] = 1.0 - th["A"]
    th.index = pd.RangeIndex(len(th))
    return th

def build_true_expected_mixture(theta_true, gene_means, libsize=1500):
    muA   = gene_means["A"].values
    muVar = gene_means["A_var"].values
    muA  /= muA.sum()
    muVar /= muVar.sum()
    mix = (theta_true["A"].values[:, None] * muA[None, :] +
           theta_true["A_var"].values[:, None] * muVar[None, :])
    lam = libsize * mix
    return pd.DataFrame(lam, index=theta_true.index, columns=gene_means.index)

# -------------------------------------------------------------
#  Fit 2‑topic model  →  θ & β mapped to ['A','A_var']
# -------------------------------------------------------------
# -------------------------------------------------------------
#  2-topic model  →  θ, β mapped to ['A', 'A_var']
# -------------------------------------------------------------
def fit_two_topic_model(counts_df, gene_means, model="lda", seed=0):
    """
    Returns
    --------
    theta_df : DataFrame (cells × 2)  columns = ["A", "A_var"]
    beta_df  : DataFrame (genes × 2)  columns = ["A", "A_var"]
    """
    X     = counts_df.values           # raw counts
    genes = counts_df.columns

    # ---------- fit model & get raw factors ------------------
    if model == "lda":
        mdl = LatentDirichletAllocation(
            n_components=2,
            random_state=seed,
            learning_method="batch"
        )
        theta = mdl.fit_transform(X)        # (N × 2)
        beta  = mdl.components_             # (2 × G)

    elif model == "nmf":
        mdl = NMF(
            n_components=2,
            init="nndsvda",
            beta_loss="kullback-leibler",   # Poisson–KL ≈ topic-model loss
            solver="mu",
            max_iter=5000,
            random_state=seed
        )
        theta = mdl.fit_transform(X)        # (N × 2)  = W
        beta  = mdl.components_             # (2 × G)  = H
    else:
        raise ValueError("model must be 'lda' or 'nmf'")

    # ---------- make θ and β probability rows ----------------
    theta = theta / theta.sum(axis=1, keepdims=True)          # each row sums to 1
    beta  = beta  / beta.sum(axis=1, keepdims=True)           # each topic sums to 1

    # ---------- map topics to A / A_var -----------------------
    de_Avar = gene_means.index[gene_means["DE_group"] == "A_var"]
    idx     = genes.get_indexer(de_Avar)          # positions of DE-A_var genes
    score   = beta[:, idx].sum(axis=1)            # topic weight on those genes
    idx_Avar = score.argmax()                     # topic with more A_var mass
    idx_A    = 1 - idx_Avar

    # ---------- build labelled DataFrames --------------------
    theta_df = pd.DataFrame(
        np.column_stack([theta[:, idx_A], theta[:, idx_Avar]]),
        index=counts_df.index,
        columns=["A", "A_var"]
    )

    beta_df = pd.DataFrame(
        np.column_stack([beta[idx_A].T, beta[idx_Avar].T]),
        index=genes,
        columns=["A", "A_var"]
    )

    return theta_df, beta_df

# -------------------------------------------------------------
#  Model expected λ  (lib_obs · θβ)
# -------------------------------------------------------------
def build_model_expected(counts_df, theta_df, beta_df):
    lib_obs = counts_df.sum(axis=1).values
    P_est   = theta_df.values.dot(beta_df.T.values)
    lam_est = lib_obs[:, None] * P_est
    return pd.DataFrame(lam_est, index=theta_df.index, columns=beta_df.index)

# -------------------------------------------------------------
#  Plot helpers
# -------------------------------------------------------------
# ------------------------------------------------------------ #
#  Plot helpers  (with legends)                                #
# ------------------------------------------------------------
##def compare_topic(beta, gene_means, target):
##    other = "A_var" if target == "A" else "A"
##    mask  = gene_means["DE_group"] == target
##    df = gene_means.loc[mask, ["DE_factor"]].copy()
##    df["gene"]    = df.index
##    df["beta_fc"] = beta.loc[df.index, target] / beta.loc[df.index, other]
##    return df.reset_index(drop=True)

def compare_topic(gm,beta,target:str):
    tag  = "A" if target=="A" else "A_var"
##    df   = gm.loc[gm["DE_group"]==tag,[target,"DE_factor"]].rename(
##           columns={target:f"gene_mean_{target}"})
    df = gm[[target]].copy().rename(columns={target: f"gene_mean_{target}"})
    df   = df.loc[df.index.intersection(beta.index)]
    other= "A_var" if target=="A" else "A"
    df["beta_prob"] = beta.loc[df.index,target]
    df[f"beta_fold_change_vs_{other}"] = beta.loc[df.index,target]/beta.loc[df.index,other]
    df["DE_group"]=tag
    return df

def build_fold_change_tables(beta_df, gene_means):
    def _one_target(target):
        other = "A_var" if target == "A" else "A"
        mask  = gene_means["DE_group"] == target
        if not mask.any():                 # no DE genes for this target
            return pd.DataFrame()

        df = gene_means.loc[mask, ["DE_factor"]].copy()
        df["gene"]          = df.index
        df[f"beta_prob_{target}"] = beta_df.loc[df.index, target]
        df[f"beta_prob_{other}"]  = beta_df.loc[df.index, other]
        df["beta_fc"]       = df[f"beta_prob_{target}"] / df[f"beta_prob_{other}"]
        df["target"]        = target
        return df.reset_index(drop=True)

    return {t: _one_target(t) for t in ("A", "A_var")}

def build_nonDE_fold_change(beta_df, gene_means):
    mask = ~gene_means["DE_group"].isin(["A", "A_var"])
    if not mask.any():
        return pd.DataFrame()

    idx = gene_means.index[mask]
    df = pd.DataFrame({
        "gene"          : idx,
        "beta_prob_A"   : beta_df.loc[idx, "A"],
        "beta_prob_Avar": beta_df.loc[idx, "A_var"],
    })
    # ratio
    df["beta_fc"] = df["beta_prob_Avar"] / df["beta_prob_A"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["beta_fc"])
    return df.reset_index(drop=True)

def plot_lambda_frac(comps,outpng):
    plt.figure(figsize=(10,4))
    for i,t in enumerate(("A","A_var"),1):
        df=comps[t]; frac=df[f"gene_mean_{t}"]/df[f"gene_mean_{t}"].sum()
        ax=plt.subplot(1,2,i)
        sns.scatterplot(x=frac,y=df["beta_prob"],alpha=.7,ax=ax)
        m,M=min(frac.min(),df["beta_prob"].min()),max(frac.max(),df["beta_prob"].max())
        ax.plot([m,M],[m,M],"--",color="gray")
        ax.set_xlabel(f"{t} λ / Σλ"); ax.set_ylabel(f"P(gene|{t})"); ax.set_title(t)
    plt.tight_layout(); plt.savefig(outpng,dpi=300); plt.close()


# ------------------------------------------------------------
#  Histogram of β_Avar / β_A   (non‑DE genes)
# ------------------------------------------------------------
def plot_nonDE_fold_change(df, outpng, jitter_width=0.02):
    """
    Parameters
    ----------
    df           : DataFrame with column 'beta_fc' (finite values)
    outpng       : Path to save PNG
    jitter_width : half‑width of uniform jitter around x=1
    """
    if df.empty:
        print("No non‑DE genes to plot.")
        return

    # x values: 1 ± jitter
    x_jitter = 1.0 + np.random.uniform(
        low=-jitter_width, high=jitter_width, size=len(df)
    )
    y = df["beta_fc"].values

    plt.figure(figsize=(5, 5))
    plt.scatter(x_jitter, y, s=20, alpha=0.6)
    plt.axhline(1.0, ls="--", color="red", label="y = 1")

    plt.xlim(1 - 1.5 * jitter_width, 1 + 1.5 * jitter_width)
    plt.xlabel("True fold change (≈1) with jitter")
    plt.ylabel("Estimated β fold change")
    plt.title("Non DE genes fold change scatter")
    plt.legend(loc="upper left", fontsize="small")

    plt.tight_layout()
    plt.savefig(outpng, dpi=300)
    plt.close()

# ------------------------------------------------------------
#  Plot fold‑change (takes the dictionary returned above)
# ------------------------------------------------------------
def plot_fold_change(comps, outpng):
    avail = [t for t, d in comps.items() if not d.empty]
    if not avail:
        print("No DE genes for any target – fold‑change plot skipped")
        return

    plt.figure(figsize=(4 * len(avail), 4))

    for i, t in enumerate(avail, 1):
        d = comps[t]
        ax = plt.subplot(1, len(avail), i)

        # scatter
        sns.scatterplot(data=d, x="DE_factor", y="beta_fc",
                        s=20, alpha=.8, legend=False, ax=ax)

        # axis labels / title
        ax.set_xlabel("DE factor")
        ax.set_ylabel("β fold change vs other")
        ax.set_title(t)

        # make y = x line span the full axes
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        lo = max(min(xmin, ymin), 0)   # allow for zero lower-bound
        hi = min(max(xmax, ymax), max(xmax, ymax))  # upper bound across axes
        ax.plot([lo, hi], [lo, hi], "--", color="gray", label="y = x")

        ax.legend(fontsize="x-small")
    plt.tight_layout(); plt.savefig(outpng, dpi=300); plt.close()

def plot_theta_true_vs_est(theta_true, theta_est, outpng):
    plt.figure(figsize=(8, 4))

    for i, col in enumerate(["A", "A_var"], 1):
        x, y = theta_true[col], theta_est[col]

        ax = plt.subplot(1, 2, i)
        ax.scatter(x, y, s=8, alpha=.6)
        ax.plot([0, 1], [0, 1], "--", color="gray", label="y = x")

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel(f"true {col}")
        ax.set_ylabel(f"est {col}")
        ax.set_title(col)
        ax.legend(fontsize="small")

    plt.tight_layout()
    plt.savefig(outpng, dpi=300)
    plt.close()

def plot_sum_true_vs_model(true_lam,
                           est_lam,
                           gm,
                           theta_est,          # ← NEW
                           outpng):
    # genes that are DE in A_var
    genes = gm.index[gm["DE_group"] == "A_var"]
    s_true = true_lam[genes].sum(axis=1)
    s_est  =  est_lam[genes].sum(axis=1)
    colour = theta_est["A_var"]            # colour = estimated θ_A_var

    plt.figure(figsize=(5, 5))
    sc = plt.scatter(s_true, s_est, c=colour, cmap="viridis",
                     s=10, alpha=.6)

    m, M = min(s_true.min(), s_est.min()), max(s_true.max(), s_est.max())
    plt.plot([m, M], [m, M], "--", color="gray", label="y = x")

    plt.xlabel("Σ expected counts (DE in A_var)")
    plt.ylabel("Σ actual counts * (theta * beta)")
    plt.title("All cells")
    plt.legend(loc="upper left", fontsize="small")

    cbar = plt.colorbar(sc)
    cbar.set_label("Estimated_theta_A_var")              # ← legend for colour

    plt.tight_layout()
    plt.savefig(outpng, dpi=300)
    plt.close()

def sum_DE_Avar_by_ranges(counts_df, true_mix, theta_est, gm,
                          ranges, title_stub, outdir):
    """
    Produce one scatter for each (lo,hi) in `ranges`
    using all cells, colouring by estimated θ_A_var.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # genes DE in A_var
    genes_Avar = gm.index[gm["DE_group"] == "A_var"]
    if genes_Avar.empty:
        print("No DE‑A_var genes — range plots skipped.")
        return

    s_true = true_mix[genes_Avar].sum(axis=1)

    # ------------------------------------------------------------
    # 2) Per‑cell Σ *model* expected counts for the same genes
    #    (need NumPy to avoid pandas multi‑dim indexing error)
    # ------------------------------------------------------------
    lib_obs = counts_df.sum(axis=1).to_numpy()               # (N,)
    theta_np = theta_est.to_numpy()                          # (N × 2)
    beta_np  = gm[["A", "A_var"]].to_numpy().T              # (2 × G)

    # mixture probability for every (cell, gene)
    P_est = theta_np @ beta_np                               # (N × G)

    # scale by observed library size
    lam_est = lib_obs[:, None] * P_est                       # NumPy OK

    # pick the columns that correspond to DE‑A_var genes
    gene_pos = counts_df.columns.get_indexer(genes_Avar)
    s_est = lam_est[:, gene_pos].sum(axis=1)

    # ------------------------------------------------------------
    # 3) DataFrame used for plotting
    # ------------------------------------------------------------
    df_base = pd.DataFrame({
        "sum_true"   : s_true.values,              # Series → ndarray
        "sum_est"    : s_est,
        "theta_Avar" : theta_est["A_var"].values
    }, index=theta_est.index)
    for lo, hi in ranges:
        df = df_base[(df_base["sum_true"] >= lo) & (df_base["sum_true"] <= hi)]
        if df.empty:
            continue

        plt.figure(figsize=(5, 5))

        sc = plt.scatter(
            df["sum_true"],               # ← df, not sub
            df["sum_est"],
            c=df["theta_Avar"],
            cmap="viridis",
            s=18, alpha=.7
        )
        plt.axline((lo, lo), (hi, hi), ls="--", color="gray")

        plt.xlim(lo - 2, hi + 2)
        plt.xlabel("Σ expected (DE A_var)")
        plt.ylabel("Σ model counts (DE A_var)")
        plt.title(f"{title_stub}  [{lo}, {hi}]")

        cbar = plt.colorbar(sc)
        cbar.set_label("θ̂ A_var")

        plt.tight_layout()
        outpng = outdir / f"sum_DE_Avar_true_{lo}_{hi}.png"
        plt.savefig(outpng, dpi=300)
        plt.close()
# ------------------------------------------------------------ #
#  Driver for one model (saves θ & β CSV)                      #
# ------------------------------------------------------------
def run_model(model_name):
    print(f"\n*** {model_name.upper()} ***")
    out_dir = EST_ROOT / model_name.upper()
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    theta_est, beta_est = fit_two_topic_model(counts_df, gm, model=model_name)
    theta_est.to_csv(out_dir / f"{model_name.upper()}_theta.csv")
    beta_est .to_csv(out_dir / f"{model_name.upper()}_beta.csv")

##    est_lam = build_model_expected(counts_df, theta_est, beta_est)
##    comps = build_fold_change_tables(beta_est, gm)
##    for t, df in comps.items():
##        if not df.empty:
##            df.to_csv(out_dir / f"{model_name.upper()}_fold_change_{t}.csv",
##                      index=False)
##            
##    nonDE_df = build_nonDE_fold_change(beta_est, gm)
##    if not nonDE_df.empty:
##        nonDE_df.to_csv(out_dir / f"{model_name.upper()}_fold_change_nonDE.csv",
##                        index=False)
##        plot_nonDE_fold_change(nonDE_df,
##                               plot_dir / f"{model_name}_fold_change_nonDE.png")
##    plot_fold_change(comps, plot_dir / f"{model_name}_fold_change.png")
##
##    plot_theta_true_vs_est(theta_true, theta_est,
##                           plot_dir / f"{model_name}_theta_true_vs_est.png")
##
##
##    plot_sum_true_vs_model(true_mix, est_lam, gm, theta_est,plot_dir / f"{model_name}_sum_true_vs_model.png")
##    # five successive non‑overlapping bins of width 5
##    ranges = [(300, 305),
##              (350, 355),
##              (375, 380),
##              (320, 325),
##              (400, 405)]
##
##    sum_DE_Avar_by_ranges(
##        counts_df, true_mix, theta_est, gm,
##        ranges,
##        title_stub=f"{model_name.upper()} – all cells",
##        outdir=plot_dir / "ranges"
##    )
##    print(f"Finished plots & CSV for {model_name.upper()}")

    comps = {t:compare_topic(gm,beta_est,t) for t in ("A","A_var")}
##    plot_fold_change(comps, p("fold_change_A_vs_Avar.png"))
    plot_lambda_frac(comps, plot_dir / "lambda_frac_vs_prob.png")

if __name__ == "__main__":
    sns.set_style("whitegrid")

    counts_df  = load_counts()
    gm         = load_gene_means()
    theta_true = load_theta_true()
    true_mix   = build_true_expected_mixture(theta_true, gm)   # 1500‑mixture expectation

    run_model("lda")
    run_model("nmf")
