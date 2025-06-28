# ---------------------------------------------------------------
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, pickle, os, random
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
# ---------------------------------------------------------------
# ROOT paths  (edit here if you move things)
EST_ROOT  = Path("../estimates/hsim")          # beta / theta / gene_means
DATA_ROOT = Path("../data/hsim")               # library_sizes.csv
SAMP_DIR  = Path("../samples/hsim")            # MCMC sample pickles   <-- UPDATED
PLOT_DIR  = EST_ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
def p(fname:str)->str: return str(PLOT_DIR / fname)

# ═══════════════════════════════════════════════════════════════
#  Loaders
# ═══════════════════════════════════════════════════════════════
def load_gene_means():
    return pd.read_csv(DATA_ROOT / "gene_means_matrix.csv", index_col=0)

def load_beta():
    beta = pd.read_csv(EST_ROOT / "NMF_beta.csv", index_col=0)
##    beta.columns = ["A", "A_var"]
    beta.columns = ['Root', 'AB', 'A', 'B', 'C']
    print(beta.columns)
    return beta

def _force_range_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.RangeIndex(len(df))
    return df

def load_theta_true():
    """
    data/hsim_A_var/theta.csv  →  DataFrame with columns ['A','A_var']
    and a clean RangeIndex.
    """
    th = pd.read_csv(DATA_ROOT / "theta.csv", index_col=False) 
    if th.columns[0].lower().startswith("unnamed"):
        th = th.drop(th.columns[0], axis=1)

    th.columns = ["A"]                 
    th["A_var"] = 1.0 - th["A"]               
    return _force_range_index(th)

def load_theta_est():

    df = pd.read_csv(EST_ROOT / "NMF_theta.csv", index_col=False)

    if df.columns[0].lower().startswith(("cell", "ident", "unnamed")) or df.iloc[:,0].dtype == object:
        df = df.drop(df.columns[0], axis=1)

    df.columns = ['Root', 'AB', 'A', 'B', 'C']
    return _force_range_index(df)

def load_library_sizes():
    return pd.read_csv(DATA_ROOT / "library_sizes.csv", index_col=0)

# ============================================================== #
# 1 ▸ build per‑cell, per‑gene “true”  and  “model” expectations #
# ============================================================== #
def build_true_expected_mixture(theta_df, gene_means, libsize=1500):
    """
    Returns a DataFrame λ_true[c, g] =
        libsize · ( θ_A[c]·μ_A_norm[g]  +  θ_Avar[c]·μ_Avar_norm[g] )

    μ_A_norm[g]  =  group1_genemean[g]      /  sum_g group1_genemean
    μ_Avar_norm  =  prog_genemean_1[g]      /  sum_g prog_genemean_1
    """
    muA   = gene_means["A"].values
    muVar = gene_means["A_var"].values
    muA  /= muA.sum()
    muVar /= muVar.sum()

    mix = (theta_df["A"].values[:, None]   * muA[None, :] +
           theta_df["A_var"].values[:, None] * muVar[None, :])

    lam = libsize * mix
    return pd.DataFrame(lam, index=theta_df.index, columns=gene_means.index)


def build_model_expected_mixture(counts_df, theta_df, beta_df):
    """
    λ_est[c, g] =  ( total_counts_in_cell c ) · P_est[c, g]
    where P_est = θ_est · β   (rows sum to 1).
    """
    total_counts = counts_df.sum(axis=1).values           # (n_cells,)
    P            = theta_df.values.dot(beta_df.T.values)  # (cells × genes)
    lam          = total_counts[:, None] * P
    return pd.DataFrame(lam, index=theta_df.index, columns=beta_df.index)

# ═══════════════════════════════════════════════════════════════
#  Differential‑expression comparisons
# ═══════════════════════════════════════════════════════════════
def compare_topic(gm,beta,target:str):
##    tag  = "A" if target=="A" else "A_var"
    tag = target
##    df   = gm.loc[gm["DE_group"]==tag,[target,"DE_factor"]].rename(
##           columns={target:f"gene_mean_{target}"})
    df = gm[[target]].copy().rename(columns={target: f"gene_mean_{target}"})
    df   = df.loc[df.index.intersection(beta.index)]
##    other= "A_var" if target=="A" else "A"
    other = target
    df["beta_prob"] = beta.loc[df.index,target]
    df[f"beta_fold_change_vs_{other}"] = beta.loc[df.index,target]/beta.loc[df.index,other]
    df["DE_group"]=tag
    return df

def plot_fold_change(comps, outpng):
    plt.figure(figsize=(10, 4))
    for i, (topic, df) in enumerate(comps.items(), 1):
        other = "A_var" if topic == "A" else "A"

        x = df["DE_factor"]
        y = df[f"beta_fold_change_vs_{other}"]

        ax = plt.subplot(1, 2, i)
        sns.scatterplot(
            x=x,
            y=y,
            hue=df["DE_group"],          # categorical colours
            palette="Set2",
            legend=False,
            ax=ax
        )

        # axes limits – *only* numeric arrays
        m, M = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([m, M], [m, M], "--", color="gray")

        ax.set_xlabel("Simulated DE factor")
        ax.set_ylabel(f"β fold‑change vs {other}")
        ax.set_title(topic)

    plt.tight_layout()
    plt.savefig(outpng, dpi=300)
    plt.close()

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

def plot_lambda_frac(comps,outpng):
    plt.figure(figsize=(10,4))
    for i,t in enumerate(('A', 'B', 'C'),1):
        df=comps[t]; frac=df[f"gene_mean_{t}"]/df[f"gene_mean_{t}"].sum()
        ax=plt.subplot(1,3,i)
        sns.scatterplot(x=frac,y=df["beta_prob"],alpha=.7,ax=ax)
        m,M=min(frac.min(),df["beta_prob"].min()),max(frac.max(),df["beta_prob"].max())
        ax.plot([m,M],[m,M],"--",color="gray")
        ax.set_xlabel(f"{t} λ / Σλ"); ax.set_ylabel(f"P(gene|{t})"); ax.set_title(t)
    plt.tight_layout(); plt.savefig(outpng,dpi=300); plt.close()

# ═══════════════════════════════════════════════════════════════
#  θ histograms
# ═══════════════════════════════════════════════════════════════
def theta_hist(theta,outpng):
    fig,ax=plt.subplots(1,5,figsize=(10,4))
    for a,c in zip(ax,('Root', 'AB', 'A', 'B', 'C')):
        sns.histplot(theta[c],bins=30,ax=a,color="C0")
        a.set_title(c); a.set_xlabel("θ"); a.set_ylabel("Count")
    plt.tight_layout(); plt.savefig(outpng,dpi=300); plt.close()

def plot_theta_true_vs_est(theta_true, theta_est, outpng):

    plt.figure(figsize=(10,4))
    for i,col in enumerate(("A","A_var"),1):
        x = theta_true[col]
        y = theta_est[col]
        ax = plt.subplot(1,2,i)
        sns.scatterplot(x=x, y=y, alpha=.6, s=15, ax=ax)
        m,M = 0,1
        ax.plot([m,M],[m,M],"--",color="gray")
        ax.set_xlim(m,M); ax.set_ylim(m,M)
        ax.set_xlabel(f"True θ_{col}")
        ax.set_ylabel(f"Estimated θ_{col}")
    plt.tight_layout()
    plt.savefig(outpng, dpi=300)
    plt.close()

# ═══════════════════════════════════════════════════════════════
#  Expected‑count scatters
# ═══════════════════════════════════════════════════════════════
def sum_DE_Avar_by_ranges(counts_df, true_exp, theta_df, gene_means_df,
                          mask_cells, ranges, basename, outdir):

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # subset once so we don’t repeat work
    idxs = theta_df.index[mask_cells]
    if idxs.empty:
        print("No cells selected for range plots.")
        return

    de_genes = gene_means_df.index[gene_means_df["DE_group"] == "A_var"]
    if de_genes.empty:
        print("No DE‑A_var genes — skipping range plots.")
        return

    data = pd.DataFrame({
        "sum_true"   : true_exp.loc[idxs, de_genes].sum(axis=1),
        "sum_actual" : counts_df.loc[idxs, de_genes].sum(axis=1),
        "theta_Avar" : 1 - theta_df.loc[idxs, "A"]
    })

    for lo, hi in ranges:
        sub = data[(data["sum_true"] >= lo) & (data["sum_true"] <= hi)]
        if sub.empty:
            print(f"Range [{lo}, {hi}] has no cells — skipped.")
            continue

        plt.figure(figsize=(5,5))
        sc = sns.scatterplot(
            data=sub,
            x="sum_true",
            y="sum_actual",
            hue="theta_Avar",
            palette="viridis",
            alpha=.8
        )
        sc.axline((lo, lo), (hi, hi), ls="--", color="gray")

        sc.set_xlim(lo-2, hi+2)
        sc.set_xlabel("Σ expected counts (DE in A_var)")
        sc.set_ylabel("Σ actual counts (DE in A_var)")
        sc.set_title(f"{basename}  —  sum_true in [{lo}, {hi}]")
        sc.legend(title="θ_A_var", loc="upper left", fontsize="x-small")

        plt.tight_layout()
        outpng = outdir / f"sum_DE_Avar_true_{lo}_{hi}.png"
        plt.savefig(outpng, dpi=300)
        plt.close()
        print("→", outpng)
        
def _sum_DE_Avar(counts_df, true_exp, theta_df, gene_means_df,
                 mask_cells, title, outpng):
    """helper shared by high/low wrappers"""
    idxs = theta_df.index[mask_cells]
    if idxs.empty:
        print(f"No cells for {outpng}")
        return

    # genes DE in A_var
    de_genes = gene_means_df.index[gene_means_df["DE_group"] == "A_var"]
    if de_genes.empty:
        print("No genes with DE_group == 'A_var'")
        return

    # per‑cell sums
    sum_true   = true_exp.loc[idxs, de_genes].sum(axis=1)
    sum_actual = counts_df.loc[idxs, de_genes].sum(axis=1)
    avar_prob  = 1 - theta_df.loc[idxs, "A"]

    df = pd.DataFrame({
        "sum_true": sum_true,
        "sum_actual": sum_actual,
        "Estimated_theta_A_var": avar_prob
    })

    plt.figure(figsize=(6,6))
    sc = sns.scatterplot(data=df,
                         x="sum_true",
                         y="sum_actual",
                         hue="Estimated_theta_A_var",
                         palette="viridis",
                         alpha=0.8)

    m,M = df[["sum_true","sum_actual"]].min().min(), df[["sum_true","sum_actual"]].max().max()
    sc.plot([m,M],[m,M],"--",color="gray")
    sc.set_xlabel("Σ expected counts (DE in A_var)")
    sc.set_ylabel("Σ actual counts * (theta * beta) (DE in A_var)")
    sc.set_title(title)

    plt.tight_layout()
    plt.savefig(outpng, dpi=300)
    plt.close()
    print("→", outpng)

def plot_sum_DE_Avar_high_cells(counts_df, true_exp, theta_df, gene_means_df,
                                hi=0.8,
                                outpng="../estimates/hsim_A_var/plots/sum_DE_Avar_high.png"):
    mask = theta_df["A_var"] > hi
    _sum_DE_Avar(counts_df, true_exp, theta_df, gene_means_df,
                 mask, f"High A_var cells (θ_A_var > {hi})", outpng)

def plot_sum_DE_Avar_low_cells(counts_df, true_exp, theta_df, gene_means_df,
                               lo=0.2,
                               outpng="../estimates/hsim_A_var/plots/sum_DE_Avar_low.png"):
    mask = theta_df["A_var"] < lo
    _sum_DE_Avar(counts_df, true_exp, theta_df, gene_means_df,
                 mask, f"Low A_var cells (θ_A_var < {lo})", outpng)

def plot_sum_DE_Avar_all_cells(counts_df, true_exp, theta_df, gene_means_df,
                               outpng="../estimates/hsim_A_var/plots/sum_DE_Avar_all.png"):
    mask_all = np.ones(len(theta_df), dtype=bool)   # keep every row
    _sum_DE_Avar(counts_df,
                 true_exp,
                 theta_df,
                 gene_means_df,
                 mask_cells=mask_all,
                 title="All cells (DE genes in A_var)",
                 outpng=outpng)

# ═══════════════════════════════════════════════════════════════
#  Geweke diagnostics
# ═══════════════════════════════════════════════════════════════
def load_chain_samples(key):
    arr=[]
    for fn in sorted(SAMP_DIR.glob("*.pkl")):
        with open(fn,"rb") as f: arr.append(pickle.load(f)[key])
    return np.stack(arr)

def geweke_z(chain,first=0.1,last=0.5):
    n=len(chain); n1=max(1,int(first*n)); n2=max(1,int(last*n))
    a,b=chain[:n1],chain[-n2:]; v1,v2=a.var(ddof=1),b.var(ddof=1)
    return (a.mean()-b.mean())/np.sqrt(v1/n1+v2/n2)

def geweke_plots(samples,key,out_hist,out_pdf,thr=3,max_chains=500):
    ns,nc,nt=samples.shape
    z=[geweke_z(samples[:,i,k]) for i in range(nc) for k in range(nt)]
    plt.figure(figsize=(6,4)); plt.hist(z,bins=50,color='steelblue',edgecolor='black')
    plt.axvline(0,color='red',ls='--'); plt.xlabel("Geweke z"); plt.ylabel("count")
    plt.title(f"{key}: Geweke scores"); plt.tight_layout(); plt.savefig(out_hist,dpi=300); plt.close()
    hi=[(i,k,zv) for (i,k),zv in np.ndenumerate(np.array(z).reshape(nc,nt)) if abs(zv)>thr]
    if not hi: return
    if len(hi)>max_chains: hi=random.sample(hi,max_chains)
    with PdfPages(out_pdf) as pdf:
        for i,k,zv in hi:
            fig,ax=plt.subplots(figsize=(8,3)); ax.plot(samples[:,i,k]); ax.set_title(f"{key} cell{i} topic{k} z={zv:.2f}")
            ax.set_xlabel("iter"); ax.set_ylabel("val"); plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

# ═══════════════════════════════════════════════════════════════
#  Main runner
# ═══════════════════════════════════════════════════════════════
def run_all():
    gm   = load_gene_means()
    beta = load_beta()
    th   = load_theta_est()
##    true_th = load_theta_true()
##    lib  = load_library_sizes()
##    counts_df = _force_range_index(pd.read_csv(DATA_ROOT / "simulated_counts.csv"))
##    counts_df = counts_df.drop(counts_df.columns[0], axis=1)

    comps = {t:compare_topic(gm,beta,t) for t in ['A', 'B', 'C']}
##    plot_fold_change(comps, p("fold_change_A_vs_Avar.png"))
    plot_lambda_frac(comps, p("NMF_lambda_frac_vs_prob.png"))
##
    theta_hist(th, p("NMF_theta_histograms.png"))
##    plot_theta_true_vs_est(
##        true_th,
##        th,
##        outpng = p("theta_true_vs_est.png")
##    )
##
##    true_mix  = build_true_expected_mixture(true_th, gm, libsize=1500)
##    est_mix   = build_model_expected_mixture(counts_df, th, beta)
##
##
##    plot_sum_DE_Avar_all_cells(
##    counts_df,
##    true_mix,        # or whatever true‑expected matrix you built
##    th,
##    gm,
##    outpng=p("sum_DE_Avar_all.png")
##    )
##
##    zoom_ranges = [(300, 305), (305, 310), (400, 405)]
##
##    mask_all = np.ones(len(th), dtype=bool)   # or any other mask
##    sum_DE_Avar_by_ranges(
##        counts_df, true_mix, th, gm,
##        mask_cells = mask_all,
##        ranges     = zoom_ranges,
##        basename   = "All cells",
##        outdir     = PLOT_DIR / "Avar_sums_by_range"
##    )
    

##    for t, df in comps.items():
##        if not df.empty:
##            df.to_csv(EST_ROOT / f"HLDA_fold_change_{t}.csv",
##                      index=False)
##            
##    nonDE_df = build_nonDE_fold_change(beta, gm)
##    if not nonDE_df.empty:
##        nonDE_df.to_csv(EST_ROOT / f"HLDA_fold_change_nonDE.csv",
##                        index=False)
##        plot_nonDE_fold_change(nonDE_df,
##                               PLOT_DIR / f"HLDA_fold_change_nonDE.png")
##    plot_fold_change(comps, PLOT_DIR / f"HLDA_fold_change.png")


##    if SAMP_DIR.exists():
##        for key in ("D","A"):
##            try:
##                arr = load_chain_samples(key)
##                geweke_plots(arr,key,
##                             out_hist=p(f"geweke_hist_{key}.png"),
##                             out_pdf =p(f"high_geweke_traces_{key}.pdf"))
##            except Exception as e:
##                print(f"Skip {key}: {e}")

if __name__ == "__main__":
    run_all()
