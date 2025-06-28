###!/usr/bin/env python3
from __future__ import annotations

import os
import itertools
import multiprocessing as mp
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

import time
import numpy as np
from numba import njit
from numba.typed import List as TypedList

from typing import List
import cvxpy as cp
from scipy.optimize import nnls

import ast
import os
import pickle
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.decomposition import LatentDirichletAllocation, NMF, PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


from hsim     import simulate_counts
from gibbs    import (compute_constants,
                      initialize_long_data,
                      gibbs_and_write_memmap,
                      compute_posterior_means_from_memmaps)

from pipeline import (run_sklearn_model,
                     match_topics,
                     cosine_similarity_matrix,
                     plot_pca_pair,
                     compute_pca_projection,
                     plot_geweke_histograms,
                     compare_topic,
                     plot_beta_vs_lambda,
                     plot_theta_histograms_by_identity,
                     plot_theta_true_vs_est_by_identity,
                     ensure_dir,
                     match_topics_sse,
                     )

from sse      import (simulate_cells,
                      incremental_sse)

# --- Parameter grid ---------------------------------------------------------
DE_MEANS     = [0.5, 1, 1.5, 2, 3, 4]
##TOPIC_TOTALS = [5, 6, 7]
TOPIC_TOTALS = [6]

TOPIC_STR2INT = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'V1': 4, 'V2': 5, 'V3': 6}
TOPIC_INT2STR = {v: k for k, v in TOPIC_STR2INT.items()}

IDENTITY_STR2INT = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
IDENTITY_INT2STR = {v: k for k, v in IDENTITY_STR2INT.items()}


def sim_counts(DE_MEAN):

    out_dir = Path(f"../data/ABCD_V1V2/DE_MEAN_{DE_MEAN}/")
    out_dir.mkdir(parents=True, exist_ok=True)

    counts_df, lib_df, gm, theta_df, gene_meta = simulate_counts(
        n_genes=2000,
        identities=['A', 'B', 'C', 'D'],
        activity_topics=['V1', 'V2'],
        cells_per_identity=3000,
        dirichlet_params=[8, 8, 8],
        activity_frac=0.3,
        n_de=250,
        de_lognorm_mean=DE_MEAN,
        de_lognorm_sigma=0.4,
        gamma_shape=0.6,
        gamma_rate=0.3,
        libsize_mean=1500,
        libsize_sd=0,
        random_state=42
    )

    counts_df.to_csv(out_dir / "counts.csv")
    lib_df.to_csv(out_dir / "library_sizes.csv")
    gm.to_csv(out_dir / "gene_means.csv")
    theta_df.to_csv(out_dir / "theta.csv")
    gene_meta.to_csv(out_dir / "gene_metadata.csv")

# --- Single-run function ----------------------------------------------------
def run(DE_MEAN, K_TOTAL):

############ SET DIRECTORIES AND PARAMETERS ############################################################
    DATA_DIR     = Path(f"../data/ABCD_V1V2/DE_MEAN_{DE_MEAN}")
    SAMPLE_DIR   = Path(f"../samples/ABCD_V1V2/{K_TOTAL}_topic_fit_DE_MEAN_{DE_MEAN}/")
    EST_DIR     = Path(f"../estimates/ABCD_V1V2/{K_TOTAL}_topic_fit_DE_MEAN_{DE_MEAN}/")

    for d in (SAMPLE_DIR, EST_DIR):
        os.makedirs(d, exist_ok=True)

    COUNTS_CSV  = DATA_DIR / "counts.csv"
    GM_FILE = DATA_DIR / "gene_means.csv"
    THETA_FILE = DATA_DIR / "theta.csv"
    GENE_META_FILE = DATA_DIR / "gene_metadata.csv"

    MIXTURE_THRESH = 0.01
    K = K_TOTAL
    N_GENES=2000

    INFER_TRUE_TOPICS = True if K == 6 else False
    
    GENE_MEANS = pd.read_csv(GM_FILE, index_col=0)
    GENE_META = pd.read_csv(GENE_META_FILE, index_col=0)
    
    TRUE_THETA = pd.read_csv(THETA_FILE, index_col=0)

    NUM_LOOPS   = 10000
    BURN_IN     = 2000
    THIN        =   20
    HYPERPARAMS = {'alpha_beta': .1, 'alpha_c': .1}
    n_activity = K_TOTAL - 4

    LEAF_TOPICS = ["A", "B", "C", "D"]
    TRUE_ACTIVITY_TOPICS = ["V1", "V2"]
    ACTIVITY_TOPICS = [f"V{i+1}" for i in range(n_activity)]
    TOPICS = LEAF_TOPICS + ACTIVITY_TOPICS

    TOPIC_HIERARCHY_INT = {
        IDENTITY_STR2INT[name]: 
            [TOPIC_STR2INT[name]] + [TOPIC_STR2INT[f"V{i+1}"] for i in range(n_activity)]
        for name in ['A', 'B', 'C', 'D']
    }

    N_CELLS_PER_ID     = 500       
    MIX_FRACTION       = 0.30 
    DIRICHLET_ALPHA    = (8, 8, 8) 
    SEED               = 2025

    SIM_COUNTS_CSV     = DATA_DIR / "inference_counts.csv"
    SSE_CSV_NAME       = "simulated_recon_sse.csv"  

    
############ RUN GIBBS SAMPLING ########################################################################

    df = pd.read_csv(COUNTS_CSV, index_col=0)
    cell_ids = list(df.index) 

    long_data, n_cells, n_genes = initialize_long_data(
        count_matrix=df,
        cell_identities=cell_ids,
        topic_hierarchy_int=TOPIC_HIERARCHY_INT,
        identity_str2int=IDENTITY_STR2INT,
        n_activity=n_activity,
        save_path=None
    )
##
##    consts = compute_constants(long_data, K, n_cells, n_genes)
##    A = consts['A'].copy()
##    B = consts['B'].copy()
##    D = consts['D'].copy() 
##
##    cell_idx_arr      = long_data['cell_idx'] 
##    gene_idx_arr      = long_data['gene_idx']   
##    cell_identity_arr = long_data['cell_identity'] 
##    z0                = long_data['z']      
##
##    alpha_beta = HYPERPARAMS['alpha_beta']
##    alpha_c    = HYPERPARAMS['alpha_c']
##
##    valid_list = TypedList()
##    for ident_int in sorted(TOPIC_HIERARCHY_INT):
##        vt = np.array(TOPIC_HIERARCHY_INT[ident_int], dtype=np.int32)
##        valid_list.append(vt)
##
    n_save = (NUM_LOOPS - BURN_IN + 1) // THIN

    A_chain_path = SAMPLE_DIR / "A_chain.memmap"
    D_chain_path = SAMPLE_DIR / "D_chain.memmap"
##
##    A_chain = np.memmap(
##        filename=str(A_chain_path),
##        mode="w+",
##        dtype=np.int32,
##        shape=(n_save, n_genes, K)
##    )
##    D_chain = np.memmap(
##        filename=str(D_chain_path),
##        mode="w+",
##        dtype=np.int32,
##        shape=(n_save, n_cells, K)
##    )
##
##    assert A.flags['C_CONTIGUOUS']
##    assert D.flags['C_CONTIGUOUS']
##    assert z0.flags['C_CONTIGUOUS']
##    assert A_chain.flags['C_CONTIGUOUS']
##    assert D_chain.flags['C_CONTIGUOUS']
##
##    total_tokens = cell_idx_arr.shape[0]
##    pilot_loops = 10
##    A_pilot = A.copy()
##    B_pilot = B.copy()
##    D_pilot = D.copy()
##    z_pilot = z0.copy()
##
##    t_start = time.time()
##    _ = gibbs_and_write_memmap(
##        cell_idx_arr.astype(np.int32),
##        gene_idx_arr.astype(np.int32),
##        cell_identity_arr.astype(np.int32),
##        z_pilot,
##        A_pilot, B_pilot, D_pilot,
##        valid_list,
##        alpha_beta, alpha_c,
##        np.int32(n_genes),
##        np.int32(K),
##        np.int32(pilot_loops),
##        np.int32(0),   
##        np.int32(1), 
##        A_chain[:0],  
##        D_chain[:0] 
##    )
##    t_pilot = time.time() - t_start
##
##    processed = pilot_loops * total_tokens
##    rate = processed / t_pilot
##    full_updates = NUM_LOOPS * total_tokens
##    est_seconds = full_updates / rate
##    est_hours = est_seconds / 3600.0
##
##    print(f"Pilot: {processed} token‐updates in {t_pilot:.2f}s  →  {rate:.0f} tokens/s")
##    print(f"Estimated full run ({NUM_LOOPS} sweeps, {full_updates} updates): {est_hours:.2f} h")
##
##    t0 = time.time()
##    saved_count = gibbs_and_write_memmap(
##        cell_idx_arr.astype(np.int32),
##        gene_idx_arr.astype(np.int32),
##        cell_identity_arr.astype(np.int32),
##        z0.astype(np.int32),
##        A, B, D,
##        valid_list,
##        alpha_beta, alpha_c,
##        np.int32(n_genes),
##        np.int32(K),
##        np.int32(NUM_LOOPS),
##        np.int32(BURN_IN),
##        np.int32(THIN),
##        A_chain,
##        D_chain
##    )
##    elapsed = time.time() - t0
##    print(f"✔ Sampling complete. (Elapsed time: {elapsed:.1f}s for {NUM_LOOPS} sweeps)")
##
##    if saved_count != n_save:
##        raise RuntimeError(f"Expected to save {n_save} slices, but Numba returned {saved_count}")
##
##    print(f"✔ Wrote {saved_count} thinned samples to memmaps:")
##    print(f"  • {A_chain_path}  (shape = {A_chain.shape})")
##    print(f"  • {D_chain_path}  (shape = {D_chain.shape})")
##
##    beta_df, theta_df = compute_posterior_means_from_memmaps(
##        A_chain_path=A_chain_path,
##        D_chain_path=D_chain_path,
##        n_save=n_save,
##        n_genes=n_genes,
##        n_cells=n_cells,
##        K=K,
##        df_genes=df.columns,
##        df_cells=cell_ids
##    )
##
##    
##    os.makedirs(EST_DIR / "HLDA", exist_ok=True)
##
##    beta_df.to_csv(EST_DIR / "HLDA/HLDA_beta.csv")
##    theta_df.to_csv(EST_DIR / "HLDA/HLDA_theta.csv")
##    print("✔ Wrote averaged Beta and Theta CSVs:")
##    print(f"  • {EST_DIR / 'HLDA_beta.csv'}")
##    print(f"  • {EST_DIR / 'HLDA_theta.csv'}")



############ RUN LDA, NMF, and ANALYSIS ########################################################################
    sns.set_style("whitegrid")

##    for mod in ("lda", "nmf"):
####    mod="nmf"
##        run_sklearn_model(df, mod, EST_DIR / mod.upper(), k=K)

    models = [
##        ("LDA", EST_DIR / "LDA"),
##        ("NMF", EST_DIR / "NMF"),
        ("HLDA", EST_DIR / "HLDA")
    ]

    topics_eval = TOPICS if INFER_TRUE_TOPICS else LEAF_TOPICS
    
    print('fitting pca')
    mixture_mask = TRUE_THETA[TRUE_ACTIVITY_TOPICS].sum(axis=1).gt(MIXTURE_THRESH).values
    true_beta = GENE_MEANS
    true_beta = true_beta.div(true_beta.sum(axis=0), axis=1)
    eigvecs, cell_proj = compute_pca_projection(df)
    true_beta_proj = eigvecs @ true_beta.values.astype(np.float32)   # (2 × K)

    for model_name, model_dir in models:

        plot_dir = ensure_dir(model_dir / "plots")
        
        beta_path  = model_dir / f"{model_name.lower()}_beta.csv"
        theta_path = model_dir / f"{model_name.lower()}_theta.csv"
        if not beta_path.exists():
            continue

        beta_raw  = pd.read_csv(beta_path,  index_col=0)
        theta_raw = pd.read_csv(theta_path, index_col=0)

        mapping = (
            match_topics(beta_raw, GENE_MEANS, topics_eval)
            if INFER_TRUE_TOPICS
            else match_topics(beta_raw, GENE_MEANS, LEAF_TOPICS)
        )
        rename_dict = {str(orig): tgt for orig, tgt in mapping.items()}
        beta_all   = beta_raw.rename(columns=rename_dict) 
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
                                save_path=model_dir / "plots",
                            )

        pc_pairs = [(0, 1), (2, 3), (4, 5)]
        for pcx, pcy in pc_pairs:
            out_png = model_dir / "plots" / f"{model_name}_PC{pcx+1}{pcy+1}.png"
            plot_pca_pair(pcx, pcy,
                          beta_est_proj, est_names, label_mask,
                          true_beta_proj, true_names,
                          cell_proj, mixture_mask,
                          model_name, out_png)

        if model_name == "HLDA":
            plot_geweke_histograms(
                sample_root=SAMPLE_DIR,    
                keys=["A", "D"],          
                out_dir=model_dir / "plots",
                n_save=n_save,
                n_cells=n_cells,
                n_genes=n_genes,
                n_topics=K
            )


        for topic in topics_eval:
            comp = compare_topic(GENE_MEANS, beta, GENE_META, topic)
            print('Plotting beta vs beta-hat')
            plot_beta_vs_lambda(comp, topic, plot_dir / f"{model_name}_{topic}_all.png", only_de=False)

        print('Plotting theta hist')
        plot_theta_histograms_by_identity(theta_raw, df.index, plot_dir / f"{model_name}_theta_by_identity.png")

        if INFER_TRUE_TOPICS and TRUE_THETA is not None:
            plot_theta_true_vs_est_by_identity(TRUE_THETA, theta, df.index.to_series(), model_name, plot_dir / f"{model_name}_theta_true_vs_est_by_id.png")

    print("✅ Pipeline finished. Outputs in", EST_DIR.resolve())

############ SSE RECONSTRUCTION ########################################################################

    X_sim, theta = simulate_cells(GENE_MEANS)
    X_sim.to_csv(SIM_COUNTS_CSV)
    theta.to_csv(DATA_DIR / "inference_theta.csv")

    identities = pd.Series(X_sim.index).str.split("_").str[0]

    for model_name in ["LDA", "NMF", "HLDA"]:
        model_dir = EST_DIR / model_name
        beta_path = model_dir / f"{model_name.lower()}_beta.csv"
        if not beta_path.exists():
            continue

        beta_hat = pd.read_csv(beta_path, index_col=0)

        mapping = match_topics_sse(beta_hat,
                                    GENE_MEANS.div(GENE_MEANS.sum(axis=0), axis=1),
                                    K_TOTAL)
        rename_dict = {str(orig): tgt for orig, tgt in mapping.items()}
        beta_hat   = beta_hat.rename(columns=rename_dict)

        activity_present = [t for t in ACTIVITY_TOPICS if t in beta_hat.columns]
        sse_df = incremental_sse(
            X_sim,
            beta_hat,
            identities,
            activity_present,
            theta_out=model_dir / "inference_theta_nnls.csv",   # saved in est/<MODEL>/
        )
        out_csv = model_dir / SSE_CSV_NAME
        ensure_dir(out_csv.parent)
        sse_df.to_csv(out_csv, index=False)
        print(f"[saved] {out_csv.relative_to(EST_DIR.parent)}")

############ RUN IN PARALLEL ###########################################################################
if __name__ == "__main__":
    
    grid = list(itertools.product(DE_MEANS, TOPIC_TOTALS))
    n_workers = max(1, os.cpu_count() - 2)

##    for DE_MEAN in DE_MEANS:
##        sim_counts(DE_MEAN)

    with mp.Pool(processes=n_workers) as pool:
        pool.starmap(run, grid)



















