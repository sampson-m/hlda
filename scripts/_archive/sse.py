#!/usr/bin/env python3
"""
simulate_recon.py  (stand‑alone)
===============================
Simulate a fresh batch of cells **without any command‑line flags** and compute
incremental reconstruction SSE using pre‑trained β̂ for each model directory.

Configuration parameters live in the *CONFIG* block below—edit them directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import List
import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from pipeline import match_topics

# -----------------------------------------------------------------------------
#  CONFIG — edit here ----------------------------------------------------------
# -----------------------------------------------------------------------------
DATA_ROOT          = Path("../data/ABCD_V1V2")
EST_ROOT           = Path("../estimates/ABCD_V1V2/6_topic_fit/")

N_CELLS_PER_ID     = 500       # simulated cells per identity (A,B,C,D)
LIB_SIZE_MEAN      = 1500      # mean of Poisson library size
MIX_FRACTION       = 0.30      # fraction of cells per identity that mix activity topics
DIRICHLET_ALPHA    = (8, 8, 8) # α_identity, α_V1, α_V2

IDENTITY_TOPS      = ["A", "B", "C", "D"]
ACTIVITY_TOPS      = ["V1", "V2"]
SEED               = 2025

SIM_COUNTS_CSV     = DATA_ROOT / "inference_counts.csv"
SSE_CSV_NAME       = "simulated_recon_sse.csv"  # written inside each model dir

# -----------------------------------------------------------------------------
#  RNG & helpers --------------------------------------------------------------
# -----------------------------------------------------------------------------
rng = np.random.default_rng(SEED)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def simulate_cells(gene_means: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame (cells × genes) using global CONFIG parameters."""
    gm = gene_means.div(gene_means.sum(axis=0), axis=1)  # col‑normalise
    
    topics_all = IDENTITY_TOPS + ACTIVITY_TOPS
    topic_idx  = {t: i for i, t in enumerate(topics_all)}

    rows, thetas, ids = [], [], []
    for ident in IDENTITY_TOPS:
        p_id = gm[ident].values
        p_v1 = gm["V1"].values
        p_v2 = gm["V2"].values
        for i in range(N_CELLS_PER_ID):
            lib = LIB_SIZE_MEAN
            if rng.random() < MIX_FRACTION:
                # Dirichlet over (identity, V1, V2)
                theta_raw = rng.dirichlet(DIRICHLET_ALPHA)
            else:
                theta_raw = np.array([1.0, 0.0, 0.0])

            theta_vec = np.zeros(len(topics_all))
            theta_vec[topic_idx[ident]] = theta_raw[0]   # identity weight
            theta_vec[topic_idx["V1"]]    = theta_raw[1]
            theta_vec[topic_idx["V2"]]    = theta_raw[2]

            prob = (theta_raw[0] * p_id +
                    theta_raw[1] * p_v1 +
                    theta_raw[2] * p_v2)
            rows.append(rng.poisson(lib * prob))
            thetas.append(theta_vec)
            ids.append(f"{ident}_{i+1:03d}")

    counts_df = pd.DataFrame(np.vstack(rows), index=ids, columns=gm.index)
    theta_df  = pd.DataFrame(np.vstack(thetas), index=ids, columns=topics_all)
    return counts_df, theta_df


def estimate_theta_nnls(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    thetas = np.empty((X.shape[0], beta.shape[1]))
    for i, x in enumerate(X):
        thetas[i], _ = nnls(beta, x)
    return thetas

def estimate_theta_simplex(X: np.ndarray, B: np.ndarray, l1) -> np.ndarray:
    n, k = X.shape[0], B.shape[1]
    Theta = np.empty((n, k))
    for i, x in enumerate(X):
        th = cp.Variable(k, nonneg=True)
        obj = cp.sum_squares(B @ th - x) + l1 * cp.sum(th)
        cp.Problem(cp.Minimize(obj), [cp.sum(th) == 1]).solve(
            solver=cp.OSQP, eps_abs=1e-6, verbose=False
        )
        Theta[i] = th.value
    return Theta

def reconstruction_sse(X: np.ndarray, Theta: np.ndarray, Beta: np.ndarray) -> float:
    recon = Theta @ Beta.T
    return np.square(X - recon).sum()


def incremental_sse(
    X_df: pd.DataFrame,
    beta_df: pd.DataFrame,
    identities: pd.Series,
    activity_order: list[str],
    theta_out: Path | None = None,
) -> pd.DataFrame:
    """Incremental SSE where **Θ is re‑fitted *per identity group*.**

    For each identity (A–D):
        step‑0: NNLS with that identity topic only
        step‑1: NNLS with identity + V1 (if present)
        step‑2: NNLS with identity + V2 (if present)
    Returns a tidy DataFrame of SSEs and, if *theta_out* is given, saves
    the concatenated Θ matrix (cells × all topics used across steps).
    """
    # --- align genes and convert to probabilities ---------------------------
    X_prop   = X_df.div(X_df.sum(axis=1), axis=0).values

    theta_accum = np.zeros((len(X_df), 0))
    theta_cols  = []

    results = []
    for ident in IDENTITY_TOPS:
        mask = identities == ident
        if not mask.any():
            continue

        topic_steps = [[ident]]  # step 0: identity only
        for i in range(1, len(activity_order) + 1):
            topic_steps.append([ident] + activity_order[:i])

        for topics_now in topic_steps:
            B = beta_df[topics_now].values  # (genes × k_now)
            Theta = estimate_theta_simplex(X_prop[mask], B, l1=0.002)

            sse = reconstruction_sse(X_prop[mask], Theta, B)
            results.append({
                "topics": "+".join(topics_now),
                "identity": ident,
                "cells": int(mask.sum()),
                "SSE_prob": sse,
            })

            # collect θ for optional saving (prefix step tag)
            col_tags = [f"{t}" for t in topics_now]
            theta_accum = np.concatenate([theta_accum, np.zeros((len(X_df), Theta.shape[1]))], axis=1)
            theta_accum[mask, -Theta.shape[1]:] = Theta
            theta_cols.extend(col_tags)

    # save θ if requested ----------------------------------------------------
    if theta_out is not None and theta_accum.size:
        theta_df = pd.DataFrame(theta_accum, index=pd.Index(X_df.index), columns=pd.Index(theta_cols))
        ensure_dir(theta_out.parent)
        theta_df.to_csv(theta_out)

    return pd.DataFrame(results)


def incremental_sse_custom(
    X_df: pd.DataFrame,
    beta_df: pd.DataFrame,
    identities: pd.Series,
    activity_topics: list[str],
    theta_out: Path | None = None,
) -> pd.DataFrame:
    """Incremental SSE with custom topic addition order: [id], [id,V1], [id,V2], [id,V1,V2]."""
    X_prop = X_df.div(X_df.sum(axis=1), axis=0).values
    theta_accum = np.zeros((len(X_df), 0))
    theta_cols = []
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
            B = beta_df[topics_now].values
            Theta = estimate_theta_simplex(X_prop[mask], B, l1=0.002)
            sse = reconstruction_sse(X_prop[mask], Theta, B)
            results.append({
                "topics": "+".join(topics_now),
                "identity": ident,
                "cells": int(mask.sum()),
                "SSE_prob": sse,
            })
            col_tags = [f"{t}" for t in topics_now]
            theta_accum = np.concatenate([theta_accum, np.zeros((len(X_df), Theta.shape[1]))], axis=1)
            theta_accum[mask, -Theta.shape[1]:] = Theta
            theta_cols.extend(col_tags)
    if theta_out is not None and theta_accum.size:
        theta_df = pd.DataFrame(theta_accum, index=pd.Index(X_df.index), columns=pd.Index(theta_cols))
        ensure_dir(theta_out.parent)
        theta_df.to_csv(theta_out)
    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
#  MAIN -----------------------------------------------------------------------
# -----------------------------------------------------------------------------
 

def main() -> None:
    # 1) Simulate counts and save
    gene_means = pd.read_csv(DATA_ROOT / "gene_means.csv", index_col=0)
    X_sim, theta = simulate_cells(gene_means)
    ensure_dir(DATA_ROOT)
    X_sim.to_csv(SIM_COUNTS_CSV)
    theta.to_csv(DATA_ROOT / "inference_theta.csv")

    identities = pd.Series(X_sim.index).str.split("_").str[0]

    for model_name in ["LDA", "NMF", "HLDA"]:
        model_dir = EST_ROOT / model_name
        beta_path = model_dir / f"{model_name.lower()}_beta.csv"
        if not beta_path.exists():
            continue

        beta_hat = pd.read_csv(beta_path, index_col=0)

        mapping = match_topics(beta_hat,
                                       gene_means.div(gene_means.sum(axis=0), axis=1),
                                       IDENTITY_TOPS+ACTIVITY_TOPS)
        rename_dict = {str(orig): tgt for orig, tgt in mapping.items()}
        beta_hat   = beta_hat.rename(columns=rename_dict)

        activity_present = [t for t in ACTIVITY_TOPS if t in beta_hat.columns]
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
        print(f"[saved] {out_csv.relative_to(EST_ROOT.parent)}")


if __name__ == "__main__":
    main()
