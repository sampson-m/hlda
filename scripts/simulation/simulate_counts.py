#!/usr/bin/env python3
"""
hsim.py  –  Synthetic scRNA‑seq generator (30 % activity‑expressing cells)
=========================================================================
*Now with differential expression (DE) per topic*.

Changes
-------
1. **Baseline gene means**: one Gamma‑distributed mean per gene.
2. **DE modulation**: for every topic (identity *and* activity) we pick
   `N_DE` genes (default = 400) and multiply their means by a Log‑Normal
   factor (`DE_LOGNORM_MEAN`, `DE_LOGNORM_SIGMA`).
3. **Gene metadata**: a table `gene_metadata.csv` records which genes are
   DE in which topics and their fold‑change factors.

Config summary
~~~~~~~~~~~~~~
```python
IDENTITIES           = ["A", "B", "C", "D"]
ACTIVITY_TOPICS      = ["V1", "V2"]
DIRICHLET_PARAMS     = [8, 2, 2]
ACTIVITY_FRAC        = 0.30
CELLS_PER_IDENTITY   = 3000
N_GENES_DEFAULT      = 2000
# DE settings
N_DE                 = 400           # number of DE genes per topic
DE_LOGNORM_MEAN      = 0.5           # log‑Normal mean (μ)
DE_LOGNORM_SIGMA     = 0.4           # log‑Normal sigma (σ)
```

Outputs
~~~~~~~~
* `counts.csv`          – cell × gene counts
* `library_sizes.csv`   – identity & library size per cell
* `gene_means.csv`      – gene means for all topics
* `theta.csv`           – theta mixture weights per cell
* `gene_metadata.csv`   – DE flags and fold‑change factors
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import os, psutil
from sklearn.model_selection import train_test_split

# =====================================================================
# USER CONFIG – edit these defaults
# =====================================================================
IDENTITIES: List[str]          = ["A", "B"]
ACTIVITY_TOPICS: List[str]     = ["V1"]
DIRICHLET_PARAMS: List[float]  = [8, 8]
ACTIVITY_FRAC: float           = 0.30
CELLS_PER_IDENTITY: int | Dict[str, int] = 3000
N_GENES_DEFAULT: int           = 2000

# DE configuration
N_DE: int               = 250   # genes per topic that are DE
DE_LOGNORM_MEAN: float  = 0.5
DE_LOGNORM_SIGMA: float = 0.4

OUT_DIR_DEFAULT: Path          = Path("data/AB_V1/")

# ---------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------

def simulate_counts(
    *,
    n_genes: int = N_GENES_DEFAULT,
    identities: Sequence[str] | None = None,
    activity_topics: Sequence[str] | None = None,
    cells_per_identity: int | Dict[str, int] = CELLS_PER_IDENTITY,
    dirichlet_params: Sequence[float] = DIRICHLET_PARAMS,
    activity_frac: float = ACTIVITY_FRAC,
    n_de: int = N_DE,
    de_lognorm_mean: float = DE_LOGNORM_MEAN,
    de_lognorm_sigma: float = DE_LOGNORM_SIGMA,
    gamma_shape: float = 0.6,
    gamma_rate: float = 0.3,
    libsize_mean: int = 1500,
    libsize_sd: int = 0,
    random_state: int | None = None,
    out: Path = OUT_DIR_DEFAULT,
):
    """Simulate counts with DE per topic."""

    rng = np.random.default_rng(random_state)

    identities      = list(identities or IDENTITIES)
    activity_topics = list(activity_topics or ACTIVITY_TOPICS)

    n_id  = len(identities)
    n_act = len(activity_topics)
    n_topics = n_id + n_act

    # ------------ cells per identity ------------
    if isinstance(cells_per_identity, int):
        n_cells_by_id = {idt: cells_per_identity for idt in identities}
    else:
        n_cells_by_id = {idt: cells_per_identity[idt] for idt in identities}

    # ------------ baseline gene means ------------
    gene_labels = [f"Gene_{g+1}" for g in range(n_genes)]
    baseline = rng.gamma(shape=gamma_shape, scale=1.0 / gamma_rate, size=n_genes)

    topic_labels = identities + activity_topics 
    gm = pd.DataFrame(
        np.repeat(baseline[:, None], n_topics, axis=1),
        index=gene_labels,
        columns=topic_labels,
    )

    # ------------ gene metadata (DE tracking) ------------
    meta_cols = {
        "DE": False,
        "DE_groups": [[] for _ in range(n_genes)],
    }
    for t in topic_labels:
        meta_cols[f"DE_factor_{t}"] = np.nan
    gene_meta = pd.DataFrame(meta_cols, index=gene_labels)

    # ------------ apply DE per topic ------------
    available_genes = set(range(n_genes))
    for t in topic_labels:
      if not available_genes:
          break
      take = min(n_de, len(available_genes))
      de_genes = rng.choice(list(available_genes), size=take, replace=False)
      available_genes -= set(de_genes)
      factors = rng.lognormal(mean=de_lognorm_mean, sigma=de_lognorm_sigma, size=de_genes.size)

      col_idx = gm.columns.get_loc(t)
      gm.values[de_genes, col_idx] *= factors

      gene_meta.loc[gene_meta.index[de_genes], "DE"] = True
      gene_meta.loc[gene_meta.index[de_genes], "DE_groups"].apply(lambda lst, tg=t: lst.append(tg))
      gene_meta.loc[gene_meta.index[de_genes], f"DE_factor_{t}"] = factors


    # ------------ Dirichlet alpha vector ------------
    if not dirichlet_params:
        raise ValueError("DIRICHLET_PARAMS must contain at least the identity alpha value.")
    if len(dirichlet_params) == 1:
        dirichlet_params = list(dirichlet_params) + [1.0] * n_act
    if len(dirichlet_params) < n_act + 1:
        dirichlet_params = list(dirichlet_params) + [dirichlet_params[-1]] * (n_act + 1 - len(dirichlet_params))
    if len(dirichlet_params) > n_act + 1:
        raise ValueError("DIRICHLET_PARAMS is longer than identity + n_activity_topics")

    alpha_identity  = dirichlet_params[0]
    alpha_activitys = dirichlet_params[1:]

    cell_records: List[Dict[str, int]] = []
    theta_records: List[Dict[str, float]] = []

    for id_idx, idt in enumerate(identities):
        n_cells_total = n_cells_by_id[idt]
        n_activity_cells = int(round(activity_frac * n_cells_total))

        activity_flags = np.zeros(n_cells_total, dtype=bool)
        activity_flags[:n_activity_cells] = True
        rng.shuffle(activity_flags)

        for c in range(n_cells_total):
            # Use unique cell names to avoid pandas memory issues
            cell_name = f"{idt}_{c+1}"

            if activity_flags[c]:
                alpha = [alpha_identity] + alpha_activitys
                theta_sub = rng.dirichlet(alpha)
                theta_full = np.zeros(n_topics)
                theta_full[id_idx] = theta_sub[0]
                for act_j in range(n_act):
                    theta_full[n_id + act_j] = theta_sub[act_j + 1]
            else:
                theta_full = np.zeros(n_topics)
                theta_full[id_idx] = 1.0

            lib_size = libsize_mean if libsize_sd == 0 else max(1, int(rng.normal(libsize_mean, libsize_sd)))

            topic_weighted = (gm.values * theta_full).sum(axis=1)
            effective_means = lib_size * topic_weighted / topic_weighted.sum()
            counts = rng.poisson(effective_means).astype(int)

            cell_records.append({"cell": cell_name,
                                 **dict(zip(gene_labels, counts)),
                                 "identity": idt,
                                 "library_size": lib_size})
            theta_records.append({"cell": cell_name,
                                   **{t: float(theta_full[i]) for i, t in enumerate(topic_labels)}})

    counts_df = pd.DataFrame(cell_records).set_index("cell")[gene_labels].astype(int)
    lib_df    = pd.DataFrame(cell_records).set_index("cell")[["identity", "library_size"]]
    theta_df  = pd.DataFrame(theta_records).set_index("cell")

    # --- Direct per-identity train/test split (20% test per identity) ---
    rng = np.random.default_rng(random_state)
    train_idx = []
    test_idx = []
    for idt in identities:
        idx = counts_df.index[counts_df.index.str.startswith(f"{idt}_")]
        idx = rng.permutation(idx)
        n_test = int(0.2 * len(idx))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    # Write out full matrix and other outputs first
    out.mkdir(parents=True, exist_ok=True)
    print(f"Writing to count matrix")
    counts_df.to_csv(out / "counts.csv")
    print(f"Writing library sizes")
    lib_df.to_csv(out / "library_sizes.csv")
    print(f"Writing gene means")
    gm.to_csv(out / "gene_means.csv")
    print(f"Writing theta")
    theta_df.to_csv(out / "theta.csv")
    print(f"Writing gene metadata")
    gene_meta.to_csv(out / "gene_metadata.csv")

    # Delete large objects we no longer need
    del lib_df, gm, theta_df, gene_meta
    import gc; gc.collect()

    # Write train split
    print(f"Writing to train count matrix")
    train_df = counts_df.loc[train_idx]
    train_df.to_csv(out / "filtered_counts_train.csv")

    # Write test split
    print(f"Writing to test count matrix")
    test_df = counts_df.loc[test_idx]
    test_df.to_csv(out / "filtered_counts_test.csv")

    # Save the row order (cell names) for train and test splits
    pd.Series(train_df.index, name='cell').to_csv(out / "train_cells.csv", index=False)
    pd.Series(test_df.index, name='cell').to_csv(out / "test_cells.csv", index=False)

    del train_idx, test_idx, train_df, test_df
    gc.collect()

    # Optionally delete counts_df
    del counts_df
    gc.collect()

    return None, None, None, None, None


# ---------------------------------------------------------------------
# CLI – defaults from USER CONFIG
# ---------------------------------------------------------------------

def _lit(text: str):
    return ast.literal_eval(text)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="scRNA‑seq simulator with DE per topic")

    p.add_argument("--n-genes", type=int, default=N_GENES_DEFAULT)
    p.add_argument("--identities", default=str(IDENTITIES))
    p.add_argument("--activity-topics", default=str(ACTIVITY_TOPICS))
    p.add_argument("--cells-per-identity", default=str(CELLS_PER_IDENTITY))
    p.add_argument("--dirichlet-params", default=str(DIRICHLET_PARAMS))
    p.add_argument("--activity-frac", type=float, default=ACTIVITY_FRAC)
    p.add_argument("--n-de", type=int, default=N_DE)
    p.add_argument("--de-mean", type=float, default=DE_LOGNORM_MEAN)
    p.add_argument("--de-sigma", type=float, default=DE_LOGNORM_SIGMA)
    p.add_argument("--out", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--seed", type=int)

    return p.parse_args()


def main():
    args = _parse_args()

    counts_df, lib_df, gm, theta_df, gene_meta = simulate_counts(
        n_genes=args.n_genes,
        identities=_lit(args.identities),
        activity_topics=_lit(args.activity_topics),
        cells_per_identity=_lit(args.cells_per_identity),
        dirichlet_params=_lit(args.dirichlet_params),
        activity_frac=args.activity_frac,
        n_de=args.n_de,
        de_lognorm_mean=args.de_mean,
        de_lognorm_sigma=args.de_sigma,
        random_state=args.seed,
        out=args.out,
    )

    print(f"✅  Simulation complete – files written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
