###!/usr/bin/env python3
##"""
##Custom Gibbs HLDA pipeline with in-Numba permutation for speed
##"""
##import os
##import glob
##import pickle
##import numpy as np
##import pandas as pd
##from pathlib import Path
##from numba import njit
##from numba.typed import List as TypedList
##from tqdm import tqdm
##
### --- Configuration ----------------------------------------------------------
##TOPIC_STR2INT = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'V1': 4, 'V2': 5}
##TOPIC_INT2STR = {v: k for k, v in TOPIC_STR2INT.items()}
##K = len(TOPIC_STR2INT)
##
##IDENTITY_STR2INT = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
##IDENTITY_INT2STR = {v: k for k, v in IDENTITY_STR2INT.items()}
##
##TOPIC_HIERARCHY_INT = {
##    IDENTITY_STR2INT['A']: [TOPIC_STR2INT['A'], TOPIC_STR2INT['V1'], TOPIC_STR2INT['V2']],
##    IDENTITY_STR2INT['B']: [TOPIC_STR2INT['B'], TOPIC_STR2INT['V1'], TOPIC_STR2INT['V2']],
##    IDENTITY_STR2INT['C']: [TOPIC_STR2INT['C'], TOPIC_STR2INT['V1'], TOPIC_STR2INT['V2']],
##    IDENTITY_STR2INT['D']: [TOPIC_STR2INT['D'], TOPIC_STR2INT['V1'], TOPIC_STR2INT['V2']]
##}
##
### Reduced sampling for debug
##NUM_LOOPS   = 10000    # fewer sweeps
##BURN_IN     = 4000     # shorter burn-in
##THIN        = 10     # thin more aggressively for logs
##HYPERPARAMS = {'alpha_beta': 1.0, 'alpha_c': 1.0}
##
##DATA_ROOT   = Path("../data/ABCD_V1V2")
##COUNTS_CSV  = DATA_ROOT / "counts.csv"
##SAMPLES_DIR = Path("../samples/ABCD_V1V2/6_topic_fit/")
##OUT_DIR     = Path("../estimates/ABCD_V1V2/6_topic_fit/HLDA/")
##
### --- Initialization --------------------------------------------------------
##def initialize_long_data(count_matrix: pd.DataFrame,
##                         cell_identities: list[str],
##                         topic_hierarchy_int: dict[int, list[int]],
##                         identity_str2int: dict[str,int],
##                         save_path: str | None = None):
##
##    cell_identities = [c.split("_")[0] for c in cell_identities]
##    cell_ints = np.array([identity_str2int[c] for c in cell_identities], dtype=np.int32)
##
##    mat = count_matrix.to_numpy().astype(np.int32)
##    n_cells, n_genes = mat.shape
##    print(f"[DEBUG] Count matrix: {n_cells} cells × {n_genes} genes")
##
##    rows, cols = np.nonzero(mat)
##    counts = mat[rows, cols]
##    total = counts.sum()
##    print(f"[DEBUG] Exploded to {total} total tokens")
##
##    cell_rep  = np.repeat(rows, counts)
##    gene_rep  = np.repeat(cols, counts)
##    ident_rep = np.repeat(cell_ints[rows], counts)
##
##    # Build valid-topic list per identity
##    max_ident = max(topic_hierarchy_int.keys())
##    # <— use comprehension so each slot is its own array
##    valid_py = [np.empty(0, dtype=np.int32) for _ in range(max_ident + 1)]
##    for iid, tlist in topic_hierarchy_int.items():
##        valid_py[iid] = np.array(tlist, dtype=np.int32)
##        print(f"[DEBUG] Identity {iid}->{IDENTITY_INT2STR[iid]} valid topics: {tlist}")
##
##    # Initialize z randomly per token
##    z0 = np.empty(total, dtype=np.int32)
##    for i in range(total):
##        vt = valid_py[ident_rep[i]]
##        if vt.size == 0:
##            raise RuntimeError(f"No valid topics for identity {ident_rep[i]} at token {i}")
##        z0[i] = vt[np.random.randint(vt.size)]
##
##    dtype = np.dtype([
##        ('cell_idx',      np.int32),
##        ('gene_idx',      np.int32),
##        ('cell_identity', np.int32),
##        ('z',             np.int32)
##    ])
##    long_data = np.zeros(total, dtype=dtype)
##    long_data['cell_idx']      = cell_rep
##    long_data['gene_idx']      = gene_rep
##    long_data['cell_identity'] = ident_rep
##    long_data['z']             = z0
##
##    if save_path:
##        os.makedirs(save_path, exist_ok=True)
##        pd.DataFrame(long_data).to_csv(
##            os.path.join(save_path, 'long_data_init.csv'),
##            index=False
##        )
##        print(f"[DEBUG] Saved long_data_init.csv ({total} rows)")
##
##    return long_data, n_cells, n_genes
##
### --- Constants computation --------------------------------------------------
##def compute_constants(long_data, K, n_cells, n_genes):
##    print("[DEBUG] Computing constants A, B, D...")
##    cell_idx = long_data['cell_idx']
##    gene_idx = long_data['gene_idx']
##    z_arr    = long_data['z']
##    total    = z_arr.size
##
##    A = np.bincount(gene_idx*K + z_arr, minlength=n_genes*K).reshape(n_genes, K)
##    B = np.bincount(z_arr, minlength=K)
##    D = np.bincount(cell_idx*K + z_arr, minlength=n_cells*K).reshape(n_cells, K)
##
##    print(f"[DEBUG] A shape {A.shape}, sum {A.sum()} vs {total}")
##    print(f"[DEBUG] B shape {B.shape}, sum {B.sum()} vs {total}")
##    print(f"[DEBUG] D shape {D.shape}, sum {D.sum()} vs {total}")
##
##    assert A.sum() == total
##    assert B.sum() == total
##    assert D.sum() == total
##
##    return {'A': A, 'B': B, 'D': D, 'G': n_genes, 'K': K, 'C': n_cells}
##
### --- Numba update with in-Numba shuffle -------------------------------------
##@njit
##def gibbs_update_numba(cell_idx_arr,
##                       gene_idx_arr,
##                       cell_identity_arr,
##                       z_arr,
##                       A, B, D,
##                       valid_list,
##                       alpha_beta, alpha_c,
##                       G, K):
##    """
##    One full sweep over all tokens in order: for i in 0..N-1 do one-site update.
##    """
##    N = cell_idx_arr.shape[0]
##    G_abeta = G * alpha_beta
##
##    for i in range(N):
##        c = cell_idx_arr[i]
##        g = gene_idx_arr[i]
##        ident = cell_identity_arr[i]
##        current_z = z_arr[i]
##
##        # remove current assignment
##        A[g, current_z] -= 1
##        B[current_z]    -= 1
##        D[c, current_z] -= 1
##
##        # only allow the valid topics for this identity
##        vt = valid_list[ident]
##        m  = vt.size
##
##        # compute unnormalized probabilities
##        probs = np.empty(m, np.float64)
##        total_p = 0.0
##        for j in range(m):
##            t = vt[j]
##            p = ((alpha_beta + A[g, t]) /
##                 (G_abeta + B[t])) * (alpha_c + D[c, t])
##            probs[j] = p
##            total_p += p
##
##        # normalize
##        for j in range(m):
##            probs[j] /= total_p
##
##        # draw a new topic for token i
##        r = np.random.rand()
##        cum = 0.0
##        new_z = vt[-1]
##        for j in range(m):
##            cum += probs[j]
##            if r < cum:
##                new_z = vt[j]
##                break
##
##        # record new assignment
##        A[g, new_z] += 1
##        B[new_z]    += 1
##        D[c, new_z] += 1
##        z_arr[i]     = new_z
##
##
### --- Sampling loop: call the deterministic sweep directly --------------------
##def run_gibbs_sampling(long_data, consts, topic_hier, hyper,
##                       num_loops, burn_in, thin, sample_dir):
##    import os, glob, pickle
##    from tqdm import tqdm
##
##    os.makedirs(sample_dir, exist_ok=True)
##    for fn in glob.glob(os.path.join(sample_dir, "*.pkl")):
##        os.remove(fn)
##
##    A, B, D = consts['A'], consts['B'], consts['D']
##    G, K     = consts['G'], consts['K']
##    ci       = long_data['cell_identity']
##    ri       = long_data['cell_idx']
##    gi       = long_data['gene_idx']
##    z0       = long_data['z']
##
##    # build valid‐topic list as a TypedList
##    valid_list = TypedList()
##    for ident_int in sorted(topic_hier):
##        valid_list.append(np.array(topic_hier[ident_int], dtype=np.int32))
##    print(f"[DEBUG] valid_list sizes: {[len(v) for v in valid_list]}")
##
##    alpha_beta, alpha_c = hyper['alpha_beta'], hyper['alpha_c']
##
##    for it in tqdm(range(num_loops), desc="Gibbs Sampling"):
##        # deterministic sweep
##        gibbs_update_numba(ri,
##                           gi,
##                           ci,
##                           z0,
##                           A, B, D,
##                           valid_list,
##                           alpha_beta, alpha_c,
##                           G, K)
##
##        if it >= burn_in and ((it - burn_in + 1) % thin == 0):
##            snap = it + 1
##            outp = os.path.join(sample_dir, f"constants_{snap}.pkl")
##            with open(outp, "wb") as f:
##                pickle.dump({'A': A.copy(), 'D': D.copy()}, f)
##
##    print(f"Done: {num_loops} its (burn={burn_in}, thin={thin})")
##
### --- Reconstruction ---------------------------------------------------------
##def reconstruct_beta_theta(sample_dir, save_dir, gene_names, cell_id_strs):
##    files = sorted([f for f in os.listdir(sample_dir) if f.endswith('.pkl')])
##    A_acc = None
##    D_acc = None
##    n = 0
##    for fn in files:
##        with open(os.path.join(sample_dir, fn), 'rb') as f:
##            d = pickle.load(f)
##        a, dmat = d['A'], d['D']
##        if A_acc is None:
##            A_acc, D_acc = a.copy(), dmat.copy()
##        else:
##            A_acc += a
##            D_acc += dmat
##        n += 1
##    beta = A_acc / n
##    theta = D_acc / n
##    beta = beta / beta.sum(axis=0)
##    theta = theta / theta.sum(axis=1, keepdims=True)
##
##    beta_df = pd.DataFrame(beta,
##                           index=gene_names,
##                           columns=[TOPIC_INT2STR[i] for i in range(K)])
##    theta_df = pd.DataFrame(theta,
##                            index=cell_id_strs,
##                            columns=[TOPIC_INT2STR[i] for i in range(K)])
##    os.makedirs(save_dir, exist_ok=True)
##    beta_df.to_csv(os.path.join(save_dir, 'HLDA_beta.csv'))
##    theta_df.to_csv(os.path.join(save_dir, 'HLDA_theta.csv'))
##
### --- Main -------------------------------------------------------------------
##def main():
##    data_root   = DATA_ROOT
##    counts_csv  = COUNTS_CSV
##    samples_dir = SAMPLES_DIR
##    out_dir     = OUT_DIR
##
##    df = pd.read_csv(counts_csv, index_col=0)
##    cell_ids = df.index.tolist()
##
##    long_data, n_cells, n_genes = initialize_long_data(
##        df, cell_ids,
##        TOPIC_HIERARCHY_INT,
##        IDENTITY_STR2INT,
##        save_path=str(out_dir)
##    )
##    consts = compute_constants(long_data, K, n_cells, n_genes)
##
##    run_gibbs_sampling(long_data, consts,
##                       TOPIC_HIERARCHY_INT, HYPERPARAMS,
##                       NUM_LOOPS, BURN_IN, THIN,
##                       sample_dir=str(samples_dir))
##
##    reconstruct_beta_theta(str(samples_dir),
##                           str(out_dir),
##                           gene_names=list(df.columns),
##                           cell_id_strs=cell_ids)
##
##if __name__ == "__main__":
##    main()

#!/usr/bin/env python3
"""
Gibbs HLDA pipeline with single-shot Numba sweeps, on-disk memmap for saving thinned iterations,
pilot timing for token-update rate, and a function to compute averaged Beta and Theta matrices
from the saved memmaps.
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit
from numba.typed import List as TypedList

# --- Configuration ----------------------------------------------------------
TOPIC_STR2INT = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'V1': 4, 'V2': 5, 'V3': 6}
TOPIC_INT2STR = {v: k for k, v in TOPIC_STR2INT.items()}
K = len(TOPIC_STR2INT)

IDENTITY_STR2INT = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
IDENTITY_INT2STR = {v: k for k, v in IDENTITY_STR2INT.items()}

# Updated, flat hierarchy lists:
TOPIC_HIERARCHY_INT = {
    IDENTITY_STR2INT['A']: [TOPIC_STR2INT['A'], TOPIC_STR2INT['V1'], TOPIC_STR2INT['V2'], TOPIC_STR2INT['V3']],
    IDENTITY_STR2INT['B']: [TOPIC_STR2INT['B'], TOPIC_STR2INT['V1'], TOPIC_STR2INT['V2'], TOPIC_STR2INT['V3']],
    IDENTITY_STR2INT['C']: [TOPIC_STR2INT['C'], TOPIC_STR2INT['V1'], TOPIC_STR2INT['V2'], TOPIC_STR2INT['V3']],
    IDENTITY_STR2INT['D']: [TOPIC_STR2INT['D'], TOPIC_STR2INT['V1'], TOPIC_STR2INT['V2'], TOPIC_STR2INT['V3']]
}

NUM_LOOPS   = 15000    # total Gibbs sweeps
BURN_IN     = 5000     # burn-in iterations
THIN        =   40     # thinning interval
HYPERPARAMS = {'alpha_beta': .1, 'alpha_c': .1}

DATA_ROOT   = Path("../data/ABCD_V1V2")
COUNTS_CSV  = DATA_ROOT / "counts.csv"
SAMPLE_DIR = Path("../samples/ABCD_V1V2/7_topic_fit/")
OUT_DIR     = Path("../estimates/ABCD_V1V2/7_topic_fit/HLDA/")
os.makedirs(OUT_DIR, exist_ok=True)


# ------------------------------------------------------------------------------
# 1) Initialize “long_data” to produce the four arrays and dimensions
# ------------------------------------------------------------------------------
def initialize_long_data(count_matrix: pd.DataFrame,
                         cell_identities: list[str],
                         topic_hierarchy_int: dict[int, list[int]],
                         identity_str2int: dict[str, int],
                         n_activity: int,
                         save_path: str | None = None):

    stripped = [c.split("_")[0] for c in cell_identities]
    cell_ints = np.array([identity_str2int[c] for c in stripped], dtype=np.int32)

    mat = count_matrix.to_numpy().astype(np.int32)
    n_cells, n_genes = mat.shape

    rows, cols = np.nonzero(mat)
    counts = mat[rows, cols]

    cell_rep = np.repeat(rows.astype(np.int32), counts.astype(np.int32))
    gene_rep = np.repeat(cols.astype(np.int32), counts.astype(np.int32))
    total_tokens = cell_rep.shape[0]

    ident_rep = cell_ints[cell_rep]

    p_activity = 0.05  
    z0 = np.empty(total_tokens, dtype=np.int32)
    for i in range(total_tokens):
        ident_val = ident_rep[i]
        valid = topic_hierarchy_int[ident_val]
        if np.random.rand() < p_activity:
            z0[i] = np.random.choice([TOPIC_STR2INT[f"V{i+1}"] for i in range(n_activity)])
##            z0[i] = TOPIC_STR2INT["V1"]

        else:
            z0[i] = ident_val  # identity leaf

    long_data = {
        'cell_idx': cell_rep,
        'gene_idx': gene_rep,
        'cell_identity': ident_rep,
        'z': z0
    }
    return long_data, n_cells, n_genes


# ------------------------------------------------------------------------------
# 2) Compute initial count arrays A (gene×topic), B (topic), D (cell×topic)
# ------------------------------------------------------------------------------
def compute_constants(long_data, K: int, n_cells: int, n_genes: int):
    """
    Given long_data with keys 'cell_idx', 'gene_idx', 'z', compute:
      - A: int32[n_genes, K]   (counts of gene-topic assignments)
      - B: int32[K]            (counts of topic assignments overall)
      - D: int32[n_cells, K]   (counts of cell-topic assignments)
    Returns a dict containing these plus dimensions.
    """
    cell_idx_arr = long_data['cell_idx']
    gene_idx_arr = long_data['gene_idx']
    z_arr         = long_data['z']

    # Build A by counting occurrences of (gene, z)
    A = np.bincount(gene_idx_arr * K + z_arr, minlength=n_genes * K).reshape(n_genes, K)
    B = np.bincount(z_arr, minlength=K)
    D = np.bincount(cell_idx_arr * K + z_arr, minlength=n_cells * K).reshape(n_cells, K)

    return {
        'A': A.astype(np.int32),
        'B': B.astype(np.int32),
        'D': D.astype(np.int32),
        'G': n_genes,
        'K': K
    }


# ------------------------------------------------------------------------------
# 3) Single-shot Numba function: run all sweeps, and write each thinned A & D to memmaps
# ------------------------------------------------------------------------------
@njit
def gibbs_and_write_memmap(
    cell_idx_arr,       # int32[total_tokens]
    gene_idx_arr,       # int32[total_tokens]
    cell_identity_arr,  # int32[total_tokens]
    z_arr,              # int32[total_tokens]
    A,                  # int32[n_genes, K]
    B,                  # int32[K]
    D,                  # int32[n_cells, K]
    valid_list,         # typed List[int32[:]] for each identity
    alpha_beta,         # float64
    alpha_c,            # float64
    G,                  # int32
    K,                  # int32
    num_loops,          # int32
    burn_in,            # int32
    thin,               # int32
    A_chain,            # int32[n_save, n_genes, K]   ← memmap
    D_chain             # int32[n_save, n_cells, K]   ← memmap
):
    """
    Run num_loops of Gibbs sampling. After burn_in, every thin-th iteration:
      - Copy current A → A_chain[cnt, :, :]
      - Copy current D → D_chain[cnt, :, :]
    z_arr, A, B, D are updated in place; A_chain & D_chain are on-disk memmaps.
    Returns the total number of saved slices (should equal n_save).
    """
    total_tokens = cell_idx_arr.shape[0]
    G_abeta = G * alpha_beta
    cnt = 0  # counter for how many slices have been written into memmaps

    for it in range(num_loops):
        # ————— One Gibbs sweep over every token —————
        for idx in range(total_tokens):
            c = cell_idx_arr[idx]
            g = gene_idx_arr[idx]
            ident = cell_identity_arr[idx]
            old_z = z_arr[idx]

            # Remove old assignment
            A[g, old_z] -= 1
            B[old_z]    -= 1
            D[c, old_z] -= 1

            # Only allow valid topics for this identity
            vt = valid_list[ident]
            m = vt.shape[0]

            # Compute unnormalized probabilities for each valid topic
            total_p = 0.0
            probs = np.empty(m, np.float64)
            for j in range(m):
                t = vt[j]
                p = ((alpha_beta + A[g, t]) / (G_abeta + B[t])) * (alpha_c + D[c, t])
                probs[j] = p
                total_p += p
            for j in range(m):
                probs[j] /= total_p

            # Sample new topic
            r = np.random.rand()
            cum = 0.0
            new_z = vt[m - 1]
            for j in range(m):
                cum += probs[j]
                if r < cum:
                    new_z = vt[j]
                    break

            # Record new assignment
            A[g, new_z] += 1
            B[new_z]    += 1
            D[c, new_z] += 1
            z_arr[idx]   = new_z
        # ————— End one sweep —————

        # After burn-in, every thin-th iteration, write to memmaps
        if it >= burn_in and ((it - burn_in + 1) % thin == 0):
            # Copy A → A_chain[cnt, :, :]
            for gg in range(G):
                for tt in range(K):
                    A_chain[cnt, gg, tt] = A[gg, tt]
            # Copy D → D_chain[cnt, :, :]
            n_cells = D.shape[0]
            for cc in range(n_cells):
                for tt in range(K):
                    D_chain[cnt, cc, tt] = D[cc, tt]

            cnt += 1

    # Return how many slices were actually written
    return cnt


# ------------------------------------------------------------------------------
# 4) Function to compute averaged Beta and Theta from memmaps
# ------------------------------------------------------------------------------
def compute_posterior_means_from_memmaps(A_chain_path: Path,
                                         D_chain_path: Path,
                                         n_save: int,
                                         n_genes: int,
                                         n_cells: int,
                                         K: int,
                                         df_genes: pd.Index,
                                         df_cells: list[str]):
    """
    Given the file paths to A_chain and D_chain memmaps, along with dimensions,
    compute the posterior mean Beta (gene × topic) and Theta (cell × topic) matrices.

    Returns:
      beta_df: pd.DataFrame of shape (n_genes, K), normalized so columns sum to 1
      theta_df: pd.DataFrame of shape (n_cells, K), normalized so rows sum to 1
    """
    # Load the memmaps in read-only mode
    A_chain = np.memmap(filename=str(A_chain_path),
                        mode="r",
                        dtype=np.int32,
                        shape=(n_save, n_genes, K))
    D_chain = np.memmap(filename=str(D_chain_path),
                        mode="r",
                        dtype=np.int32,
                        shape=(n_save, n_cells, K))

    # Sum over the first axis (iterations) in a streaming fashion
    Beta_sum = np.zeros((n_genes, K), dtype=np.float64)
    Theta_sum = np.zeros((n_cells, K), dtype=np.float64)

    for idx in range(n_save):
        Beta_sum += A_chain[idx, :, :].astype(np.float64)
        Theta_sum += D_chain[idx, :, :].astype(np.float64)

    # Compute means
    Beta_mean_counts = Beta_sum / n_save
    Theta_mean_counts = Theta_sum / n_save

    # Normalize: Beta columns sum to 1; Theta rows sum to 1
    Beta_mean = Beta_mean_counts / Beta_mean_counts.sum(axis=0, keepdims=True)
    Theta_mean = Theta_mean_counts / Theta_mean_counts.sum(axis=1, keepdims=True)

    # Convert to DataFrames with appropriate indexes/columns
    beta_df = pd.DataFrame(Beta_mean,
                           index=df_genes,
                           columns=[TOPIC_INT2STR[t] for t in range(K)])
    theta_df = pd.DataFrame(Theta_mean,
                            index=df_cells,
                            columns=[TOPIC_INT2STR[t] for t in range(K)])

    return beta_df, theta_df


# ------------------------------------------------------------------------------
# 5) Main workflow: load data, build long_data, preallocate memmaps, pilot timing,
#                   full Numba run, write CSVs
# ------------------------------------------------------------------------------
def main():
    # Debug: print the (flat) topic hierarchy
    print(f"DEBUG: Topic hierarchy = {TOPIC_HIERARCHY_INT}")

    # 5.1) Load raw count matrix (cells × genes)
    df = pd.read_csv(COUNTS_CSV, index_col=0)
    cell_ids = list(df.index)  # e.g. ['A_1', 'A_2', …]

    # 5.2) Initialize “long format” data and get dimensions
    long_data, n_cells, n_genes = initialize_long_data(
        count_matrix=df,
        cell_identities=cell_ids,
        topic_hierarchy_int=TOPIC_HIERARCHY_INT,
        identity_str2int=IDENTITY_STR2INT,
        save_path=None
    )

    # 5.3) Compute initial A, B, D
    consts = compute_constants(long_data, K, n_cells, n_genes)
    A = consts['A'].copy()   # int32[n_genes, K]
    B = consts['B'].copy()   # int32[K]
    D = consts['D'].copy()   # int32[n_cells, K]

    cell_idx_arr      = long_data['cell_idx']       # int32[total_tokens]
    gene_idx_arr      = long_data['gene_idx']       # int32[total_tokens]
    cell_identity_arr = long_data['cell_identity']  # int32[total_tokens]
    z0                = long_data['z']              # int32[total_tokens]

    alpha_beta = HYPERPARAMS['alpha_beta']
    alpha_c    = HYPERPARAMS['alpha_c']

    # 5.4) Build valid_list = numba.typed.List of int32 arrays for each identity
    valid_list = TypedList()
    for ident_int in sorted(TOPIC_HIERARCHY_INT):
        vt = np.array(TOPIC_HIERARCHY_INT[ident_int], dtype=np.int32)
        valid_list.append(vt)

    # 5.5) Compute how many thinned samples we will save
    n_save = (NUM_LOOPS - BURN_IN + 1) // THIN  # integer division

    # 5.6) Preallocate two memmaps on disk for saving each thinned A & D
    A_chain_path = SAMPLE_DIR / "A_chain.memmap"
    D_chain_path = SAMPLE_DIR / "D_chain.memmap"

    A_chain = np.memmap(
        filename=str(A_chain_path),
        mode="w+",
        dtype=np.int32,
        shape=(n_save, n_genes, K)
    )
    D_chain = np.memmap(
        filename=str(D_chain_path),
        mode="w+",
        dtype=np.int32,
        shape=(n_save, n_cells, K)
    )

    # 5.7) Ensure contiguity for best Numba performance
    assert A.flags['C_CONTIGUOUS']
    assert D.flags['C_CONTIGUOUS']
    assert z0.flags['C_CONTIGUOUS']
    assert A_chain.flags['C_CONTIGUOUS']
    assert D_chain.flags['C_CONTIGUOUS']

    # === PILOT RUN FOR TIMING ===
    total_tokens = cell_idx_arr.shape[0]
    pilot_loops = 10
    A_pilot = A.copy()
    B_pilot = B.copy()
    D_pilot = D.copy()
    z_pilot = z0.copy()

    t_start = time.time()
    _ = gibbs_and_write_memmap(
        cell_idx_arr.astype(np.int32),
        gene_idx_arr.astype(np.int32),
        cell_identity_arr.astype(np.int32),
        z_pilot,
        A_pilot, B_pilot, D_pilot,
        valid_list,
        alpha_beta, alpha_c,
        np.int32(n_genes),
        np.int32(K),
        np.int32(pilot_loops),
        np.int32(0),      # burn_in=0 for pilot
        np.int32(1),      # thin=1 for pilot
        A_chain[:0],      # empty slices (no actual writing)
        D_chain[:0]       # empty slices (no actual writing)
    )
    t_pilot = time.time() - t_start

    processed = pilot_loops * total_tokens
    rate = processed / t_pilot
    full_updates = NUM_LOOPS * total_tokens
    est_seconds = full_updates / rate
    est_hours = est_seconds / 3600.0

    print(f"Pilot: {processed} token‐updates in {t_pilot:.2f}s  →  {rate:.0f} tokens/s")
    print(f"Estimated full run ({NUM_LOOPS} sweeps, {full_updates} updates): {est_hours:.2f} h")

    # === FULL Numba RUN ===
    t0 = time.time()
    saved_count = gibbs_and_write_memmap(
        cell_idx_arr.astype(np.int32),
        gene_idx_arr.astype(np.int32),
        cell_identity_arr.astype(np.int32),
        z0.astype(np.int32),
        A, B, D,
        valid_list,
        alpha_beta, alpha_c,
        np.int32(n_genes),
        np.int32(K),
        np.int32(NUM_LOOPS),
        np.int32(BURN_IN),
        np.int32(THIN),
        A_chain,
        D_chain
    )
    elapsed = time.time() - t0
    print(f"✔ Sampling complete. (Elapsed time: {elapsed:.1f}s for {NUM_LOOPS} sweeps)")

    # 5.9) Sanity check: saved_count should equal n_save
    if saved_count != n_save:
        raise RuntimeError(f"Expected to save {n_save} slices, but Numba returned {saved_count}")

    print(f"✔ Wrote {saved_count} thinned samples to memmaps:")
    print(f"  • {A_chain_path}  (shape = {A_chain.shape})")
    print(f"  • {D_chain_path}  (shape = {D_chain.shape})")

    # 5.10) Compute averaged Beta and Theta from the memmaps
    beta_df, theta_df = compute_posterior_means_from_memmaps(
        A_chain_path=A_chain_path,
        D_chain_path=D_chain_path,
        n_save=n_save,
        n_genes=n_genes,
        n_cells=n_cells,
        K=K,
        df_genes=df.columns,
        df_cells=cell_ids
    )

    # 5.11) Write the averaged Beta and Theta to CSV
    beta_df.to_csv(OUT_DIR / "HLDA_beta.csv")
    theta_df.to_csv(OUT_DIR / "HLDA_theta.csv")
    print("✔ Wrote averaged Beta and Theta CSVs:")
    print(f"  • {OUT_DIR / 'HLDA_beta.csv'}")
    print(f"  • {OUT_DIR / 'HLDA_theta.csv'}")


if __name__ == "__main__":
    main()
