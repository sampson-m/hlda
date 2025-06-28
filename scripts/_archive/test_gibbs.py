#!/usr/bin/env python3
"""
Custom Gibbs HLDA only pipeline
"""
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit, typed
from tqdm import tqdm
import glob
from numba.typed import List as TypedList


TOPIC_STR2INT = {'A': 0, 'B': 1, 'VAR': 2}
TOPIC_INT2STR = {v: k for k, v in TOPIC_STR2INT.items()}
K = len(TOPIC_STR2INT)

IDENTITY_STR2INT = {'A': 0, 'B': 1}
IDENTITY_INT2STR = {v: k for k, v in IDENTITY_STR2INT.items()}

TOPIC_HIERARCHY_INT = {
    IDENTITY_STR2INT['A']: [TOPIC_STR2INT['A'], TOPIC_STR2INT['VAR']],
    IDENTITY_STR2INT['B']: [TOPIC_STR2INT['B'], TOPIC_STR2INT['VAR']]
}


NUM_LOOPS = 20000
BURN_IN   = 10000
THIN      = 50
HYPERPARAMS = { 'alpha_beta': 1.0, 'alpha_c': 1.0 }

# --- Initialization --------------------------------------------------------
def initialize_long_data(count_matrix: pd.DataFrame,
                         cell_identities: list[str],
                         topic_hierarchy_int: dict[int, list[int]],
                         identity_str2int: dict[str,int],
                         save_path: str | None = None):
    # Map cell identities to ints
    cell_id_arr = np.array(
        [identity_str2int[c] for c in cell_identities],
        dtype=np.int32
    )
    # Convert counts to numpy
    mat = count_matrix.to_numpy().astype(np.int32)
    G = mat.shape[1]
    # Explode counts
    row_idx, col_idx = np.nonzero(mat)
    counts = mat[row_idx, col_idx]
    total = counts.sum()
    # Repeat indices
    cell_idx_rep = np.repeat(row_idx, counts)
    gene_idx_rep = np.repeat(col_idx, counts)
    identity_rep = np.repeat(cell_id_arr[row_idx], counts)
    # Build valid-topic list
    n_id = max(topic_hierarchy_int.keys())
    valid_topics_py = [None] * n_id
    for ident_int, tlist in topic_hierarchy_int.items():
        valid_topics_py[ident_int-1] = np.array(tlist, dtype=np.int32)
    # Initialize z randomly per token
    z0 = np.empty(total, dtype=np.int32)
    for i in range(total):
        vt = valid_topics_py[identity_rep[i]-1]
        z0[i] = vt[np.random.randint(0, vt.shape[0])]
    # Assemble record array
    dtype = np.dtype([
        ('cell_idx', np.int32),
        ('gene_idx', np.int32),
        ('cell_identity', np.int32),
        ('z', np.int32)
    ])
    long_data = np.zeros(total, dtype=dtype)
    long_data['cell_idx'] = cell_idx_rep
    long_data['gene_idx'] = gene_idx_rep
    long_data['cell_identity'] = identity_rep
    long_data['z'] = z0
    # Optionally save for debugging
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        pd.DataFrame({
            'cell_idx': long_data['cell_idx'],
            'gene_idx': long_data['gene_idx'],
            'cell_identity': long_data['cell_identity'],
            'z': long_data['z']
        }).to_csv(os.path.join(save_path, 'long_data_init.csv'), index=False)
    return long_data

# --- Constants computation --------------------------------------------------
def compute_constants(long_data, K, n_cells, n_genes):

    # unpack fields
    cell_idx = long_data['cell_idx']
    gene_idx = long_data['gene_idx']
    z_arr    = long_data['z']
    total    = len(z_arr)

    # --- A[g, t]: counts of gene g assigned to topic t ---
    gt_pairs = gene_idx * K + z_arr
    A_counts = np.bincount(gt_pairs, minlength=n_genes * K)
    A = A_counts.reshape(n_genes, K)
    assert A.sum() == total, f"A sum {A.sum()} != total counts {total}"

    # --- B[t]: total counts assigned to topic t (global) ---
    B = np.bincount(z_arr, minlength=K)
    assert B.sum() == total, f"B sum {B.sum()} != total counts {total}"

    # --- D[c, t]: counts in cell c for topic t ---
    ct_pairs = cell_idx * K + z_arr
    D_counts = np.bincount(ct_pairs, minlength=n_cells * K)
    D = D_counts.reshape(n_cells, K)
    assert D.sum() == total, f"D sum {D.sum()} != total counts {total}"

    return {
        'A': A,
        'B': B,
        'D': D,
        'G': n_genes,
        'K': K,
        'C': n_cells
    }
# --- Numba update -----------------------------------------------------------
@njit
def gibbs_update_numba(perm, cell_idx_arr, gene_idx_arr, cell_identity_arr, z_arr,
                       A, B, D, valid_topic_list,
                       alpha_beta, alpha_c, G, K):

    for loop in range(perm.size):
        i = perm[loop] 
        c = cell_idx_arr[i]
        g = gene_idx_arr[i]
        ident = cell_identity_arr[i]
        current_z = z_arr[i]
        A[g, current_z] -= 1
        B[current_z]    -= 1
        D[c, current_z] -= 1
        vt = valid_topic_list[ident-1]
        nvt = vt.shape[0]

        # compute probabilities
        probs = np.empty(vt.shape[0], np.float64)
        s = 0.0
        for j in range(vt.shape[0]):
            t = vt[j]
            p = ((alpha_beta + A[g,t]) / (G*alpha_beta + B[t])) * (alpha_c + D[c,t])
            probs[j] = p; s += p
        for j in range(vt.shape[0]): probs[j] /= s
        r = np.random.rand(); cum = 0.0; new_z = vt[-1]
        for j in range(vt.shape[0]):
            cum += probs[j]
            if r < cum:
                new_z = vt[j]; break
        A[g, new_z] += 1
        B[new_z]    += 1
        D[c, new_z] += 1
        z_arr[i]     = new_z

        valid = False
        for j in range(nvt):
            if new_z == vt[j]:
                valid = True
                break
        # this assertion will raise if you ever assign an invalid topic
        assert valid

# --- Sampling loop ----------------------------------------------------------
def run_gibbs_sampling(long_data, constants, topic_hierarchy_int,
                       hyperparams,
                       num_loops=NUM_LOOPS,
                       burn_in=BURN_IN,
                       thin=THIN,
                       sample_dir="samples"):

    os.makedirs(sample_dir, exist_ok=True)
    for fn in glob.glob(os.path.join(sample_dir, "*.pkl")):
        os.remove(fn)

    A = constants['A']
    B = constants['B']
    D = constants['D']
    G = constants['G']
    K = constants['K']

    cell_identity_arr = long_data['cell_identity']
    cell_idx_arr      = long_data['cell_idx']
    gene_idx_arr      = long_data['gene_idx']
    z_arr             = long_data['z']

    valid_py = []
    for ident_int in sorted(topic_hierarchy_int):
        arr = np.array(topic_hierarchy_int[ident_int], dtype=np.int32)
        valid_py.append(arr)
    valid_list = TypedList()
    for arr in valid_py:
        valid_list.append(arr)

    alpha_beta = hyperparams['alpha_beta']
    alpha_c    = hyperparams['alpha_c']

    for it in tqdm(range(num_loops), desc="Gibbs Sampling"):
        
        perm = np.random.permutation(cell_idx_arr.shape[0])
        gibbs_update_numba(
            perm,
            cell_idx_arr,
            gene_idx_arr,
            cell_identity_arr,
            z_arr,
            A, B, D,
            valid_list,
            alpha_beta, alpha_c,
            G, K
        )

        if it >= burn_in and ((it - burn_in + 1) % thin == 0):
            snap = it + 1
            outp = os.path.join(sample_dir, f"constants_{snap}.pkl")
            with open(outp, "wb") as f:
                pickle.dump({'A': A.copy(), 'D': D.copy()}, f)

    print(f"Done: {num_loops} iters, burn_in={burn_in}, thinned every {thin}.")
# --- Reconstruction ---------------------------------------------------------
def reconstruct_beta_theta(sample_dir, save_dir, gene_names, cell_id_strs):
    files = sorted([f for f in os.listdir(sample_dir) if f.endswith('.pkl')])
    A_acc = None; D_acc = None; n=0
    for fn in files:
        c = pickle.load(open(os.path.join(sample_dir, fn),'rb'))
        a, d = c['A'], c['D']
        if A_acc is None:
            A_acc, D_acc = a.copy(), d.copy()
        else:
            A_acc += a; D_acc += d
        n += 1
    beta = A_acc / n; theta = D_acc / n
    # normalize
    beta = beta / beta.sum(axis=0)
    theta= theta/ theta.sum(axis=1, keepdims=True)
    beta_df = pd.DataFrame(beta, index=gene_names,
                           columns=[TOPIC_INT2STR[i] for i in range(K)])
    theta_df= pd.DataFrame(theta,index=cell_id_strs,
                           columns=[TOPIC_INT2STR[i] for i in range(K)])
    os.makedirs(save_dir, exist_ok=True)
    beta_df.to_csv(os.path.join(save_dir,'HLDA_beta.csv'))
    theta_df.to_csv(os.path.join(save_dir,'HLDA_theta.csv'))

# --- Main -------------------------------------------------------------------
def main():
    # Paths
    data_root   = Path("../data/AB")
    counts_csv  = data_root / "simulated_counts.csv"
    samples_dir = Path("../samples/AB")
    out_dir     = Path("../estimates/AB/HLDA")
    ensure = lambda p: p.mkdir(parents=True, exist_ok=True)
##    ensure(out_dir)
##    # Load counts
    df = pd.read_csv(counts_csv, index_col=0)
    cell_ids = df.index.tolist() 
##    # Initialize
    long_data = initialize_long_data(df, cell_ids,
                                     TOPIC_HIERARCHY_INT,
                                     IDENTITY_STR2INT,
                                     save_path=str(out_dir))
    n_cells = df.shape[0]
    n_genes = df.shape[1]
    constants = compute_constants(long_data, K, n_cells=n_cells, n_genes=n_genes)    # Sample
    run_gibbs_sampling(long_data, constants,
                       TOPIC_HIERARCHY_INT,
                       HYPERPARAMS, NUM_LOOPS, BURN_IN, THIN,
                       sample_dir=str(samples_dir))
    # Reconstruct
    reconstruct_beta_theta(str(samples_dir), str(out_dir),
                           gene_names=list(df.columns),
                           cell_id_strs=cell_ids)

if __name__ == "__main__":
    main()
