import argparse
import pandas as pd
from pathlib import Path
import numpy as np
from numba import njit
from numba.typed import List as TypedList
import time

# --- PBMC Cell Type Definitions ---------------------------------------------------------
# Define the PBMC cell types from your data
PBMC_CELL_TYPES = [
    'T cells',
    'CD19+ B', 
    'CD56+ NK',
    'CD34+',
    'Dendritic',
    'CD14+ Monocyte'
]

# Identity mappings (just the cell types, not activity topics)
IDENTITY_STR2INT = {cell_type: i for i, cell_type in enumerate(PBMC_CELL_TYPES)}
IDENTITY_INT2STR = {v: k for k, v in IDENTITY_STR2INT.items()}

# --- HLDA/Gibbs functions and constants moved from gibbs.py ---
def initialize_long_data(count_matrix: pd.DataFrame,
                         cell_identities: list[str],
                         topic_hierarchy_int: dict[int, list[int]],
                         identity_str2int: dict[str, int],
                         n_activity: int,
                         save_path: str | None = None):
    """Initialize long data format for Gibbs sampling."""
    # For PBMC data, cell identities are already the cell type labels
    cell_ints = np.array([identity_str2int[c] for c in cell_identities], dtype=np.int32)
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
        if np.random.rand() < p_activity and n_activity > 0:
            z0[i] = np.random.choice([valid[j] for j in range(1, len(valid))])
        else:
            z0[i] = valid[0]
    long_data = {
        'cell_idx': cell_rep,
        'gene_idx': gene_rep,
        'cell_identity': ident_rep,
        'z': z0
    }
    return long_data, n_cells, n_genes

def compute_constants(long_data, K: int, n_cells: int, n_genes: int):
    """Compute initial constants for Gibbs sampling."""
    cell_idx_arr = long_data['cell_idx']
    gene_idx_arr = long_data['gene_idx']
    z_arr         = long_data['z']
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

@njit
def gibbs_and_write_memmap(
    cell_idx_arr,
    gene_idx_arr,
    cell_identity_arr,
    z_arr,
    A,
    B,
    D,
    valid_list,
    alpha_beta,
    alpha_c,
    G,
    K,
    num_loops,
    burn_in,
    thin,
    A_chain,
    D_chain
):
    """Run Gibbs sampling and write results to memmap files."""
    total_tokens = cell_idx_arr.shape[0]
    G_abeta = G * alpha_beta
    cnt = 0
    for it in range(num_loops):
        for idx in range(total_tokens):
            c = cell_idx_arr[idx]
            g = gene_idx_arr[idx]
            ident = cell_identity_arr[idx]
            old_z = z_arr[idx]
            A[g, old_z] -= 1
            B[old_z]    -= 1
            D[c, old_z] -= 1
            vt = valid_list[ident]
            m = vt.shape[0]
            total_p = 0.0
            probs = np.empty(m, np.float64)
            for j in range(m):
                t = vt[j]
                p = ((alpha_beta + A[g, t]) / (G_abeta + B[t])) * (alpha_c + D[c, t])
                probs[j] = p
                total_p += p
            for j in range(m):
                probs[j] /= total_p
            r = np.random.rand()
            cum = 0.0
            new_z = vt[m - 1]
            for j in range(m):
                cum += probs[j]
                if r < cum:
                    new_z = vt[j]
                    break
            A[g, new_z] += 1
            B[new_z]    += 1
            D[c, new_z] += 1
            z_arr[idx]   = new_z
        if it >= burn_in and ((it - burn_in + 1) % thin == 0):
            for gg in range(G):
                for tt in range(K):
                    A_chain[cnt, gg, tt] = A[gg, tt]
            n_cells = D.shape[0]
            for cc in range(n_cells):
                for tt in range(K):
                    D_chain[cnt, cc, tt] = D[cc, tt]
            cnt += 1
    return cnt

def compute_posterior_means_from_memmaps(A_chain_path: Path,
                                         D_chain_path: Path,
                                         n_save: int,
                                         n_genes: int,
                                         n_cells: int,
                                         K: int,
                                         df_genes: pd.Index,
                                         df_cells: list[str]):
    """Compute posterior means from saved memmap files."""
    A_chain = np.memmap(filename=str(A_chain_path),
                        mode="r",
                        dtype=np.int32,
                        shape=(n_save, n_genes, K))
    D_chain = np.memmap(filename=str(D_chain_path),
                        mode="r",
                        dtype=np.int32,
                        shape=(n_save, n_cells, K))
    Beta_sum = np.zeros((n_genes, K), dtype=np.float64)
    Theta_sum = np.zeros((n_cells, K), dtype=np.float64)
    for idx in range(n_save):
        Beta_sum += A_chain[idx, :, :].astype(np.float64)
        Theta_sum += D_chain[idx, :, :].astype(np.float64)
    Beta_mean_counts = Beta_sum / n_save
    Theta_mean_counts = Theta_sum / n_save
    Beta_mean = Beta_mean_counts / Beta_mean_counts.sum(axis=0, keepdims=True)
    Theta_mean = Theta_mean_counts / Theta_mean_counts.sum(axis=1, keepdims=True)
    beta_df = pd.DataFrame(Beta_mean,
                           index=pd.Index(df_genes),
                           columns=pd.Index([str(t) for t in range(K)]))
    theta_df = pd.DataFrame(Theta_mean,
                            index=pd.Index(df_cells),
                            columns=pd.Index([str(t) for t in range(K)]))
    return beta_df, theta_df
# --- End of moved functions ---

def main():
    parser = argparse.ArgumentParser(description="Fit HLDA (Gibbs) to PBMC count matrix with cell type identity topics.")
    parser.add_argument("--counts_csv", type=str, required=True, help="Path to filtered count matrix CSV (index: cell type)")
    parser.add_argument("--n_extra_topics", type=int, default=2, help="Number of extra activity topics (V1, V2, ...) (default: 2)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save HLDA outputs")
    parser.add_argument("--n_loops", type=int, default=10000, help="Number of Gibbs sweeps (default: 10000)")
    parser.add_argument("--burn_in", type=int, default=4000, help="Burn-in iterations (default: 4000)")
    parser.add_argument("--thin", type=int, default=20, help="Thinning interval (default: 20)")
    parser.add_argument("--alpha_beta", type=float, default=0.1, help="Dirichlet prior for beta (default: 0.1)")
    parser.add_argument("--alpha_c", type=float, default=0.1, help="Dirichlet prior for theta (default: 0.1)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Set up topic structure
    n_identity = len(PBMC_CELL_TYPES)
    n_activity = args.n_extra_topics
    total_topics = n_identity + n_activity

    # Build topic names: identity topics + V1, V2, ...
    identity_topics = PBMC_CELL_TYPES
    activity_topics = [f"V{i+1}" for i in range(n_activity)]
    all_topics = identity_topics + activity_topics

    # Build TOPIC_STR2INT mapping dynamically based on number of activity topics
    TOPIC_STR2INT = {}
    # Identity topics (one per cell type)
    for i, cell_type in enumerate(PBMC_CELL_TYPES):
        TOPIC_STR2INT[cell_type] = i
    # Activity topics (V1, V2, ...)
    for i in range(n_activity):
        TOPIC_STR2INT[f"V{i+1}"] = n_identity + i

    TOPIC_INT2STR = {v: k for k, v in TOPIC_STR2INT.items()}

    # Create topic hierarchy: each cell type maps to its own topic + all activity topics
    topic_hierarchy_int = {
        IDENTITY_STR2INT[cell_type]: 
            [TOPIC_STR2INT[cell_type]] + [TOPIC_STR2INT[f"V{i+1}"] for i in range(n_activity)]
        for cell_type in PBMC_CELL_TYPES
    }

    # Debug: Print topic mappings and hierarchy
    print(f"\n=== DEBUG: Topic Mappings ===")
    print(f"TOPIC_STR2INT:")
    for topic, idx in TOPIC_STR2INT.items():
        print(f"  '{topic}' -> {idx}")
    
    print(f"\nTOPIC_INT2STR:")
    for idx, topic in TOPIC_INT2STR.items():
        print(f"  {idx} -> '{topic}'")
    
    print(f"\nIDENTITY_STR2INT:")
    for cell_type, idx in IDENTITY_STR2INT.items():
        print(f"  '{cell_type}' -> {idx}")
    
    print(f"\nTopic Hierarchy (topic_hierarchy_int):")
    for ident_int, valid_topics in topic_hierarchy_int.items():
        cell_type = IDENTITY_INT2STR[ident_int]
        valid_names = [TOPIC_INT2STR[t] for t in valid_topics]
        print(f"  {ident_int} ({cell_type}) -> {valid_topics} ({valid_names})")
    
    # Debug validation checks
    print(f"\n=== DEBUG: Validation Checks ===")
    print(f"Expected total topics: {total_topics}")
    print(f"Actual TOPIC_STR2INT size: {len(TOPIC_STR2INT)}")
    print(f"Identity topics: {n_identity}")
    print(f"Activity topics: {n_activity}")
    
    # Check that all identity topics are in TOPIC_STR2INT
    missing_identity = [ct for ct in PBMC_CELL_TYPES if ct not in TOPIC_STR2INT]
    if missing_identity:
        print(f"ERROR: Missing identity topics: {missing_identity}")
    else:
        print(f"✓ All identity topics present in TOPIC_STR2INT")
    
    # Check that all activity topics are in TOPIC_STR2INT
    expected_activity = [f"V{i+1}" for i in range(n_activity)]
    missing_activity = [at for at in expected_activity if at not in TOPIC_STR2INT]
    if missing_activity:
        print(f"ERROR: Missing activity topics: {missing_activity}")
    else:
        print(f"✓ All activity topics present in TOPIC_STR2INT")
    
    # Check topic hierarchy structure
    print(f"\nTopic hierarchy validation:")
    for ident_int, valid_topics in topic_hierarchy_int.items():
        cell_type = IDENTITY_INT2STR[ident_int]
        # Check that identity topic is first
        if valid_topics[0] != TOPIC_STR2INT[cell_type]:
            print(f"ERROR: {cell_type} identity topic not first in hierarchy")
        else:
            print(f"✓ {cell_type}: identity topic {valid_topics[0]} is first")
        
        # Check that all activity topics are included
        expected_activity_indices = [TOPIC_STR2INT[f"V{i+1}"] for i in range(n_activity)]
        missing_activity_indices = [idx for idx in expected_activity_indices if idx not in valid_topics[1:]]
        if missing_activity_indices:
            print(f"ERROR: {cell_type} missing activity topics: {missing_activity_indices}")
        else:
            print(f"✓ {cell_type}: all {n_activity} activity topics included")

    print(f"\nPBMC HLDA Configuration:")
    print(f"  Cell types (identity topics): {PBMC_CELL_TYPES}")
    print(f"  Activity topics: {activity_topics}")
    print(f"  Total topics: {total_topics}")
    print(f"  Topic hierarchy:")
    for cell_type in PBMC_CELL_TYPES:
        ident_int = IDENTITY_STR2INT[cell_type]
        valid_topics = topic_hierarchy_int[ident_int]
        valid_names = [TOPIC_INT2STR[t] for t in valid_topics]
        print(f"    {cell_type} -> {valid_names}")

    # Read count matrix
    df = pd.read_csv(args.counts_csv, index_col=0)
    cell_identities = list(df.index)
    n_genes = df.shape[1]
    n_cells = df.shape[0]

    print(f"\nData summary:")
    print(f"  Count matrix shape: {df.shape}")
    print(f"  Cell types in data: {sorted(set(cell_identities))}")
    print(f"  Cell type distribution:")
    cell_type_counts = pd.Series(cell_identities).value_counts()
    for cell_type, count in cell_type_counts.items():
        print(f"    {cell_type}: {count} cells")

    # Initialize long data
    long_data, n_cells, n_genes = initialize_long_data(
        count_matrix=df,
        cell_identities=cell_identities,
        topic_hierarchy_int=topic_hierarchy_int,
        identity_str2int=IDENTITY_STR2INT,
        n_activity=n_activity,
        save_path=None
    )

    # Compute constants
    consts = compute_constants(long_data, total_topics, n_cells, n_genes)
    A = consts['A'].copy()
    B = consts['B'].copy()
    D = consts['D'].copy()

    cell_idx_arr      = long_data['cell_idx']
    gene_idx_arr      = long_data['gene_idx']
    cell_identity_arr = long_data['cell_identity']
    z0                = long_data['z']

    alpha_beta = args.alpha_beta
    alpha_c    = args.alpha_c

    # Create valid_list for Gibbs sampling
    valid_list = TypedList()
    for ident_int in sorted(topic_hierarchy_int):
        vt = np.array(topic_hierarchy_int[ident_int], dtype=np.int32)
        valid_list.append(vt)

    n_save = (args.n_loops - args.burn_in + 1) // args.thin

    A_chain_path = sample_dir / "A_chain.memmap"
    D_chain_path = sample_dir / "D_chain.memmap"

    A_chain = np.memmap(
        filename=str(A_chain_path),
        mode="w+",
        dtype=np.int32,
        shape=(n_save, n_genes, total_topics)
    )
    D_chain = np.memmap(
        filename=str(D_chain_path),
        mode="w+",
        dtype=np.int32,
        shape=(n_save, n_cells, total_topics)
    )

    # Run Gibbs sampler
    print(f"\nRunning HLDA Gibbs sampler with {total_topics} topics ({n_identity} identity, {n_activity} activity)...")
    start_time = time.time()
    saved_count = gibbs_and_write_memmap(
        cell_idx_arr.astype(np.int32),
        gene_idx_arr.astype(np.int32),
        cell_identity_arr.astype(np.int32),
        z0.astype(np.int32),
        A, B, D,
        valid_list,
        np.float64(alpha_beta), np.float64(alpha_c),
        np.int32(n_genes), np.int32(total_topics),
        np.int32(args.n_loops), np.int32(args.burn_in), np.int32(args.thin),
        A_chain, D_chain
    )
    elapsed = time.time() - start_time
    print(f"HLDA Gibbs sampling completed in {elapsed:.2f} seconds ({elapsed/60:.2f} min)")

    if saved_count != n_save:
        raise RuntimeError(f"Expected to save {n_save} slices, but Numba returned {saved_count}")

    # Compute posterior means
    beta_df, theta_df = compute_posterior_means_from_memmaps(
        A_chain_path=A_chain_path,
        D_chain_path=D_chain_path,
        n_save=n_save,
        n_genes=n_genes,
        n_cells=n_cells,
        K=total_topics,
        df_genes=df.columns,
        df_cells=cell_identities
    )

    # Rename columns to use topic names instead of numbers
    topic_names = [TOPIC_INT2STR.get(i, f"Topic_{i}") for i in range(total_topics)]
    beta_df.columns = topic_names
    theta_df.columns = topic_names

    # Save outputs
    hlda_dir = output_dir
    hlda_dir.mkdir(parents=True, exist_ok=True)
    beta_df.to_csv(hlda_dir / "HLDA_beta.csv")
    theta_df.to_csv(hlda_dir / "HLDA_theta.csv")
    # Save topic mapping info
    topic_info = pd.DataFrame({
        'topic_id': range(total_topics),
        'topic_name': topic_names,
        'topic_type': ['identity'] * n_identity + ['activity'] * n_activity
    })
    topic_info.to_csv(hlda_dir / "topic_info.csv", index=False)
    print(f"\n✔ HLDA fit complete. Results saved to {hlda_dir}")
    print(f"  • HLDA_beta.csv: Gene-topic distributions")
    print(f"  • HLDA_theta.csv: Cell-topic distributions") 
    print(f"  • topic_info.csv: Topic mapping information")
    print(f"  • samples/: Gibbs sampling chains")

if __name__ == "__main__":
    main() 