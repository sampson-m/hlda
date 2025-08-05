import argparse
import pandas as pd
from pathlib import Path
import numpy as np
from numba import njit
from numba.typed import List as TypedList
import time
import yaml
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize

# --- HLDA/Gibbs functions and constants moved from gibbs.py ---
def initialize_long_data_old(count_matrix: pd.DataFrame,
                         cell_identities: list[str],
                         topic_hierarchy_int: dict[int, list[int]],
                         identity_str2int: dict[str, int],
                         n_activity: int,
                         save_path: str | None = None):
    """OLD: Initialize long data format for Gibbs sampling with random assignment."""
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

def estimate_identity_betas(count_matrix: pd.DataFrame, 
                           cell_identities: list[str], 
                           identity_topics: list[str]):
    """Estimate identity topic betas from average gene expression per cell type."""
    identity_betas = {}
    
    print("[INIT] Estimating identity topic betas from data...")
    
    for cell_type in identity_topics:
        # Find cells of this type
        type_mask = np.array([cell_type in cell_id for cell_id in cell_identities])
        n_cells_type = type_mask.sum()
        
        if n_cells_type > 0:
            # Average gene expression for this cell type
            type_data = count_matrix.loc[type_mask]
            avg_expr = type_data.mean(axis=0).values
            
            # Add small pseudocount and normalize to probability distribution
            avg_expr = avg_expr + 1e-10
            identity_betas[cell_type] = avg_expr / avg_expr.sum()
            
            print(f"[INIT]   {cell_type}: {n_cells_type} cells, avg total counts = {avg_expr.sum():.1f}")
        else:
            # Fallback to uniform if no cells of this type
            n_genes = count_matrix.shape[1]
            identity_betas[cell_type] = np.ones(n_genes) / n_genes
            print(f"[INIT]   {cell_type}: 0 cells, using uniform beta")
    
    return identity_betas



def compute_cell_identity_thetas(count_matrix: pd.DataFrame,
                                cell_identities: list[str],
                                identity_topics: list[str],
                                identity_betas: dict):
    """Compute identity theta for each cell based on how well it matches its own identity topic."""
    print("[INIT] Computing cell-specific identity thetas...")
    
    cell_identity_thetas = []
    own_similarities = []
    
    for i, cell_id in enumerate(cell_identities):
        cell_data = count_matrix.iloc[i].values.astype(float)
        cell_total = cell_data.sum()
        
        if cell_total == 0:
            cell_identity_thetas.append(0.5)  # default for empty cells
            own_similarities.append(0.0)
            continue
            
        # Normalize to probability distribution
        cell_data_norm = cell_data / cell_total
        
        # Find which identity this cell belongs to
        cell_type = None
        for identity in identity_topics:
            if identity in cell_id:
                cell_type = identity
                break
        
        if cell_type and cell_type in identity_betas:
            identity_beta = identity_betas[cell_type]
            
            # Compute correlation with its OWN identity topic only
            corr = np.corrcoef(cell_data_norm, identity_beta)[0, 1]
            # Handle NaN correlations (e.g., constant vectors)
            own_sim = 0.0 if np.isnan(corr) else corr
            
            own_similarities.append(own_sim)
            
            # Convert correlation to identity theta using sigmoid transformation
            # Higher correlation -> higher identity theta
            # Clamp correlation to [0,1] first
            corr_clamped = np.clip(own_sim, 0.0, 1.0)
            # Sigmoid transformation: smooth S-curve from 0.2 to 0.9
            # Centers transition around correlation = 0.7
            identity_theta = 0.2 + 0.7 / (1 + np.exp(-10 * (corr_clamped - 0.7)))
        else:
            identity_theta = 0.6  # default for unidentified cells
            own_similarities.append(0.0)
            
        cell_identity_thetas.append(identity_theta)
    
    # Report statistics
    own_similarities = np.array(own_similarities)
    cell_identity_thetas = np.array(cell_identity_thetas)
    
    print(f"[INIT]   Correlation with own identity: mean={np.mean(own_similarities):.3f}, std={np.std(own_similarities):.3f}")
    print(f"[INIT]   Correlation range: [{np.min(own_similarities):.3f}, {np.max(own_similarities):.3f}]")
    print(f"[INIT]   Identity theta stats: mean={np.mean(cell_identity_thetas):.3f}, std={np.std(cell_identity_thetas):.3f}")
    print(f"[INIT]   Identity theta range: [{np.min(cell_identity_thetas):.3f}, {np.max(cell_identity_thetas):.3f}]")
    
    return cell_identity_thetas.tolist()



def initialize_long_data(count_matrix: pd.DataFrame,
                        cell_identities: list[str],
                        topic_hierarchy_int: dict[int, list[int]],
                        identity_str2int: dict[str, int],
                        identity_topics: list[str],
                        n_activity: int,
                        save_path: str | None = None):
    """NEW: Initialize long data using improved data-driven approach with random activity topic assignment."""
    print("[INIT] ========== Improved Data-Driven HLDA Initialization ===========")
    
    # Step 1: Estimate identity topic betas from data
    identity_betas = estimate_identity_betas(count_matrix, cell_identities, identity_topics)
    
    # Step 2: Compute cell-specific identity thetas
    cell_identity_thetas = compute_cell_identity_thetas(count_matrix, cell_identities,
                                                       identity_topics, identity_betas)
    
    # Step 3: Convert to token-level assignments
    print("[INIT] Converting to token-level topic assignments...")
    
    cell_ints = np.array([identity_str2int[c] for c in cell_identities], dtype=np.int32)
    mat = count_matrix.to_numpy().astype(np.int32)
    n_cells, n_genes = mat.shape
    rows, cols = np.nonzero(mat)
    counts = mat[rows, cols]
    cell_rep = np.repeat(rows.astype(np.int32), counts.astype(np.int32))
    gene_rep = np.repeat(cols.astype(np.int32), counts.astype(np.int32))
    total_tokens = cell_rep.shape[0]
    ident_rep = cell_ints[cell_rep]
    
    print(f"[INIT]   Total tokens: {total_tokens:,}")
    print(f"[INIT]   Cells: {n_cells}, Genes: {n_genes}")
    
    z0 = np.empty(total_tokens, dtype=np.int32)
    identity_assignments = 0
    activity_assignments = 0
    activity_topic_counts = {f"V{i+1}": 0 for i in range(n_activity)}
    
    for i in range(total_tokens):
        cell_idx = cell_rep[i]
        ident_val = ident_rep[i]
        valid = topic_hierarchy_int[ident_val]
        
        # Use precomputed identity theta for this cell
        identity_theta = cell_identity_thetas[cell_idx]
        
        if len(valid) > 1 and np.random.rand() > identity_theta:
            # Assign to activity topic randomly (uniform distribution)
            activity_topics = valid[1:]  # [V1, V2, ...]
            z0[i] = np.random.choice(activity_topics)
            
            activity_assignments += 1
            # Track which activity topic was assigned
            topic_name = f"V{z0[i] - len(identity_topics) + 1}"
            if topic_name in activity_topic_counts:
                activity_topic_counts[topic_name] += 1
        else:
            # Assign to identity topic  
            z0[i] = valid[0]
            identity_assignments += 1
    
    identity_frac = identity_assignments / total_tokens
    activity_frac = activity_assignments / total_tokens
    
    print(f"[INIT]   Final assignments: {identity_frac:.1%} identity, {activity_frac:.1%} activity")
    if n_activity > 0:
        print(f"[INIT]   Activity topic breakdown:")
        for topic, count in activity_topic_counts.items():
            topic_frac = count / total_tokens
            print(f"[INIT]     {topic}: {topic_frac:.1%} ({count:,} tokens)")
    print(f"[INIT] ========== Initialization Complete ===========")
    
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

def get_default_parameters():
    """Return default parameters for HLDA fitting."""
    return {
        'n_loops': 15000,
        'burn_in': 5000,
        'thin': 40,
        'alpha_beta': 0.1,
        'alpha_c': 0.1
    }

def main():
    parser = argparse.ArgumentParser(description="Fit HLDA (Gibbs) to count matrix with cell type identity topics.")
    parser.add_argument("--counts_csv", type=str, required=True, help="Path to filtered count matrix CSV (index: cell type)")
    parser.add_argument("--n_extra_topics", type=int, default=2, help="Number of extra activity topics (V1, V2, ...) (default: 2)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save HLDA outputs")
    parser.add_argument("--n_loops", type=int, default=15000, help="Number of Gibbs sweeps (default: 15000)")
    parser.add_argument("--burn_in", type=int, default=5000, help="Burn-in iterations (default: 5000)")
    parser.add_argument("--thin", type=int, default=40, help="Thinning interval (default: 40)")
    parser.add_argument("--alpha_beta", type=float, default=0.1, help="Dirichlet prior for beta (default: 0.1)")
    parser.add_argument("--alpha_c", type=float, default=0.1, help="Dirichlet prior for theta (default: 0.1)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (must match a key in the config file)")
    parser.add_argument("--config_file", type=str, default="dataset_identities.yaml", help="Path to dataset identity config YAML file")
    parser.add_argument("--use_lda_init", action="store_true", help="If set, use LDA to initialize HLDA sampler.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Load identity topics from config file
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    if args.dataset not in config:
        raise ValueError(f"Dataset '{args.dataset}' not found in config file {args.config_file}")
    identity_topics = config[args.dataset]['identities']
    n_identity = len(identity_topics)
    n_activity = args.n_extra_topics
    total_topics = n_identity + n_activity

    # Build topic names: identity topics + V1, V2, ...
    activity_topics = [f"V{i+1}" for i in range(n_activity)]
    all_topics = identity_topics + activity_topics

    # Build mappings dynamically
    IDENTITY_STR2INT = {cell_type: i for i, cell_type in enumerate(identity_topics)}
    IDENTITY_INT2STR = {v: k for k, v in IDENTITY_STR2INT.items()}
    TOPIC_STR2INT = {cell_type: i for i, cell_type in enumerate(identity_topics)}
    for i in range(n_activity):
        TOPIC_STR2INT[f"V{i+1}"] = n_identity + i
    TOPIC_INT2STR = {v: k for k, v in TOPIC_STR2INT.items()}

    # Create topic hierarchy: each cell type maps to its own topic + all activity topics
    topic_hierarchy_int = {
        IDENTITY_STR2INT[cell_type]: [TOPIC_STR2INT[cell_type]] + [TOPIC_STR2INT[f"V{i+1}"] for i in range(n_activity)]
        for cell_type in identity_topics
    }

    # Read count matrix
    df = pd.read_csv(args.counts_csv, index_col=0)
    cell_identities = list(df.index)
    n_genes = df.shape[1]
    n_cells = df.shape[0]

    print(f"Data: {df.shape[0]} cells, {df.shape[1]} genes")

    lda_theta = None
    lda_beta = None
    if args.use_lda_init:
        print("[INFO] Running LDA for HLDA initialization...")
        X = df.values.astype(float)
        lda = LatentDirichletAllocation(n_components=total_topics, random_state=0, max_iter=100)
        lda_theta = lda.fit_transform(X)
        lda_theta = lda_theta / lda_theta.sum(axis=1, keepdims=True)
        lda_beta = normalize(lda.components_, norm="l1", axis=1).T
        print("[INFO] LDA initialization complete. Using LDA theta to bias HLDA initialization.")
    
    if args.use_lda_init and lda_theta is not None:
        # Use LDA theta to bias initial topic assignments
        mat = df.to_numpy().astype(np.int32)
        n_cells, n_genes = mat.shape
        rows, cols = np.nonzero(mat)
        counts = mat[rows, cols]
        cell_rep = np.repeat(rows.astype(np.int32), counts.astype(np.int32))
        gene_rep = np.repeat(cols.astype(np.int32), counts.astype(np.int32))
        total_tokens = cell_rep.shape[0]
        ident_rep = np.array([IDENTITY_STR2INT[cell_identities[c]] for c in cell_rep], dtype=np.int32)
        z0 = np.empty(total_tokens, dtype=np.int32)
        for i in range(total_tokens):
            cell_idx = cell_rep[i]
            valid = topic_hierarchy_int[ident_rep[i]]
            # Use LDA theta for this cell to bias topic assignment
            probs = np.array([lda_theta[cell_idx, t] for t in valid])
            if probs.sum() == 0:
                probs = np.ones(len(valid)) / len(valid)
            else:
                probs = probs / probs.sum()
            z0[i] = valid[np.random.choice(len(valid), p=probs)]
        long_data = {
            'cell_idx': cell_rep,
            'gene_idx': gene_rep,
            'cell_identity': ident_rep,
            'z': z0
        }
        n_cells = df.shape[0]
        n_genes = df.shape[1]
    else:
        long_data, n_cells, n_genes = initialize_long_data(
            count_matrix=df,
            cell_identities=cell_identities,
            topic_hierarchy_int=topic_hierarchy_int,
            identity_str2int=IDENTITY_STR2INT,
            identity_topics=identity_topics,
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
    print(f"HLDA Gibbs sampler parameters: n_loops={args.n_loops}, burn_in={args.burn_in}, thin={args.thin}")
    print(f"Running HLDA Gibbs sampler ({total_topics} topics)...")
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
    print("Computing posterior means...")
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
    print(f"HLDA fit complete. Results saved to {hlda_dir}")

if __name__ == "__main__":
    main() 