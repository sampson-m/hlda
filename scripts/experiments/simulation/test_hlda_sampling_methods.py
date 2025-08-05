#!/usr/bin/env python3
"""
test_hlda_sampling_methods.py

Test different HLDA sampling methods to address dominant node issues.
Methods: standard Gibbs, temperature annealing, blocked sampling, tempered sampling, multi-start
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import argparse
import time
from numba import njit
from numba.typed import List as TypedList
import sys
import os
# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.shared.fit_hlda import initialize_long_data, compute_constants, compute_posterior_means_from_memmaps
from scripts.simulation.simulation_evaluation_functions import compare_theta_true_vs_estimated, compare_beta_true_vs_estimated, create_stacked_comparison_plots

# Temperature annealing parameters
initial_temp = 2.0
final_temp = 0.5

@njit
def gibbs_standard_sampling(
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
    """Standard Gibbs sampling - can get stuck in dominant nodes."""
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
                total_p += probs[j]
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

@njit
def gibbs_annealing_sampling(
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
    """Temperature annealing Gibbs sampling."""
    total_tokens = cell_idx_arr.shape[0]
    G_abeta = G * alpha_beta
    cnt = 0

    temp_schedule = np.exp(np.linspace(np.log(initial_temp), np.log(final_temp), num_loops))

    for it in range(num_loops):
        current_temp = temp_schedule[it]
        
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
                # Temperature annealing
                if current_temp <= 1.0:
                    probs[j] = p
                else:
                    log_p = np.log(p + 1e-10)
                    probs[j] = np.exp(log_p / current_temp)
                total_p += probs[j]
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

@njit
def gibbs_blocked_sampling(
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
    """Blocked Gibbs sampling - sample all tokens for a cell together."""
    total_tokens = cell_idx_arr.shape[0]
    G_abeta = G * alpha_beta
    cnt = 0
    
    # Find unique cells and their token indices
    max_cell = 0
    for idx in range(total_tokens):
        if cell_idx_arr[idx] > max_cell:
            max_cell = cell_idx_arr[idx]
    
    # Create arrays to store cell token information
    cell_token_counts = np.zeros(max_cell + 1, dtype=np.int32)
    cell_token_starts = np.zeros(max_cell + 1, dtype=np.int32)
    
    # Count tokens per cell
    for idx in range(total_tokens):
        c = cell_idx_arr[idx]
        cell_token_counts[c] += 1
    
    # Compute starting positions
    for c in range(1, max_cell + 1):
        cell_token_starts[c] = cell_token_starts[c-1] + cell_token_counts[c-1]
    
    # Create array to store token indices for each cell
    cell_token_indices = np.zeros(total_tokens, dtype=np.int32)
    cell_token_positions = np.zeros(max_cell + 1, dtype=np.int32)
    
    # Fill token indices for each cell
    for idx in range(total_tokens):
        c = cell_idx_arr[idx]
        pos = cell_token_starts[c] + cell_token_positions[c]
        cell_token_indices[pos] = idx
        cell_token_positions[c] += 1

    for it in range(num_loops):
        # Sample all tokens for each cell together
        for c in range(max_cell + 1):
            if cell_token_counts[c] == 0:
                continue
                
            start_idx = cell_token_starts[c]
            end_idx = start_idx + cell_token_counts[c]
            
            # Remove all tokens for this cell
            for pos in range(start_idx, end_idx):
                idx = cell_token_indices[pos]
                g = gene_idx_arr[idx]
                old_z = z_arr[idx]
                A[g, old_z] -= 1
                B[old_z]    -= 1
                D[c, old_z] -= 1
            
            # Sample all tokens for this cell together
            for pos in range(start_idx, end_idx):
                idx = cell_token_indices[pos]
                g = gene_idx_arr[idx]
                ident = cell_identity_arr[idx]
                vt = valid_list[ident]
                m = vt.shape[0]
                total_p = 0.0
                probs = np.empty(m, np.float64)
                for j in range(m):
                    t = vt[j]
                    p = ((alpha_beta + A[g, t]) / (G_abeta + B[t])) * (alpha_c + D[c, t])
                    probs[j] = p
                    total_p += probs[j]
                
                # Handle case where all probabilities are zero
                if total_p == 0.0:
                    # Use uniform distribution over valid topics
                    for j in range(m):
                        probs[j] = 1.0 / m
                else:
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

@njit
def gibbs_tempered_sampling(
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
    """Tempered Gibbs sampling with periodic high-temperature moves."""
    total_tokens = cell_idx_arr.shape[0]
    G_abeta = G * alpha_beta
    cnt = 0

    for it in range(num_loops):
        # Every 100 iterations, do a high-temperature sweep to escape local optima
        if it % 100 == 0 and it > 0:
            escape_temp = 5.0  # High temperature for exploration
        else:
            escape_temp = 1.0  # Normal temperature
        
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
                # Use escape temperature for periodic high-T moves
                if escape_temp <= 1.0:
                    probs[j] = p
                else:
                    log_p = np.log(p + 1e-10)
                    probs[j] = np.exp(log_p / escape_temp)
                total_p += probs[j]
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

def apply_balanced_initialization(z0, cell_identities, topic_hierarchy_int, identity_str2int, n_identity, n_activity):
    """Apply balanced initialization to prevent dominant nodes."""
    print("  Using balanced initialization - equal topic representation")
    
    # Create mapping from cell identity to valid topics
    cell_to_topics = {}
    for cell_identity in cell_identities:
        if cell_identity in identity_str2int:
            ident_int = identity_str2int[cell_identity]
            valid_topics = topic_hierarchy_int[ident_int]
            cell_to_topics[cell_identity] = valid_topics
    
    topic_counts = {i: 0 for i in range(n_identity + n_activity)}
    
    for i, cell_identity in enumerate(cell_identities):
        if cell_identity in cell_to_topics:
            valid_topics = cell_to_topics[cell_identity]
            # Choose topic with lowest count among valid topics
            min_count = float('inf')
            chosen_topic = valid_topics[0]
            for topic in valid_topics:
                if topic_counts[topic] < min_count:
                    min_count = topic_counts[topic]
                    chosen_topic = topic
            z0[i] = chosen_topic
            topic_counts[chosen_topic] += 1
    
    # Print initialization statistics
    unique, counts = np.unique(z0, return_counts=True)
    topic_dist = dict(zip(unique, counts))
    print(f"  Initial topic distribution: {topic_dist}")
    
    return z0

def run_sampling_method(method_name, sampling_func, cell_idx_arr, gene_idx_arr, cell_identity_arr, z0, 
                       A, B, D, valid_list, alpha_beta, alpha_c, n_genes, total_topics, 
                       n_loops, burn_in, thin, A_chain, D_chain):
    """Run a specific sampling method and return results."""
    print(f"\nRunning {method_name} sampling...")
    start_time = time.time()
    
    # Reset arrays to initial state
    A_reset = A.copy()
    B_reset = B.copy()
    D_reset = D.copy()
    z0_reset = z0.copy()
    
    saved_count = sampling_func(
        cell_idx_arr.astype(np.int32),
        gene_idx_arr.astype(np.int32),
        cell_identity_arr.astype(np.int32),
        z0_reset.astype(np.int32),
        A_reset, B_reset, D_reset,
        valid_list,
        np.float64(alpha_beta), np.float64(alpha_c),
        np.int32(n_genes), np.int32(total_topics),
        np.int32(n_loops), np.int32(burn_in), np.int32(thin),
        A_chain, D_chain
    )
    
    elapsed = time.time() - start_time
    print(f"  {method_name} completed in {elapsed:.2f} seconds ({elapsed/60:.2f} min)")
    
    return saved_count, A_reset, B_reset, D_reset

def main():
    parser = argparse.ArgumentParser(description="Test different HLDA sampling methods")
    parser.add_argument("--counts_csv", type=str, required=True, help="Path to filtered count matrix CSV")
    parser.add_argument("--n_extra_topics", type=int, default=2, help="Number of extra activity topics")
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory to save results")
    parser.add_argument("--n_loops", type=int, default=2000, help="Number of Gibbs sweeps")
    parser.add_argument("--burn_in", type=int, default=1000, help="Burn-in iterations")
    parser.add_argument("--thin", type=int, default=10, help="Thinning interval")
    parser.add_argument("--alpha_beta", type=float, default=0.1, help="Dirichlet prior for beta")
    parser.add_argument("--alpha_c", type=float, default=0.1, help="Dirichlet prior for theta")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--config_file", type=str, default="dataset_identities.yaml", help="Config file")
    parser.add_argument("--true_theta", type=str, help="Path to true theta.csv")
    parser.add_argument("--true_beta", type=str, help="Path to true beta.csv")
    parser.add_argument("--train_cells", type=str, help="Path to train_cells.csv")
    parser.add_argument("--methods", type=str, nargs='+', default=['standard', 'annealing', 'blocked', 'tempered'], 
                       help="Sampling methods to run (default: all)")
    parser.add_argument("--run_comparison", action='store_true', 
                       help="Run comparison plots after sampling")
    parser.add_argument("--true_theta", type=str, help="Path to true theta.csv for comparison")
    parser.add_argument("--true_beta", type=str, help="Path to true beta.csv for comparison")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    if args.dataset not in config:
        raise ValueError(f"Dataset '{args.dataset}' not found in config file {args.config_file}")
    identity_topics = config[args.dataset]['identities']
    n_identity = len(identity_topics)
    n_activity = args.n_extra_topics
    total_topics = n_identity + n_activity

    # Build mappings
    IDENTITY_STR2INT = {cell_type: i for i, cell_type in enumerate(identity_topics)}
    TOPIC_STR2INT = {cell_type: i for i, cell_type in enumerate(identity_topics)}
    for i in range(n_activity):
        TOPIC_STR2INT[f"V{i+1}"] = n_identity + i
    TOPIC_INT2STR = {v: k for k, v in TOPIC_STR2INT.items()}

    # Create topic hierarchy
    topic_hierarchy_int = {
        IDENTITY_STR2INT[cell_type]: [TOPIC_STR2INT[cell_type]] + [TOPIC_STR2INT[f"V{i+1}"] for i in range(n_activity)]
        for cell_type in identity_topics
    }

    # Read count matrix
    print(f"Loading count matrix from {args.counts_csv}...")
    df = pd.read_csv(args.counts_csv, index_col=0)
    df = df[df.index != 'cell']
    df.index = pd.Series(df.index).str.split('_').str[0]
    
    cell_identities = list(df.index)
    n_genes = df.shape[1]
    n_cells = df.shape[0]
    print(f"Data: {df.shape[0]} cells, {df.shape[1]} genes")

    # Initialize long data format
    print("Initializing long data format...")
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

    cell_idx_arr = long_data['cell_idx']
    gene_idx_arr = long_data['gene_idx']
    cell_identity_arr = long_data['cell_identity']
    z0 = long_data['z']

    # Apply balanced initialization
    z0 = apply_balanced_initialization(z0, cell_identities, topic_hierarchy_int, 
                                     IDENTITY_STR2INT, n_identity, n_activity)
    long_data['z'] = z0

    # Create valid_list for Gibbs sampling
    valid_list = TypedList()
    for ident_int in sorted(topic_hierarchy_int):
        vt = np.array(topic_hierarchy_int[ident_int], dtype=np.int32)
        valid_list.append(vt)

    n_save = (args.n_loops - args.burn_in + 1) // args.thin

    # Define sampling methods
    all_sampling_methods = {
        'standard': gibbs_standard_sampling,
        'annealing': gibbs_annealing_sampling,
        'blocked': gibbs_blocked_sampling,
        'tempered': gibbs_tempered_sampling
    }
    
    # Filter to requested methods
    sampling_methods = {k: v for k, v in all_sampling_methods.items() if k in args.methods}
    print(f"Running methods: {list(sampling_methods.keys())}")

    # Run each sampling method
    results = {}
    for method_name, sampling_func in sampling_methods.items():
        print(f"\n{'='*60}")
        print(f"Testing {method_name.upper()} sampling method")
        print(f"{'='*60}")
        
        # Create method-specific output directory
        method_dir = output_dir / method_name
        method_dir.mkdir(exist_ok=True)
        sample_dir = method_dir / "samples"
        sample_dir.mkdir(exist_ok=True)

        # Clear existing memmap files
        A_chain_path = sample_dir / "A_chain.memmap"
        D_chain_path = sample_dir / "D_chain.memmap"
        if A_chain_path.exists():
            A_chain_path.unlink()
        if D_chain_path.exists():
            D_chain_path.unlink()

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

        # Run sampling
        saved_count, A_final, B_final, D_final = run_sampling_method(
            method_name, sampling_func, cell_idx_arr, gene_idx_arr, cell_identity_arr, z0,
            A, B, D, valid_list, args.alpha_beta, args.alpha_c, n_genes, total_topics,
            args.n_loops, args.burn_in, args.thin, A_chain, D_chain
        )

        if saved_count != n_save:
            print(f"  Warning: Expected {n_save} samples, got {saved_count}")

        # Compute posterior means
        print("  Computing posterior means...")
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

        # Rename columns
        topic_names = [TOPIC_INT2STR.get(i, f"Topic_{i}") for i in range(total_topics)]
        beta_df.columns = topic_names
        theta_df.columns = topic_names

        # Save results
        beta_df.to_csv(method_dir / f"{method_name}_beta.csv")
        theta_df.to_csv(method_dir / f"{method_name}_theta.csv")
        
        # Restore cell names if train_cells provided
        if args.train_cells and Path(args.train_cells).exists():
            train_cells = pd.read_csv(args.train_cells)['cell']
            if len(theta_df) == len(train_cells):
                theta_df.index = train_cells
                theta_df.to_csv(method_dir / f"{method_name}_theta.csv")

        # Generate scatter plots
        if args.true_theta and args.true_beta and Path(args.true_theta).exists() and Path(args.true_beta).exists():
            print("  Generating comparison scatter plots...")
            plot_dir = method_dir / "plots"
            plot_dir.mkdir(exist_ok=True)
            
            # Theta scatter plot
            compare_theta_true_vs_estimated(
                args.true_theta, 
                str(method_dir / f"{method_name}_theta.csv"), 
                str(plot_dir / f"theta_scatter_{method_name}.png"), 
                f"{method_name.upper()} Train"
            )
            
            # Beta scatter plot
            compare_beta_true_vs_estimated(
                args.true_beta, 
                str(method_dir / f"{method_name}_beta.csv"), 
                str(plot_dir / f"beta_scatter_{method_name}.png"), 
                method_name.upper()
            )
            print(f"  ✓ Plots saved to {plot_dir}")

        results[method_name] = {
            'beta_df': beta_df,
            'theta_df': theta_df,
            'method_dir': method_dir
        }

    print(f"\n{'='*60}")
    print("ALL SAMPLING METHODS COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print("Methods tested:")
    for method in results.keys():
        print(f"  - {method.upper()}: {results[method]['method_dir']}")
    
    # Run comparison plots if requested
    if args.run_comparison and args.true_theta and args.true_beta:
        print(f"\n{'='*60}")
        print("RUNNING COMPARISON PLOTS")
        print(f"{'='*60}")
        
        comparison_dir = output_dir / "comparison_plots"
        create_stacked_comparison_plots(
            results_dir=str(output_dir),
            output_dir=str(comparison_dir),
            true_theta=args.true_theta,
            true_beta=args.true_beta,
            dataset_name=args.dataset
        )

if __name__ == "__main__":
    main() 