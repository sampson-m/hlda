#!/usr/bin/env python3
"""
combined_evaluation.py

Combined evaluation script that creates combined theta scatter plots and 
runs noise analysis using true parameters.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the simulation directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulation_evaluation_functions import (
    compare_theta_true_vs_estimated, 
    compare_beta_true_vs_estimated, 
    combine_theta_scatter_plots,
    combine_beta_scatter_plots
)
# Import noise analysis function from shared directory
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared'))

def generate_noise_analysis(simulation_dir, output_dir, libsize_mean=1500, n_identity_topics=2):
    """
    Generate noise analysis using true parameters from simulation.
    Computes three Frobenius norms from noise_analysis.tex derivation.
    
    Args:
        simulation_dir: Path to simulation output directory
        output_dir: Path to save noise analysis results
        libsize_mean: Library size parameter used in simulation
        n_identity_topics: Number of identity topics
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from scipy.linalg import qr
    
    simulation_dir = Path(simulation_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load simulation data
    theta_path = simulation_dir / "theta.csv"
    beta_path = simulation_dir / "gene_means.csv" 
    counts_path = simulation_dir / "counts.csv"
    
    if not all(p.exists() for p in [theta_path, beta_path, counts_path]):
        print("Warning: Required simulation files not found for noise analysis")
        return
    
    try:
        theta = pd.read_csv(theta_path, index_col=0).values  # Shape: (C, K)
        beta = pd.read_csv(beta_path, index_col=0).values    # Shape: (G, K)
        X = pd.read_csv(counts_path, index_col=0).values     # Shape: (C, G)
        
        print(f"    Loaded data shapes - X: {X.shape}, theta: {theta.shape}, beta: {beta.shape}")
        
        # Normalize beta to probabilities
        beta = beta / np.sum(beta, axis=0)
        
        # Expected values: L * theta * beta^T 
        L_theta_beta = libsize_mean * np.dot(theta, beta.T)  # Shape: (C, G)
        
        # Normalized count matrix: X / sqrt(L * theta * beta^T)
        X_normalized = X / np.sqrt(L_theta_beta)
        
        # QR decomposition on beta matrix (G x K)
        Q, R = qr(beta, mode='economic')  # Q: (G, K), R: (K, K)
        
        # Transformed parameters
        theta_tilde = np.dot(theta, R.T)  # Shape: (C, K)
        beta_tilde = Q  # Shape: (G, K) - orthogonal
        
        # Split into identity and activity topics
        identity_indices = list(range(n_identity_topics))
        activity_index = n_identity_topics  # Assuming last topic is activity
        
        # Identity topic components
        theta_i_tilde = theta_tilde[:, identity_indices]  # Shape: (C, n_identity)
        beta_i_tilde = beta_tilde[:, identity_indices]    # Shape: (G, n_identity)
        
        # Activity topic components
        theta_A_tilde = theta_tilde[:, activity_index]    # Shape: (C,)
        beta_A_tilde = beta_tilde[:, activity_index]      # Shape: (G,)
        
        # Decompose activity topic: beta_A = sum_i s_i * beta_i + s_A_perp * beta_A_perp
        s_coefficients = np.dot(beta_i_tilde.T, beta_A_tilde)  # Shape: (n_identity,)
        identity_component = np.dot(beta_i_tilde, s_coefficients)  # Shape: (G,)
        beta_A_perp = beta_A_tilde - identity_component
        s_A_perp = np.linalg.norm(beta_A_perp)
        
        if s_A_perp > 1e-10:
            beta_A_perp_unit = beta_A_perp / s_A_perp
        else:
            beta_A_perp_unit = beta_A_perp
        
        print(f"    s_A_perp = {s_A_perp:.6f}")
        
        # Compute the three Frobenius norms from derivation
        
        # 1. NOISE: ||X/√(Lθβ^T) - (Poisson(Lθβ^T) - Lθβ^T)/√(Lθβ^T)||_F
        expected_normalized = np.sqrt(L_theta_beta)
        noise_component = X_normalized - expected_normalized
        frobenius_noise = np.linalg.norm(noise_component, 'fro')
        
        # 2. IDENTITY SIGNAL: Remove identity + projected activity components
        # Identity contribution: sum_i theta_i_tilde * beta_i_tilde^T
        identity_contribution = np.dot(theta_i_tilde, beta_i_tilde.T)  # Shape: (C, G)
        
        # Projected activity contribution: theta_A_tilde * (sum_i s_i * beta_i_tilde)^T  
        projected_activity = np.outer(theta_A_tilde, identity_component)  # Shape: (C, G)
        
        # Combined identity + projected activity signal
        identity_plus_projected = libsize_mean * (identity_contribution + projected_activity)
        identity_signal_normalized = identity_plus_projected / np.sqrt(L_theta_beta)
        identity_residual = X_normalized - identity_signal_normalized
        frobenius_identity_signal = np.linalg.norm(identity_residual, 'fro')
        
        # 3. ACTIVITY SIGNAL: Pure orthogonal activity component
        # Activity signal: theta_A_tilde * (s_A_perp * beta_A_perp_unit)^T
        pure_activity_signal = libsize_mean * np.outer(theta_A_tilde, s_A_perp * beta_A_perp_unit)  # Shape: (C, G)
        activity_signal_normalized = pure_activity_signal / np.sqrt(L_theta_beta)
        activity_residual = X_normalized - activity_signal_normalized  
        frobenius_activity_signal = np.linalg.norm(activity_residual, 'fro')
        
        # Save results
        results = {
            'frobenius_noise': frobenius_noise,
            'frobenius_identity_signal': frobenius_identity_signal, 
            'frobenius_activity_signal': frobenius_activity_signal,
            's_A_perp': s_A_perp,
            'libsize_mean': libsize_mean,
            'n_identity_topics': n_identity_topics
        }
        
        results_df = pd.DataFrame([results])
        results_df.to_csv(output_dir / "noise_analysis_results.csv", index=False)
        
        print(f"    Noise analysis results:")
        print(f"      Frobenius noise: {frobenius_noise:.6f}")
        print(f"      Frobenius identity signal: {frobenius_identity_signal:.6f}")
        print(f"      Frobenius activity signal: {frobenius_activity_signal:.6f}")
        print(f"    Saved to: {output_dir / 'noise_analysis_results.csv'}")
        
    except Exception as e:
        print(f"Error in noise analysis: {e}")
        import traceback
        traceback.print_exc()
        return

def run_combined_evaluation(simulation_dir, output_dir, models=['HLDA', 'LDA', 'NMF'], 
                          topic_config="3_topic_fit", libsize_mean=1500, n_identity_topics=2):
    """
    Run combined evaluation including theta scatter plots and noise analysis.
    
    Args:
        simulation_dir: Directory containing simulation outputs
        output_dir: Directory containing model estimation results
        models: List of model names to evaluate
        topic_config: Topic configuration subdirectory name
        libsize_mean: Library size parameter used in simulation
        n_identity_topics: Number of identity topics
    """
    simulation_dir = Path(simulation_dir)
    output_dir = Path(output_dir)
    
    # Paths
    true_theta_path = simulation_dir / "theta.csv"
    true_beta_path = simulation_dir / "gene_means.csv"
    
    if not true_theta_path.exists():
        raise FileNotFoundError(f"True theta file not found: {true_theta_path}")
    if not true_beta_path.exists():
        raise FileNotFoundError(f"True beta file not found: {true_beta_path}")
    
    # Find estimated theta and beta files
    estimated_theta_paths = []
    estimated_beta_paths = []
    method_names = []
    
    for model in models:
        # Check if topic_config is already part of the output_dir
        if topic_config and not str(output_dir).endswith(topic_config):
            model_dir = output_dir / topic_config / model
        else:
            model_dir = output_dir / model
            
        est_theta_path = model_dir / f"{model}_theta.csv"
        est_beta_path = model_dir / f"{model}_beta.csv"
        
        if est_theta_path.exists() and est_beta_path.exists():
            estimated_theta_paths.append(est_theta_path)
            estimated_beta_paths.append(est_beta_path)
            method_names.append(model)
            print(f"Found {model} files: {est_theta_path}")
            
            # Generate individual beta scatter plot
            beta_output_path = model_dir / "plots" / "beta_scatter.png"
            beta_output_path.parent.mkdir(parents=True, exist_ok=True)
            compare_beta_true_vs_estimated(true_beta_path, est_beta_path, beta_output_path, model)
        else:
            print(f"Warning: {model} files not found: {est_theta_path} or {est_beta_path}")
    
    if not estimated_theta_paths:
        raise ValueError("No estimated theta/beta files found")
    
    # Generate combined theta scatter plot with topic matching
    if topic_config and not str(output_dir).endswith(topic_config):
        combined_output_path = output_dir / topic_config / "model_comparison" / "combined_theta_scatter.png"
    else:
        combined_output_path = output_dir / "model_comparison" / "combined_theta_scatter.png"
    combined_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    combine_theta_scatter_plots(
        true_theta_path, 
        estimated_theta_paths, 
        method_names, 
        combined_output_path,
        true_beta_path,
        estimated_beta_paths
    )
    
    # Generate combined beta scatter plot
    if topic_config and not str(output_dir).endswith(topic_config):
        combined_beta_output_path = output_dir / topic_config / "model_comparison" / "combined_beta_scatter.png"
    else:
        combined_beta_output_path = output_dir / "model_comparison" / "combined_beta_scatter.png"
    combined_beta_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    combine_beta_scatter_plots(
        true_beta_path,
        estimated_beta_paths,
        method_names,
        combined_beta_output_path
    )
    
    # Generate noise analysis using true parameters (save to estimates folder)
    if topic_config and not str(output_dir).endswith(topic_config):
        noise_output_dir = output_dir / topic_config / "model_comparison"
    else:
        noise_output_dir = output_dir / "model_comparison"
    generate_noise_analysis(simulation_dir, noise_output_dir, libsize_mean, n_identity_topics)
    
    print(f"Combined evaluation complete!")
    print(f"- Combined theta scatter plot: {combined_output_path}")
    print(f"- Combined beta scatter plot: {combined_beta_output_path}")
    print(f"- Noise analysis: {noise_output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run combined simulation evaluation")
    parser.add_argument("--simulation_dir", required=True, help="Directory containing simulation outputs")
    parser.add_argument("--output_dir", required=True, help="Directory containing model estimation results")
    parser.add_argument("--models", nargs='+', default=['HLDA', 'LDA', 'NMF'], help="List of model names to evaluate")
    parser.add_argument("--topic_config", default="3_topic_fit", help="Topic configuration subdirectory name")
    parser.add_argument("--libsize_mean", type=int, default=1500, help="Library size parameter used in simulation")
    parser.add_argument("--n_identity_topics", type=int, default=2, help="Number of identity topics")
    
    args = parser.parse_args()
    
    run_combined_evaluation(
        args.simulation_dir,
        args.output_dir,
        args.models,
        args.topic_config,
        args.libsize_mean,
        args.n_identity_topics
    )

if __name__ == "__main__":
    main()