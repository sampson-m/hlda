import numpy as np
from scipy.linalg import qr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def compute_signal_to_noise(simulation_dir, libsize_mean=1500, n_identity_topics=2, verbose=False):
    """
    Compute signal-to-noise ratio for activity topics in simulation data.
    
    Args:
        simulation_dir: Path to simulation output directory
        libsize_mean: Expected library size (used if not available in data)
        n_identity_topics: Number of identity topics
        verbose: Whether to print detailed output (default: False)
    
    Returns:
        dict: Signal, noise, and SNR values
    """
    simulation_dir = Path(simulation_dir)
    
    # Load simulation data
    if verbose:
        print("=== Step 1: Load Data ===")
    theta_path = simulation_dir / "theta.csv"
    beta_path = simulation_dir / "gene_means.csv"
    counts_path = simulation_dir / "counts.csv"
    libsize_path = simulation_dir / "library_sizes.csv"
    
    if not all(p.exists() for p in [theta_path, beta_path, counts_path]):
        raise FileNotFoundError(f"Required simulation files not found in {simulation_dir}")
    
    theta = pd.read_csv(theta_path, index_col=0).values  # Shape: (C, K)
    beta = pd.read_csv(beta_path, index_col=0).values    # Shape: (G, K)
    X = pd.read_csv(counts_path, index_col=0).values     # Shape: (C, G)
    
    # Load library sizes if available
    if libsize_path.exists():
        libsizes = pd.read_csv(libsize_path, index_col=0)['library_size'].values
        L = np.mean(libsizes)
    else:
        L = libsize_mean
    
    C, K = theta.shape
    G = beta.shape[0]
    n_identity = n_identity_topics
    
    if verbose:
        print(f"L = {L}")
        print(f"C = {C}, G = {G}, K = {K}")
        print(f"Number of identity topics: {n_identity}")
    
    # Normalize beta to probabilities
    beta = beta / np.sum(beta, axis=0)
    
    # Split into identity and activity topics
    if verbose:
        print("\n=== Step 2: Separate Identity vs Activity Topics ===")
    theta_i = theta[:, :n_identity]  
    theta_A = theta[:, n_identity:]  # All activity topics
    beta_i = beta[:, :n_identity] 
    beta_A = beta[:, n_identity:]    # All activity topics
    
    if verbose:
        print(f"Identity theta_i shape: {theta_i.shape}")
        print(f"Activity theta_A shape: {theta_A.shape}")
        print(f"Identity beta_i shape: {beta_i.shape}")
        print(f"Activity beta_A shape: {beta_A.shape}")
    
    # Compute expected counts: L * theta * beta^T
    L_theta_beta = L * np.dot(theta, beta.T)  # Shape: (C, G)
    if verbose:
        print(f"L*theta*beta.T shape: {L_theta_beta.shape}")
    
    # Subtract mean structure from count matrix
    X_centered = X - L_theta_beta
    if verbose:
        print(f"Centered matrix shape: {X_centered.shape}")
    
    # Apply Pearson residual normalization
    if verbose:
        print("\n=== Step 3: Pearson Residual Normalization ===")
        print("Computing Pearson residuals: (X - μ) / √μ where μ = L * θ * β^T")
    
    # Avoid division by zero
    sqrt_expected = np.sqrt(np.maximum(L_theta_beta, 1e-10))
    X_pearson = X_centered / sqrt_expected
    
    if verbose:
        print("\n=== Step 4: QR Decomposition ===")
    # Use beta matrix directly (G x K) for QR decomposition
    Q, R = qr(beta, mode='economic')  # Q: (G, K), R: (K, K)
    
    # β̃ vectors (Q matrix columns)
    if verbose:
        print(f"Q matrix shape: {Q.shape}")
    beta_tilde = Q  # (G x K) orthogonal gene patterns
    theta_tilde = theta @ R.T  # (C x K)
    
    # Verify the transformation preserves information
    theta_beta_original = theta @ beta.T  # (C x K) @ (K x G) = (C x G)
    theta_tilde_Q = theta_tilde @ beta_tilde.T  # (C x K) @ (K x G) = (C x G)
    reconstruction_error = np.linalg.norm(theta_beta_original - theta_tilde_Q)
    if verbose:
        print(f"Expression reconstruction error ||θβ^T - θ̃Q^T||: {reconstruction_error:.6e}")
    
    # Extract activity topic orthogonal vectors
    activity_indices = list(range(n_identity, K))
    if verbose:
        print(f"Activity topic indices: {activity_indices}")
    
    # For multiple activity topics, we'll compute signal for each
    activity_signals = []
    activity_noises = []
    
    for act_idx in activity_indices:
        if verbose:
            print(f"\n--- Processing Activity Topic {act_idx} ---")
        
        # Extract activity topic orthogonal vector
        beta_A_tilde = beta_tilde[:, act_idx]  # Shape: (G,)
        if verbose:
            print(f"Activity topic β̃_A norm: {np.linalg.norm(beta_A_tilde):.6f}")
        
        # Extract identity topic orthogonal vectors
        identity_beta_tildes = []
        identity_topic_indices = list(range(n_identity))
        for i in identity_topic_indices:
            identity_beta_tildes.append(beta_tilde[:, i])  # Shape: (G,)
        
        if len(identity_beta_tildes) > 0:
            # Stack identity vectors as columns
            identity_matrix = np.column_stack(identity_beta_tildes)
            if verbose:
                print(f"Identity matrix shape: {identity_matrix.shape}")
            
            # Compute projection coefficients sᵢ = ⟨β̃ᵢ, β̃_A⟩
            s_coefficients = []
            if verbose:
                print(f"--- PROJECTION COEFFICIENTS sᵢ ---")
            for i, identity_idx in enumerate(identity_topic_indices):
                beta_i_tilde = beta_tilde[:, identity_idx]  # Shape: (G,)
                s_i = np.dot(beta_i_tilde, beta_A_tilde)
                s_coefficients.append(s_i)
                if verbose:
                    print(f"s_{identity_idx} = ⟨β̃_{identity_idx}, β̃_A⟩ = {s_i:.6f}")
            
            # Compute identity component: Σᵢ sᵢ β̃ᵢ
            identity_component = np.zeros_like(beta_A_tilde)
            for i, s_i in enumerate(s_coefficients):
                identity_component += s_i * identity_beta_tildes[i]
            
            if verbose:
                print(f"Identity component norm: {np.linalg.norm(identity_component):.6f}")
            
            # Compute orthogonal component: β̃_A^⊥ = β̃_A - Σᵢ sᵢ β̃ᵢ
            beta_A_orthogonal = beta_A_tilde - identity_component
            if verbose:
                print(f"Orthogonal component norm: {np.linalg.norm(beta_A_orthogonal):.6f}")
            
            # Compute s_A^⊥ = ||β̃_A^⊥||
            s_A_perp = np.linalg.norm(beta_A_orthogonal)
            if verbose:
                print(f"s_A^⊥ = ||β̃_A^⊥|| = {s_A_perp:.6f}")
            
            # Normalize to get unit vector
            if s_A_perp > 1e-10:
                beta_A_orthogonal_unit = beta_A_orthogonal / s_A_perp
                if verbose:
                    print(f"β̃_A^⊥ unit vector norm: {np.linalg.norm(beta_A_orthogonal_unit):.6f}")
            else:
                beta_A_orthogonal_unit = beta_A_orthogonal
                if verbose:
                    print("Warning: s_A^⊥ is very small, activity topic is nearly in identity subspace")
        else:
            if verbose:
                print("No identity topics found, s_A^⊥ = 1")
            s_A_perp = 1.0
            beta_A_orthogonal_unit = beta_A_tilde
        
        if verbose:
            print(f"\n=== STEP 5: Signal and Noise Computations ===")
        
        # Normalize the count matrix by sqrt of expected counts
        X_normalized = X / sqrt_expected  # Element-wise division
        
        if verbose:
            print(f"X_normalized shape: {X_normalized.shape}")
            print(f"X_normalized statistics - Mean: {np.mean(X_normalized):.4f}, Std: {np.std(X_normalized):.4f}")
        
        # Compute signal per cell
        signal_per_cell = np.zeros(X_normalized.shape[0])
        for c in range(X_normalized.shape[0]):
            # Signal_c = (X_normalized_c · β̃_A^⊥)
            signal_per_cell[c] = np.dot(X_normalized[c, :], beta_A_orthogonal_unit)
        
        if verbose:
            print(f"Signal per cell shape: {signal_per_cell.shape}")
            print(f"Signal statistics - Mean: {np.mean(signal_per_cell):.6f}, Std: {np.std(signal_per_cell):.6f}")
            print(f"Signal range: [{np.min(signal_per_cell):.6f}, {np.max(signal_per_cell):.6f}]")
        
        # Total signal strength
        total_signal = np.sum(signal_per_cell**2)
        if verbose:
            print(f"TOTAL SIGNAL: Σ_c (Signal_c)² = {total_signal:.6f}")
        
        if verbose:
            print(f"\n--- NOISE COMPUTATION ---")
        # The noise is the normalized deviation from expected counts
        expected_normalized = np.sqrt(L_theta_beta)  # This is the signal term √(L*θ*β^T)
        noise_per_cell = X_normalized - expected_normalized
        
        if verbose:
            print(f"Noise per cell matrix shape: {noise_per_cell.shape}")
            print(f"Noise statistics - Mean: {np.mean(noise_per_cell):.6f}, Std: {np.std(noise_per_cell):.6f}")
        
        # Total noise magnitude
        total_noise = np.sum(noise_per_cell**2)
        if verbose:
            print(f"TOTAL NOISE: Σ_{{c,g}} (Noise_{{c,g}})² = {total_noise:.6f}")
        
        # Verify this matches our Pearson residuals
        pearson_noise_check = np.sum(X_pearson**2)
        if verbose:
            print(f"Pearson residuals check: Σ (X_pearson)² = {pearson_noise_check:.6f}")
            print(f"Noise computations match: {np.isclose(total_noise, pearson_noise_check)}")
        
        # Alternative signal computation
        if verbose:
            print(f"\n--- ALTERNATIVE SIGNAL COMPUTATION ---")
        signal_projection = np.dot(X_normalized, beta_A_orthogonal_unit)
        alternative_signal = np.sum(signal_projection**2)
        if verbose:
            print(f"Alternative signal computation: {alternative_signal:.6f}")
        
        if verbose:
            print(f"\n=== SIGNAL AND NOISE RESULTS FOR ACTIVITY TOPIC {act_idx} ===")
            print(f"SIGNAL: {total_signal:.6f}")
            print(f"NOISE: {total_noise:.6f}")
            print(f"SIGNAL-TO-NOISE RATIO: {total_signal/total_noise:.6f}")
        
        activity_signals.append(total_signal)
        activity_noises.append(total_noise)
    
    # Aggregate results across all activity topics
    total_activity_signal = np.sum(activity_signals)
    total_activity_noise = np.mean(activity_noises)  # Average noise across activity topics
    overall_snr = total_activity_signal / total_activity_noise
    
    # Compute identity signal (signal from identity topics only)
    if verbose:
        print(f"\n=== IDENTITY SIGNAL COMPUTATION ===")
    
    # Identity signal: projection of normalized data onto identity topic directions
    identity_signals = []
    for identity_idx in range(n_identity):
        beta_i_tilde = beta_tilde[:, identity_idx]
        identity_signal_per_cell = np.dot(X_normalized, beta_i_tilde)
        total_identity_signal = np.sum(identity_signal_per_cell**2)
        identity_signals.append(total_identity_signal)
        if verbose:
            print(f"Identity topic {identity_idx} signal: {total_identity_signal:.6f}")
    
    total_identity_signal = np.sum(identity_signals)
    
    # Compute noise-to-signal ratios
    noise_to_identity_ratio = total_activity_noise / total_identity_signal if total_identity_signal > 0 else float('inf')
    noise_to_activity_ratio = total_activity_noise / total_activity_signal if total_activity_signal > 0 else float('inf')
    
    if verbose:
        print(f"Total identity signal: {total_identity_signal:.6f}")
        print(f"Noise to identity ratio: {noise_to_identity_ratio:.6f}")
        print(f"Noise to activity ratio: {noise_to_activity_ratio:.6f}")
    
    results = {
        'total_activity_signal': total_activity_signal,
        'total_identity_signal': total_identity_signal,
        'total_activity_noise': total_activity_noise,
        'overall_snr': overall_snr,
        'noise_to_identity_ratio': noise_to_identity_ratio,
        'noise_to_activity_ratio': noise_to_activity_ratio,
        'activity_signals': activity_signals,
        'identity_signals': identity_signals,
        'activity_noises': activity_noises,
        'n_activity_topics': len(activity_indices),
        'n_identity_topics': n_identity,
        'libsize_mean': L,
        'n_cells': C,
        'n_genes': G
    }
    
    # Always print final results
    print(f"SNR: {overall_snr:.6f} (Activity Signal: {total_activity_signal:.2f}, Identity Signal: {total_identity_signal:.2f}, Noise: {total_activity_noise:.2f})")
    
    return results

# Example usage (commented out)
if __name__ == "__main__":
    # Example: compute signal-to-noise for a simulation
    # results = compute_signal_to_noise("data/AB_V1/DE_mean_0.5", libsize_mean=1500, n_identity_topics=2)
    # print(results)
    pass
