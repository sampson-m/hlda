import numpy as np
from scipy.linalg import qr
import pandas as pd
import matplotlib.pyplot as plt

# Load your existing data files
print("=== Step 1: Load Data ===")
# Replace these with your actual file paths
theta = pd.read_csv('data/AB_VAR/theta.csv', index_col=0).values  # Shape: (n_topics,)
beta = pd.read_csv('data/AB_VAR/gene_means_matrix.csv', index_col=0)
beta = beta.iloc[:,:3].values
X = pd.read_csv('data/AB_VAR/simulated_counts.csv', index_col=0).values

beta = beta / np.sum(beta, axis = 0)

L = 1500
C = len(theta)
n_identity = len(beta.T) - 1

print(f"L = {L}")
print(f"C = {C}")
print(f"Number of identity topics: {n_identity}")

# Split into identity and activity topics
print("\n=== Step 2: Separate Identity vs Activity Topics ===")
theta_i = theta[:,:-1]  
theta_A = theta[:, -1]
beta_i = beta[:, :-1] 
beta_A = beta[:,-1]

print(f"Identity theta_i shape: {theta_i.shape}")
print(f"Activity theta_A shape: {theta_A.shape}")
print(f"Identity beta_i shape: {beta_i.shape}")
print(f"Activity beta_A shape: {beta_A.shape}")

# Compute original expression: sum_i theta_i * beta_i + theta_A * beta_A
##print("\n=== Step 3: Compute Original Expression ===")
##identity_contribution = np.sum(theta_i[:, np.newaxis] * beta_i, axis=0)
##activity_contribution = theta_A * beta_A
##original_theta_beta = identity_contribution + activity_contribution

L_theta_beta = L * np.dot(theta, beta.T)  # Note: beta.T because beta is (G x K) but should be (K x G)
print(f"L*theta*beta.T shape: {L_theta_beta.shape}")
print("Need to verify: theta should be (C x K), beta should be (K x G)")
print(f"Current theta shape: {theta.shape}")
print(f"Current beta shape: {beta.shape}")
print(f"Current beta.T shape: {beta.T.shape}")

# The correct formulation should be L * theta * beta where:
# theta: (C x K) - cell-topic probabilities
# beta: (K x G) - topic-gene probabilities  
# Result: (C x G) - expected counts per cell-gene pair

# Subtract mean structure from count matrix
X_centered = X - L_theta_beta
print(f"Centered matrix shape: {X_centered.shape}")

# Apply Pearson residual normalization
print("\n=== Step 3: Pearson Residual Normalization ===")
print("Computing Pearson residuals: (X - μ) / √μ where μ = L * θ * β^T")

# Compute the mean matrix μ = L * θ * β^T
if beta.shape[0] > beta.shape[1]:  # If beta is (G x K) instead of (K x G)
    mu = L * np.dot(theta, beta.T)  # theta: (C x K), beta.T: (K x G) -> (C x G)
else:
    mu = L * np.dot(theta, beta)   # theta: (C x K), beta: (K x G) -> (C x G)

print(f"Mean matrix μ = L * θ * β^T shape: {mu.shape}")
print(f"μ statistics - Min: {np.min(mu):.4f}, Max: {np.max(mu):.4f}, Mean: {np.mean(mu):.4f}")

# Compute Pearson residuals element-wise
print("Computing Pearson residuals...")
X_pearson = (X - mu) / np.sqrt(mu)
print(f"Pearson residual matrix shape: {X_pearson.shape}")
print(f"Pearson residuals statistics - Mean: {np.mean(X_pearson):.4f}, Std: {np.std(X_pearson):.4f}")

# Check for any issues with the normalization
print(f"Number of zero/negative μ values: {np.sum(mu <= 0)}")
print(f"Number of infinite/NaN values in Pearson residuals: {np.sum(~np.isfinite(X_pearson))}")

# Replace any problematic values if needed
if np.sum(~np.isfinite(X_pearson)) > 0:
    print("Warning: Found infinite/NaN values, replacing with 0")
    X_pearson[~np.isfinite(X_pearson)] = 0

# Use Pearson residuals for eigenvalue analysis
X_normalized = X_pearson
print(f"Using Pearson residuals for eigenvalue analysis")

# Compute PCA using SVD
print("\n=== Step 4: Eigenvalue Analysis ===")
print("Computing SVD on Pearson residual matrix...")
U, s, Vt = np.linalg.svd(X_normalized, full_matrices=False)
eigenvalues = s**2 / (X_normalized.shape[0] - 1)  # Convert singular values to eigenvalues

print(f"SVD completed:")
print(f"  U shape: {U.shape}")
print(f"  Singular values shape: {s.shape}")
print(f"  Vt shape: {Vt.shape}")
print(f"  Number of eigenvalues: {len(eigenvalues)}")

print(f"Eigenvalue statistics:")
print(f"  Singular values range: {np.min(s):.4f} to {np.max(s):.4f}")
print(f"  Top 5 eigenvalues: {eigenvalues[:5]}")
print(f"  Eigenvalue mean: {np.mean(eigenvalues):.4f}")
print(f"  Eigenvalue std: {np.std(eigenvalues):.4f}")

# Debug: Check matrix conditioning
print(f"Pearson residual matrix condition number: {np.linalg.cond(X_normalized):.2e}")

print("\n=== Step 5: Step-by-Step Orthogonalization ===")
print("Following the exact derivation from noise_analysis.tex")
print("We need to compute: λ, θ̃, β̃, s terms, and final signal")

print(f"\n=== STEP 5A: Input Data Verification ===")
print(f"theta shape: {theta.shape}")
print(f"beta shape: {beta.shape}")
print(f"X shape: {X.shape}")

# Check if we need to transpose beta to get correct dimensions
if beta.shape[0] > beta.shape[1]:  # If beta is (G x K) instead of (K x G)
    print("Beta appears to be (G x K), transposing to get (K x G)")
    beta_correct = beta.T  # Now (K x G)
else:
    beta_correct = beta

print(f"beta_correct shape: {beta_correct.shape}")
print(f"theta range: [{np.min(theta):.4f}, {np.max(theta):.4f}]")
print(f"beta range: [{np.min(beta_correct):.4f}, {np.max(beta_correct):.4f}]")

# Identify identity vs activity topics explicitly
n_topics = theta.shape[1] if len(theta.shape) > 1 else len(theta)
print(f"\nTotal topics: {n_topics}")
activity_topic_idx = n_topics - 1  # Last topic is activity
identity_topic_indices = list(range(n_topics - 1))
print(f"Identity topics: {identity_topic_indices}")
print(f"Activity topic: {activity_topic_idx}")

print(f"\n=== STEP 5B: Construct Individual Topic Vectors θᵢβᵢ ===")
print("We need to create θᵢβᵢ vectors for each topic to orthogonalize")

topic_vectors = []
topic_info = []

for i in range(n_topics):
    if len(theta.shape) > 1:
        theta_i = np.mean(theta[:, i])  # Average across cells for topic i
    else:
        theta_i = theta[i]
    
    beta_i = beta_correct[i, :]  # Topic i gene probabilities
    
    # Create θᵢβᵢ vector
    theta_beta_i = theta_i * beta_i
    topic_vectors.append(theta_beta_i)
    
    topic_type = "activity" if i == activity_topic_idx else "identity"
    topic_info.append({
        'index': i,
        'type': topic_type,
        'theta': theta_i,
        'beta_norm': np.linalg.norm(beta_i),
        'theta_beta_norm': np.linalg.norm(theta_beta_i)
    })
    
    print(f"Topic {i} ({topic_type}):")
    print(f"  θ_{i} = {theta_i:.6f}")
    print(f"  ||β_{i}|| = {np.linalg.norm(beta_i):.6f}")
    print(f"  ||θ_{i}β_{i}|| = {np.linalg.norm(theta_beta_i):.6f}")

# Stack as matrix for QR decomposition
A = np.column_stack(topic_vectors)  # (G x K) matrix
print(f"\nMatrix A shape: {A.shape} (genes x topics)")
print(f"A = [θ₁β₁, θ₂β₂, ..., θₖβₖ]")
print(f"A condition number: {np.linalg.cond(A):.2e}")

print(f"\n=== STEP 5C: QR Decomposition ===")
print("Applying QR decomposition: A = Q*R")
Q, R = qr(A, mode='economic')
print(f"Q shape: {Q.shape} (orthogonal basis)")
print(f"R shape: {R.shape} (transformation matrix)")
print(f"||Q^T Q - I||: {np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1])):.2e}")
print(f"QR reconstruction error: {np.linalg.norm(A - Q @ R):.2e}")

# The orthogonalization is applied to individual topic vectors
print(f"\nA matrix first 3x3 entries:")
print(A[:3, :3])
print(f"\nQ matrix first 3x3 entries:")
print(Q[:3, :3])
print(f"\nR matrix:")
print(R)

print(f"\n=== STEP 5D: Extract Orthogonal Terms ===")
print("From QR decomposition A = Q*R, we extract:")
print("- λ matrix (R): transformation coefficients")
print("- β̃ vectors (Q): orthogonal basis vectors") 
print("- θ̃ coefficients: solved from the relationship")

# λ matrix (R matrix)
print(f"\n--- λ MATRIX (R) ---")
print(f"λ shape: {R.shape}")
print(f"λ is upper triangular: {np.allclose(R, np.triu(R))}")
print(f"λ diagonal elements: {np.diag(R)}")

# Extract specific λ values for each topic
lambda_values = {}
for i in range(n_topics):
    lambda_i = R[i, i]  # Diagonal element
    lambda_values[i] = lambda_i
    topic_type = "activity" if i == activity_topic_idx else "identity"
    print(f"λ_{i} ({topic_type}): {lambda_i:.6f}")

# β̃ vectors (Q matrix columns)
print(f"\n--- β̃ VECTORS (Q columns) ---")
print(f"β̃ matrix shape: {Q.shape}")
print(f"β̃ vectors are orthonormal: {np.allclose(Q.T @ Q, np.eye(Q.shape[1]))}")

beta_tilde_vectors = {}
for i in range(n_topics):
    beta_tilde_i = Q[:, i]
    beta_tilde_vectors[i] = beta_tilde_i
    topic_type = "activity" if i == activity_topic_idx else "identity"
    print(f"β̃_{i} ({topic_type}): norm = {np.linalg.norm(beta_tilde_i):.6f}")

# θ̃ coefficients (solved from A = Q*R relationship)
print(f"\n--- θ̃ COEFFICIENTS ---")
print("From fundamental relationship: θᵢβᵢ = Σⱼ λᵢⱼ θ̃ⱼ β̃ⱼ")
print("We solve: R * θ̃ = θ_original")

# Extract original theta coefficients
theta_original = np.array([topic_info[i]['theta'] for i in range(n_topics)])
print(f"Original θ coefficients: {theta_original}")

# Solve for θ̃
theta_tilde = np.linalg.solve(R, theta_original)
print(f"Solved θ̃ coefficients: {theta_tilde}")

# Verify the solution
theta_reconstructed = R @ theta_tilde
reconstruction_error = np.linalg.norm(theta_reconstructed - theta_original)
print(f"Reconstruction error ||R*θ̃ - θ||: {reconstruction_error:.6e}")

# Store θ̃ values for each topic
theta_tilde_values = {}
for i in range(n_topics):
    theta_tilde_values[i] = theta_tilde[i]
    topic_type = "activity" if i == activity_topic_idx else "identity"
    print(f"θ̃_{i} ({topic_type}): {theta_tilde[i]:.6f}")

print(f"\n=== VERIFICATION ===")
print(f"QR decomposition error: {np.linalg.norm(A - Q @ R):.2e}")
print(f"Orthogonality error: {np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1])):.2e}")
print(f"Coefficient reconstruction error: {reconstruction_error:.2e}")

print(f"\n=== STEP 5E: Compute s Terms (Activity Topic Decomposition) ===")
print("From Step 4 of derivation: β̃_A = Σᵢ sᵢ β̃ᵢ + s_A^⊥ β̃_A^⊥")
print("We need to decompose activity topic into identity + orthogonal components")

# Extract activity topic orthogonal vector
beta_A_tilde = beta_tilde_vectors[activity_topic_idx]
print(f"\nActivity topic β̃_A norm: {np.linalg.norm(beta_A_tilde):.6f}")

# Extract identity topic orthogonal vectors
identity_beta_tildes = []
for i in identity_topic_indices:
    identity_beta_tildes.append(beta_tilde_vectors[i])

if len(identity_beta_tildes) > 0:
    # Stack identity vectors as columns
    identity_matrix = np.column_stack(identity_beta_tildes)
    print(f"Identity matrix shape: {identity_matrix.shape}")
    
    # Compute projection coefficients sᵢ = ⟨β̃ᵢ, β̃_A⟩
    s_coefficients = []
    print(f"\n--- PROJECTION COEFFICIENTS sᵢ ---")
    for i, identity_idx in enumerate(identity_topic_indices):
        beta_i_tilde = beta_tilde_vectors[identity_idx]
        s_i = np.dot(beta_i_tilde, beta_A_tilde)
        s_coefficients.append(s_i)
        print(f"s_{identity_idx} = ⟨β̃_{identity_idx}, β̃_A⟩ = {s_i:.6f}")
    
    # Compute identity component: Σᵢ sᵢ β̃ᵢ
    identity_component = np.zeros_like(beta_A_tilde)
    for i, s_i in enumerate(s_coefficients):
        identity_component += s_i * identity_beta_tildes[i]
    
    print(f"Identity component norm: {np.linalg.norm(identity_component):.6f}")
    
    # Compute orthogonal component: β̃_A^⊥ = β̃_A - Σᵢ sᵢ β̃ᵢ
    beta_A_orthogonal = beta_A_tilde - identity_component
    print(f"Orthogonal component norm: {np.linalg.norm(beta_A_orthogonal):.6f}")
    
    # Compute s_A^⊥ = ||β̃_A^⊥||
    s_A_perp = np.linalg.norm(beta_A_orthogonal)
    print(f"s_A^⊥ = ||β̃_A^⊥|| = {s_A_perp:.6f}")
    
    # Normalize to get unit vector
    if s_A_perp > 1e-10:
        beta_A_orthogonal_unit = beta_A_orthogonal / s_A_perp
        print(f"β̃_A^⊥ unit vector norm: {np.linalg.norm(beta_A_orthogonal_unit):.6f}")
    else:
        beta_A_orthogonal_unit = beta_A_orthogonal
        print("Warning: s_A^⊥ is very small, activity topic is nearly in identity subspace")
    
    # Verification: β̃_A = Σᵢ sᵢ β̃ᵢ + s_A^⊥ β̃_A^⊥
    reconstructed_beta_A = identity_component + s_A_perp * beta_A_orthogonal_unit
    decomposition_error = np.linalg.norm(reconstructed_beta_A - beta_A_tilde)
    print(f"Decomposition verification error: {decomposition_error:.6e}")
    
else:
    print("No identity topics found, s_A^⊥ = 1")
    s_A_perp = 1.0
    beta_A_orthogonal_unit = beta_A_tilde

print(f"\n=== STEP 5F: Signal Computations ===")
print("Computing both the initial and refined signal formulas")

# Extract components for refined formula
lambda_A = lambda_values[activity_topic_idx]
s_A = s_A_perp
theta_A_tilde = theta_tilde_values[activity_topic_idx]
theta_A_tilde_norm = np.abs(theta_A_tilde)

print(f"\nComponents for refined formula:")
print(f"λ_A = {lambda_A:.6f}")
print(f"s_A = {s_A:.6f}")
print(f"θ̃_A = {theta_A_tilde:.6f}")
print(f"||θ̃_A|| = {theta_A_tilde_norm:.6f}")

# μ normalization removed for now (to be clarified in derivation)
# mu_mean = np.mean(mu)
# mu_sqrt = np.sqrt(mu_mean)
# print(f"μ_mean = {mu_mean:.6f}")
# print(f"√μ = {mu_sqrt:.6f}")

print(f"\n--- INITIAL SIGNAL COMPUTATION ---")
print("signal = ||X · β̃_A^T||²")

# Get the activity topic orthogonal vector
beta_A_tilde = beta_tilde_vectors[activity_topic_idx]
print(f"β̃_A shape: {beta_A_tilde.shape}")
print(f"β̃_A norm: {np.linalg.norm(beta_A_tilde):.6f}")

# Compute initial signal: ||X · β̃_A^T||²
print(f"X shape: {X.shape}, β̃_A shape: {beta_A_tilde.shape}")
print("Computing X · β̃_A^T where X is (C x G) and β̃_A^T is (1 x C)")

# X · β̃_A^T means each row of X (cell) dots with β̃_A
signal_projection_initial = np.dot(X, beta_A_tilde)  # (C x G) * (C,) = (G,)
signal_initial = np.linalg.norm(signal_projection_initial)**2

print(f"Signal projection shape: {signal_projection_initial.shape}")
print(f"INITIAL SIGNAL: ||X · β̃_A^T||² = {signal_initial:.6f}")

# Also compute with Pearson residuals
signal_projection_pearson = np.dot(X_pearson, beta_A_tilde)
signal_initial_pearson = np.linalg.norm(signal_projection_pearson)**2
print(f"INITIAL SIGNAL (Pearson): ||X_pearson · β̃_A^T||² = {signal_initial_pearson:.6f}")

print(f"\n--- REFINED SIGNAL COMPUTATION ---")
print("signal = λ_A² s_A² ||θ̃_A||² (√μ normalization removed for now)")

# Refined signal computation without √μ normalization
signal_refined = lambda_A**2 * s_A**2 * theta_A_tilde_norm**2
print(f"signal = {lambda_A:.6f}² × {s_A:.6f}² × {theta_A_tilde_norm:.6f}²")
print(f"REFINED SIGNAL: {signal_refined:.6f}")

print(f"\n=== SIGNAL COMPARISON ===")
print(f"Initial signal (raw): {signal_initial:.6f}")
print(f"Initial signal (Pearson): {signal_initial_pearson:.6f}")
print(f"Refined signal: {signal_refined:.6f}")
print(f"Ratio (refined/initial_pearson): {signal_refined/signal_initial_pearson:.6f}")

print(f"\n=== SUMMARY ===")
print(f"✓ λ_A (transformation coefficient): {lambda_A:.6f}")
print(f"✓ s_A (orthogonal component magnitude): {s_A:.6f}")
print(f"✓ θ̃_A (orthogonal activity coefficient): {theta_A_tilde:.6f}")
print(f"✓ Initial signal value: {signal_initial_pearson:.6f}")
print(f"✓ Refined signal value: {signal_refined:.6f}")

# Store for later use
activity_signal_components = {
    'lambda_A': lambda_A,
    's_A': s_A,
    'theta_A_tilde': theta_A_tilde,
    'signal_initial': signal_initial_pearson,
    'signal_refined': signal_refined
}

print(f"\n=== STEP 6: Noise Computation ===")
print("Computing noise component: (Poisson(Lθβ^T) - Lθβ^T) / √(Lθβ^T)")

# The noise is the Pearson-normalized random Poisson deviation
noise_component_raw = X - mu  # X - Lθβ^T (raw deviations)
noise_component_pearson = (X - mu) / np.sqrt(mu)  # Pearson normalized deviations

noise_value_raw = np.linalg.norm(noise_component_raw)**2
noise_value_pearson = np.linalg.norm(noise_component_pearson)**2

print(f"Noise (raw): ||X - Lθβ^T||² = {noise_value_raw:.6f}")
print(f"Noise (Pearson): ||(X - Lθβ^T)/√(Lθβ^T)||² = {noise_value_pearson:.6f}")

# Verify this matches our pre-computed Pearson residuals
noise_value_pearson_check = np.linalg.norm(X_pearson)**2
print(f"Noise verification: ||X_pearson||² = {noise_value_pearson_check:.6f}")
print(f"Pearson noise values match: {np.isclose(noise_value_pearson, noise_value_pearson_check)}")

print(f"\n=== FINAL RESULTS ===")
print(f"SIGNAL (initial formula): {activity_signal_components['signal_initial']:.6f}")
print(f"SIGNAL (refined formula): {activity_signal_components['signal_refined']:.6f}")
print(f"NOISE (Pearson): {noise_value_pearson:.6f}")

snr_initial = activity_signal_components['signal_initial'] / noise_value_pearson if noise_value_pearson > 0 else np.inf
snr_refined = activity_signal_components['signal_refined'] / noise_value_pearson if noise_value_pearson > 0 else np.inf

print(f"SIGNAL-TO-NOISE RATIO (initial): {snr_initial:.6f}")
print(f"SIGNAL-TO-NOISE RATIO (refined): {snr_refined:.6f}")

print(f"\n=== COMPLETE COMPONENT BREAKDOWN ===")
print(f"✓ λ_A (activity transformation): {lambda_A:.6f}")
print(f"✓ s_A (orthogonal magnitude): {s_A:.6f}")
print(f"✓ θ̃_A (orthogonal coefficient): {theta_A_tilde:.6f}")
print(f"✓ Initial signal: {activity_signal_components['signal_initial']:.6f}")
print(f"✓ Refined signal: {activity_signal_components['signal_refined']:.6f}")
print(f"✓ Final noise: {noise_value_pearson:.6f}")
print(f"✓ Initial SNR: {snr_initial:.6f}")
print(f"✓ Refined SNR: {snr_refined:.6f}")
print(f"✓ Note: √μ normalization removed pending derivation clarification")

# Use the SVD eigenvalues for the rest of the analysis
eigenvalues_to_use = eigenvalues

print(f"Eigenvalue statistics:")
print(f"  Mean: {np.mean(eigenvalues_to_use):.4f}")
print(f"  Std: {np.std(eigenvalues_to_use):.4f}")
print(f"  Min: {np.min(eigenvalues_to_use):.4f}")
print(f"  Max: {np.max(eigenvalues_to_use):.4f}")

# Count eigenvalues in range [0.75, 1.25]
eigenvalues_in_range = np.sum((eigenvalues_to_use >= 0.4) & (eigenvalues_to_use <= 1.4))
total_eigenvalues = len(eigenvalues_to_use)
percentage_in_range = (eigenvalues_in_range / total_eigenvalues) * 100

print(f"\nEigenvalues in range [0.75, 1.25]:")
print(f"  Count: {eigenvalues_in_range}")
print(f"  Total eigenvalues: {total_eigenvalues}")
print(f"  Percentage: {percentage_in_range:.2f}%")

# Additional analysis for signal detection
print(f"\n=== Signal Analysis ===")
print(f"Eigenvalues > 1.0: {np.sum(eigenvalues_to_use > 1.0)}")
print(f"Eigenvalues > 2.0: {np.sum(eigenvalues_to_use > 2.0)}")
print(f"Ratio of max to median eigenvalue: {np.max(eigenvalues_to_use) / np.median(eigenvalues_to_use):.2f}")


# Plot histogram of eigenvalues
plt.figure(figsize=(10, 6))
plt.hist(eigenvalues, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.title('Histogram of PCA Eigenvalues')
plt.grid(True, alpha=0.3)
plt.show()
