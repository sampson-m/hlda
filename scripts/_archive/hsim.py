import numpy as np
import pandas as pd

def simulate_counts(
    n_genes=2000,
    n_de=400,
    cells_per_group=3000,
    groups=['A', 'A_var'],
    gamma_shape=0.6,
    gamma_rate=0.3,
    de_lognormal_mean=0.5,
    de_lognormal_sigma=0.4
):
    baseline_means = np.random.gamma(shape=gamma_shape, scale=1/gamma_rate, size=n_genes)

    gene_labels = [f'Gene_{i+1}' for i in range(n_genes)]
    gene_means_matrix = pd.DataFrame({
        grp: baseline_means.copy() for grp in groups
    }, index=gene_labels)
    
    gene_means_matrix['DE'] = False
    gene_means_matrix['DE_group'] = ""
    gene_means_matrix['DE_factor'] = np.nan
    
    de_gene_indices = np.random.choice(np.arange(n_genes), size=n_de, replace=False)
    de_patterns = ['A_var']
    
    de_assignments = {}
    
    for gene_idx in de_gene_indices:
        pattern = np.random.choice(de_patterns)
        if pattern == 'A_and_B':
            affected_groups = ['A', 'B']
            group_string = "A,B"
        else:
            affected_groups = [pattern]
            group_string = pattern

        de_factor = np.random.lognormal(mean=de_lognormal_mean, sigma=de_lognormal_sigma)
        de_assignments[gene_idx] = {'groups': affected_groups, 'DE_factor': de_factor}
        
        for grp in affected_groups:
            gene_means_matrix.at[gene_labels[gene_idx], grp] *= de_factor
        
        gene_means_matrix.at[gene_labels[gene_idx], 'DE'] = True
        gene_means_matrix.at[gene_labels[gene_idx], 'DE_group'] = group_string
        gene_means_matrix.at[gene_labels[gene_idx], 'DE_factor'] = de_factor

    cell_data = []        
    cell_ids = []          
    cell_identities = []
    theta_vals = []
    sampled_ls, true_ls = [], []
    A_means = gene_means_matrix['A']
    A_var_means = gene_means_matrix['A_var']
    
    for i in range(cells_per_group):
        theta = np.random.uniform(0.0, 1.0)
        theta_vals.append(theta)

        mixture_means = theta * A_means + (1.0 - theta) * A_var_means
        total_mean = mixture_means.sum()

        lib_size = 1500
        effective_means = lib_size * (mixture_means / total_mean)
        counts = np.random.poisson(lam=effective_means)

        cell_data.append(counts)
        cell_ids.append(f"A_{i+1}")
        cell_identities.append('A')
        sampled_ls.append(lib_size)
        true_ls.append(counts.sum())
    
    counts_df = pd.DataFrame(cell_data, columns=gene_labels)
    counts_df.insert(0, 'cell_identity', cell_identities)
    lib_df = pd.DataFrame({
        'cell_id': cell_identities,
        'sampled_library_size': sampled_ls,
        'true_library_size': true_ls
    })
    theta_df = pd.DataFrame(theta_vals, columns = ['theta'])
    
    return counts_df, lib_df, gene_means_matrix, de_assignments, de_gene_indices, theta_df

if __name__ == '__main__':
    # Run the simulation.
    counts_df, lib_df, gene_means_matrix, de_assignments, de_gene_indices, theta_df = simulate_counts()
    
    counts_df.to_csv('../data/hsim_A_var/simulated_counts.csv', index=False)
    gene_means_matrix.to_csv('../data/hsim_A_var/gene_means_matrix.csv')
    lib_df.to_csv('../data/hsim_A_var/library_sizes.csv')
    theta_df.to_csv('../data/hsim_A_var/theta.csv')
    


