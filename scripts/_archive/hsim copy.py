##import numpy as np
##import pandas as pd
##
##def simulate_counts(
##    n_genes=2000,
##    n_de=400,
##    cells_per_group=3000,
##    groups=['A', 'A_var', 'B'],
##    gamma_shape=0.6,
##    gamma_rate=0.3,
##    de_lognormal_mean=0.5,
##    de_lognormal_sigma=0.4
##):
##    """
##    Simulate counts for three topics: A, A_var, and B.
##    - A vs A_var mixture cells labeled 'A'
##    - Pure-B cells labeled 'B'
##    - DE genes: n_de in A_var, n_de in B (possibly overlapping)
##    Returns:
##      counts_df, lib_df, gene_means (gm),
##      de_assignments, de_Avar_idx, de_B_idx, true_theta
##    """
##    # 1) Baseline gene means
##    baseline = np.random.gamma(shape=gamma_shape,
##                               scale=1.0/gamma_rate,
##                               size=n_genes)
##    gene_labels = [f"Gene_{i+1}" for i in range(n_genes)]
##    gm = pd.DataFrame({g: baseline.copy() for g in groups},
##                      index=gene_labels)
##
##    gm['DE'] = False
##    gm['DE_group'] = ""
##    gm['DE_factor'] = np.nan
##    de_assignments = {}
##
##    # A_var DE
##    de_Avar = np.random.choice(n_genes, size=n_de, replace=False)
##    for idx in de_Avar:
##        factor = np.random.lognormal(de_lognormal_mean,
##                                     de_lognormal_sigma)
##        gm.iat[idx, gm.columns.get_loc('A_var')] *= factor
##        gm.iat[idx, gm.columns.get_loc('DE')] = True
##        gm.iat[idx, gm.columns.get_loc('DE_group')] = 'A_var'
##        gm.iat[idx, gm.columns.get_loc('DE_factor')] = factor
##        de_assignments[idx] = {'groups': ['A_var'], 'DE_factor': factor}
##
##    # B DE (random, may overlap)
##    de_B = np.random.choice(n_genes, size=n_de, replace=False)
##    for idx in de_B:
##        factor = np.random.lognormal(de_lognormal_mean,
##                                     de_lognormal_sigma)
##        gm.iat[idx, gm.columns.get_loc('B')] *= factor
##        gm.iat[idx, gm.columns.get_loc('DE')] = True
##        gm.iat[idx, gm.columns.get_loc('DE_group')] = 'B'
##        gm.iat[idx, gm.columns.get_loc('DE_factor')] = factor
##        de_assignments[idx] = {'groups': ['B'], 'DE_factor': factor}
##
##    # 3) Prepare for simulation
##    cell_data     = []
##    cell_ids      = []
##    cell_identity = []
##    theta_list    = []
##
##    A_means    = gm['A'].values
##    Avar_means = gm['A_var'].values
##    B_means    = gm['B'].values
##
##    # 3a) Mixture cells (A vs A_var), labeled 'A'
##    for i in range(cells_per_group):
##        th = np.random.uniform()
##        theta_vec = [th, 1.0 - th, 0.0]
##        mixture  = th * A_means + (1-th) * Avar_means
##        lib      = 1500
##        eff      = lib * (mixture / mixture.sum())
##        counts   = np.random.poisson(lam=eff)
##
##        cell_data.append(counts)
##        cell_ids.append(f"{groups[0]}_{i+1}")     # e.g. "A_1", "A_2", …
##        cell_identity.append(groups[0])           # always "A"
##        theta_list.append(theta_vec)
##
##    # 3b) Pure-B cells
##    for i in range(cells_per_group):
##        theta_vec = [0.0, 0.0, 1.0]
##        lib       = 1500
##        eff       = lib * (B_means / B_means.sum())
##        counts    = np.random.poisson(lam=eff)
##
##        cell_data.append(counts)
##        cell_ids.append(f"{groups[2]}_{i+1}")     # "B_1", "B_2", …
##        cell_identity.append(groups[2])           # "B"
##        theta_list.append(theta_vec)
##
##    # 4) Assemble DataFrames
##    counts_df = pd.DataFrame(cell_data,
##                             columns=gene_labels,
##                             index=cell_identity)
##
##    lib_df = pd.DataFrame({
##        'cell_identity':         cell_identity,
##        'sampled_library_size': [1500]*len(cell_ids),
##        'true_library_size':    counts_df[gene_labels].sum(axis=1).values
##    })
##
##    theta_df = pd.DataFrame(theta_list,
##                            columns=groups,
##                            index=cell_identity)
##
##    return counts_df, lib_df, gm, de_assignments, de_Avar, de_B, theta_df
##
##if __name__ == '__main__':
##    # Run the simulation.
##    counts_df, lib_df, gene_means_matrix, de_assignments, de_gene_indices, theta_df = simulate_counts()
##    
##    counts_df.to_csv('../data/hsim_A_var/simulated_counts.csv', index=False)
##    gene_means_matrix.to_csv('../data/hsim_A_var/gene_means_matrix.csv')
##    lib_df.to_csv('../data/hsim_A_var/library_sizes.csv')
##    theta_df.to_csv('../data/hsim_A_var/theta.csv')
##    
##
##
import numpy as np
import pandas as pd
from pathlib import Path

##def simulate_counts(
##    n_genes=2000,
##    n_de=400,
##    cells_per_group=3000,
##    groups=('A','B','C'),
##    gamma_shape=0.6,
##    gamma_rate=0.3,
##    de_lognormal_mean=0.5,
##    de_lognormal_sigma=0.4
##):
##    """
##    Simulate counts for three pure topics: A, B, and C.
##    - Each cell belongs 100% to one topic.
##    - For each topic, select n_de genes to be differentially expressed (DE) in that topic.
##      Genes may be DE in multiple topics, and each group has its own DE factor.
##
##    Returns:
##      counts_df       : DataFrame (n_cells x n_genes)
##      lib_df          : DataFrame (n_cells x [cell_identity, sampled/true lib])
##      gene_means      : DataFrame (n_genes x 3 topics + DE metadata)
##      de_assignments  : dict mapping gene_idx -> dict[group]->factor
##      de_A            : array of DE gene indices for A
##      de_B            : array of DE gene indices for B
##      de_C            : array of DE gene indices for C
##      theta_df        : DataFrame (n_cells x 3 topics)
##    """
##    # 1) Baseline gene means
##    gene_labels = [f"Gene_{i+1}" for i in range(n_genes)]
##    baseline = np.random.gamma(shape=gamma_shape,
##                               scale=1.0/gamma_rate,
##                               size=n_genes)
##    gm = pd.DataFrame({g: baseline.copy() for g in groups},
##                      index=gene_labels)
##
##    # 2) Prepare DE metadata columns
##    gm['DE'] = False
##    gm['DE_group'] = [[] for _ in range(n_genes)]  # list of groups per gene
##    # one DE_factor column per group to record factor if DE
##    for g in groups:
##        gm[f'DE_factor_{g}'] = np.nan
##    # dict mapping gene_idx -> {group: factor}
##    de_assignments = {i: {} for i in range(n_genes)}
##
##    # 3) Assign DE genes for each group (may overlap)
##    de_A = np.random.choice(n_genes, size=n_de, replace=False)
##    de_B = np.random.choice(n_genes, size=n_de, replace=False)
##    de_C = np.random.choice(n_genes, size=n_de, replace=False)
##
##    for idx in de_A:
##        factor = np.random.lognormal(de_lognormal_mean,
##                                     de_lognormal_sigma)
##        gm.at[gene_labels[idx], 'A'] *= factor
##        gm.at[gene_labels[idx], 'DE'] = True
##        gm.at[gene_labels[idx], 'DE_group'].append('A')
##        gm.at[gene_labels[idx], 'DE_factor_A'] = factor
##        de_assignments[idx]['A'] = factor
##
##    for idx in de_B:
##        factor = np.random.lognormal(de_lognormal_mean,
##                                     de_lognormal_sigma)
##        gm.at[gene_labels[idx], 'B'] *= factor
##        gm.at[gene_labels[idx], 'DE'] = True
##        gm.at[gene_labels[idx], 'DE_group'].append('B')
##        gm.at[gene_labels[idx], 'DE_factor_B'] = factor
##        de_assignments[idx]['B'] = factor
##
##    for idx in de_C:
##        factor = np.random.lognormal(de_lognormal_mean,
##                                     de_lognormal_sigma)
##        gm.at[gene_labels[idx], 'C'] *= factor
##        gm.at[gene_labels[idx], 'DE'] = True
##        gm.at[gene_labels[idx], 'DE_group'].append('C')
##        gm.at[gene_labels[idx], 'DE_factor_C'] = factor
##        de_assignments[idx]['C'] = factor
##
##    # 4) extract means for simulation
##    means = {g: gm[g].values for g in groups}
##
##    # 5) simulate pure-group counts
##    cell_data     = []
##    cell_ids      = []
##    cell_identity = []
##    theta_list    = []
##    lib_size = 1500
##
##    for grp in groups:
##        mean_vec = means[grp]
##        for i in range(cells_per_group):
##            theta = [1.0 if g==grp else 0.0 for g in groups]
##            eff   = lib_size * (mean_vec / mean_vec.sum())
##            counts = np.random.poisson(lam=eff)
##            cell_data.append(counts)
##            cell_ids.append(f"{grp}")
##            cell_identity.append(grp)
##            theta_list.append(theta)
##
##    # 6) assemble DataFrames
##    counts_df = pd.DataFrame(cell_data,
##                             index=cell_ids,
##                             columns=gene_labels)
##
##    lib_df = pd.DataFrame({
##        'cell_identity':         cell_identity,
##        'sampled_library_size':  [lib_size]*len(cell_ids),
##        'true_library_size':     counts_df.sum(axis=1).values
##    }, index=cell_ids)
##
##    theta_df = pd.DataFrame(theta_list,
##                            index=cell_ids,
##                            columns=list(groups))
##
##    return counts_df, lib_df, gm, de_assignments, de_A, de_B, de_C, theta_df

##def simulate_counts(
##    n_genes=2000,
##    n_de=400,
##    cells_per_group=3000,
##    groups=('A','B'),
##    gamma_shape=0.6,
##    gamma_rate=0.3,
##    de_lognormal_mean=0.5,
##    de_lognormal_sigma=0.4
##):
##
##    # 1) Baseline gene means
##    gene_labels = [f"Gene_{i+1}" for i in range(n_genes)]
##    baseline = np.random.gamma(shape=gamma_shape,
##                               scale=1.0/gamma_rate,
##                               size=n_genes)
##    gm = pd.DataFrame({g: baseline.copy() for g in groups},
##                      index=gene_labels)
##
##    gm['DE'] = False
##    gm['DE_group'] = [[] for _ in range(n_genes)]  
##    for g in groups:
##        gm[f'DE_factor_{g}'] = np.nan
##    de_assignments = {i: {} for i in range(n_genes)}
##
##    # 3) Assign DE genes for each group (may overlap)
##    de_A = np.random.choice(n_genes, size=n_de, replace=False)
##    de_B = np.random.choice(n_genes, size=n_de, replace=False)
##
##    for idx in de_A:
##        factor = np.random.lognormal(de_lognormal_mean,
##                                     de_lognormal_sigma)
##        gm.at[gene_labels[idx], 'A'] *= factor
##        gm.at[gene_labels[idx], 'DE'] = True
##        gm.at[gene_labels[idx], 'DE_group'].append('A')
##        gm.at[gene_labels[idx], 'DE_factor_A'] = factor
##        de_assignments[idx]['A'] = factor
##
##    for idx in de_B:
##        factor = np.random.lognormal(de_lognormal_mean,
##                                     de_lognormal_sigma)
##        gm.at[gene_labels[idx], 'B'] *= factor
##        gm.at[gene_labels[idx], 'DE'] = True
##        gm.at[gene_labels[idx], 'DE_group'].append('B')
##        gm.at[gene_labels[idx], 'DE_factor_B'] = factor
##        de_assignments[idx]['B'] = factor
##
##    means = {g: gm[g].values for g in groups}
##
##    cell_data     = []
##    cell_ids      = []
##    cell_identity = []
##    theta_list    = []
##    lib_size = 1500
##
##    for grp in groups:
##        mean_vec = means[grp]
##        for i in range(cells_per_group):
##            theta = [1.0 if g==grp else 0.0 for g in groups]
##            eff   = lib_size * (mean_vec / mean_vec.sum())
##            counts = np.random.poisson(lam=eff)
##            cell_data.append(counts)
##            cell_ids.append(f"{grp}")
##            cell_identity.append(grp)
##            theta_list.append(theta)
##
##    counts_df = pd.DataFrame(cell_data,
##                             index=cell_ids,
##                             columns=gene_labels)
##
##    lib_df = pd.DataFrame({
##        'cell_identity':         cell_identity,
##        'sampled_library_size':  [lib_size]*len(cell_ids),
##        'true_library_size':     counts_df.sum(axis=1).values
##    }, index=cell_ids)
##
##    theta_df = pd.DataFrame(theta_list,
##                            index=cell_ids,
##                            columns=list(groups))
##
##    return counts_df, lib_df, gm, de_assignments, de_A, de_B, theta_df

def simulate_counts(
    n_genes=2000,
    n_de=400,
    cells_per_identity=3000,
    gamma_shape=0.6,
    gamma_rate=0.3,
    de_lognormal_mean=0.5,
    de_lognormal_sigma=0.4,
    p_var=0.3
):
    # — 1) baseline gene‐means for A, VAR, B
    baseline = np.random.gamma(gamma_shape, 1.0/gamma_rate, size=n_genes)
    genes = [f"Gene_{i+1}" for i in range(n_genes)]
    gm = pd.DataFrame({
        'A':    baseline.copy(),
        'B':    baseline.copy(),
        'VAR':  baseline.copy()
    }, index=genes)

    # set up DE bookkeeping
    gm['DE'] = False
    gm['DE_group'] = [[] for _ in genes]
    gm['DE_factor_A']   = np.nan
    gm['DE_factor_B']   = np.nan
    gm['DE_factor_VAR'] = np.nan

    # — 2) choose DE genes in each topic (can overlap)
    de_A   = np.random.choice(n_genes, size=n_de, replace=False)
    de_VAR = np.random.choice(n_genes, size=n_de, replace=False)
    de_B   = np.random.choice(n_genes, size=n_de, replace=False)

    for idx in de_A:
        factor = np.random.lognormal(de_lognormal_mean, de_lognormal_sigma)
        gm.iat[idx, gm.columns.get_loc('A')] *= factor
        gm.at[genes[idx],'DE'] = True
        gm.at[genes[idx],'DE_group'].append('A')
        gm.at[genes[idx],'DE_factor_A'] = factor

    for idx in de_VAR:
        factor = np.random.lognormal(de_lognormal_mean, de_lognormal_sigma)
        gm.iat[idx, gm.columns.get_loc('VAR')] *= factor
        gm.at[genes[idx],'DE'] = True
        gm.at[genes[idx],'DE_group'].append('VAR')
        gm.at[genes[idx],'DE_factor_VAR'] = factor

    for idx in de_B:
        factor = np.random.lognormal(de_lognormal_mean, de_lognormal_sigma)
        gm.iat[idx, gm.columns.get_loc('B')] *= factor
        gm.at[genes[idx],'DE'] = True
        gm.at[genes[idx],'DE_group'].append('B')
        gm.at[genes[idx],'DE_factor_B'] = factor

    # — 3) simulate counts & θ
    A_means   = gm['A'].values
    VAR_means = gm['VAR'].values
    B_means   = gm['B'].values

    cell_data     = []
    cell_ids      = []
    cell_idents   = []
    theta_list    = []
    lib_size_list = []
    true_lib_list = []

    for ident in ['A','B']:
        id_means = A_means if ident=='A' else B_means

        for i in range(cells_per_identity):
            coin = np.random.rand()
            if coin < p_var:
                th = np.random.rand()  # mix weight
                mix = th*id_means + (1-th)*VAR_means
                if ident=='A':
                    theta_full = [th, 0, 1-th]
                else:
                    theta_full = [0, th, 1-th]
            else:
                mix = id_means
                if ident=='A':
                    theta_full = [1, 0, 0]
                else:
                    theta_full = [0, 1, 0]

            lib = 1500
            eff = lib * (mix / mix.sum())
            counts = np.random.poisson(eff)

            cell_data.append(counts)
            cell_ids.append(f"{ident}")
            cell_idents.append(ident)
            theta_list.append(theta_full)
            lib_size_list.append(lib)
            true_lib_list.append(counts.sum())

    # — 4) assemble DataFrames
    data_root = Path("../data/AB_VAR")
    data_root.mkdir(parents=True, exist_ok=True)

    counts_df = pd.DataFrame(cell_data, columns=genes, index=cell_ids)
    counts_df.to_csv(data_root/"simulated_counts.csv")

    lib_df = pd.DataFrame({
        'cell_identity':         cell_idents,
        'sampled_library_size':  lib_size_list,
        'true_library_size':     true_lib_list
    }, index=cell_ids)
    lib_df.to_csv(data_root/"library_sizes.csv")

    gm.to_csv(data_root/"gene_means_matrix.csv")

    theta_df = pd.DataFrame(theta_list, columns=['A','B', 'VAR'], index=cell_ids)
    theta_df.to_csv(data_root/"theta.csv")

    return counts_df, lib_df, gm, de_A, de_VAR, de_B, theta_df


if __name__ == '__main__':
    counts_df, lib_df, gm, de_A, de_VAR, de_B, theta_df = simulate_counts()
##    counts_df.to_csv('../data/AB/simulated_counts.csv')
##    lib_df.to_csv('../data/AB/library_sizes.csv')
##    gene_means.to_csv('../data/AB/gene_means_matrix.csv')
##    theta_df.to_csv('../data/AB/theta.csv')
