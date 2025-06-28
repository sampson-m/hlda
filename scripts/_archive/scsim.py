import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_data(
    n_singlets=3000,
    total_genes=500,
    n_identity=3,
    n_activity=2,
    activity_genes_per_topic=50,
    de_prob=0.025,
    de_loc=0.75,
    lib_loc=4,
    lib_scale=0.5,
    mean_rate=7.68,
    mean_shape=0.34,
    bcv_dispersion=0.448,
    activity_factor=2.0,
    activity_fraction=0.3,
    activity_usage_range=(0.1, 0.7),
    seed=1234
):
    np.random.seed(seed)
    nb_k = 1 / bcv_dispersion
    gamma_scale = mean_rate / mean_shape

    # Identity programs with DE gene info
    identity_programs = np.zeros((n_identity, total_genes))
    de_masks = []  # list of boolean arrays for each identity program
    for i in range(n_identity):
        program = np.random.gamma(shape=mean_shape, scale=gamma_scale, size=total_genes)
        de_mask = np.random.rand(total_genes) < de_prob
        program[de_mask] *= de_loc
        identity_programs[i, :] = program
        de_masks.append(de_mask)

    # Activity programs and record the gene indices used
    activity_programs = np.ones((n_activity, total_genes))
    act_gene_indices = []  # list of arrays of gene indices for each activity program
    remaining_genes = np.arange(total_genes)
    for j in range(n_activity):
        act_genes = np.random.choice(remaining_genes, size=activity_genes_per_topic, replace=False)
        act_gene_indices.append(act_genes)
        remaining_genes = np.setdiff1d(remaining_genes, act_genes)
        act_vals = np.random.gamma(shape=mean_shape, scale=gamma_scale, size=activity_genes_per_topic)
        act_vals *= activity_factor
        activity_programs[j, act_genes] = act_vals

    library_sizes = np.random.lognormal(mean=lib_loc, sigma=lib_scale, size=n_singlets)
    identities = np.random.choice(n_identity, size=n_singlets)

    # Activity usage per cell for each activity program
    activity_usage = np.zeros((n_singlets, n_activity))
    for i in range(n_singlets):
        for j in range(n_activity):
            if np.random.rand() < activity_fraction:
                activity_usage[i, j] = np.random.uniform(*activity_usage_range)

    # Theta (cell-topic proportions)
    theta = np.zeros((n_singlets, n_identity + n_activity))
    for i in range(n_singlets):
        w_id = max(0, 1 - activity_usage[i].sum())
        theta[i, identities[i]] = w_id
        theta[i, n_identity:] = activity_usage[i, :]

    # Beta (topic-gene profiles)
    beta = np.vstack([identity_programs, activity_programs])

    # Mean expression and counts
    lambda_cells = np.zeros((n_singlets, total_genes))
    for i in range(n_singlets):
        lam = (theta[i, identities[i]] * identity_programs[identities[i], :] +
               theta[i, n_identity] * activity_programs[0, :] +
               theta[i, n_identity+1] * activity_programs[1, :])
        lam *= library_sizes[i]
        lambda_cells[i, :] = lam

    counts = np.zeros((n_singlets, total_genes), dtype=int)
    for i in range(n_singlets):
        mu_vec = lambda_cells[i, :]
        p_vec = nb_k / (nb_k + mu_vec)
        counts[i, :] = np.random.negative_binomial(nb_k, p_vec)

    metadata = pd.DataFrame({
        'CellID': np.arange(n_singlets),
        'CellType': [f"Identity_{x+1}" for x in identities],
        'Activity1': activity_usage[:, 0],
        'Activity2': activity_usage[:, 1],
        'LibrarySize': library_sizes
    })

    return theta, beta, counts, metadata, de_masks, act_gene_indices

def plot_theta_stacked(
    theta,
    metadata,
    group_col='CellType',
    output_dir='.',
    output_prefix='theta_plot',
    bin_size=50
):

    os.makedirs(output_dir, exist_ok=True)
    groups = metadata[group_col].unique()
    n_groups = len(groups)
    n_topics = theta.shape[1]
    cmap = plt.cm.get_cmap('tab10', n_topics)
    topic_colors = [cmap(i) for i in range(n_topics)]

    fig, axes = plt.subplots(nrows=1, ncols=n_groups, figsize=(4*n_groups, 4), sharey=True)
    if n_groups == 1:
        axes = [axes]

    for ax, group in zip(axes, groups):
        idx = metadata[group_col] == group
        sub_theta = theta[idx, :]
        order = np.argsort(-np.max(sub_theta, axis=1))
        sub_theta = sub_theta[order, :]

        # Bin the cells to smooth out the plot
        n_sub = sub_theta.shape[0]
        n_bins = n_sub // bin_size
        binned = []
        for b in range(n_bins):
            chunk = sub_theta[b*bin_size:(b+1)*bin_size, :]
            binned.append(chunk.mean(axis=0))
        remainder = n_sub % bin_size
        if remainder > 0:
            chunk = sub_theta[n_bins*bin_size:, :]
            binned.append(chunk.mean(axis=0))
        binned = np.array(binned)

        x_vals = np.arange(binned.shape[0])
        ax.stackplot(
            x_vals,
            *binned.T,
            colors=topic_colors,
            alpha=1.0,
            linewidth=0,
            labels=[f"Topic {t+1}" for t in range(n_topics)]
        )
        ax.set_title(str(group))
        ax.set_xlim([0, binned.shape[0]-1])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Binned Cells")
        if ax == axes[0]:
            ax.set_ylabel("Membership Proportion")

    axes[-1].legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    outpath = os.path.join(output_dir, f"{output_prefix}.png")
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved stacked theta plot to: {outpath}")

if __name__ == "__main__":
    theta, beta, counts, metadata, de_masks, act_gene_indices = simulate_data()
    output_dir = "../data/scsim/"
    os.makedirs(output_dir, exist_ok=True)

    theta_df = pd.DataFrame(theta, columns=[f"Topic_{i+1}" for i in range(theta.shape[1])])
    beta_df = pd.DataFrame(beta, columns=[f"Gene_{i+1}" for i in range(beta.shape[1])])
    counts_df = pd.DataFrame(counts, columns=[f"Gene_{i+1}" for i in range(counts.shape[1])])
    metadata.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    theta_df.to_csv(os.path.join(output_dir, "true_theta.csv"), index=False)
    beta_df.to_csv(os.path.join(output_dir, "true_beta.csv"), index=False)
    counts_df.to_csv(os.path.join(output_dir, "counts.csv"), index=False)

    # Save DE gene metadata for identity programs
    gene_names = [f"Gene_{i+1}" for i in range(beta.shape[1])]
    de_meta = pd.DataFrame({'Gene': gene_names})
    for i, de_mask in enumerate(de_masks):
        de_meta[f"DE_Identity_{i+1}"] = de_mask
    de_meta.to_csv(os.path.join(output_dir, "de_genes_identity.csv"), index=False)

    # Save activity gene metadata (each column lists genes selected for that activity)
    act_meta = {}
    for j, act_genes in enumerate(act_gene_indices):
        act_meta[f"Activity_{j+1}"] = [f"Gene_{g+1}" for g in sorted(act_genes)]
    act_meta_df = pd.DataFrame(act_meta)
    act_meta_df.to_csv(os.path.join(output_dir, "activity_genes.csv"), index=False)

    plot_theta_stacked(theta, metadata, group_col='CellType', output_dir=output_dir, output_prefix='theta_stacked')
