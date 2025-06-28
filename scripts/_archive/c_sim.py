import numpy as np
import pandas as pd
import pickle
import os
from functions import (gibbs_update_numba,
                       run_gibbs_sampling_numba,
                       initialize_long_data,
                       compute_constants)
from sklearn.decomposition import LatentDirichletAllocation, NMF
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def compute_normalized_avg_expr_by_celltype(counts_df):

    avg_expr = counts_df.groupby(level=0).mean() 
    avg_expr = avg_expr.div(avg_expr.sum(axis=1), axis=0).fillna(0)
    avg_expr_df = avg_expr.T
    
    return avg_expr_df

def plot_betas_with_avg_expr(model_beta_dict, counts_df, output_file):

    avg_expr_df = compute_normalized_avg_expr_by_celltype(counts_df)

    all_vals = []
    for df in model_beta_dict.values():
        all_vals.extend(df.values.flatten())  
    all_vals.extend(avg_expr_df.values.flatten())  

    vmin = np.min(all_vals)
    vmax = np.max(all_vals)

    model_names = list(model_beta_dict.keys())
    n_models = len(model_names)
    total_panels = n_models + 1  

    fig, axs = plt.subplots(1, total_panels, figsize=(5 * total_panels, 6))
    if total_panels == 1:
        axs = [axs]

    for ax, model in zip(axs[:n_models], model_names):
        beta_df = model_beta_dict[model]
        sns.heatmap(
            beta_df, ax=ax, cmap="viridis", annot=True, fmt=".2f",
            vmin=vmin, vmax=vmax
        )
        ax.set_title(f"{model} Beta")
        ax.set_xlabel("Topic")
        ax.set_ylabel("Gene")

    ax = axs[-1]
    sns.heatmap(
        avg_expr_df, ax=ax, cmap="viridis", annot=True, fmt=".2f",
        vmin=vmin, vmax=vmax
    )
    ax.set_title("Avg Expr by Cell Type")
    ax.set_xlabel("Cell Type")
    ax.set_ylabel("Gene")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved Beta + AvgExpr heatmap to {output_file}")

def plot_all_thetas_heatmaps(model_theta_dict, output_file):
    
    all_vals = []
    for df in model_theta_dict.values():
        all_vals.extend(df.values.flatten())
    vmin = np.min(all_vals)
    vmax = np.max(all_vals)
    
    model_names = list(model_theta_dict.keys())
    n_models = len(model_names)
    
    fig, axs = plt.subplots(1, n_models, figsize=(5*n_models, 6))
    if n_models == 1:
        axs = [axs]
    for ax, model in zip(axs, model_names):
        df = model_theta_dict[model]
        sns.heatmap(df, ax=ax, cmap="viridis", annot=True, fmt=".2f", vmin=vmin, vmax=vmax)
        ax.set_title(f"{model} Theta")
        ax.set_xlabel("Topic")
        ax.set_ylabel("Group")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved combined Theta heatmap to {output_file}")

def plot_true_thetas_histograms_by_topic(theta_df, output_file):
    
    theta_df_reset = theta_df.reset_index().rename(columns={'index': 'cell_type'})
    
    topic_cols = [col for col in theta_df_reset.columns if col.lower().startswith("topic")]
    
    theta_long = theta_df_reset.melt(id_vars="cell_type", value_vars=topic_cols,
                                     var_name="Topic", value_name="Theta")
    
    g = sns.FacetGrid(
        theta_long, 
        col="Topic", 
        hue="cell_type", 
        col_wrap=3,
        sharex=True, 
        sharey=False
    )
    bins = np.linspace(0, 1, 30)
    g.map(sns.histplot, "Theta", bins=bins, alpha=0.6)
    g.set(xlim=(0, 1))
    g.add_legend(title="Leaf ID", loc="lower right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved true theta histograms to {output_file}")

def match_topics(true_mat, est_mat):

    n_topics = true_mat.shape[1]
    cost_matrix = np.zeros((n_topics, n_topics))
    for i in range(n_topics):
        for j in range(n_topics):
            corr = np.corrcoef(true_mat[:, i], est_mat[:, j])[0, 1]
            cost_matrix[i, j] = -corr 
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return col_ind

def plot_est_vs_true_theta_by_model(true_theta, model_estimated_theta_dict, output_file):

    if hasattr(true_theta, 'values'):
        true_theta = true_theta.values

    n_topics = true_theta.shape[1]
    n_models = len(model_estimated_theta_dict)
    
    fig, axs = plt.subplots(n_topics, n_models, figsize=(5 * n_models, 4 * n_topics), squeeze=False)
    
    for col_idx, (model_name, est_theta) in enumerate(model_estimated_theta_dict.items()):
        
        if hasattr(est_theta, 'values'):
            est_theta = est_theta.values
            
        perm = match_topics(true_theta, est_theta)
        est_theta_aligned = est_theta[:, perm]
        
        for topic in range(n_topics):
            ax = axs[topic, col_idx]
            ax.scatter(true_theta[:, topic], est_theta_aligned[:, topic], alpha=0.5)
            ax.plot([0, 1], [0, 1], 'r--', lw=1)  # identity line for reference
            ax.set_title(f"{model_name}, Topic {topic}")
            ax.set_xlabel("True θ")
            ax.set_ylabel("Estimated θ")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved θ comparison plot to {output_file}")

def run_plots(estimate_folder, plot_folder, count_matrix_file, theta_file, compare_cell_level_theta=False):

    os.makedirs(plot_folder, exist_ok=True)
    counts_df = pd.read_csv(count_matrix_file, index_col=0)
    cell_level_thetas = pd.read_csv(theta_file, index_col=0)

    hier_beta_df = pd.read_csv(os.path.join(estimate_folder, "HLDA_beta.csv"), index_col=0)
    hier_theta_df = pd.read_csv(os.path.join(estimate_folder, "HLDA_theta.csv"), index_col=0)
    hier_theta_df_collapsed = pd.read_csv(os.path.join(estimate_folder, "HLDA_theta_collapsed.csv"), index_col=0)

    lda_beta_df = pd.read_csv(os.path.join(estimate_folder, "LDA_beta.csv"), index_col=0)
    lda_theta_df = pd.read_csv(os.path.join(estimate_folder, "LDA_theta.csv"), index_col=0)
    lda_theta_df_collapsed = pd.read_csv(os.path.join(estimate_folder, "LDA_theta_collapsed.csv"), index_col=0)

    nmf_beta_df = pd.read_csv(os.path.join(estimate_folder, "NMF_beta.csv"), index_col=0)
    nmf_theta_df = pd.read_csv(os.path.join(estimate_folder, "NMF_theta.csv"), index_col=0)
    nmf_theta_df_collapsed = pd.read_csv(os.path.join(estimate_folder, "NMF_theta_collapsed.csv"), index_col=0)

    model_beta_dict = {
        "hierarchical": hier_beta_df,
        "LDA": lda_beta_df,
        "NMF": nmf_beta_df
    }
    model_theta_dict_collapsed = {
        "hierarchical": hier_theta_df_collapsed,
        "LDA": lda_theta_df_collapsed,
        "NMF": nmf_theta_df_collapsed
    }
    model_theta_dict = {
        "hierarchical": hier_theta_df,
        "LDA": lda_theta_df,
        "NMF": nmf_theta_df
    }

    beta_output_file = os.path.join(plot_folder, "beta_plot.png")
    plot_betas_with_avg_expr(model_beta_dict, counts_df, beta_output_file)

    theta_output_file = os.path.join(plot_folder, "theta_plot.png")
    plot_all_thetas_heatmaps(model_theta_dict_collapsed, theta_output_file)

    true_theta_hist_output_file = os.path.join(plot_folder, "true_theta_histograms.png")
    plot_true_thetas_histograms_by_topic(cell_level_thetas, true_theta_hist_output_file)

    if compare_cell_level_theta:

        cell_theta_output_file = os.path.join(plot_folder, "cell_theta_plot.png")
        plot_est_vs_true_theta_by_model(cell_level_thetas, model_theta_dict, cell_theta_output_file)


def create_beta_matrix(total_topics=5):

    beta = np.eye(total_topics)
    return beta

def generate_cells(concentration_parameters, leaves, leaf_topics, beta, cells_per_leaf=3000, lambda_count=150):
    num_topics = beta.shape[0]
    num_genes = beta.shape[1]
    total_cells = len(leaves) * cells_per_leaf

    theta_matrix = np.zeros((total_cells, num_topics))
    cell_expressions = np.zeros((total_cells, num_genes), dtype=int)
    leaf_ids = np.empty(total_cells, dtype=int)

    cell_index = 0
    for leaf in leaves:
        available = leaf_topics[leaf]  # e.g., for leaf 2: [2, 0, 1]
        
        theta_leaf = np.random.dirichlet(concentration_parameters, size=cells_per_leaf)
        theta_matrix[cell_index:cell_index+cells_per_leaf, available] = theta_leaf
        
        for i in range(cells_per_leaf):

            leaf_ids[cell_index + i] = leaf
            
            total_counts = np.random.poisson(lambda_count)
            counts = np.zeros(num_genes, dtype=int)
            
            p_available = theta_matrix[cell_index + i, available]
                
            for _ in range(total_counts):
                chosen_topic = np.random.choice(available, p=p_available)
                gene = np.random.choice(num_genes, p=beta[chosen_topic])
                counts[gene] += 1
                
            cell_expressions[cell_index + i, :] = counts
        
        cell_index += cells_per_leaf

    return cell_expressions, theta_matrix, leaf_ids

def run_generation():

    total_topics = 5  # topics 0 and 1 are roots; topics 2,3,4 are leaves.
    cells_per_leaf = 3000
    lambda_count = 100
    random_seed = 42

    output_folder = '../data/c_sim/'
    
    np.random.seed(random_seed)
    
    beta = create_beta_matrix(total_topics)
    
    leaves = [2, 3, 4]
    
    leaf_topics = {
        2: [2, 0, 1],
        3: [3, 0, 1],
        4: [4, 0, 1]
    }
    
    concentration_parameters = [8, 2, 2]
    
    count_matrix, theta_matrix, leaf_ids = generate_cells(
        concentration_parameters,
        leaves=leaves,
        leaf_topics=leaf_topics,
        beta=beta,
        cells_per_leaf=cells_per_leaf,
        lambda_count=lambda_count
    )
    
    gene_names = [f"Gene{i}" for i in range(beta.shape[1])]
    
    counts_df = pd.DataFrame(count_matrix, columns=gene_names, index=leaf_ids)
    
    theta_columns = [f"Topic{i}" for i in range(total_topics)]
    theta_df = pd.DataFrame(theta_matrix, columns=theta_columns, index=leaf_ids)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    counts_df.to_csv(os.path.join(output_folder, "simulated_count_matrix.csv"), index=True)
    theta_df.to_csv(os.path.join(output_folder, "simulated_theta_matrix.csv"), index=True)


def run_lda_nmf_pipeline(num_topics, input_folder, input_file, output_folder):
    
    counts_df = pd.read_csv(os.path.join(input_folder, input_file), index_col=0)
    cell_ids = counts_df.index.to_list()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print("Fitting LDA...")
    lda = LatentDirichletAllocation(n_components=num_topics)
    lda.fit(counts_df)
    
    theta_lda = lda.transform(counts_df)
    theta_lda = theta_lda / theta_lda.sum(axis=1, keepdims=True)
    
    raw_beta_lda = lda.components_
    beta_lda = raw_beta_lda / raw_beta_lda.sum(axis=1, keepdims=True)
    
    beta_lda_df = pd.DataFrame(beta_lda, 
                               columns=counts_df.columns, 
                               index=[f"Topic_{i}" for i in range(num_topics)])
    theta_lda_df = pd.DataFrame(theta_lda, 
                                index=counts_df.index, 
                                columns=[f"Topic_{i}" for i in range(num_topics)])

    collapsed_theta_lda = theta_lda_df.groupby(level=0).mean()
    
    lda_beta_out = os.path.join(output_folder, "LDA_beta.csv")
    beta_lda_df.T.to_csv(lda_beta_out)
    
    lda_grouped_theta_out = os.path.join(output_folder, "LDA_theta_collapsed.csv")
    lda_theta_out = os.path.join(output_folder, "LDA_theta.csv")
    
    collapsed_theta_lda.to_csv(lda_grouped_theta_out)
    theta_lda_df.to_csv(lda_theta_out)

    print("Fitting NMF...")
    nmf_init = 'nndsvd' if num_topics <= counts_df.shape[1] else 'random'
    
    nmf = NMF(n_components=num_topics, max_iter=1000, init=nmf_init)
    W = nmf.fit_transform(counts_df)  # document-topic matrix
    H = nmf.components_               # topic-term matrix
    
    W_norm = W / W.sum(axis=1, keepdims=True)
    theta_nmf = W_norm
    H_norm = H / H.sum(axis=1, keepdims=True)
    beta_nmf = H_norm
    
    beta_nmf_df = pd.DataFrame(beta_nmf, 
                               columns=counts_df.columns, 
                               index=[f"Topic_{i}" for i in range(num_topics)])
    theta_nmf_df = pd.DataFrame(theta_nmf, 
                                index=counts_df.index, 
                                columns=[f"Topic_{i}" for i in range(num_topics)])
    
    collapsed_theta_nmf = theta_nmf_df.groupby(level=0).mean()
    
    nmf_beta_out = os.path.join(output_folder, "NMF_beta.csv")
    beta_nmf_df.T.to_csv(nmf_beta_out)
    
    nmf_theta_out = os.path.join(output_folder, "NMF_theta.csv")
    nmf_theta_collapsed_out = os.path.join(output_folder, "NMF_theta_collapsed.csv")

    theta_nmf_df.to_csv(nmf_theta_out)
    collapsed_theta_nmf.to_csv(nmf_theta_collapsed_out)

def reconstruct_beta_theta(sample_dir, save_dir, gene_names, cell_ids):

    beta_accumulator = None
    theta_accumulator = None
    sample_files = sorted([f for f in os.listdir(sample_dir) if f.endswith(".pkl")])
    
    print(f"Found {len(sample_files)} sample files for reconstruction.")
    
    for file in sample_files:
        with open(os.path.join(sample_dir, file), 'rb') as f:
            constants_sample = pickle.load(f)
            A = constants_sample['A']   # Assume shape: (num_genes, num_topics)
            D = constants_sample['D']   # Assume shape: (num_cells, num_topics)
        
        if beta_accumulator is None:
            beta_accumulator = A.copy()
        else:
            beta_accumulator += A
        
        if theta_accumulator is None:
            theta_accumulator = D.copy()
        else:
            theta_accumulator += D
    
    n_files = len(sample_files)
    beta_matrix = beta_accumulator / n_files
    theta_matrix = theta_accumulator / n_files

    beta_matrix /= beta_matrix.sum(axis=0)
    theta_matrix /= theta_matrix.sum(axis=1, keepdims=True)

    beta_df = pd.DataFrame(beta_matrix, index=gene_names, columns=[f"Topic_{k}" for k in range(beta_matrix.shape[1])])
    theta_df = pd.DataFrame(theta_matrix, index=cell_ids, columns=[f"Topic_{k}" for k in range(theta_matrix.shape[1])])

    theta_collapsed = theta_df.groupby(level=0).mean()
    
    # Create the save directory if it does not exist.
    os.makedirs(save_dir, exist_ok=True)
    
    beta_output_path = os.path.join(save_dir, "HLDA_beta.csv")
    theta_output_path = os.path.join(save_dir, "HLDA_theta.csv")
    theta_collapsed_output_path = os.path.join(save_dir, "HLDA_theta_collapsed.csv")
    
    beta_df.to_csv(beta_output_path)
    theta_df.to_csv(theta_output_path)
    theta_collapsed.to_csv(theta_collapsed_output_path)

def run_gibbs_pipeline(num_topics, input_folder, sample_output_dir, estimate_output_dir):

    num_loops = 1000
    burn_in = 500
    hyperparams = {'alpha_beta': 1, 'alpha_c': 1}

    count_matrix_file = os.path.join(input_folder, input_file)
    count_matrix = pd.read_csv(count_matrix_file, index_col=0)
    
    cell_identities = count_matrix.index.tolist()
    gene_names = count_matrix.columns.tolist()

    long_data, identity_mapping = initialize_long_data(count_matrix,
                                                       cell_identities,
                                                       num_topics,
                                                       input_folder)

    constants = compute_constants(long_data, num_topics)

    run_gibbs_sampling_numba(long_data, constants, hyperparams,
                       num_loops, burn_in, sample_output_dir, identity_mapping)

    reconstruct_beta_theta(sample_output_dir, estimate_output_dir, gene_names, cell_identities)
    

if __name__ == '__main__':
    
    folder_name = 'c_sim_5_topic'
    num_topics=5

    input_folder = '../data/c_sim/'
    input_file = "simulated_count_matrix.csv"
    theta_file = 'simulated_theta_matrix.csv'
    output_folder = os.path.join("../estimates/", folder_name)
    sample_output_dir = os.path.join('../samples/', folder_name)
    estimate_output_dir = os.path.join('../estimates/', folder_name)
    plot_folder = os.path.join(estimate_output_dir, '_plots')
    
##    run_generation()
    run_lda_nmf_pipeline(num_topics, input_folder, input_file, output_folder)
    run_gibbs_pipeline(num_topics, input_folder, sample_output_dir, estimate_output_dir)
    
    run_plots(estimate_output_dir, plot_folder,
              os.path.join(input_folder, input_file),
              os.path.join(input_folder, theta_file),
              True)


























