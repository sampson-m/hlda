
import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List as TypedList
from tqdm import tqdm
import time
import pandas as pd
import os
import re
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from scipy.sparse import issparse
import scanpy as sc
from sklearn.decomposition import LatentDirichletAllocation, NMF
import subprocess
from config import TOPICS, K

######################################################################################################
#### RAW PROCESSING ##################################################################################
######################################################################################################

def process_raw_data(input_file, output_folder):

    print('Processing sampled data')
    
    adata = sc.read_h5ad(input_file)
    sc.pp.filter_genes(adata, min_counts=10)
    adata_raw = adata.copy()
    
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    top_genes = adata.var_names[adata.var["highly_variable"]]
    
    adata_raw = adata_raw[:, adata.var['highly_variable']]
    sc.pp.filter_cells(adata_raw, min_genes=10)
    print("Shape after subsetting raw data to highly variable genes and filtering cells:", adata_raw.shape)

    if issparse(adata_raw.X):
        dense_matrix = adata_raw.X.toarray().astype(int)
    else:
        dense_matrix = adata_raw.X.astype(int)

    dense_df = pd.DataFrame(
        dense_matrix, 
        index=adata_raw.obs['celltype'], 
        columns=adata_raw.var_names
    )
    
    outfile = os.path.join(output_folder, 'count_matrix.csv')
    dense_df.to_csv(outfile)


######################################################################################################
#### UTIL FUNCTIONS ##################################################################################
######################################################################################################

def clear_directory(dir_path):
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        return
    
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)        
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)   
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def create_topic_hierarchy(cell_identities, extra_node_count):

    unique_cells = sorted(set(cell_identities)) 
    hierarchy = {cell: [i] for i, cell in enumerate(unique_cells)}
    
    start = len(unique_cells)
    extra_nodes = list(range(start, start + extra_node_count))
    
    for cell in hierarchy:
        hierarchy[cell] += extra_nodes
    
    return hierarchy

def map_topics_to_nodes(cell_identities, n_fc_nodes):

    unique_cells = sorted(set(cell_identities))
    
    mapping = {}
    for i, cell in enumerate(unique_cells):
        mapping[cell] = [i]
    
    start = len(unique_cells)
    for j in range(n_fc_nodes):
        key = f"fc {j+1}"
        mapping[key] = [start + j]
    
    return mapping


def get_filenames(simdir):
    ## Input files
    countfn = '%s/counts.npz' % simdir      
    countfilt_fn = '%s/counts_filt.npz' % simdir
    TPM_fn = '%s/TPM.npz' % simdir

    genestats_fn = '%s/genestats.txt' % simdir

    highvargenes_fn = '%s/highvargenes.txt' % simdir

    return(countfn, countfilt_fn, TPM_fn, genestats_fn, highvargenes_fn)

def load_model_betas(estimate_folder):
    hier_beta_df = pd.read_csv(os.path.join(estimate_folder, "HLDA_beta.csv"), index_col=0)
    lda_beta_df = pd.read_csv(os.path.join(estimate_folder, "LDA_beta.csv"), index_col=0)
    nmf_beta_df = pd.read_csv(os.path.join(estimate_folder, "fastTopics_beta.csv"), index_col=0)
    return {
        "hierarchical": hier_beta_df,
        "LDA": lda_beta_df,
        "NMF": nmf_beta_df
    }

def load_model_thetas(estimate_folder):
    hier_theta_df = pd.read_csv(os.path.join(estimate_folder, "HLDA_theta.csv"), index_col=0)
    lda_theta_df = pd.read_csv(os.path.join(estimate_folder, "LDA_theta.csv"), index_col=0)
    nmf_theta_df = pd.read_csv(os.path.join(estimate_folder, "fastTopics_theta.csv"), index_col=0)
    return {
        "hierarchical": hier_theta_df,
        "LDA": lda_theta_df,
        "NMF": nmf_theta_df
    }

def format_genemean_matrix(genemean_path, beta_df):
    gene_means = pd.read_csv(genemean_path, index_col=0)
    
    group_cols = [col for col in gene_means.columns if re.match(r"group\d+_genemean", col)]
    prog_cols = [col for col in gene_means.columns if re.match(r"prog_genemean(_\d+)?", col)]
    
    all_cols = group_cols + prog_cols
    gene_means = gene_means[all_cols].copy()
    gene_means = gene_means.loc[gene_means.index.intersection(beta_df.index)]
    
    new_names = {}
    group_topic_numbers = []
    for col in group_cols:
        m = re.search(r"group(\d+)_genemean", col)
        if m:
            topic_num = int(m.group(1)) - 1
            new_names[col] = f"topic{topic_num}"
            group_topic_numbers.append(topic_num)
    
    if group_topic_numbers:
        start_prog_topic = max(group_topic_numbers) + 1
    else:
        start_prog_topic = 0

    def extract_prog_num(col):
        m = re.search(r"_(\d+)$", col)
        return int(m.group(1)) if m else 0
    prog_cols_sorted = sorted(prog_cols, key=extract_prog_num)
    
    for i, col in enumerate(prog_cols_sorted):
        new_names[col] = f"topic{start_prog_topic + i}"
    
    gene_means.rename(columns=new_names, inplace=True)
    return gene_means

def compute_topic_cosine_similarity(beta_df, gene_means_df):
    sim_matrix = cosine_similarity(gene_means_df.T, beta_df.T)
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    topic_matching = {gene_means_df.columns[i]: beta_df.columns[j] for i, j in zip(row_ind, col_ind)}
    matched_cos_sims = {}
    for g_topic, b_topic in topic_matching.items():
        v1 = gene_means_df[g_topic].values
        v2 = beta_df[b_topic].values
        matched_cos_sims[g_topic] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return topic_matching, matched_cos_sims


######################################################################################################
#### SAMPLING FUNCTIONS ##############################################################################
######################################################################################################

def initialize_long_data(count_matrix, cell_identities, topic_hierarchy, save_path=None):

    unique_identities = list(set(cell_identities))
    identity_to_int = {identity: idx for idx, identity in enumerate(unique_identities)}
    int_to_identity = {idx: identity for identity, idx in identity_to_int.items()} 

    cell_identities_int = np.array([identity_to_int[cell] for cell in cell_identities], dtype=int)

    count_matrix = count_matrix.to_numpy().astype(int)

    long_data_dtype = np.dtype([
        ('cell_idx', np.int32),
        ('gene_idx', np.int32),
        ('cell_identity', np.int32),
        ('z', np.int32)
    ])

    total_gene_counts = count_matrix.sum()
    print("Total Gene Counts:", total_gene_counts)

    long_data = np.zeros(total_gene_counts, dtype=long_data_dtype)

    row_indices, col_indices = np.where(count_matrix > 0)
    counts = count_matrix[row_indices, col_indices]

    expanded_rows = np.repeat(row_indices, counts)
    expanded_cols = np.repeat(col_indices, counts)
    expanded_cell_ids = np.repeat(cell_identities_int[row_indices], counts)
    
    topic_constraints = np.array([
        topic_hierarchy[int_to_identity[cell_identity]]
        for cell_identity in expanded_cell_ids
    ], dtype=object)

    z_array = np.array([
        np.random.choice(valid_topics)
        for valid_topics in topic_constraints
    ])

    long_data['cell_idx'] = expanded_rows
    long_data['gene_idx'] = expanded_cols
    long_data['cell_identity'] = expanded_cell_ids
    long_data['z'] = z_array

    long_data_df = pd.DataFrame({
        'cell_idx': long_data['cell_idx'],
        'gene_idx': long_data['gene_idx'],
        'cell_identity': long_data['cell_identity'],
        'z': long_data['z']
    })
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        long_data_df.to_csv(os.path.join(save_path, 'long_data_init.csv'), index=False)
        
        topic_int_df = pd.DataFrame.from_dict(int_to_identity, orient="index")
        topic_int_df.to_csv(os.path.join(save_path, 'topic_int_mapping.csv'))
        
        topic_hierarchy_df = pd.DataFrame.from_dict(topic_hierarchy, orient="index")
        topic_hierarchy_df.to_csv(os.path.join(save_path, 'topic_hierarchy.csv'))

##        topic_node_map = map_topics_to_nodes(unique_identities, n_fc_nodes)
##        topic_node_map_df = pd.DataFrame.from_dict(topic_node_map, orient="index")
##        topic_node_map_df.to_csv(os.path.join(save_path, 'topic_node_map.csv'))

    return long_data, int_to_identity

def compute_constants(long_data, K):
    
    cell_identity = long_data['cell_identity']
    unique_cell_ids = np.unique(cell_identity)
    n_cell_ids = len(unique_cell_ids)
        
    gene_idx = long_data['gene_idx']
    cell_idx = long_data['cell_idx']
    z = long_data['z']
    
    unique_genes = np.unique(gene_idx)
    unique_cells = np.unique(cell_idx)
    n_cells = len(unique_cells)
    
    n_expected_genes = np.max(gene_idx) + 1
    all_genes = np.arange(n_expected_genes)
    
    missing_genes = np.setdiff1d(all_genes, unique_genes)
    print("Missing gene indices (columns with no entries):", missing_genes)
    
    print("Computing A...")
    # A[g, k]: Number of times gene g is sampled from topic k across all cells.
    gene_topic_pairs = gene_idx * K + z
    counts = np.bincount(gene_topic_pairs, minlength=n_expected_genes * K)
    A = counts.reshape(n_expected_genes, K)
    print("A matrix computed successfully.")
    
    assert np.sum(A) == len(long_data), "A computation failed!"
    
    print("Computing B...")
    # B[k]: Number of times topic k is sampled across all cells.
    B = np.bincount(z, minlength=K)
    print("B vector computed successfully.")
    
    assert np.sum(B) == len(long_data), "B computation failed!"
        
    print("Computing D...")
    # D[c, k]: Number of times topic k is sampled in cell c
    cell_topic_pairs = cell_idx * K + z
    counts = np.bincount(cell_topic_pairs, minlength=n_cells * K)
    D = counts.reshape(n_cells, K)
    print("D matrix computed successfully.")
    
    assert np.sum(D) == len(long_data), "D computation failed!"
    
    constants = {
        'A': A,
        'B': B,
        'D': D,
        'unique_genes': unique_genes,
        'unique_cells': unique_cells,
        'unique_topics': np.arange(K),
        'G': len(unique_genes),
        'K': K,
        'C': n_cells
    }
    
    return constants

@njit
def gibbs_update_numba(cell_idx_arr, gene_idx_arr, cell_identity_arr, z_arr,
                       A, B, D,
                       valid_topic_list,
                       alpha_beta, alpha_c, G, K):

    total_counts = cell_idx_arr.shape[0]
    G_alpha_beta = G * alpha_beta

##    for i in range(total_counts):
    for _ in range(total_counts):
        i = np.random.randint(0, total_counts)
        c = cell_idx_arr[i]
        g = gene_idx_arr[i]
        ident = cell_identity_arr[i]
        current_z = z_arr[i]

        A[g, current_z] -= 1
        B[current_z] -= 1
        D[c, current_z] -= 1

        valid_topics = valid_topic_list[ident]
        n_valid = valid_topics.shape[0]

        probs = np.empty(n_valid, dtype=np.float64)
        sum_probs = 0.0
        for j in range(n_valid):
            t = valid_topics[j]
            prob = ((alpha_beta + A[g, t]) / (G_alpha_beta + B[t])) * (alpha_c + D[c, t])
            probs[j] = prob
            sum_probs += prob

        for j in range(n_valid):
            probs[j] /= sum_probs

        r = np.random.rand()
        cumulative = 0.0
        new_z = valid_topics[-1]
        for j in range(n_valid):
            cumulative += probs[j]
            if r < cumulative:
                new_z = valid_topics[j]
                break

        A[g, new_z] += 1
        B[new_z] += 1
        D[c, new_z] += 1
        z_arr[i] = new_z

def run_gibbs_sampling_numba(topic_hierarchy, long_data, constants, hyperparams,
                             num_loops, burn_in, output_dir, identity_mapping):

    start_time = time.time()
    clear_directory(output_dir)

    A = constants['A']
    B = constants['B']
    D = constants['D']
    G = constants['G']
    K = constants['K']

    alpha_beta = hyperparams['alpha_beta']
    alpha_c = hyperparams['alpha_c']

    total_counts = len(long_data)
    valid_topic_list_py = []
    num_identities = len(identity_mapping)

    cell_ids = np.unique(long_data['cell_identity'])
    unique_cell_ids = [identity_mapping[t] for t in cell_ids]

    for i in range(num_identities):
        cell_identity_str = identity_mapping[i] 
        vtopics = topic_hierarchy[cell_identity_str]
        valid_topic_list_py.append(np.array(vtopics, dtype=np.int32))

    valid_topic_list = TypedList()
    for arr in valid_topic_list_py:
        valid_topic_list.append(arr)

    cell_idx_arr = long_data['cell_idx']
    gene_idx_arr = long_data['gene_idx']
    cell_identity_arr = long_data['cell_identity']
    z_arr = long_data['z']

    with tqdm(total=num_loops, desc="Gibbs Sampling", unit="iteration") as pbar:
        for loop in range(num_loops):
            gibbs_update_numba(cell_idx_arr, gene_idx_arr, cell_identity_arr, z_arr,
                               A, B, D,
                               valid_topic_list,
                               alpha_beta, alpha_c, G, K)

##            if (loop >= burn_in) & ((loop + 1) % 100 == 0):
            if (loop >= burn_in):

                constants_filename = os.path.join(output_dir, f'constants_sample_{loop + 1}.pkl')
                with open(constants_filename, 'wb') as f:
                    pickle.dump(constants, f)
                pbar.set_postfix_str(f"Sampling (Iteration {loop + 1})")
            else:
                pbar.set_postfix_str(f"Burn-in (Iteration {loop + 1})")

            pbar.update(1)

    end_time = time.time()
    print(f"Time taken for {total_counts * num_loops} gene count samples: {end_time - start_time:.2f} seconds")
    
def reconstruct_beta_theta(sample_dir, save_dir, gene_names, cell_ids):

    beta_accumulator = None
    theta_accumulator = None
    sample_files = sorted([f for f in os.listdir(sample_dir) if f.endswith(".pkl")])

    print(f"Found {len(sample_files)} sample files for reconstruction.")
    
    for file in sample_files:
        with open(os.path.join(sample_dir, file), 'rb') as f:
            constants_sample = pickle.load(f)
            A = constants_sample['A']
            D = constants_sample['D']
        
        if beta_accumulator is None:
            beta_accumulator = A.copy()
        else:
            beta_accumulator += A

        if theta_accumulator is None:
            theta_accumulator = D.copy()
        else:
            theta_accumulator += D

    beta_matrix = beta_accumulator / len(sample_files)
    theta_matrix = theta_accumulator / len(sample_files)

    beta_matrix /= beta_matrix.sum(axis=0)
    theta_matrix /= theta_matrix.sum(axis=1, keepdims=True)

    beta_df = pd.DataFrame(beta_matrix, index=gene_names, columns=[f"Topic_{k}" for k in range(beta_matrix.shape[1])])
    theta_df = pd.DataFrame(theta_matrix, index=cell_ids, columns=[f"Topic_{k}" for k in range(theta_matrix.shape[1])])

    os.makedirs(save_dir, exist_ok=True)

    beta_output_path = os.path.join(save_dir, "HLDA_beta.csv")
    theta_output_path = os.path.join(save_dir, "HLDA_theta.csv")
    beta_df.to_csv(beta_output_path)
    theta_df.to_csv(theta_output_path)

######################################################################################################
#### PLOTTING FUNCTIONS ##############################################################################
######################################################################################################

def cosine_similarity_pd(matrix: pd.DataFrame) -> pd.DataFrame:

    mat = matrix.values
    norms = np.sqrt((mat ** 2).sum(axis=0))
    mat_norm = mat / norms
    sim = mat_norm.T @ mat_norm
    return pd.DataFrame(sim, index=matrix.columns, columns=matrix.columns)

def plot_grouped_stacked_membership(
    theta_df,
    topic_cols=None,
    figsize=(15,5),
    sort_cells=False,
    output_path="grouped_stacked_membership.png",
    smoothing_window=10,
    palette='tab20'
):
    if topic_cols is None:
        topic_cols = list(theta_df.columns)
    n_topics = len(topic_cols)
    colors = sns.color_palette(palette, n_topics)
    color_map = dict(zip(topic_cols, colors))
    
    # Use provided order if available; otherwise, use unique index order.
##    groups = ["CD8+ Cytotoxic T", "CD34+", "T cell", "CD19+ B", "CD56+ NK", "CD14+ Monocyte"]
    groups = list(set(theta_df.index))
    
    group_dfs = []
    group_boundaries = []
    current_index = 0
    for group in groups:
        # Use .loc[...] to get the sub-dataframe for this group.
        sub_df = theta_df.loc[theta_df.index == group, topic_cols].copy()
        if sort_cells:
            max_topics = sub_df.idxmax(axis=1)
            sub_df["max_topic"] = max_topics
            sub_df.sort_values("max_topic", inplace=True)
            sub_df.drop(columns="max_topic", inplace=True)
        group_dfs.append(sub_df)
        start_idx = current_index
        end_idx = current_index + sub_df.shape[0]
        group_boundaries.append((group, start_idx, end_idx))
        current_index = end_idx
    
    all_df = pd.concat(group_dfs, axis=0)
    if smoothing_window > 1:
        all_df = all_df.rolling(window=smoothing_window, center=True, min_periods=1).mean()
    
    x = np.arange(all_df.shape[0])
    y_values = [all_df[topic].values for topic in topic_cols]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.stackplot(x, *y_values, colors=[color_map[topic] for topic in topic_cols], labels=topic_cols)
    
    # Draw a box and label for each group
    for group, start, end in group_boundaries:
        width = end - start
        rect = patches.Rectangle(
            (start, 0),     # bottom-left corner
            width,          # width
            1,              # height (assuming proportions 0-1)
            linewidth=1,
            edgecolor='black',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Place the group label above the box
        mid = (start + end) / 2
        ax.text(mid, 1.02, group, ha='center', va='bottom', fontsize=10,
                transform=ax.get_xaxis_transform(), clip_on=False)
    
    ax.set_xlim(0, all_df.shape[0])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Topic Proportion")
    ax.set_xlabel("Cells")
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_beta_cosine_heatmap(beta_df: pd.DataFrame, ax=None, title="Beta Cosine Similarity"):

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    sim_df = cosine_similarity_pd(beta_df)
    sns.heatmap(sim_df, ax=ax, cmap="viridis", annot=False, fmt=".2f", square=True)
    ax.set_title(title)
    return ax

def compute_log_likelihood(theta, beta, count_matrix, epsilon=1e-12):

    C, K = theta.shape
    K2, G = beta.shape
    assert K == K2, f"Mismatch: theta has {K} topics, but beta has {K2} topics."
    
    C2, G2 = count_matrix.shape
    assert C == C2 and G == G2, f"Mismatch in doc_word_counts shape. Expected ({D},{V}), got ({D2},{V2})."

    log_likelihood = 0.0

    for c in range(C):
        for g in range(G):
            count = count_matrix[c, g]
            if count > 0:
                p_count = np.dot(theta[c, :], beta[:, g])
                log_likelihood += count * np.log(np.maximum(p_count, epsilon))
    
    return log_likelihood

def create_elbow_plot(folder_dir, data_dir):

    data_path = os.path.join(data_dir, 'count_matrix.csv')
    count_matrix = pd.read_csv(data_path, index_col=0).values

    folders = [os.path.join(folder_dir, d) for d in os.listdir(folder_dir) if re.match(r'pbmc_sample_\d+_fc', d)]
    results = []
    
    for folder in folders:
        
        match = re.search(r'pbmc_sample_(\d+)_fc', folder)
        
        if match:
            fc_num = int(match.group(1))
            
            beta_path = os.path.join(folder, 'HLDA_beta.csv')
            theta_path = os.path.join(folder, 'HLDA_theta.csv')
            
            if os.path.exists(beta_path) and os.path.exists(theta_path):
                
                beta = pd.read_csv(beta_path, index_col=0).values.T
                theta = pd.read_csv(theta_path, index_col=0).values
                
                log_likelihood = compute_log_likelihood(theta, beta, count_matrix)
                
                results.append((fc_num, log_likelihood))
    
    results.sort(key=lambda x: x[0])
    
    x_vals = [r[0] for r in results]
    y_vals = [r[1] for r in results]
    
    # Plot the elbow curve
    plt.plot(x_vals, y_vals, marker='o')
    plt.title('Elbow Plot of Log-Likelihood')
    plt.xlabel('Number of Fully Connected Nodes')
    plt.ylabel('Log-Likelihood')
    plt.grid(True)

    output_path = os.path.join(folder_dir, 'elbow_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  
    plt.close()

##def match_topics(true_mat, est_mat):
##
##    n_topics = true_mat.shape[1]
##    cost_matrix = np.zeros((n_topics, n_topics))
##    for i in range(n_topics):
##        for j in range(n_topics):
##            corr = np.corrcoef(true_mat[:, i], est_mat[:, j])[0, 1]
##            cost_matrix[i, j] = -corr 
##    row_ind, col_ind = linear_sum_assignment(cost_matrix)
##    return col_ind

def match_topics_greedy(true_mat, est_mat):
    n_topics = true_mat.shape[1]
    corr_matrix = np.empty((n_topics, n_topics))
    for i in range(n_topics):
        for j in range(n_topics):
            corr_matrix[i, j] = np.corrcoef(true_mat[:, i], est_mat[:, j])[0, 1]
    corr_copy = corr_matrix.copy()
    mapping = np.empty(n_topics, dtype=int)
    for _ in range(n_topics):
        i, j = np.unravel_index(np.argmax(corr_copy), corr_copy.shape)
        mapping[i] = j
        corr_copy[i, :] = -np.inf
        corr_copy[:, j] = -np.inf
    return mapping

def plot_est_vs_true_theta_by_model(true_theta, model_estimated_theta_dict, output_file):

    if hasattr(true_theta, 'values'):
        true_theta = true_theta.values

    n_topics = true_theta.shape[1]
    n_models = len(model_estimated_theta_dict)
    
    fig, axs = plt.subplots(n_topics, n_models, figsize=(5 * n_models, 4 * n_topics), squeeze=False)
    
    for col_idx, (model_name, est_theta) in enumerate(model_estimated_theta_dict.items()):
        
        if hasattr(est_theta, 'values'):
            est_theta = est_theta.values
            
        perm = match_topics_greedy(true_theta, est_theta)
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

def plot_matched_heatmaps(model_beta_dict, formatted_gene_means, plot_folder, prefix):
    fig_cos, axes_cos = plt.subplots(1, 3, figsize=(20, 6))
    fig_corr, axes_corr = plt.subplots(1, 3, figsize=(20, 6))

    for ax_cos, ax_corr, (model_name, beta_df) in zip(axes_cos, axes_corr, model_beta_dict.items()):
        topic_matching, matched_cos_sims = compute_topic_cosine_similarity(beta_df, formatted_gene_means)
        gene_topics = list(formatted_gene_means.columns)
        beta_topics = list(beta_df.columns)
        n_rows, n_cols = len(gene_topics), len(beta_topics)

        cos_full = np.zeros((n_rows, n_cols))
        corr_full = np.zeros((n_rows, n_cols))

        for i, g_topic in enumerate(gene_topics):
            b_topic = topic_matching[g_topic]
            j = beta_topics.index(b_topic)
            val_cos = matched_cos_sims[g_topic]
            cos_full[i, j] = val_cos
            v1 = formatted_gene_means[g_topic].values
            v2 = beta_df[b_topic].values
            val_corr = np.corrcoef(v1, v2)[0, 1]
            corr_full[i, j] = val_corr

        im_cos = ax_cos.imshow(cos_full, aspect='auto', cmap='RdBu_r', norm=Normalize(vmin=-1, vmax=1))
        ax_cos.set_title(f"{model_name} - Cosine (Matched)")
        ax_cos.set_xlabel("Beta Topics")
        ax_cos.set_ylabel("Gene Mean Topics")
        ax_cos.set_xticks(np.arange(n_cols))
        ax_cos.set_yticks(np.arange(n_rows))
        ax_cos.set_xticklabels(beta_topics, rotation=45, ha="right")
        ax_cos.set_yticklabels(gene_topics)
        for i in range(n_rows):
            for j in range(n_cols):
                val = cos_full[i, j]
                if val != 0:
                    c = "white" if val > 0.5 else "black"
                    ax_cos.text(j, i, f"{val:.2f}", ha="center", va="center", color=c, fontsize=8)
        fig_cos.colorbar(im_cos, ax=ax_cos, fraction=0.046, pad=0.04)

        im_corr = ax_corr.imshow(corr_full, aspect='auto', cmap='RdBu_r', norm=Normalize(vmin=-1, vmax=1))
        ax_corr.set_title(f"{model_name} - Corr (Matched)")
        ax_corr.set_xlabel("Beta Topics")
        ax_corr.set_ylabel("Gene Mean Topics")
        ax_corr.set_xticks(np.arange(n_cols))
        ax_corr.set_yticks(np.arange(n_rows))
        ax_corr.set_xticklabels(beta_topics, rotation=45, ha="right")
        ax_corr.set_yticklabels(gene_topics)
        for i in range(n_rows):
            for j in range(n_cols):
                val = corr_full[i, j]
                if val != 0:
                    c = "white" if abs(val) > 0.5 else "black"
                    ax_corr.text(j, i, f"{val:.2f}", ha="center", va="center", color=c, fontsize=8)
        fig_corr.colorbar(im_corr, ax=ax_corr, fraction=0.046, pad=0.04)

    fig_cos.tight_layout()
    fig_cos.savefig(os.path.join(plot_folder, f"{prefix}_cosine_matched_heatmap.png"),
                    dpi=300, bbox_inches='tight')
    plt.close(fig_cos)
    
    fig_corr.tight_layout()
    fig_corr.savefig(os.path.join(plot_folder, f"{prefix}_corr_matched_heatmap.png"),
                     dpi=300, bbox_inches='tight')
    plt.close(fig_corr)


######################################################################################################
#### PIPELINE AND LOOP FUNCTIONS #####################################################################
######################################################################################################

def run_lda_pipeline(n_fc_nodes, input_folder, input_file, output_folder):

    counts_df = pd.read_csv(os.path.join(input_folder, input_file), index_col=0)
    cell_ids = counts_df.index.to_list()
    num_topics = len(set(cell_ids)) + n_fc_nodes

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

    
    lda_beta_out = os.path.join(output_folder, "LDA_beta.csv")
    beta_lda_df.T.to_csv(lda_beta_out)
    
    lda_theta_out = os.path.join(output_folder, "LDA_theta.csv")
    theta_lda_df.to_csv(lda_theta_out)

    subprocess.run(["Rscript", "fastTopic_fit.R"], check=True)

def run_plots(estimate_folder, plot_folder):

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder, exist_ok=True)
        
    theta_path = os.path.join(estimate_folder, 'HLDA_theta.csv')
    theta_df = pd.read_csv(theta_path, index_col=0)
    
    plot_grouped_stacked_membership(
                 theta_df=theta_df,
                 sort_cells=True,
                 output_path=os.path.join(plot_folder, f"stacked_theta_plot.png")
             )

def run_gibbs_pipeline(topic_hierarchy, K, num_loops, burn_in, hyperparams, sample_dir, est_dir, input_file):

    count_matrix = pd.read_csv(input_file, index_col=0)

    cell_identities = count_matrix.index.tolist()
    gene_names = count_matrix.columns.tolist()

    print('Initializing long data')
    long_data, identity_mapping = initialize_long_data(count_matrix,
                                                       cell_identities,
                                                       topic_hierarchy,
                                                       est_dir)

    constants = compute_constants(long_data, K)
    
    run_gibbs_sampling_numba(topic_hierarchy, long_data, constants, hyperparams,
                       num_loops, burn_in, sample_dir, identity_mapping)

    reconstruct_beta_theta(sample_dir, est_dir, gene_names, cell_identities)

    
def run_gibbs_loop(min_, max_):

    data_dir = '../data/pbmc/zheng/'
    X_path = os.path.join(data_dir, 'count_matrix.csv')

    num_loops=500
    burn_in=200
    hyperparams = {'alpha_beta': 1, 'alpha_c': 1}

    for i in range(min_, max_):

        print(f"Running sampling for N FC Nodes = {i}")

        sample_dir = f"../samples/zheng_{i}_fc/"
        est_dir = f'../estimates/zheng_{i}_fc/'
        plot_dir = os.path.join(est_dir, '_plots/')

        run_gibbs_pipeline(i, num_loops, burn_in, hyperparams, sample_dir, est_dir, X_path)
####        run_lda_pipeline(i, data_dir, 'counts_filt_highvar.csv', est_dir)
        run_plots(est_dir, plot_dir)


##    create_elbow_plot('../estimates/', data_dir)




















