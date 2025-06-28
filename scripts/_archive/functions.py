
import pandas as pd
import scanpy as sc
import numpy as np
import os
import re
from scipy.sparse import issparse
import pickle
from tqdm import tqdm
import time
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import umap.umap_ as umap
from numba import njit
from numba.typed import List as TypedList

def clear_directory(dir_path):
    """
    Remove all files and subdirectories in 'dir_path', 
    but keep the directory itself. If 'dir_path' does not
    exist, create it.
    """
    if not os.path.exists(dir_path):
        # If the folder doesn't exist, just create it
        os.makedirs(dir_path)
        return
    
    # If the folder exists, remove everything inside it
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)           # remove file or link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)      # remove directory recursively
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def compute_perplexity(E, theta, beta):
    """
    Compute perplexity from a document-word count matrix E,
    document-topic estimates theta (n_docs x n_topics) and
    topic-word estimates beta (n_topics x n_words).

    This function computes the log likelihood as:
    
        LL = sum_{d,w} E[d,w] * log( sum_{k} theta[d,k] * beta[k,w] )

    and then returns perplexity = exp(-LL / total_words).

    A small constant is added inside the log to avoid log(0).
    """
    # Compute the reconstruction: (n_docs x n_words)
    reconstruction = np.dot(theta, beta)
    # Add a small value to avoid log(0)
    log_likelihood = np.sum(E * np.log(reconstruction + 1e-12))
    total_words = np.sum(E)
    perplexity = np.exp(-log_likelihood / total_words)
    return perplexity

def fit_and_save_lda(E, out_path, n_topics, max_iter, random_state):

    os.makedirs(out_path, exist_ok=True)
    lda = LatentDirichletAllocation(n_components=n_topics,
                                    max_iter=max_iter,
                                    random_state=random_state)
    lda.fit(E)
    theta = lda.transform(E)  # shape: (n_docs, n_topics)
    beta = lda.components_
    beta = beta / beta.sum(axis=1, keepdims=True)
    
    theta_df = pd.DataFrame(theta, columns=[f"Topic_{i+1}" for i in range(n_topics)])
    theta_df.to_csv(os.path.join(out_path, 'lda_theta.csv'))
    
    beta_df = pd.DataFrame(beta, index=[f"Topic_{i+1}" for i in range(n_topics)])
    beta_df.to_csv(os.path.join(out_path, 'lda_beta.csv'))
    
    return theta, beta, lda

def fit_and_save_nmf(E, out_path, n_topics, max_iter, random_state):

    os.makedirs(out_path, exist_ok=True)
    nmf_model = NMF(n_components=n_topics, max_iter=max_iter, random_state=random_state)
    # W: document-topic matrix, H: topic-word matrix
    W = nmf_model.fit_transform(E)
    H = nmf_model.components_

    theta = W / (W.sum(axis=1, keepdims=True) + 1e-12)
    beta = H / (H.sum(axis=1, keepdims=True) + 1e-12)

    theta_df = pd.DataFrame(theta, columns=[f"Topic_{i+1}" for i in range(n_topics)])
    theta_df.to_csv(os.path.join(out_path, 'nmf_theta.csv'))

    beta_df = pd.DataFrame(beta, index=[f"Topic_{i+1}" for i in range(n_topics)])
    beta_df.to_csv(os.path.join(out_path, 'nmf_beta.csv'))

    return theta, beta, nmf_model

def sample_pbmc_by_group(adata, n_per_group):

    sampled_indices = []
    
    for cell_type, group in adata.obs.groupby("cell_type"):    

        sample_size = min(n_per_group, len(group))
        sampled_group = group.sample(n=sample_size, replace=False)
        
        sampled_indices.extend(sampled_group.index)

        sampled_indices = [idx for idx in sampled_indices if idx in adata.obs_names]

    sampled_indices = [idx for idx in sampled_indices if idx in adata.obs_names]
    adata_sampled = adata[sampled_indices].copy()

    return adata_sampled

def randomly_sample_pbmc(adata, n):

    total_cells = adata.shape[0]
    if n >= total_cells:
        print("Requested sample size exceeds or equals the total number of cells. Returning all cells.")
        return adata.copy()

    sampled_indices = np.random.choice(adata.obs_names, size=n, replace=False)
    adata_sampled = adata[sampled_indices].copy()
    
    return adata_sampled

def process_split_sampled_data(input_file, output_file_base, test_split):
    
    adata = sc.read_h5ad(input_file)
    print("Original data shape:", adata.shape)
    
    sc.pp.filter_genes(adata, min_counts=10)
    print("Shape after gene filtering (removing genes with no counts):", adata.shape)
    
    adata_raw = adata.copy()
    
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000)
    top_genes = adata.var_names[adata.var["highly_variable"]]
    print("Number of highly variable genes selected:", len(top_genes))
    
    adata_raw = adata_raw[:, adata.var['highly_variable']]
    sc.pp.filter_cells(adata_raw, min_genes=1)
    print("Shape after subsetting raw data to highly variable genes and filtering cells:", adata_raw.shape)

    if issparse(adata_raw.X):
        print("Converting adata.X from sparse to dense format...")
        dense_matrix = adata_raw.X.toarray().astype(int)
    else:
        print("adata.X is already dense.")
        dense_matrix = adata_raw.X.astype(int)

    print("Dense matrix shape:", dense_matrix.shape)

    dense_df = pd.DataFrame(
        dense_matrix, 
        index=adata_raw.obs['cell_type'], 
        columns=adata_raw.var_names
    )
    train_dfs = []
    test_dfs = []

    for cell_type, group in dense_df.groupby(level=0):
        train_group, test_group = train_test_split(group, test_size=test_split, random_state=42)
        train_dfs.append(train_group)
        test_dfs.append(test_group)
    
    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)

    train_output_file = output_file_base + "_train.csv"
    test_output_file = output_file_base + "_test.csv"

    train_df.to_csv(train_output_file)
    test_df.to_csv(test_output_file)

    print(f"Train data saved to: {train_output_file}")
    print(f"Test data saved to: {test_output_file}")


def get_topic_hierarchy(output_path=None):

##    topic_hierarchy = {
##        7:  [0, 1, 3, 7],
##        8:  [0, 1, 3, 8],
##        9:  [0, 1, 4, 9],
##        10: [0, 1, 4, 10],
##        11: [0, 2, 5, 11],
##        12: [0, 2, 5, 12],
##        13: [0, 2, 6, 13],
##        14: [0, 2, 6, 14],
##    }

##    topic_hierarchy = {
##        2: [2,0,1],
##        3: [3,0,1],
##        4: [4,0,1]
##    }

##    topic_hierarchy = {
##        2: [2,0,1,5],
##        3: [3,0,1,5],
##        4: [4,0,1,5]
##    }

##    topic_hierarchy = {
##        2: [0],
##        3: [1],
##        4: [2]
##    }

##    topic_hierarchy = {
##        2: [0,1],
##        3: [0,2],
##        4: [0,3]
##    }
    
##    topic_hierarchy = {
##        # -- Dendritic cells (DC) -----------------------------------------
##        "CD1c-positive myeloid dendritic cell":    [0, 1, 2],
##        "CD141-positive myeloid dendritic cell":   [0, 1, 3],
##        "plasmacytoid dendritic cell":             [0, 1, 4],
##        
##        # -- Lymphocytes --------------------------------------------------
##        # B-cell subsets
##        "B cell":                                  [0, 5, 6, 7],
##        "naive B cell":                            [0, 5, 6, 8],
##        "memory B cell":                           [0, 5, 6, 9],
##        "plasmablast":                             [0, 5, 6, 10],
##        
##        # T-cell subsets
##        #   (CD4 trunk)
##        "CD4-positive, alpha-beta T cell":         [0, 5, 11, 12, 13],
##        "central memory CD4-positive, alpha-beta T cell": [0, 5, 11, 12, 14],
##        "effector memory CD4-positive, alpha-beta T cell": [0, 5, 11, 12, 15],
##        "naive thymus-derived CD4-positive, alpha-beta T cell": [0, 5, 11, 12, 33],
##        
##        #   (CD8 trunk)
##        "naive thymus-derived CD8-positive, alpha-beta T cell": [0, 5, 11, 16, 17],
##        "central memory CD8-positive, alpha-beta T cell":       [0, 5, 11, 16, 18],
##        "effector memory CD8-positive, alpha-beta T cell":      [0, 5, 11, 16, 19],
##        
##        #   (Other T subsets)
##        "double negative thymocyte":              [0, 5, 11, 20],
##        "gamma-delta T cell":                     [0, 5, 11, 21],
##        "mucosal invariant T cell":               [0, 5, 11, 22],
##        "regulatory T cell":                      [0, 5, 11, 23],
##        
##        # NK / ILC
##        "natural killer cell":                    [0, 5, 24, 25],
##        "CD16-negative, CD56-bright natural killer cell, human": [0, 5, 24, 26],
##        "innate lymphoid cell":                   [0, 5, 24, 27],
##        
##        # -- Monocytes ----------------------------------------------------
##        "CD14-positive monocyte":                  [0, 28, 29],
##        "CD14-low, CD16-positive monocyte":        [0, 28, 30],
##        
##        # -- Single-leaf lineages -----------------------------------------
##        "platelet":                                [0, 31],
##        "hematopoietic precursor cell":            [0, 32],
##    }

    topic_hierarchy = {
        # -- Dendritic cells (DC) -----------------------------------------
        "CD1c-positive myeloid dendritic cell":    [0],
        "CD141-positive myeloid dendritic cell":   [1],
        "plasmacytoid dendritic cell":             [2],
        
        # -- Lymphocytes --------------------------------------------------
        # B-cell subsets
        "B cell":                                  [3],
        "naive B cell":                            [4],
        "memory B cell":                           [5],
        "plasmablast":                             [6],
        
        # T-cell subsets
        #   (CD4 trunk)
        "CD4-positive, alpha-beta T cell":         [7],
        "central memory CD4-positive, alpha-beta T cell": [8],
        "effector memory CD4-positive, alpha-beta T cell": [9],
        "naive thymus-derived CD4-positive, alpha-beta T cell": [10],
        
        #   (CD8 trunk)
        "naive thymus-derived CD8-positive, alpha-beta T cell": [11],
        "central memory CD8-positive, alpha-beta T cell":       [12],
        "effector memory CD8-positive, alpha-beta T cell":      [13],
        
        #   (Other T subsets)
        "double negative thymocyte":              [14],
        "gamma-delta T cell":                     [15],
        "mucosal invariant T cell":               [16],
        "regulatory T cell":                      [17],
        
        # NK / ILC
        "natural killer cell":                    [18],
        "CD16-negative, CD56-bright natural killer cell, human": [19],
        "innate lymphoid cell":                   [20],
        
        # -- Monocytes ----------------------------------------------------
        "CD14-positive monocyte":                  [21],
        "CD14-low, CD16-positive monocyte":        [22],
        
        # -- Single-leaf lineages -----------------------------------------
        "platelet":                                [23],
        "hematopoietic precursor cell":            [24]
    }

    topic_hierarchy = { key: value + [25,26,27,28,29,30,31,32,33] for key, value in topic_hierarchy.items() }
    
    # Optionally convert to a DataFrame for easier viewing or saving
    df = pd.DataFrame({
        "Cell_Type": list(topic_hierarchy.keys()),
        "Topic_Path": list(topic_hierarchy.values())
    })
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Topic hierarchy saved to: {output_path}")
    
    return topic_hierarchy

def map_topics_to_nodes():

##    mapping = {
##    "Root": [0],
##    "Dendritic cells (DC)": [1],
##    "CD1c-positive myeloid dendritic cell": [2],
##    "CD141-positive myeloid dendritic cell": [3],
##    "plasmacytoid dendritic cell": [4],
##    "Lymphocytes": [5],
##    "B-cell trunk": [6],
##    "B cell": [7],
##    "naive B cell": [8],
##    "memory B cell": [9],
##    "plasmablast": [10],
##    "T-cell trunk": [11],
##    "CD4 trunk": [12],
##    "CD4-positive, alpha-beta T cell": [13],
##    "central memory CD4-positive, alpha-beta T cell": [14],
##    "effector memory CD4-positive, alpha-beta T cell": [15],
##    "naive thymus-derived CD4-positive, alpha-beta T cell": [33],
##    "CD8 trunk": [16],
##    "naive thymus-derived CD8-positive, alpha-beta T cell": [17],
##    "central memory CD8-positive, alpha-beta T cell": [18],
##    "effector memory CD8-positive, alpha-beta T cell": [19],
##    "double negative thymocyte": [20],
##    "gamma-delta T cell": [21],
##    "mucosal invariant T cell": [22],
##    "regulatory T cell": [23],
##    "NK_ILC": [24],
##    "natural killer cell": [25],
##    "CD16-negative, CD56-bright natural killer cell, human": [26],
##    "innate lymphoid cell": [27],
##    "Monocytes": [28],
##    "CD14-positive monocyte": [29],
##    "CD14-low, CD16-positive monocyte": [30],
##    "platelet": [31],
##    "hematopoietic precursor cell": [32],
##    }

    mapping = {
        # -- Dendritic cells (DC) -----------------------------------------
        "CD1c-positive myeloid dendritic cell":    [0],
        "CD141-positive myeloid dendritic cell":   [1],
        "plasmacytoid dendritic cell":             [2],
        
        # -- Lymphocytes --------------------------------------------------
        # B-cell subsets
        "B cell":                                  [3],
        "naive B cell":                            [4],
        "memory B cell":                           [5],
        "plasmablast":                             [6],
        
        # T-cell subsets
        #   (CD4 trunk)
        "CD4-positive, alpha-beta T cell":         [7],
        "central memory CD4-positive, alpha-beta T cell": [8],
        "effector memory CD4-positive, alpha-beta T cell": [9],
        "naive thymus-derived CD4-positive, alpha-beta T cell": [10],
        
        #   (CD8 trunk)
        "naive thymus-derived CD8-positive, alpha-beta T cell": [11],
        "central memory CD8-positive, alpha-beta T cell":       [12],
        "effector memory CD8-positive, alpha-beta T cell":      [13],
        
        #   (Other T subsets)
        "double negative thymocyte":              [14],
        "gamma-delta T cell":                     [15],
        "mucosal invariant T cell":               [16],
        "regulatory T cell":                      [17],
        
        # NK / ILC
        "natural killer cell":                    [18],
        "CD16-negative, CD56-bright natural killer cell, human": [19],
        "innate lymphoid cell":                   [20],
        
        # -- Monocytes ----------------------------------------------------
        "CD14-positive monocyte":                  [21],
        "CD14-low, CD16-positive monocyte":        [22],
        
        # -- Single-leaf lineages -----------------------------------------
        "platelet":                                [23],
        "hematopoietic precursor cell":            [24],
        "fc 1": [25],
        "fc 2": [26],
        "fc 3": [27],
        "fc 4": [28],
        "fc 5": [29],
        "fc 6": [30],
        "fc 7": [31],
        "fc 8": [32],
        "fc 9": [33]
    }

    return mapping

def create_hierarchy_dict(topic_hierarchy):
    
    hierarchy_dict = {}
    # Iterate over all paths
    for cell_type, path in topic_hierarchy.items():
        for i in range(len(path) - 1):
            parent = path[i]
            child = path[i + 1]
            # Add child to parent's set of children
            if parent not in hierarchy_dict:
                hierarchy_dict[parent] = set()
            hierarchy_dict[parent].add(child)
    # Ensure that nodes with no children are included
    all_nodes = set()
    for path in topic_hierarchy.values():
        all_nodes.update(path)
    for node in all_nodes:
        if node not in hierarchy_dict:
            hierarchy_dict[node] = set()
    # Convert sets to sorted lists
    hierarchy_dict = {node: sorted(list(children)) for node, children in hierarchy_dict.items()}
    return hierarchy_dict

@njit
def gibbs_update_numba(cell_idx_arr, gene_idx_arr, cell_identity_arr, z_arr,
                       A, B, D,
                       valid_topic_list,  # typed list of np.array (int32)
                       alpha_beta, alpha_c, G, K):
    """
    Perform one iteration over all gene counts, updating topics in place.
    """
    total_counts = cell_idx_arr.shape[0]
    G_alpha_beta = G * alpha_beta

    for i in range(total_counts):
        c = cell_idx_arr[i]
        g = gene_idx_arr[i]
        ident = cell_identity_arr[i]
        current_z = z_arr[i]

        # Remove current assignment
        A[g, current_z] -= 1
        B[current_z] -= 1
        D[c, current_z] -= 1

        # Retrieve valid topics for this cell identity from typed list
        valid_topics = valid_topic_list[ident]
        n_valid = valid_topics.shape[0]

        # Compute probabilities
        probs = np.empty(n_valid, dtype=np.float64)
        sum_probs = 0.0
        for j in range(n_valid):
            t = valid_topics[j]
            prob = ((alpha_beta + A[g, t]) / (G_alpha_beta + B[t])) * (alpha_c + D[c, t])
            probs[j] = prob
            sum_probs += prob

        # Normalize
        for j in range(n_valid):
            probs[j] /= sum_probs

        # Sample
        r = np.random.rand()
        cumulative = 0.0
        new_z = valid_topics[-1]
        for j in range(n_valid):
            cumulative += probs[j]
            if r < cumulative:
                new_z = valid_topics[j]
                break

        # Update with new assignment
        A[g, new_z] += 1
        B[new_z] += 1
        D[c, new_z] += 1
        z_arr[i] = new_z

def run_gibbs_sampling_numba(long_data, constants, hyperparams, num_loops, burn_in, output_dir, identity_mapping):
    """
    Run Gibbs sampling for gene count topics with hierarchy constraints, using Numba for the inner loop.
    """
    start_time = time.time()
    clear_directory(output_dir)

    # Unpack constants
    A = constants['A']
    B = constants['B']
    D = constants['D']
    G = constants['G']
    K = constants['K']

    # Retrieve or build your topic hierarchy (dict: identity -> list of valid topics)
    topic_hierarchy = get_topic_hierarchy()

    # Unpack hyperparameters
    alpha_beta = hyperparams['alpha_beta']
    alpha_c = hyperparams['alpha_c']

    total_counts = len(long_data)

    # Convert the "topic_hierarchy" dict into a typed list of np.array for Numba
    # 1) For each integer identity i, find the valid topics in Python
    # 2) Convert them to a NumPy array of int32
    # 3) Append them to a Numba typed list
    from numba.typed import List as TypedList
    valid_topic_list_py = []
    num_identities = len(identity_mapping)
    # Build the reverse mapping from int->identity
    # identity_mapping is int->string, so let's confirm that is the case
    for i in range(num_identities):
        cell_identity_str = identity_mapping[i]  # e.g. "B cell"
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
            # Call the Numba-compiled update
            gibbs_update_numba(cell_idx_arr, gene_idx_arr, cell_identity_arr, z_arr,
                               A, B, D,
                               valid_topic_list,
                               alpha_beta, alpha_c, G, K)

            # Save constants after burn-in
            if loop >= burn_in:
                constants_filename = os.path.join(output_dir, f'constants_sample_{loop + 1}.pkl')
                with open(constants_filename, 'wb') as f:
                    pickle.dump(constants, f)
                pbar.set_postfix_str(f"Sampling (Iteration {loop + 1})")
            else:
                pbar.set_postfix_str(f"Burn-in (Iteration {loop + 1})")

            pbar.update(1)

    end_time = time.time()
    print(f"Time taken for {total_counts * num_loops} gene count samples: {end_time - start_time:.2f} seconds")
    

def initialize_long_data(count_matrix, cell_identities, num_topics, long_data_csv_path):

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

    topic_hierarchy=get_topic_hierarchy()
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
    long_data_df.to_csv(os.path.join(long_data_csv_path, 'long_data.csv'), index=False)
    pd.DataFrame.from_dict(int_to_identity, orient="index").to_csv(os.path.join(long_data_csv_path, 'topic_int_mapping.csv'))

    return long_data, int_to_identity

def compute_constants(long_data, K):

    cell_idx = long_data['cell_idx']
    gene_idx = long_data['gene_idx']
    z = long_data['z']

    unique_genes = np.unique(gene_idx)
    unique_cells = np.unique(cell_idx)

    n_expected_genes = np.max(gene_idx) + 1  # +1 because indices are 0-based
    all_genes = np.arange(n_expected_genes)

    missing_genes = np.setdiff1d(all_genes, unique_genes)
    print("Missing gene indices (columns with no entries):", missing_genes)

    print("Computing A...")
    # A[g, k]: Number of times gene g is sampled from topic k across all cells
    gene_topic_pairs = gene_idx * K + z
    counts = np.bincount(gene_topic_pairs, minlength=n_expected_genes * K)
    A = counts.reshape(n_expected_genes, K)
    print("A matrix computed successfully.")

    assert np.sum(A) == len(long_data), "A computation failed!"

    print("Computing B...")
    # B[k]: Number of times topic k is sampled across all cells
    B = np.bincount(z, minlength=K)
    print("B vector computed successfully.")

    assert np.sum(B) == len(long_data), "B computation failed!"

    print("Computing D...")
    # D[c, k]: Number of times topic k is sampled in cell c
    cell_topic_pairs = cell_idx * K + z
    counts = np.bincount(cell_topic_pairs, minlength=len(unique_cells) * K)
    D = counts.reshape(len(unique_cells), K)
    print("D matrix computed successfully.")

    assert np.sum(D) == len(long_data), "D computation failed!"

    # Return the computed constants
    constants = {
        'A': A,
        'B': B,
        'D': D,
        'unique_genes': unique_genes,
        'unique_cells': unique_cells,
        'unique_topics': np.arange(K),
        'G': len(unique_genes),
        'K': K,
        'C': len(unique_cells)
    }
    
    return constants

def run_gibbs_sampling(long_data, constants, hyperparams, num_loops, burn_in, output_dir, identity_mapping):

    start_time = time.time()
    clear_directory(output_dir)

    # Unpack constants
    A = constants['A']
    B = constants['B']
    D = constants['D']
    G = constants['G']
    K = constants['K']

    topic_hierarchy = get_topic_hierarchy()

    # Unpack hyperparameters
    alpha_beta = hyperparams['alpha_beta']  # Dirichlet prior for beta
    alpha_c = hyperparams['alpha_c']       # Dirichlet prior for topic distribution

    total_counts = len(long_data)

    with tqdm(total=num_loops, desc="Gibbs Sampling", unit="iteration") as pbar:
        for loop in range(num_loops):
            for idx in range(total_counts):
                # Extract current gene count information
                cell_idx = long_data['cell_idx'][idx]
                gene = long_data['gene_idx'][idx]
                cell_identity = long_data['cell_identity'][idx]
                current_z = long_data['z'][idx]

                # Remove current assignment
                A[gene, current_z] -= 1
                B[current_z] -= 1
                D[cell_idx, current_z] -= 1

                # Get valid topics for this cell identity
                valid_topics = topic_hierarchy[identity_mapping[cell_identity]]

                # Compute probabilities for valid topics
                topic_probs = []
                for topic in valid_topics:
                    prob = (
                        ((alpha_beta + A[gene, topic]) / (G * alpha_beta + B[topic])) *
                        (alpha_c + D[cell_idx, topic])
                    )
                    topic_probs.append(prob)

                # Normalize probabilities
                topic_probs = np.array(topic_probs)
                topic_probs /= topic_probs.sum()

                # Sample a new topic
                new_z = np.random.choice(valid_topics, p=topic_probs)

                # Update counts
                A[gene, new_z] += 1
                B[new_z] += 1
                D[cell_idx, new_z] += 1

                # Update long_data
                long_data['z'][idx] = new_z

            # Save constants after burn-in
            if loop >= burn_in:
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


def compute_dot_product_scores(adata, beta_matrix, topic_mapping, prefix="dot_product_"):

    # Convert adata.X to dense if needed:
    X_data = adata.X
    if hasattr(X_data, "toarray"):
        X_data = X_data.toarray()

    n_cells = X_data.shape[0]
    n_genes = X_data.shape[1]
    n_topics = beta_matrix.shape[1]

    # Just an example: if topic_mapping is a dict {topic_name: [column_index]}...
    # or if it's {topic_name: column_index}, adapt as needed.
    for topic_name, column_indices in topic_mapping.items():
        # If your mapping stores a list like [7], take the first item:
        col_idx = column_indices[0]
        # Dot product for each cell:
        # shape => (n_cells,)
        dot_scores = X_data.dot(beta_matrix[:, col_idx])
        # Store in obs
        adata.obs[f"{prefix}{topic_name}"] = dot_scores

    return adata

def plot_avg_theta_by_celltype(theta_csv_path, output_folder, title="Average Theta by Cell Type"):

    theta_df = pd.read_csv(theta_csv_path, index_col=0)
    avg_theta = theta_df.groupby(theta_df.index).mean()

    out_path = os.path.join(output_folder, 'avg_theta_heatmap.png')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_theta, annot=False, fmt=".f", cmap="viridis")
    plt.title(title)
    plt.xlabel("Topic")
    plt.ylabel("Cell Type")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_avg_theta_for_selected_topics(theta_csv_path, selected_topics, output_folder, title="Average Theta (Selected Topics)"):

    out_path = os.path.join(output_folder, 'avg_theta_heatmap_fc.png')

    theta_df = pd.read_csv(theta_csv_path, index_col=0)
    avg_theta = theta_df.groupby(theta_df.index).mean()

    valid_topics = [t for t in selected_topics if t in theta_df.columns]
    if not valid_topics:
        raise ValueError("None of the selected topics exist in the DataFrame columns.")
    
    avg_theta_subset = avg_theta[valid_topics]

    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_theta_subset, annot=True, fmt=".1f", cmap="viridis")
    plt.title(title)
    plt.xlabel("Topic")
    plt.ylabel("Cell Type")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_topic_cosine_similarity(input_csv, output_folder):
    
    df = pd.read_csv(input_csv, index_col=0)
    os.makedirs(output_folder, exist_ok=True)
    
    sim = cosine_similarity(df.T.values)
    sim_df = pd.DataFrame(sim, index=df.columns, columns=df.columns)
    
    out_png = os.path.join(output_folder, "topic_cosine_similarity_heatmap.png")
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_df, annot=False, fmt=".2f", cmap="viridis")
    plt.title("Topic Cosine Similarity")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def run_umap_and_save_plots(adata, output_folder, dot_product_prefix="dot_product_"):

    os.makedirs(output_folder, exist_ok=True)
    sc.settings.figdir = output_folder

    if "X_umap" not in adata.obsm:
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)

    # Save a reference UMAP plot colored by "cell_type".
    sc.pl.umap(adata, color="cell_type", save="_hierarchy_key.png", show=False)

    dot_cols = [col for col in adata.obs.columns if col.startswith(dot_product_prefix)]
    for col in dot_cols:
        sc.pl.umap(adata, color=col, save=f"_{col}.png", show=False)

    return adata


def collect_descendants(node, node_hierarchy):
    """
    Recursively collect descendants of a node. Leaf nodes include themselves as descendants.

    Parameters:
        node (str): The current node to process.
        node_hierarchy (dict): The hierarchy of nodes.

    Returns:
        set: A set of all descendants for the node, including itself if it's a leaf.
    """
    if not node_hierarchy[node]:
        return {node}

    # Otherwise, collect all descendants recursively
    descendants = set()
    for child in node_hierarchy[node]:
        child_descendants = collect_descendants(child, node_hierarchy)
        descendants.update(child_descendants)
    return descendants

def create_node_descendants():
    """
    Create a mapping of each node ID to its descendant node IDs, based on the new topic hierarchy.
    The new topic hierarchy (from get_topic_hierarchy) provides cell-type (leaf) paths, from which
    we reconstruct the entire tree. Then, for each internal node, we compute all descendant leaves.
    
    Returns:
        dict: A dictionary mapping each node ID (int) to a list of descendant node IDs.
    """
    # Get the mapping from cell types (leaves) to their topic paths.
    topic_paths = get_topic_hierarchy()
    
    # Build the tree: for each topic path, add the parent-child relationships.
    tree = {}
    for cell_type, path in topic_paths.items():
        for i in range(len(path)):
            node = path[i]
            if node not in tree:
                tree[node] = set()
            # If there is a child in the path, add it to the parent's set.
            if i < len(path) - 1:
                child = path[i + 1]
                tree[node].add(child)
    
    # Convert the sets of children to lists for consistency.
    node_hierarchy = {node: list(children) for node, children in tree.items()}
    
    # Now, for each node in the tree, compute its descendants.
    descendant_mapping = {}
    for node in node_hierarchy:
        descendant_mapping[node] = list(collect_descendants(node, node_hierarchy))
    
    return descendant_mapping



def reverse_mapping(mapping):
    reversed_mapping = {}
    for key, values in mapping.items():
        for value in values:
            reversed_mapping[value] = key
    return reversed_mapping

def compute_descendant_expression(adata, descendant_mapping, output_folder, prefix="dot_product_"):
    """
    For each node in descendant_mapping, compute fraction of dot product
    in descendant cells. Uses columns in adata.obs that start with prefix.
    """

    os.makedirs(output_folder, exist_ok=True)
    results = []

    # Library size for each cell
    if hasattr(adata.X, "tocsc"):
        library_sizes = adata.X.sum(axis=1).A1
    else:
        library_sizes = adata.X.sum(axis=1)
    adata.obs["library_size"] = library_sizes
    total_library_size = adata.obs["library_size"].sum()

    # reversed_mapping to get the node name from ID
    reversed_map = reverse_mapping(map_topics_to_nodes())

    # We'll look for columns that match f"{prefix}{reversed_map[node]}"
    # E.g. if prefix="dot_product_gmdf_" and reversed_map[node]="B cell",
    # then the column is "dot_product_gmdf_B cell"
    
    for node, descendants in descendant_mapping.items():
        node_name = reversed_map[node]  # e.g. "B cell"
        node_score_col = f"{prefix}{node_name}"

        # total score for this node across all cells
        total_score = adata.obs[node_score_col].sum()

        # Score for the node in descendant cells
        descendant_cell_types = [reversed_map[d] for d in descendants]
        descendant_cells = adata.obs["cell_type"].isin(descendant_cell_types)
        descendant_score = adata.obs.loc[descendant_cells, node_score_col].sum()

        # fraction
        fraction = round(descendant_score / total_score, 4) * 100 if total_score > 0 else 0

        # library size fraction
        descendant_library_size = adata.obs.loc[descendant_cells, "library_size"].sum()
        library_size_percentage = (
            round(descendant_library_size / total_library_size, 4) * 100 if total_library_size > 0 else 0
        )

        cell_count = descendant_cells.sum()

        results.append({
            "node": node_name,
            "descendants": ", ".join([reversed_map[d] for d in descendants]),
            "dot_product_percentage_in_descendants": fraction,
            "node_library_size_percentage": library_size_percentage,
            "node_cell_count": cell_count
        })

    df = pd.DataFrame(results)
    out_csv = os.path.join(output_folder, f"node_expression_fractions_{prefix}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Node expression fractions saved to {out_csv}.")
##
##def run_theta_inference(E, train, hlda_beta, num_topics, hyperparams, num_loops, burn_in,
##                      output_dir, long_data_path):
##
##    cell_identities = E.index.tolist()
##    gene_names = E.columns.tolist()
##
##    long_data, identity_mapping = initialize_long_data(E,
##                                                       cell_identities,
##                                                       num_topics,
##                                                       long_data_path)
##
##    topic_hierarchy = get_topic_hierarchy()
##
##    constants = compute_constants(long_data, num_topics)
##    
##    infer_theta_gibbs(long_data, hlda_beta, hyperparams, num_loops, burn_in,
##                      identity_mapping, topic_hierarchy, output_dir)
##
##def infer_theta_gibbs(held_out_data, beta, hyperparams, num_loops, burn_in,
##                      identity_mapping, topic_hierarchy, output_dir):
##    """
##    Infer theta (cell–topic distributions) for held–out cells via Gibbs sampling,
##    holding the corpus–level beta fixed.
##
##    Parameters:
##        held_out_data (np.ndarray): Structured array (or similar)
##            containing held–out data in long–format. Must have at least these fields:
##            'cell_idx', 'gene_idx', 'cell_identity', and 'z' (for topic assignment).
##        beta (np.ndarray): Fixed beta matrix of shape (G, K) that gives the probability
##            of each gene given each topic.
##        hyperparams (dict): Dictionary with at least:
##            'alpha_c': Dirichlet prior for the cell–topic distribution.
##        num_loops (int): Total number of Gibbs sampling iterations.
##        burn_in (int): Number of iterations to treat as burn–in (not used for averaging).
##        identity_mapping (dict): Mapping from cell identity (as stored in held_out_data)
##            to a key used in topic_hierarchy.
##        topic_hierarchy (dict): Dictionary mapping identity keys to a list of valid topics.
##    
##    Returns:
##        theta (np.ndarray): Inferred cell–topic distributions of shape (num_cells, K).
##            Each row sums to 1.
##    """
##    clear_directory(output_dir)
##    alpha_c = hyperparams['alpha_c']
##    alpha_beta = hyperparams['alpha_beta']
##    # Determine number of held-out cells (assumes cell indices start at 0)
##    num_cells = held_out_data['cell_idx'].max() + 1
##    # Assume beta has shape (num_genes, num_topics)
##    K = beta.shape[0]
##    print(K)
##
##    # Initialize the cell–topic count matrix D for held–out cells.
##    D = np.zeros((num_cells, K), dtype=int)
##
##    total_counts = len(held_out_data)
##    
##    # --- Initialize latent assignments randomly ---
##    for idx in range(total_counts):
##        cell_idx = held_out_data['cell_idx'][idx]
##        cell_identity = held_out_data['cell_identity'][idx]
##        valid_topics = topic_hierarchy[identity_mapping[cell_identity]]
##        # Randomly choose one of the valid topics
##        initial_z = np.random.choice(valid_topics)
##        held_out_data['z'][idx] = initial_z
##        D[cell_idx, initial_z] += 1
##
##    # --- Gibbs sampling loop ---
##    with tqdm(total=num_loops, desc="Gibbs Sampling", unit="iteration") as pbar:
##        for loop in range(num_loops):
##            for idx in range(total_counts):
##                cell_idx = held_out_data['cell_idx'][idx]
##                gene = held_out_data['gene_idx'][idx]
##                cell_identity = held_out_data['cell_identity'][idx]
##                current_z = held_out_data['z'][idx]
##
##                # Remove the current assignment from the cell–topic count
##                D[cell_idx, current_z] -= 1
##
##                # Get the allowed (valid) topics for this cell identity
##                valid_topics = topic_hierarchy[identity_mapping[cell_identity]]
##
##                # Compute the (unnormalized) probability for each valid topic.
##                # Here, beta[gene, topic] gives p(gene|topic) and (alpha_c + D[cell, topic])
##                # is the current count (plus prior) for the cell.
##                topic_probs = []
##                for topic in valid_topics:
##                    prob = (
##                            beta[topic, gene] *
##                            (alpha_c + D[cell_idx, topic])
##                        )
##                    topic_probs.append(prob)
##                topic_probs = np.array(topic_probs)
##
##                if topic_probs.sum() == 0:
##                    D[cell_idx, current_z] += 1
##                else:
##                    topic_probs = topic_probs / topic_probs.sum()
##                    new_z = np.random.choice(valid_topics, p=topic_probs)
##                    held_out_data['z'][idx] = new_z
##                    D[cell_idx, new_z] += 1                
##
##            # (Optional) If you wish, you can store D or compute intermediate θ values
##            # after the burn–in period. For brevity, this example only prints progress.
##            if loop < burn_in:
##                pbar.set_postfix_str(f"Burn-in (Iteration {loop + 1})")
##            else:
##                pbar.set_postfix_str(f"Sampling (Iteration {loop + 1})")
##                constants_filename = os.path.join(output_dir, f'theta_sample_{loop + 1}.pkl')
##                with open(constants_filename, 'wb') as f:
##                    pickle.dump(D, f)
##            pbar.update(1)

def run_theta_inference(E, train, hlda_beta, num_topics, hyperparams, num_loops, burn_in,
                        output_dir, long_data_path):
    """
    Wrapper function that prepares data and runs Gibbs sampling for theta inference.
    """
    cell_identities = E.index.tolist()      # E is assumed to have cells as rows
    gene_names = E.columns.tolist()         # and genes as columns

    # Initialize the 'held_out_data' in a structured array, similar to your existing code:
    long_data, identity_mapping = initialize_long_data(
        E,
        cell_identities,
        num_topics,
        long_data_path
    )

    topic_hierarchy = get_topic_hierarchy()  # e.g. dict[str, list[int]]
    
    # Compute any necessary constants (not strictly needed here if we only fix beta)
    constants = compute_constants(long_data, num_topics)
    
    # Actually run the inference
    infer_theta_gibbs(
        held_out_data=long_data,
        beta=hlda_beta,
        hyperparams=hyperparams,
        num_loops=num_loops,
        burn_in=burn_in,
        identity_mapping=identity_mapping,
        topic_hierarchy=topic_hierarchy,
        output_dir=output_dir
    )

def infer_theta_gibbs(held_out_data, beta, hyperparams, num_loops, burn_in,
                      identity_mapping, topic_hierarchy, output_dir):
    """
    Infer theta (cell–topic distributions) for held–out cells via Gibbs sampling,
    holding the corpus–level beta fixed (shape: [K, G]).
    """
    clear_directory(output_dir)
    
    alpha_c = hyperparams['alpha_c']
    alpha_beta = hyperparams['alpha_beta']  # might be unused if beta is fixed
    # Number of held-out cells
    num_cells = held_out_data['cell_idx'].max() + 1
    
    # beta is shape (K, G) => beta[topic, gene]
    K, G = beta.shape
    
    # Initialize the cell–topic count matrix D for held–out cells.
    D = np.zeros((num_cells, K), dtype=np.int32)

    total_counts = len(held_out_data)

    # --- Extract arrays from held_out_data for Numba ---
    cell_idx_arr = held_out_data['cell_idx']
    gene_idx_arr = held_out_data['gene_idx']
    cell_identity_arr = held_out_data['cell_identity']
    z_arr = held_out_data['z']

    # --- Build a typed list of valid topics for each cell identity ---
    # identity_mapping is int->string, topic_hierarchy is string->list[int]
    max_identity = max(identity_mapping.keys())  # assume 0..max
    valid_topic_list_py = []
    for i in range(max_identity+1):
        # If i is not in identity_mapping, you might handle it or skip
        if i not in identity_mapping:
            valid_topic_list_py.append(np.array([], dtype=np.int32))
            continue
        cell_identity_str = identity_mapping[i]
        valid_topics = topic_hierarchy[cell_identity_str]  # e.g. [0,1,2,...]
        valid_topic_list_py.append(np.array(valid_topics, dtype=np.int32))

    valid_topic_list = TypedList()
    for arr in valid_topic_list_py:
        valid_topic_list.append(arr)

    # --- Initialize the z assignments randomly (if not already) and update D ---
    # (If you already did this in initialize_long_data, you can skip.)
    for idx in range(total_counts):
        c = cell_idx_arr[idx]
        ident = cell_identity_arr[idx]
        valid_topics = valid_topic_list[ident]
        z_val = np.random.choice(valid_topics)
        z_arr[idx] = z_val
        D[c, z_val] += 1

    # --- Main Gibbs sampling loop ---
    with tqdm(total=num_loops, desc="Gibbs Sampling", unit="iteration") as pbar:
        for loop in range(num_loops):
            # Use the Numba-compiled update
            theta_update_numba(
                cell_idx_arr, gene_idx_arr, cell_identity_arr, z_arr,
                D,
                valid_topic_list,
                beta,
                alpha_c,
                total_counts
            )

            # Optionally save D after burn-in
            if loop >= burn_in:
                constants_filename = os.path.join(output_dir, f'theta_sample_{loop + 1}.pkl')
                with open(constants_filename, 'wb') as f:
                    pickle.dump(D, f)

            # Update progress bar
            if loop < burn_in:
                pbar.set_postfix_str(f"Burn-in (Iteration {loop + 1})")
            else:
                pbar.set_postfix_str(f"Sampling (Iteration {loop + 1})")
            pbar.update(1)

    # After sampling, compute final theta
    # theta[c, k] = (D[c, k] + alpha_c) / sum_k(D[c, k] + alpha_c)
    theta = np.zeros((num_cells, K), dtype=np.float64)
    for c in range(num_cells):
        row_sum = (D[c, :] + alpha_c).sum()
        theta[c, :] = (D[c, :] + alpha_c) / row_sum

    # Save final theta
    theta_path = os.path.join(output_dir, "theta_final.npy")
    np.save(theta_path, theta)
    print(f"Final theta saved to {theta_path}.")

# -----------------------------
# Numba-compiled function
# -----------------------------
@njit
def theta_update_numba(cell_idx_arr, gene_idx_arr, cell_identity_arr, z_arr,
                       D,
                       valid_topic_list,  # typed list of np.array (int32)
                       beta,             # shape (K, G), so beta[topic, gene]
                       alpha_c,
                       total_counts):
    """
    Single iteration of Gibbs updates for held-out data, holding beta fixed.
    D is the cell-topic count matrix, updated in place.
    z_arr is updated in place.
    """
    K, G = beta.shape
    for i in range(total_counts):
        c = cell_idx_arr[i]
        gene = gene_idx_arr[i]
        ident = cell_identity_arr[i]
        old_z = z_arr[i]

        # Remove old assignment
        D[c, old_z] -= 1

        # Get valid topics for this cell identity
        valid_topics = valid_topic_list[ident]

        # Compute probabilities for each valid topic
        topic_probs = np.empty(valid_topics.shape[0], dtype=np.float64)
        sum_probs = 0.0
        for j in range(valid_topics.shape[0]):
            t = valid_topics[j]
            # beta[t, gene] is p(gene|topic)
            # (alpha_c + D[c, t]) is the prior + current cell–topic count
            prob = beta[t, gene] * (alpha_c + D[c, t])
            topic_probs[j] = prob
            sum_probs += prob

        # If sum_probs == 0, revert to old assignment
        if sum_probs == 0.0:
            D[c, old_z] += 1
            continue

        # Normalize
        for j in range(valid_topics.shape[0]):
            topic_probs[j] /= sum_probs

        # Sample new topic
        r = np.random.rand()
        cum_sum = 0.0
        new_z = valid_topics[-1]
        for j in range(valid_topics.shape[0]):
            cum_sum += topic_probs[j]
            if r < cum_sum:
                new_z = valid_topics[j]
                break

        # Update with new assignment
        z_arr[i] = new_z
        D[c, new_z] += 1

def compute_theta(sample_dir):
    
    theta_accumulator = None
    sample_files = sorted([f for f in os.listdir(sample_dir) if f.endswith(".pkl")])

    print(f"Found {len(sample_files)} sample files for reconstruction.")
    
    for file in sample_files:
        with open(os.path.join(sample_dir, file), 'rb') as f:
            D = pickle.load(f)
            
        if theta_accumulator is None:
            theta_accumulator = D.copy()
        else:
            theta_accumulator += D

    theta_matrix = theta_accumulator / len(sample_files)

    theta_matrix /= theta_matrix.sum(axis=1, keepdims=True)

    theta_df = pd.DataFrame(theta_matrix,
##                            index=cell_ids,
                            columns=[f"Topic_{k}" for k in range(theta_matrix.shape[1])])

##    theta_output_path = os.path.join(save_dir, "theta_est.csv")
##    theta_df.to_csv(theta_output_path)
    
    return theta_matrix
    

def infer_theta_lda(held_out_counts, lda_model):
    """
    Infer theta for held–out cells using a pre–fitted LDA model.

    Parameters:
        held_out_counts (array-like or sparse matrix): Document–term matrix for held–out cells,
            where each row corresponds to a cell and each column to a gene.
        lda_model (LatentDirichletAllocation): A pre–fitted LDA model.

    Returns:
        theta (np.ndarray): Inferred cell–topic distributions (each row sums to 1).
    """
    theta = lda_model.transform(held_out_counts)
    return theta


def infer_theta_nmf(held_out_counts, nmf_model):
    """
    Infer theta for held–out cells using a pre–fitted NMF model.
    
    Parameters:
        held_out_counts (array-like or sparse matrix): Document–term matrix for held–out cells.
        nmf_model (NMF): A pre–fitted NMF model.
    
    Returns:
        theta (np.ndarray): Inferred cell–topic distributions, with each row normalized to sum to 1.
    """
    # The transform method returns the W matrix (cell–topic loadings)
    W = nmf_model.transform(held_out_counts)
    # Normalize each row to sum to 1 (adding a tiny constant to avoid division by zero)
    theta = W / (W.sum(axis=1, keepdims=True) + 1e-10)
    return theta

def union_theta_and_umap(
    theta_train, 
    theta_test,
    random_state=42,
    n_neighbors=15,
    min_dist=0.5,
    spread=1.0
):
    """
    Concatenate train & test usage, run UMAP to reduce to 2D.

    Parameters
    ----------
    theta_train : np.ndarray, shape (n_train, k_topics)
    theta_test : np.ndarray, shape (n_test, k_topics)
    random_state : int
    n_neighbors : int
    min_dist : float
    spread : float

    Returns
    -------
    embedding : np.ndarray, shape (n_train + n_test, 2)
        The 2D UMAP coordinates for all cells (train first, then test).
    labels_train_test : np.ndarray of shape (n_train + n_test,)
        An array of strings: "train" for the first n_train, "test" for the next n_test.
    """
    n_train = theta_train.shape[0]
    n_test = theta_test.shape[0]

    # 1) Stack the theta matrices row-wise.
    theta_all = np.vstack([theta_train, theta_test])
    # 2) Create a train/test label array.
    labels_train_test = np.array(["train"] * n_train + ["test"] * n_test)

    # 3) Run UMAP on the combined matrix.
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, 
        min_dist=min_dist,
        spread=spread,
        random_state=random_state
    )
    embedding = reducer.fit_transform(theta_all)  # shape => (n_train + n_test, 2)

    return embedding, labels_train_test

def plot_umap_by_celltype(
    embedding,
    cell_types_train,
    cell_types_test,
    save_path,
    method_name="Method"
):
    """
    Create a UMAP scatterplot colored by cell type.
    The first len(cell_types_train) points are training,
    next len(cell_types_test) are test.
    """
    n_train = len(cell_types_train)
    n_test = len(cell_types_test)

    # Combine the cell types into one array (train then test)
    cell_types_all = np.concatenate([cell_types_train, cell_types_test], axis=0)
    
    # Build a color mapping.
    unique_types = np.unique(cell_types_all)
    color_map = {ctype: i for i, ctype in enumerate(unique_types)}
    cmap = plt.get_cmap("tab20")
    num_colors = cmap.N  # typically 20 for tab20

    # Convert each cell's type to a color.
    colors = [cmap(color_map[ct] % num_colors) for ct in cell_types_all]

    plt.figure(figsize=(7, 6))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        alpha=0.7
    )
    plt.title(f"UMAP Colored by Cell Type: {method_name}")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    # Build legend handles: one Patch per cell type.
    handles = []
    for ctype, idx in color_map.items():
        patch_color = cmap(idx % num_colors)
        patch = mpatches.Patch(color=patch_color, label=ctype)
        handles.append(patch)

    plt.legend(
        handles=handles,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0,
        title="Cell Types"
    )

    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, f"{method_name}_colorby_celltype.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

def plot_umap_by_traintest(
    embedding,
    labels_train_test,
    save_path,
    method_name="Method"
):
    """
    Create a UMAP scatterplot colored by "train" vs. "test".
    labels_train_test is an array of shape (n_cells_total,) with 
    entries "train" or "test".
    """
    plt.figure(figsize=(7, 6))
    # We'll assign "train" -> 0, "test" -> 1.
    color_map = {"train": 0, "test": 1}
    # Use the bwr colormap.
    cmap = plt.get_cmap("bwr")
    numeric_colors = [color_map[lbl] for lbl in labels_train_test]

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=numeric_colors,
        cmap="bwr",
        alpha=0.7
    )
    plt.title(f"UMAP Colored by Train/Test: {method_name}")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    # Build a legend: train => blue, test => red.
    patch_train = mpatches.Patch(color=cmap(0.0), label="train")
    patch_test  = mpatches.Patch(color=cmap(1.0), label="test")
    plt.legend(
        handles=[patch_train, patch_test],
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0,
        title="Label"
    )

    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, f"{method_name}_colorby_train_test.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

def plot_umap_per_celltype_traintest(
    theta_train, 
    theta_test,
    cell_types_train, 
    cell_types_test,
    save_path,
    method_name="Method",
    random_state=42,
    n_neighbors=15,
    min_dist=0.5,
    spread=1.0
):
    """
    For each unique cell type in the union of cell_types_train and cell_types_test,
    filter the theta matrices, run UMAP, and create a UMAP plot colored by train/test.
    All plots are saved in the specified folder (which is method-specific).

    Parameters
    ----------
    theta_train : np.ndarray, shape (n_train, k_topics)
    theta_test  : np.ndarray, shape (n_test, k_topics)
    cell_types_train : array-like, shape (n_train,)
        The cell type labels for the training data.
    cell_types_test  : array-like, shape (n_test,)
        The cell type labels for the test data.
    save_path : str
        Folder in which to save the plots.
    method_name : str
        The name of the method (used in file names).
    random_state : int
        Random state for UMAP.
    n_neighbors : int
        Number of neighbors for UMAP.
    min_dist : float
        Minimum distance for UMAP.
    spread : float
        Spread parameter for UMAP.
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Ensure cell type labels are NumPy arrays.
    cell_types_train = np.array(cell_types_train)
    cell_types_test  = np.array(cell_types_test)
    
    # Identify unique cell types from both train and test.
    unique_cell_types = np.unique(np.concatenate([cell_types_train, cell_types_test]))
    
    for ctype in unique_cell_types:
        # Get indices for this cell type.
        idx_train = np.where(cell_types_train == ctype)[0]
        idx_test  = np.where(cell_types_test == ctype)[0]
        
        # Skip if no samples are present.
        if len(idx_train) + len(idx_test) == 0:
            continue

        # Filter the theta matrices.
        theta_train_ct = theta_train[idx_train, :]
        theta_test_ct  = theta_test[idx_test, :]
        
        # Run UMAP on the filtered subset.
        embedding, labels_train_test = union_theta_and_umap(
            theta_train_ct,
            theta_test_ct,
            random_state=random_state,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            spread=spread
        )
        
        # Clean cell type string for filenames.
        ctype_safe = ctype.replace(" ", "_")
        
        # Save the UMAP plot for this cell type into the method folder.
        plot_umap_by_traintest(
            embedding,
            labels_train_test,
            save_path,
            method_name=f"{method_name}_{ctype_safe}"
        )
        print(f"Saved UMAP train/test plot for cell type '{ctype}'")
        
def compute_mixing_ratio(
    embedding,
    labels,
    k=15
):
    """
    Compute a normalized mixing ratio of 'train' vs 'test' points 
    in the given embedding.

    Parameters
    ----------
    embedding : np.ndarray of shape (N, d)
        A 2D or high-D embedding of all N points (train + test).
        Must match the length of 'labels'.
    labels : array-like of shape (N,)
        Each entry is 'train' or 'test' (or 0/1). 
    k : int
        Number of neighbors to consider.

    Returns
    -------
    norm_mix_score : float
        The average ratio of (observed fraction of opposite-label neighbors)
        over (expected fraction of opposite-label cells).
        If ~1 => random mixing,
        If <1 => more segregated than random,
        If >1 => more intermixed than random.
    """
    N = embedding.shape[0]
    if len(labels) != N:
        raise ValueError("Embedding and labels must have the same length")

    # Fit nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)

    fractions = np.zeros(N, dtype=float)

    for i in range(N):
        # The first neighbor is often the point itself, so skip it
        neighbor_ids = indices[i, 1:k+1]  # shape (k,)
        my_label = labels[i]
        # Count how many neighbors have the opposite label
        opposite_count = np.sum(labels[neighbor_ids] != my_label)
        fractions[i] = opposite_count / k

    mix_score = fractions.mean()
    return mix_score

def plot_all_methods_umap(
    hlda_theta_train, hlda_theta_test,
    gmdf_theta_train, gmdf_theta_test,
    lda_theta_train, lda_theta_test,
    nmf_theta_train, nmf_theta_test,
    cell_types_train,
    cell_types_test,
    save_path,
    random_state=42
):
    """
    Demonstration: create two UMAP plots for each method:
      1) colored by cell type
      2) colored by train/test
    """

    methods = {
        "GMDF": (gmdf_theta_train, gmdf_theta_test),
        "LDA":  (lda_theta_train, lda_theta_test),
        "NMF":  (nmf_theta_train, nmf_theta_test),
        "HLDA": (hlda_theta_train, hlda_theta_test)
    }

    mixing_results = []
    # For each method, build & plot a single UMAP
    for method_name, (theta_tr, theta_te) in methods.items():
        # 1) Union & run UMAP
        print(method_name)
        embedding, labels_train_test = union_theta_and_umap(
            theta_tr,
            theta_te,
            random_state=42,
            n_neighbors=15,
            min_dist=0.5,
            spread=1.0
        )

        # 2) Plot colored by cell type
        plot_umap_by_celltype(
            embedding=embedding,
            cell_types_train=cell_types_train,
            cell_types_test=cell_types_test,
            save_path=save_path,
            method_name=method_name
        )

        # 3) Plot colored by train/test
        plot_umap_by_traintest(
            embedding=embedding,
            labels_train_test=labels_train_test,
            save_path=save_path,
            method_name=method_name
        )

        mixing_score = compute_mixing_ratio(
            embedding=embedding,
            labels=labels_train_test,  # array of "train"/"test"
            k=15
        )
        # Store (method_name, mixing_score) so we can write them out later
        mixing_results.append((method_name, mixing_score))
            
    csv_file = os.path.join(save_path, "mixing_scores.csv")
    mixing_df = pd.DataFrame(
        mixing_results,
        columns=["Method", "NormalizedMixScore(k=10)"]
    )

    mixing_df.to_csv(csv_file, index=False)
    print(f"UMAP plots and mixing scores have been saved to: {save_path}")






















    

    
