##import numpy as np
import pandas as pd
import os
import pickle
import numpy as np
from tqdm import tqdm
import time
from numba import njit
from numba.typed import List as TypedList
from functions import (clear_directory,
                       reconstruct_beta_theta)
from simulation import collapse_thetas_by_index
from sklearn.decomposition import LatentDirichletAllocation, NMF
import seaborn as sns
import matplotlib.pyplot as plt

def write_theta_table_csv(base_folder, output_file, num_topics=5):
    """
    Loops over subfolders in base_folder (each corresponding to a cell identity),
    reads in both "LDA_filtered_theta.csv" and "NMF_filtered_theta.csv" for that cell identity,
    averages them if needed, rounds the estimates to two decimals, and outputs a CSV table.
    
    The CSV will have columns: "Cell Identity", "Model", "Topic_0", "Topic_1", ..., "Topic_{num_topics-1}".
    For each cell identity, two rows are written (one for LDA, one for NMF), and then a blank row is inserted.
    
    Parameters:
      base_folder (str): Folder containing subfolders (e.g. identity_7, identity_8, etc.).
      output_file (str): Path to the CSV file to be written.
      num_topics (int): Number of topics expected in the theta files.
    """
    rows = []
    # Define column names.
    col_names = ["Cell Identity", "Model"] + [f"Topic_{i}" for i in range(num_topics)]
    
    # Process subfolders in sorted order.
    for subfolder in sorted(os.listdir(base_folder)):
        subpath = os.path.join(base_folder, subfolder)
        if not os.path.isdir(subpath):
            continue
        
        # Parse cell identity from subfolder name.
        if subfolder.startswith("identity_"):
            cell_identity = subfolder.replace("identity_", "")
        else:
            cell_identity = subfolder
        
        try:
            cell_identity_val = int(cell_identity)
        except ValueError:
            cell_identity_val = cell_identity
        
        # Paths for LDA and NMF theta files.
        lda_theta_file = os.path.join(subpath, "LDA_filtered_theta.csv")
        nmf_theta_file = os.path.join(subpath, "NMF_filtered_theta.csv")
        
        if not os.path.exists(lda_theta_file) or not os.path.exists(nmf_theta_file):
            print(f"Skipping {subfolder}: missing one or both theta files.")
            continue
        
        # Load LDA theta.
        df_lda = pd.read_csv(lda_theta_file, index_col=0)
        if df_lda.shape[0] > 1:
            lda_values = df_lda.mean(axis=0).values
        else:
            lda_values = df_lda.iloc[0].values
        lda_values = np.round(lda_values, 2)
        
        # Load NMF theta.
        df_nmf = pd.read_csv(nmf_theta_file, index_col=0)
        if df_nmf.shape[0] > 1:
            nmf_values = df_nmf.mean(axis=0).values
        else:
            nmf_values = df_nmf.iloc[0].values
        nmf_values = np.round(nmf_values, 2)
        
        # Create rows for this cell identity.
        lda_row = [cell_identity_val, "LDA"] + lda_values.tolist()
        nmf_row = [cell_identity_val, "NMF"] + nmf_values.tolist()
        rows.append(lda_row)
        rows.append(nmf_row)
        # Insert a blank row.
        rows.append([""] * len(col_names))
    
    # Create DataFrame and write to CSV.
    df_out = pd.DataFrame(rows, columns=col_names)
    df_out.to_csv(output_file, index=False)
    print(f"Saved combined theta table to {output_file}")

##def plot_beta_heatmap_grid(base_folder, model_types, beta_filenames, avg_expr_file, output_file):
##    if avg_expr_file is not None:
##        df_avg_expr = pd.read_csv(avg_expr_file, index_col=0)
##    else:
##        df_avg_expr = None
##
##    identities = []
##    for subfolder in os.listdir(base_folder):
##        subpath = os.path.join(base_folder, subfolder)
##        if os.path.isdir(subpath):
##            if subfolder.startswith("identity_"):
##                identity = subfolder.replace("identity_", "")
##            else:
##                identity = subfolder
##            identities.append(identity)
##    n_identities = len(identities)
##    n_models = len(model_types)
##    
##    fig, axes = plt.subplots(n_identities, n_models, figsize=(n_models * 4, n_identities * 3), squeeze=False)
##    
##    for i, identity in enumerate(identities):
##        subfolder = f"identity_{identity}"
##        cell_folder = os.path.join(base_folder, subfolder)
##        if df_avg_expr is not None and str(identity) in df_avg_expr.index:
##            avg_expr = df_avg_expr.loc[str(identity)]
##        else:
##            avg_expr = None
##        
##        for j, model in enumerate(model_types):
##            beta_file = os.path.join(cell_folder, beta_filenames[model])
##            if not os.path.exists(beta_file):
##                print(f"Beta file for {model} not found in {cell_folder}.")
##                axes[i, j].axis("off")
##                continue
##            
##            df_beta = pd.read_csv(beta_file, index_col=0)
##            df_beta = df_beta.loc[:, (df_beta.sum(axis=0) > 0)]
##            df_beta = df_beta.sort_index()
##            
##            if avg_expr is not None:
##                avg_expr_reordered = avg_expr.reindex(df_beta.index)
##                new_labels = []
##                for gene in df_beta.index:
##                    if pd.notnull(avg_expr_reordered.get(gene, np.nan)):
##                        new_labels.append(f"{gene}\n(avg={avg_expr_reordered[gene]:.2f})")
##                    else:
##                        new_labels.append(gene)
##                df_beta.index = new_labels
##            
##            sns.heatmap(df_beta, ax=axes[i, j], cmap="viridis", annot=True, fmt=".2f", cbar=False)
##            axes[i, j].set_title(f"{model} Beta")
##            axes[i, j].set_xlabel("Topic")
##            if j == 0:
##                axes[i, j].set_ylabel(f"Cell {identity}\nGene")
##            else:
##                axes[i, j].set_ylabel("")
##    
##    plt.tight_layout()
##    plt.savefig(output_file, dpi=150)
##    plt.close()
##    print(f"Saved combined beta heatmap grid to {output_file}")

def plot_beta_heatmap_grid(base_folder, model_types, beta_filenames, avg_expr_file, output_file):
    # If avg_expr_file is provided, load it. Rows = leaf IDs (or cell IDs), columns = genes.
    if avg_expr_file is not None:
        df_avg_expr = pd.read_csv(avg_expr_file, index_col=0)
    else:
        df_avg_expr = None

    # Collect subfolders (each named like "identity_7", "identity_14", etc.).
    identities = []
    for subfolder in os.listdir(base_folder):
        subpath = os.path.join(base_folder, subfolder)
        if os.path.isdir(subpath) and subfolder.startswith("identity_"):
            identity = subfolder.replace("identity_", "")
            identities.append(identity)

    # We'll create a 3-subplot row for each identity: [LDA Beta | NMF Beta | Avg Expr].
    n_identities = len(identities)
    fig, axes = plt.subplots(n_identities, 3, figsize=(3 * 4, n_identities * 3), squeeze=False)

    for i, identity in enumerate(identities):
        subfolder = f"identity_{identity}"
        cell_folder = os.path.join(base_folder, subfolder)

        # Paths for LDA and NMF beta
        lda_beta_path = os.path.join(cell_folder, beta_filenames.get("LDA", "LDA_filtered_beta.csv"))
        nmf_beta_path = os.path.join(cell_folder, beta_filenames.get("NMF", "NMF_filtered_beta.csv"))

        # Check if LDA/NMF beta files exist
        if not os.path.exists(lda_beta_path) or not os.path.exists(nmf_beta_path):
            for col in range(3):
                axes[i, col].axis("off")
            axes[i, 0].set_title(f"Missing beta files for identity {identity}")
            continue

        # Load LDA beta (genes x topics)
        df_lda = pd.read_csv(lda_beta_path, index_col=0)
        df_lda = df_lda.loc[:, df_lda.sum(axis=0) > 0].sort_index()

        # Load NMF beta
        df_nmf = pd.read_csv(nmf_beta_path, index_col=0)
        df_nmf = df_nmf.loc[:, df_nmf.sum(axis=0) > 0].sort_index()

        if df_avg_expr is not None and int(identity) in df_avg_expr.index:
            avg_row = df_avg_expr.loc[int(identity)]  # Series of gene -> expression
            avg_row = avg_row[avg_row != 0]
            # Unify the gene set across LDA, NMF, and avg_expr
            all_genes = set(df_lda.index) | set(df_nmf.index) | set(avg_row.index)
        else:
            avg_row = None
            # Unify gene set just across LDA and NMF
            all_genes = set(df_lda.index) | set(df_nmf.index)

        # Reindex LDA and NMF to the union of genes
        df_lda = df_lda.reindex(all_genes, fill_value=0).sort_index()
        df_nmf = df_nmf.reindex(all_genes, fill_value=0).sort_index()

        # Round LDA/NMF
        df_lda = df_lda.round(2)
        df_nmf = df_nmf.round(2)

        # Heatmap for LDA Beta
        sns.heatmap(df_lda, ax=axes[i, 0], cmap="viridis", annot=True, fmt=".2f", cbar=False)
        axes[i, 0].set_title(f"Identity {identity}\nLDA Beta")
        axes[i, 0].set_xlabel("Topic")
        axes[i, 0].set_ylabel("Gene")

        # Heatmap for NMF Beta
        sns.heatmap(df_nmf, ax=axes[i, 1], cmap="viridis", annot=True, fmt=".2f", cbar=False)
        axes[i, 1].set_title("NMF Beta")
        axes[i, 1].set_xlabel("Topic")
        axes[i, 1].set_ylabel("")

        # Create a single-column DF for Avg Expr if available
        if avg_row is not None:
            avg_df = pd.DataFrame({"AvgExpr": avg_row.reindex(all_genes, fill_value=0)}).sort_index()
            avg_df = avg_df.round(2)
        else:
            # If no average expression, create a single-column DF with zeros
            avg_df = pd.DataFrame({"AvgExpr": [0]*len(all_genes)}, index=sorted(all_genes))

        sns.heatmap(avg_df, ax=axes[i, 2], cmap="viridis", annot=True, fmt=".2f", cbar=False)
        axes[i, 2].set_title("Avg Expr")
        axes[i, 2].set_xlabel("")
        axes[i, 2].set_ylabel("")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved combined 3-subplot beta heatmap to {output_file}")

def plot_theta_grid(lda_theta_file, nmf_theta_file, output_file):
    # Load LDA and NMF theta CSV files (assumed collapsed theta matrices with cell types as index and topics as columns)
    df_lda = pd.read_csv(lda_theta_file, index_col=0)
    df_nmf = pd.read_csv(nmf_theta_file, index_col=0)
    
    # Create a figure with 2 subplots (vertical layout)
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot the LDA theta heatmap on the top
    sns.heatmap(df_lda, ax=axes[0], cmap="viridis", annot=True, fmt=".2f", cbar=True)
    axes[0].set_title("LDA Theta Estimates")
    axes[0].set_xlabel("Topic")
    axes[0].set_ylabel("Cell Type")
    
    # Plot the NMF theta heatmap on the bottom
    sns.heatmap(df_nmf, ax=axes[1], cmap="viridis", annot=True, fmt=".2f", cbar=True)
    axes[1].set_title("NMF Theta Estimates")
    axes[1].set_xlabel("Topic")
    axes[1].set_ylabel("Cell Type")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved combined theta grid heatmap to {output_file}")

def plot_beta_grid(lda_beta_file, nmf_beta_file, output_file):
    # Load LDA and NMF beta CSV files (assumed beta matrices with genes as index and topics as columns)
    df_lda = pd.read_csv(lda_beta_file, index_col=0)
    df_nmf = pd.read_csv(nmf_beta_file, index_col=0)
    
    # Create a figure with 2 subplots (vertical layout)
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot the LDA beta heatmap on the top
    sns.heatmap(df_lda, ax=axes[0], cmap="viridis", annot=True, fmt=".2f", cbar=True)
    axes[0].set_title("LDA Beta Estimates")
    axes[0].set_xlabel("Topic")
    axes[0].set_ylabel("Gene")
    
    # Plot the NMF beta heatmap on the bottom
    sns.heatmap(df_nmf, ax=axes[1], cmap="viridis", annot=True, fmt=".2f", cbar=True)
    axes[1].set_title("NMF Beta Estimates")
    axes[1].set_xlabel("Topic")
    axes[1].set_ylabel("Gene")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved combined beta grid heatmap to {output_file}")
        
def compute_avg_expression_by_cell_type(input_file, output_file):
    """
    Reads in a count matrix from input_file, where rows are cells (indexed by cell type)
    and columns are genes. Computes the average gene expression for each cell type 
    and writes the result to output_file as a CSV.
    
    Parameters:
      input_file (str): Path to the count matrix CSV.
      output_file (str): Path where the average expression CSV will be saved.
    """
    # Load the count matrix.
    df = pd.read_csv(input_file, index_col=0)
    print("Input data shape:", df.shape)
    
    # Group by cell type (the index) and compute the average expression.
    df_avg = df.groupby(df.index).mean()
    df_avg = df_avg.div(df_avg.sum(axis=1), axis=0)
    
    # Save the average expression matrix to CSV.
    df_avg.to_csv(output_file)
    print(f"Average gene expression by cell type saved to {output_file}")



def get_topic_hierarchy(output_path=None):
    """
    Returns a dictionary mapping each cell type (leaf) to a unique path of integer node IDs
    from the root (0) down to that leaf. No ID is reused for different nodes.
    
    Updated to include all cell types in the data.
    """

    topic_hierarchy = {
        7:  [0, 1, 3, 7],
        8:  [0, 1, 3, 8],
        9:  [0, 1, 4, 9],
        10: [0, 1, 4, 10],
        11: [0, 2, 5, 11],
        12: [0, 2, 5, 12],
        13: [0, 2, 6, 13],
        14: [0, 2, 6, 14],
    }

    df = pd.DataFrame({
        "Cell_Type": list(topic_hierarchy.keys()),
        "Topic_Path": list(topic_hierarchy.values())
    })
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Topic hierarchy saved to: {output_path}")
    
    return topic_hierarchy

def create_ancestor_list(topic_hierarchy, K):
    """
    Given a topic_hierarchy dictionary where each value is a full path 
    (list) from the root to that topic, compute the proper ancestors for 
    each topic (excluding the topic itself) and return a Numba typed list 
    of np.array(int32) of length K.
    
    Parameters:
      topic_hierarchy (dict): e.g. {7: [0, 1, 3, 7], 8: [0, 1, 3, 8], ...}
      K (int): total number of topics.
      
    Returns:
      ancestor_list: a Numba typed list such that ancestor_list[k] is an 
                     np.array of int32 containing the proper ancestors of topic k.
                     If a topic is not in topic_hierarchy, it returns an empty array.
    """
    ancestors_dict = {}
    if topic_hierarchy is not None:
        for path in topic_hierarchy.values():
            # For each index in the path, record the proper ancestors.
            for i, topic in enumerate(path):
                # proper_ancestors are all topics before index i (empty if i == 0)
                proper_ancestors = path[:i]
                if topic in ancestors_dict:
                    if ancestors_dict[topic] != proper_ancestors:
                        print(f"Warning: inconsistent ancestors for topic {topic}. Overriding with latest.")
                        ancestors_dict[topic] = proper_ancestors
                else:
                    ancestors_dict[topic] = proper_ancestors

    # Create a Numba typed list (from numba.typed) where each entry is a np.array of int32.
    ancestor_list = TypedList()
    for k in range(K):
        if k in ancestors_dict:
            arr = np.array(ancestors_dict[k], dtype=np.int32)
        else:
            arr = np.empty(0, dtype=np.int32)
        ancestor_list.append(arr)
    
    return ancestor_list

def create_children_list(topic_hierarchy, K):
    """
    Given a topic_hierarchy dictionary where each value is a full path
    (list) from the root to a leaf topic, create a dictionary mapping each topic 
    (0,...,K-1) to the set of all nodes that follow it in any path (its full children).
    
    Parameters:
      topic_hierarchy (dict): e.g. {7: [0, 1, 3, 7], 8: [0, 1, 3, 8], ...}
      K (int): total number of topics.
      
    Returns:
      children_dict (dict): Dictionary mapping each topic (0,...,K-1) to a list
                            of its children (transitively). For example, if all paths 
                            start with 0, then children_dict[0] will be the union of 
                            all other topics.
    """
    children_dict = {k: set() for k in range(K)}
    if topic_hierarchy is not None:
        for path in topic_hierarchy.values():
            # For each topic in the path, add all later topics in the same path as children.
            for i in range(len(path) - 1):
                parent = path[i]
                for j in range(i+1, len(path)):
                    child = path[j]
                    children_dict[parent].add(child)
    # Convert sets to sorted lists (optional, but helps with reproducibility)
    for k in range(K):
        children_dict[k] = sorted(list(children_dict[k]))
    return children_dict

def create_children_list_numba(topic_hierarchy, K):
    children_dict = create_children_list(topic_hierarchy, K)
    children_list = TypedList()
    for k in range(K):
        # Explicitly set dtype to int32 so that indexing works.
        arr = np.array(children_dict[k], dtype=np.int32)
        children_list.append(arr)
    return children_list

def initialize_long_data(count_matrix, cell_identities, num_topics):
    """
    Initialize long data for Gibbs sampling.

    Parameters:
        count_matrix (pd.DataFrame): Pandas DataFrame where rows are cells and columns are genes.
        cell_identities (list of str): List of strings representing cell identities.
        num_topics (int): Number of topics.
        topic_hierarchy (dict): Mapping of cell identities to valid topic constraints.

    Returns:
        tuple: (long_data, beta_matrix)
    """
    unique_identities = list(set(cell_identities))
    identity_to_int = {identity: idx for idx, identity in enumerate(unique_identities)}
    int_to_identity = {idx: identity for identity, idx in identity_to_int.items()} 

    cell_identities_int = np.array([identity_to_int[cell] for cell in cell_identities], dtype=int)

    count_matrix = count_matrix.to_numpy().astype(int)

    long_data_dtype = np.dtype([
        ('cell_idx', np.int32),
        ('gene_idx', np.int32),
        ('cell_identity', np.int32),
        ('x1', np.int32),
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
        'x1': long_data['x1'],
        'z': long_data['z']
    })
##    long_data_df.to_csv(os.path.join(long_data_csv_path, 'long_data.csv'), index=False)
##    pd.DataFrame.from_dict(int_to_identity, orient="index").to_csv(os.path.join(long_data_csv_path, 'topic_int_mapping.csv'))

    return long_data, int_to_identity

def compute_constants(long_data, K):
    """
    Compute constants for Gibbs sampling with children-based penalty.
    
    Parameters:
      long_data (np.ndarray): Structured array with fields 'cell_idx', 'gene_idx', 'z'.
      K (int): Total number of topics.
      children_dict (dict): Dictionary mapping each topic to a list of its children,
                            computed from the full topic hierarchy.
                            
    Returns:
      dict: A dictionary containing:
            - A: 2D array ([G, K]) with counts for gene g in topic k.
            - B: 1D array (length K) with overall counts per topic.
            - D: 2D array ([#cells, K]) with counts per cell.
            - P: 2D array ([G, K]) with P[g,k] = sum_{child in children_dict[k]} A[g,child].
            - L: 1D array (length K) with L[k] = sum_{child in children_dict[k]} B[child].
            - Other metadata.
    """
    cell_idx = long_data['cell_idx']
    gene_idx = long_data['gene_idx']
    z = long_data['z']

    unique_genes = np.unique(gene_idx)
    unique_cells = np.unique(cell_idx)

    children_dict = create_children_list(get_topic_hierarchy(), K)

    n_expected_genes = np.max(gene_idx) + 1  # indices are 0-based

    # Compute A: counts for gene g in topic k.
    gene_topic_pairs = gene_idx * K + z
    counts = np.bincount(gene_topic_pairs, minlength=n_expected_genes * K)
    A = counts.reshape(n_expected_genes, K)

    # Compute B: total counts for each topic.
    B = np.bincount(z, minlength=K)

    # Compute D: per-cell topic counts.
    cell_topic_pairs = cell_idx * K + z
    counts = np.bincount(cell_topic_pairs, minlength=len(unique_cells) * K)
    D = counts.reshape(len(unique_cells), K)

    # Now compute P and L using children_dict.
    P = np.zeros_like(A, dtype=A.dtype)
    L = np.zeros_like(B, dtype=B.dtype)
    for k in range(K):
        if k in children_dict and len(children_dict[k]) > 0:
            # Convert the list to a NumPy integer array (fixes index error).
            child_list = np.array(children_dict[k], dtype=np.int32)
            # Sum counts for each gene over all children.
            P[:, k] = A[:, child_list].sum(axis=1)
            L[k] = B[child_list].sum()
        else:
            P[:, k] = 0
            L[k] = 0

    constants = {
        'A': A,
        'B': B,
        'D': D,
        'P': P,
        'L': L,
        'unique_genes': unique_genes,
        'unique_cells': unique_cells,
        'unique_topics': np.arange(K),
        'G': len(unique_genes),
        'K': K,
        'C': len(unique_cells)
    }
    return constants

@njit
def gibbs_update_numba(cell_idx_arr, gene_idx_arr, cell_identity_arr, z_arr,
                       A, B, D, P, L,
                       valid_topic_list, children_list,  
                       alpha_beta, alpha_c, G, K):
    """
    Perform one iteration over all gene counts, updating topics in place.
    
    With the children-based penalty definition:
      P[g,t] = sum_{child in children_list[t]} A[g, child]
      L[t]   = sum_{child in children_list[t]} B[child]
      
    The sampling probability for a candidate topic t (which lies in the valid pathway
    valid_chain for the current cell) is computed as:
    
      prob(t) ∝ [ max(αβ + A[g,t] – P[g,t], eps) / max(Gαβ + B[t] – L[t], eps) ]
                × (α_c + D[c,t])
                × ∏_{child in valid_chain[j+1:]} 
                     { max(αβ + A[g,child] – P[g,child] + 1, eps) / max(Gαβ + B[child] – L[child], eps) }.
    
    Parameters:
      cell_idx_arr, gene_idx_arr, cell_identity_arr, z_arr : arrays of assignments.
      A : 2D array ([#genes, K]) with counts for gene g in topic t.
      B : 1D array (length K) with overall counts for topic t.
      D : 2D array ([#cells, K]) with per-cell topic counts.
      P : 2D array ([#genes, K]), where P[g,t] = sum_{child in children_list[t]} A[g, child].
      L : 1D array (length K), where L[t] = sum_{child in children_list[t]} B[child].
      valid_topic_list : typed list of np.array(int32) giving the valid topics (ordered from root to leaf)
                          for each cell identity.
      children_list : typed list of np.array(int32) where children_list[t] gives the children of topic t.
      alpha_beta, alpha_c, G, K : hyperparameters and dimensions.
      
    Note: A small constant eps is used to ensure positivity in divisions.
    """
    eps = 1e-6
    total_counts = cell_idx_arr.shape[0]
    G_alpha_beta = G * alpha_beta

    for i in range(total_counts):
        c = cell_idx_arr[i]
        g = gene_idx_arr[i]
        ident = cell_identity_arr[i]
        current_z = z_arr[i]

        # --- Decrement the current assignment counts ---
        A[g, current_z] -= 1
        B[current_z] -= 1
        D[c, current_z] -= 1

        # Update P and L for removal:
        # For each topic t, if current_z is in its children list then update.
        for t in range(K):
            # Loop over children_list[t] (which is a Numba-typed array of int32).
            for child in children_list[t]:
                if child == current_z:
                    P[g, t] -= 1
                    L[t] -= 1
                    break

        # --- Compute the sampling probabilities over the valid pathway ---
        valid_chain = valid_topic_list[ident]  # e.g., [0, 1, 4] (ordered from root to leaf)
        n_valid = valid_chain.shape[0]
        probs = np.empty(n_valid, dtype=np.float64)
        sum_probs = 0.0

        for j in range(n_valid):
            t = valid_chain[j]
            # Direct ratio for candidate topic t.
            num = alpha_beta + A[g, t] - P[g, t]
            if num < eps:
                num = eps
            den = G_alpha_beta + B[t] - L[t]
            if den < eps:
                den = eps
            ratio = num / den

            # Cell-topic factor.
            cell_factor = alpha_c + D[c, t]

            # Product over the child terms for every valid topic to the right of index j.
            child_term = 1.0
            for k in range(j+1, n_valid):
                child_topic = valid_chain[k]
                child_num = alpha_beta + A[g, child_topic] - P[g, child_topic] + 1.0
                if child_num < eps:
                    child_num = eps
                child_den = G_alpha_beta + B[child_topic] - L[child_topic]
                if child_den < eps:
                    child_den = eps
                child_term *= (child_num / child_den)

            prob = ratio * cell_factor * child_term
            probs[j] = prob
            sum_probs += prob

        # Normalize probabilities.
        for j in range(n_valid):
            probs[j] /= sum_probs

        # --- Sample a new topic ---
        r = np.random.rand()
        cumulative = 0.0
        new_z = valid_chain[-1]
        for j in range(n_valid):
            cumulative += probs[j]
            if r < cumulative:
                new_z = valid_chain[j]
                break

        # --- Increment the counts for the new assignment ---
        A[g, new_z] += 1
        B[new_z] += 1
        D[c, new_z] += 1
        z_arr[i] = new_z

        # Update P and L for the new assignment.
        for t in range(K):
            for child in children_list[t]:
                if child == new_z:
                    P[g, t] += 1
                    L[t] += 1
                    break
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
    P = constants['P']
    L = constants['L']
    K = constants['K']

    # Retrieve or build your topic hierarchy (dict: identity -> list of valid topics)
    topic_hierarchy = get_topic_hierarchy()
    children_list = create_children_list_numba(topic_hierarchy, K)

    # Unpack hyperparameters
    alpha_beta = hyperparams['alpha_beta']
    alpha_c = hyperparams['alpha_c']

    total_counts = len(long_data)

    # Convert the "topic_hierarchy" dict into a typed list of np.array for Numba
    # 1) For each integer identity i, find the valid topics in Python
    # 2) Convert them to a NumPy array of int32
    # 3) Append them to a Numba typed list
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
                       A, B, D, P, L,
                       valid_topic_list, children_list,
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

def run_gibbs_pipeline(num_topics, num_loops, burn_in, hyperparams, output_dir, save_dir, input_file):

    count_matrix = pd.read_csv(input_file, index_col=0)

    cell_identities = count_matrix.index.tolist()
    gene_names = count_matrix.columns.tolist()

    long_data, identity_mapping = initialize_long_data(count_matrix,
                                                       cell_identities,
                                                       num_topics)

    constants = compute_constants(long_data, num_topics)

    run_gibbs_sampling_numba(long_data, constants, hyperparams,
                       num_loops, burn_in, output_dir, identity_mapping)

    reconstruct_beta_theta(output_dir, save_dir, gene_names, cell_identities)
    collapse_thetas_by_index(save_dir)

def run_lda_pipeline(input_file, num_topics, output_folder):
    """
    Reads a full count matrix from input_file, runs LDA with num_topics topics, 
    collapses the document–topic matrix by cell type, and writes the collapsed theta
    and normalized beta matrices to CSV files in output_folder.
    """
    # Load the count matrix. Assumes the CSV has cell types as its row index.
    df = pd.read_csv(input_file, index_col=0)
    print("Full data shape:", df.shape)
    
    # Fit LDA on the full dataset.
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(df)
    
    # Get document–topic matrix (theta) and normalize rows.
    raw_theta = lda.transform(df)
    theta = raw_theta / raw_theta.sum(axis=1, keepdims=True)
    theta_df = pd.DataFrame(theta, index=df.index,
                            columns=[f"Topic_{i}" for i in range(num_topics)])
    
    # Collapse by cell type (average across cells with same index).
    theta_by_cell_type = theta_df.groupby(theta_df.index).mean()
    
    # Get raw topic-term matrix and normalize to get beta.
    raw_beta = lda.components_  # shape: (num_topics, n_terms)
    beta = raw_beta / raw_beta.sum(axis=1, keepdims=True)
    beta_df = pd.DataFrame(beta, index=[f"Topic_{i}" for i in range(num_topics)],
                           columns=df.columns)
    
    # Create output folder if it doesn't exist.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save outputs.
    theta_by_cell_type.to_csv(os.path.join(output_folder, "lda_theta_by_cell_type.csv"))
    # Save beta transposed so that genes are rows.
    beta_df.T.to_csv(os.path.join(output_folder, "lda_beta_normalized.csv"))
    print("LDA pipeline complete. Outputs written to", output_folder)

def run_filtered_lda(input_file, cell_type, num_topics, output_folder):
    """
    Reads a count matrix from input_file, filters to only rows corresponding to cell_type,
    runs LDA with num_topics topics on the filtered data, collapses the theta matrix (averaging by cell type),
    and writes the outputs (collapsed theta and normalized beta) to CSV files in output_folder.
    """
    df = pd.read_csv(input_file, index_col=0)
    print("Full data shape:", df.shape)
    
    # Filter by the given cell type.
    df_filtered = df.loc[df.index == cell_type]
    print(f"Filtered data shape for cell type {cell_type}:", df_filtered.shape)
##    df_filtered = df_filtered.loc[:, (df_filtered.sum(axis=0) > 0)]
    print(f"Filtered data shape for cell type {cell_type} (after dropping zeros):", df_filtered.shape)

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(df_filtered)
    
    raw_theta = lda.transform(df_filtered)
    theta = raw_theta / raw_theta.sum(axis=1, keepdims=True)
    theta_df = pd.DataFrame(theta, index=df_filtered.index,
                            columns=[f"Topic_{i}" for i in range(num_topics)])
    
    # Collapse by cell type (should result in one row if all rows share the same cell type).
    theta_collapsed = theta_df.groupby(theta_df.index).mean()
    
    raw_beta = lda.components_
    beta = raw_beta / raw_beta.sum(axis=1, keepdims=True)
    beta_df = pd.DataFrame(beta, index=[f"Topic_{i}" for i in range(num_topics)],
                           columns=df_filtered.columns)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    theta_collapsed.to_csv(os.path.join(output_folder, "LDA_filtered_theta.csv"))
    beta_df.T.to_csv(os.path.join(output_folder, "LDA_filtered_beta.csv"))
    print("Filtered LDA pipeline complete. Outputs written to", output_folder)

def run_nmf_pipeline(input_file, num_topics, output_folder, max_iter=1000):
    """
    Reads the full count matrix from input_file, runs NMF on the full dataset with num_topics topics 
    (using max_iter iterations), normalizes the resulting document–topic (theta) and topic–term (beta) matrices,
    and writes them to CSV files in output_folder.
    """
    df = pd.read_csv(input_file, index_col=0)
    print("Full data shape:", df.shape)
    
    nmf_model = NMF(n_components=num_topics, random_state=42, max_iter=max_iter)
    W = nmf_model.fit_transform(df)  # Document–topic matrix.
    H = nmf_model.components_         # Topic–term matrix.
    
    # Normalize W: each row sums to 1.
    W_norm = W / W.sum(axis=1, keepdims=True)
    theta_nmf = pd.DataFrame(W_norm, index=df.index,
                             columns=[f"Topic_{i}" for i in range(num_topics)])
    theta_nmf_collapsed = theta_nmf.groupby(theta_nmf.index).mean()
    
    # Normalize H: each row sums to 1.
    H_norm = H / H.sum(axis=1, keepdims=True)
    beta_nmf = pd.DataFrame(H_norm, index=[f"Topic_{i}" for i in range(num_topics)],
                            columns=df.columns)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    theta_nmf_collapsed.to_csv(os.path.join(output_folder, "NMF_theta.csv"))
    beta_nmf.T.to_csv(os.path.join(output_folder, "NMF_beta.csv"))
    print("NMF pipeline complete. Outputs written to", output_folder)

def run_filtered_nmf(input_file, cell_type, num_topics, output_folder, max_iter=1000):
    """
    Reads the count matrix from input_file, filters to only rows corresponding to cell_type,
    runs NMF with num_topics topics on the filtered data (using max_iter iterations),
    normalizes the document–topic (theta) and topic–term (beta) matrices,
    collapses the theta matrix by cell type, and writes the outputs to CSV files in output_folder.
    """
    df = pd.read_csv(input_file, index_col=0)
    print("Full data shape:", df.shape)
    
    df_filtered = df.loc[df.index == cell_type]
    print(f"Filtered data shape for cell type {cell_type}:", df_filtered.shape)
##    df_filtered = df_filtered.loc[:, (df_filtered.sum(axis=0) > 0)]
    print(f"Filtered data shape for cell type {cell_type} (after dropping zeros):", df_filtered.shape)

    
    nmf_model = NMF(n_components=num_topics, random_state=42, max_iter=max_iter)
    W = nmf_model.fit_transform(df_filtered)
    H = nmf_model.components_
    
    W_norm = W / W.sum(axis=1, keepdims=True)
    theta_nmf = pd.DataFrame(W_norm, index=df_filtered.index,
                             columns=[f"Topic_{i}" for i in range(num_topics)])
    theta_nmf_collapsed = theta_nmf.groupby(theta_nmf.index).mean()
    
    H_norm = H / H.sum(axis=1, keepdims=True)
    beta_nmf = pd.DataFrame(H_norm, index=[f"Topic_{i}" for i in range(num_topics)],
                            columns=df_filtered.columns)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    theta_nmf_collapsed.to_csv(os.path.join(output_folder, "NMF_filtered_theta.csv"))
    beta_nmf.T.to_csv(os.path.join(output_folder, "NMF_filtered_beta.csv"))
    print("Filtered NMF pipeline complete. Outputs written to", output_folder)

def run_all_pipelines():
    """
    Top-level function that runs all four pipelines:
      - Full LDA
      - Filtered LDA (for a given cell type)
      - Full NMF
      - Filtered NMF (for a given cell type)
    with preset parameters. Adjust the parameters as needed.
    """
    # Parameters
    num_topics_full = 15
    num_topics_filtered = 4
    max_iter_nmf = 1000
    input_file = '../data/simulation/simulated_count_matrix.csv'
    base_output_folder = '../estimates/simulation/lda_nmf_test/'
    
    # Run full LDA.
    run_lda_pipeline(input_file, num_topics_full, base_output_folder)
    run_nmf_pipeline(input_file, num_topics_full, base_output_folder, max_iter=max_iter_nmf)

    cell_types = [7,8,9,10,11,12,13,14]
    for cell_type in cell_types:
        output_folder = os.path.join(base_output_folder, f"identity_{cell_type}")
        run_filtered_lda(input_file, cell_type, num_topics_filtered, output_folder)
        run_filtered_nmf(input_file, cell_type, num_topics_filtered, output_folder, max_iter=max_iter_nmf)


def run_gibbs():

    num_topics = 15
    num_loops = 500
    burn_in = 200
    hyperparams = {'alpha_beta': 1, 'alpha_c': 1}
    
    output_dir = f'../samples/simulation/penalized/'
    save_dir = f'../estimates/simulation/penalized/'

    input_file = '../data/simulation/simulated_count_matrix.csv'

    run_gibbs_pipeline(num_topics, num_loops, burn_in, hyperparams, output_dir, save_dir, input_file)

def plot():

    input_file = "../data/simulation/simulated_count_matrix.csv"
    avg_expr_file = "../estimates/simulation/lda_nmf_test/average_expression_by_cell_type.csv"
    
    # Ensure the output folder exists.
    output_folder = os.path.dirname(avg_expr_file)
    if not os.path.exists(avg_expr_file):
        os.makedirs(avg_expr_file)
    
##    compute_avg_expression_by_cell_type(input_file, avg_expr_file)

    base_folder="../estimates/simulation/lda_nmf_test/"
    avg_expr_file = "../estimates/simulation/lda_nmf_test/average_expression_by_cell_type.csv"
    beta_output_file = "../estimates/simulation/lda_nmf_test/plots/combined_beta_heatmap_grid_sparse.png"

    # Example usage:
    base_folder = "../estimates/simulation/lda_nmf_test/"  # e.g., folder containing identity_7, identity_8, etc.
    theta_output_file = "../estimates/simulation/lda_nmf_test/combined_identity_theta_table_sparse.csv"
    write_theta_table_csv(base_folder, theta_output_file, num_topics=4)

    model_types = ["LDA", "NMF"]
    beta_filenames = {
        "LDA": "LDA_filtered_beta.csv",
        "NMF": "NMF_filtered_beta.csv"
    }

    plot_beta_heatmap_grid(base_folder, model_types, beta_filenames, avg_expr_file, beta_output_file)

    lda_beta_file = "../estimates/simulation/lda_nmf_test/lda_beta_normalized.csv"
    nmf_beta_file = "../estimates/simulation/lda_nmf_test/NMF_beta.csv"
    combined_beta_out = "../estimates/simulation/lda_nmf_test/plots/combined_beta_heatmap.png"
##    plot_beta_grid(lda_beta_file, nmf_beta_file, combined_beta_out)

    lda_theta_file = "../estimates/simulation/lda_nmf_test/lda_theta_by_cell_type.csv"
    nmf_theta_file = "../estimates/simulation/lda_nmf_test/NMF_theta.csv"
    combined_theta_out = "../estimates/simulation/lda_nmf_test/plots/combined_theta_heatmap.png"
##    plot_theta_grid(lda_theta_file, nmf_theta_file, combined_theta_out)

if __name__ == "__main__":
    
##    run_all_pipelines()
##    plot()

    def simulate_dirichlet(alpha, n_samples=10000):
        samples = np.random.dirichlet(alpha, n_samples)
        return samples[:, 0]  # returning the first coordinate

    # Define parameters for each Dirichlet distribution
    params = {
        'Dirichlet(100,100,800)': [100, 100, 800],
        'Dirichlet(50,50,400)': [50, 50, 400],
        'Dirichlet(5,5,40)': [5, 5, 40],
        'Dirichlet(2,2,8)': [2, 2, 8],
        'Dirichlet(1,1,3)': [1,1,3]
    }

    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Plot the histogram of the first coordinate for each distribution
    bins = np.linspace(0, 0.3, 50)
    for label, alpha in params.items():
        data = simulate_dirichlet(alpha)
        plt.hist(data, bins=bins, alpha=0.5, label=label, density=True)

    plt.xlabel('First coordinate value')
    plt.ylabel('Density')
    plt.title('Comparison of the First Coordinate Across Different Dirichlet Distributions')
    plt.legend()
    plt.show()
    




























