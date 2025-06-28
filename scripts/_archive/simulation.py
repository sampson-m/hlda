import numpy as np
import pandas as pd
import os
from sklearn.decomposition import LatentDirichletAllocation
from functions import (initialize_long_data,
                       compute_constants,
                       reconstruct_beta_theta,
                       gibbs_update_numba,
                       run_gibbs_sampling_numba,
                       get_topic_hierarchy)
from gmdf import (create_condition_matrix,
                  bcd_solve)
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns

def extract_hyperparams_from_path(filepath):
    """
    Extracts alpha_c and alpha_beta values from a folder name.
    Expected folder name pattern: run_alpha_c_<alpha_c>_alpha_beta_<alpha_beta>
    """
    folder = os.path.basename(os.path.dirname(filepath))
    # Using regex to extract numbers after 'alpha_c_' and 'alpha_beta_'
    match = re.search(r'alpha_c_(\d+)_alpha_beta_(\d+)', folder)
    if match:
        alpha_c = int(match.group(1))
        alpha_beta = int(match.group(2))
        return alpha_c, alpha_beta
    else:
        return None, None

def pivot_topics_wide_to_long(df_wide):
    """
    Reads a wide CSV of topic values (e.g., columns: Topic_0, Topic_1, etc.)
    and pivots it into a long format DataFrame with columns:
      [doc, topic, value]
    
    Parameters:
    -----------
    csv_path : str
        Path to the wide-format CSV file.
    doc_id_name : str
        Column name for the row index in the resulting DataFrame (default is 'doc').

    Returns:
    --------
    pd.DataFrame
        Long-format DataFrame with columns: [doc_id_name, 'topic', 'value'].
    """
    # If your CSV does not have a unique ID column and you want to keep track 
    # of the original row index as "doc", you can reset the index:
    df_long = df_wide.melt(
        id_vars=["leaf_id", "alpha_c", "alpha_beta"],           # which columns to keep fixed
        var_name="topic",          # new column for the melted variable names
        value_name="value"         # new column for the melted values
    )

    # (Optional) Convert "Topic_0" -> 0, "Topic_1" -> 1, etc. 
    # so the 'topic' column is numeric:
    df_long["topic"] = df_long["topic"].str.replace("Topic_", "")

    return df_long

def pivot_true_thetas(df_wide):
    """
    Reads a CSV containing columns [leaf_id, theta_0, theta_1, ..., theta_n]
    and pivots it into a long format with columns: [leaf_id, topic, value].
    """
    # Convert from wide to long format
    df_long = df_wide.melt(
        id_vars="leaf_id",      # which column(s) to keep fixed
        var_name="topic",       # name for the melted topic columns (e.g., 'theta_0')
        value_name="value"      # name for the numeric values
    )
    
    # Remove the "theta_" prefix so that topic is just the number
    df_long["topic"] = df_long["topic"].str.replace("theta_", "")

    return df_long

def plot_topic_values(true_csv, simulation_pattern, save_path):
    """
    Plots a scatter plot for topic values and saves the plot.
    
    Parameters:
    - true_csv: path to the CSV containing the true topic values (with columns 'topic' and 'value').
    - simulation_pattern: a glob pattern for simulation CSV files (theta_est_grouped.csv) 
                          with folder names that include the hyperparameter values.
    - save_path: file path where the plot image will be saved.
    """
    # Read the true values CSV (assumed to have columns: topic, value)
    df_true = pd.read_csv(true_csv)
    df_true = pivot_true_thetas(df_true)
    
    # Find all simulation files using the provided pattern
    simulation_files = glob.glob(simulation_pattern)
    sim_data = []
    
    for file in simulation_files:
        # Extract hyperparameter values from the file path
        alpha_c, alpha_beta = extract_hyperparams_from_path(file)
        if alpha_c is None:
            continue  # skip if we can't parse the hyperparams
        
        # Read the simulation CSV (assumed to have columns: topic, value)
        df = pd.read_csv(file, index_col=0)
        df["leaf_id"]=df.index.tolist()
        df['alpha_c'] = alpha_c
        df['alpha_beta'] = alpha_beta
        sim_data.append(df)
    
    # Combine all simulation data
    if sim_data:
        df_sim = pd.concat(sim_data, ignore_index=True)
        df_sim = pivot_topics_wide_to_long(df_sim)
    else:
        print("No simulation data found.")
        return
    
    for leaf in df_true["leaf_id"].unique():
        
        plt.figure(figsize=(10, 6))
        
        df_true_ct = df_true[df_true["leaf_id"] == leaf]
        df_sim_ct = df_sim[df_sim["leaf_id"] == leaf]

        
        custom_palette = ["#4E79A7", "#59A14F", "#EDC948", "#B07AA1"]
        sns.scatterplot(data=df_true_ct, x='topic', y='value', color='red', s=100, label='True', zorder=10)
        sns.scatterplot(data=df_sim_ct, x='topic', y='value', hue='alpha_c', palette=custom_palette, style='alpha_beta', s=50)
        
        plt.xlabel("Topic")
        plt.ylabel("Topic Value")
        plt.title(f"Scatter Plot of Topic Values for Leaf: {leaf}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        
        cell_save_path = os.path.join(save_path, f"topic_values_celltype_{leaf}.png")
        plt.savefig(cell_save_path)
        plt.close()
        print(f"Plot saved for cell type '{leaf}' to {cell_save_path}")

def pivot_true_beta(df):
    """
    Reads the TRUE beta CSV in wide format where:
        Rows   = topics (e.g., Topic_0, Topic_1, ...)
        Columns = gene_0, gene_1, ...
    and returns a DataFrame with columns: [topic, gene, value].
    """
    # Read CSV (assuming row index is 'Topic_0', 'Topic_1', etc.)
      # index = Topic_0, Topic_1, ...
    
    # Reset index so 'Topic_0' becomes a column named 'topic'
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'topic'}, inplace=True)
    
    df_long = df.melt(
        id_vars='topic',
        var_name='gene',
        value_name='value'
    )
    
    # Convert to numeric
    df_long['gene']  = df_long['gene'].str.replace("gene_", "", regex=False)
    df_long['value'] = df_long['value'].round(2)
    
    return df_long

def pivot_sim_beta(csv_path):
    """
    Reads a wide-format CSV of simulated beta:
        Rows   = genes (e.g. 'gene_0', 'gene_1', ...)
        Columns = Topic_0, Topic_1, ...
    Returns a DataFrame with columns: [gene, topic, value], numeric IDs,
    and values rounded to two decimals.
    """
    # 1) Read CSV, using the first column as the row index (which should be gene_0, gene_1, etc.)
    df = pd.read_csv(csv_path, index_col=0)
    
    # 2) Move the row index into a 'gene' column
    df.reset_index(inplace=True)  # 'index' now becomes a column
    df.rename(columns={'index': 'gene'}, inplace=True)
    # 3) Melt (pivot) so that each Topic_* column becomes a row in 'topic'
    df_long = df.melt(
        id_vars='gene',
        var_name='topic',
        value_name='value'
    )
    
    df_long['gene'] = df_long['gene'].str.replace("gene_", "", regex=False)
    df_long['topic'] = df_long['topic'].str.replace("Topic_", "", regex=False)
    
    # 6) Round values to two decimals
    df_long['value'] = df_long['value'].round(2)
    
    return df_long

def plot_beta_values(true_csv, simulation_pattern, save_dir):
    """
    For each topic, compares the true vs. simulated beta values.
    
    - true_csv: Path to the wide CSV where rows=topics, columns=gene_*
    - simulation_pattern: Glob pattern for the simulated beta CSVs 
                         (where rows=genes, columns=Topic_*)
                         plus hyperparams in the folder name.
    - save_dir: Directory to store one plot per topic (PNG files).
    
    Each plot:
      X-axis: gene
      Y-axis: beta value
      True in red
      Simulated colored by alpha_c, shaped by alpha_beta
    """
    os.makedirs(save_dir, exist_ok=True)
    
    df_true = pd.read_csv(true_csv, index_col=0)
    df_true = pivot_true_beta(df_true)
    
    # 2) Gather & pivot all simulated betas
    sim_files = glob.glob(simulation_pattern)
    sim_data = []
    
    for file in sim_files:
        alpha_c, alpha_beta = extract_hyperparams_from_path(file)
        if alpha_c is None:
            continue
        
        df_sim = pivot_sim_beta(file)
        df_sim['alpha_c'] = alpha_c
        df_sim['alpha_beta'] = alpha_beta
        sim_data.append(df_sim)
    
    if not sim_data:
        print("No simulation data found. Check the pattern or file paths.")
        return
    
    df_sim_all = pd.concat(sim_data, ignore_index=True)
    topics = df_true['topic'].unique()
    
    # 4) Plot each topic
    for topic in topics:
        df_true_sub = df_true[df_true['topic'] == topic]
        df_sim_sub  = df_sim_all[df_sim_all['topic'] == str(topic)]
        
        plt.figure(figsize=(10, 6))
        
        # Plot true values in red
        sns.scatterplot(
            data=df_true_sub,
            x='gene',
            y='value',
            color='red',
            s=100,
            label='True'
        )
        
        custom_palette = ["#4E79A7", "#59A14F", "#EDC948", "#B07AA1"]
        sns.scatterplot(
            data=df_sim_sub,
            x='gene',
            y='value',
            hue='alpha_c',
            palette=custom_palette,
            style='alpha_beta',
            s=50
        )
        
        plt.xlabel("Gene")
        plt.ylabel("Beta Value")
        plt.title(f"Beta Values for Topic {topic}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        out_path = os.path.join(save_dir, f"beta_values_topic_{topic}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Plot saved for topic {topic} => {out_path}")

def create_binary_tree(num_leaves=8):
    """
    Create a complete binary tree with a given number of leaves.
    For a perfect binary tree, total nodes = num_leaves + (num_leaves - 1).
    Returns a dictionary representing internal nodes (with their children)
    and the total number of nodes.
    """
    total_nodes = num_leaves + (num_leaves - 1)
    tree = {}
    for i in range(total_nodes):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < total_nodes and right < total_nodes:
            tree[i] = (left, right)
    return tree, total_nodes

def get_path(node):
    """
    Returns the path from the root (node 0) to the given node.
    """
    path = []
    while node != 0:
        path.append(node)
        node = (node - 1) // 2
    path.append(0)
    return list(reversed(path))

def simulate_thetas(tree, total_nodes, num_leaves=8, path_alpha=1.0, non_path_alpha=1e-6):
    """
    For each leaf, sample a 15-dimensional theta vector from a Dirichlet distribution where:
      - For indices in the leaf's path, the alpha parameter is set to 1.
      - For indices not in the path, the alpha parameter is set to a near-zero value.
    Returns:
      - thetas: dict mapping leaf id to its theta vector.
      - paths: dict mapping leaf id to its path (list of node indices).
      - leaves: list of leaf node indices.
    """
    leaves = [i for i in range(total_nodes) if i not in tree]
    thetas = {}
    paths = {}
    for leaf in leaves:
        path = get_path(leaf)
        paths[leaf] = path
        alpha_vector = np.array([path_alpha if i in path else non_path_alpha for i in range(total_nodes)])
        theta_full = np.random.dirichlet(alpha_vector)
        thetas[leaf] = theta_full
    return thetas, paths, leaves

def create_beta_matrix(total_topics=15):
    """
    Create a beta matrix of shape (total_topics, total_topics) with one-hot rows.
    Each topic corresponds uniquely to one gene.
    """
    beta = np.eye(total_topics)
    return beta

def generate_cells(thetas, leaves, beta, cells_per_leaf=3000, total_counts=50):
    """
    For each leaf, generate cells following an LDA-style sampling procedure:
      For each cell:
        - For each of the total_counts, sample a new topic from the leaf's theta.
        - Then, sample a gene from the topic's beta distribution (here beta is one-hot,
          but the procedure is general).
      The cell's count vector is the sum over these individual draws.
    Returns:
      - count_matrix: an array of shape (total_cells, num_genes) with gene counts.
      - cell_leaf_assignments: list of leaf labels (one per cell).
      - cell_topic_assignments: list of lists containing the sampled topics for each count.
    """
    total_topics = beta.shape[0]
    num_genes = beta.shape[1]
    cell_expressions = []
    cell_leaf_assignments = []
    cell_topic_assignments = []
    
    for leaf in leaves:
        theta = thetas[leaf]
        for _ in range(cells_per_leaf):
            counts = np.zeros(num_genes, dtype=int)
            topics_assigned = []
            # For each count, sample a new topic and then sample a gene from beta.
            for _ in range(total_counts):
                topic = np.random.choice(np.arange(total_topics), p=theta)
                # Sample a gene from the topic's beta distribution.
                gene = np.random.choice(np.arange(num_genes), p=beta[topic])
                counts[gene] += 1
                topics_assigned.append(topic)
            cell_expressions.append(counts)
            cell_leaf_assignments.append(leaf)
            cell_topic_assignments.append(topics_assigned)
    
    count_matrix = np.array(cell_expressions)
    return count_matrix, cell_leaf_assignments, cell_topic_assignments

def save_csv(df, folder, filename):
    """
    Helper function to save a DataFrame as CSV in the specified folder.
    """
    path = os.path.join(folder, filename)
    df.to_csv(path, index=True)
    print(f"Saved {filename} to {folder}")

def save_beta_to_csv(beta, folder, filename="simulated_beta.csv"):
    num_genes = beta.shape[1]
    col_names = [f"gene_{i}" for i in range(num_genes)]
    df_beta = pd.DataFrame(beta, columns=col_names)
    df_beta.index.name = "topic"
    save_csv(df_beta, folder, filename)

def save_thetas_to_csv(thetas, folder, filename="simulated_thetas.csv"):
    rows = []
    total_nodes = len(next(iter(thetas.values())))
    for leaf, theta_vec in thetas.items():
        row = {"leaf_id": leaf}
        for i, val in enumerate(theta_vec):
            row[f"theta_{i}"] = val
        rows.append(row)
    df_thetas = pd.DataFrame(rows)
    df_thetas = df_thetas.sort_values(by="leaf_id").set_index("leaf_id")
    save_csv(df_thetas, folder, filename)

def save_leaf_paths_to_csv(paths, folder, filename="leaf_paths.csv"):
    rows = []
    for leaf, path in paths.items():
        rows.append({"leaf_id": leaf, "path": ",".join(map(str, path))})
    df_paths = pd.DataFrame(rows)
    df_paths = df_paths.sort_values(by="leaf_id").set_index("leaf_id")
    save_csv(df_paths, folder, filename)

def save_count_matrix_to_csv(count_matrix, cell_labels, folder, filename="simulated_count_matrix.csv"):
    """
    Save the count matrix to CSV.
    The DataFrame index is set to the cell's leaf label.
    """
    num_genes = count_matrix.shape[1]
    col_names = [f"gene_{i}" for i in range(num_genes)]
    df_counts = pd.DataFrame(count_matrix, columns=col_names, index=cell_labels)
    df_counts.index.name = "leaf_id"
    save_csv(df_counts, folder, filename)

def save_topic_mapping_to_csv(total_topics, folder, filename="topic_mapping.csv"):
    """
    Create and save a topic mapping file.
    Each row maps a topic to its unique gene.
    """
    rows = []
    for i in range(total_topics):
        rows.append({"topic": i, "gene": f"gene_{i}"})
    df_mapping = pd.DataFrame(rows).set_index("topic")
    save_csv(df_mapping, folder, filename)

def load_thetas_from_csv(folder, filename="simulated_thetas.csv"):
    """
    Load the theta vectors from CSV into a dictionary.
    """
    path = os.path.join(folder, filename)
    df = pd.read_csv(path, index_col=0)
    thetas = {}
    theta_cols = [col for col in df.columns if col.startswith("theta_")]
    for leaf, row in df.iterrows():
        thetas[int(leaf)] = row[theta_cols].values.astype(float)
    return thetas

def load_beta_from_csv(folder, filename="simulated_beta.csv"):
    """
    Load the beta matrix from CSV.
    """
    path = os.path.join(folder, filename)
    df = pd.read_csv(path, index_col=0)
    beta = df.values
    return beta

def run_simulation_sampling():
    # Specify the save folder and create it if it doesn't exist.
    save_folder = os.path.join("..", "data", "simulation")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Created directory: {save_folder}")
    else:
        print(f"Using existing directory: {save_folder}")
    
    # 1. Create a binary tree with 8 leaves.
    num_leaves = 8
    tree, total_nodes = create_binary_tree(num_leaves)
    print(f"Created binary tree with {total_nodes} nodes (internal: {len(tree)}, leaves: {num_leaves}).")
    
    # 2. Simulate theta values for each leaf using an overspecified Dirichlet.
    thetas, paths, leaves = simulate_thetas(tree, total_nodes, num_leaves, path_alpha=1.0, non_path_alpha=1e-6)
    print("Sampled theta vectors for each leaf.")
    
    # 3. Create the beta matrix (15 topics x 15 genes; one-hot per topic).
    beta = create_beta_matrix(total_topics=15)
    print("Created beta matrix (one-hot per topic).")
    
    # 4. Generate cell data: 3000 cells per leaf with 50 counts per cell following LDA sampling.
    cells_per_leaf = 3000
    total_counts = 150
    count_matrix, cell_leaf_assignments, _ = generate_cells(thetas, leaves, beta, cells_per_leaf, total_counts)
    print(f"Generated count matrix with shape {count_matrix.shape}.")
    
    # 5. Save simulated data to CSV files.
    save_beta_to_csv(beta, save_folder, filename="simulated_beta.csv")
    save_thetas_to_csv(thetas, save_folder, filename="simulated_thetas.csv")
    save_leaf_paths_to_csv(paths, save_folder, filename="leaf_paths.csv")
    save_count_matrix_to_csv(count_matrix, cell_leaf_assignments, save_folder, filename="simulated_count_matrix.csv")
    save_topic_mapping_to_csv(total_topics=15, folder=save_folder, filename="topic_mapping.csv")
    
    # 6. Read in the beta and theta files.
    loaded_beta = load_beta_from_csv(save_folder, filename="simulated_beta.csv")
    loaded_thetas = load_thetas_from_csv(save_folder, filename="simulated_thetas.csv")
    print("Loaded beta and theta values from CSV.")
    
    # 8. (Optional) Run your custom numba Gibbs sampler.
    # from custom_sampler import run_gibbs_sampler, compute_inferred_parameters
    # sampler_results = run_gibbs_sampler(hlda_count_matrix, topic_mapping=...)  # Provide the topic mapping if needed.
    # inferred_beta, inferred_thetas = compute_inferred_parameters(sampler_results)
    # Save or further analyze the inferred parameters.
    
    print("Simulation and hierarchical LDA sampling complete.")

def collapse_thetas_by_index(theta_dir):
    """
    Reads a CSV of per-leaf theta values, groups by the index (leaf ID),
    computes the mean for each leaf, and saves a collapsed CSV.

    Parameters:
    -----------
    input_csv : str
        Path to the CSV file with columns like 'theta_0', 'theta_1', etc.
        The leaf IDs must be in the CSV index.
    output_csv : str
        Path to the CSV file to save the collapsed thetas.

    Returns:
    --------
    pd.DataFrame
        DataFrame of the averaged thetas, one row per leaf ID.
    """
    # 1. Read the CSV into a DataFrame. Ensure leaf IDs are used as the index.
    df = pd.read_csv(os.path.join(theta_dir, 'theta_est.csv'), index_col=0)

    # 2. Group by the index (leaf ID) and compute the mean of each theta column.
    #    If each leaf ID only has a single row, this is effectively a no-op.
    df_collapsed = df.groupby(df.index).mean()

    # 3. Save to CSV. This yields one row per leaf ID, columns are theta_0, theta_1, etc.
    df_collapsed.to_csv(os.path.join(theta_dir, 'theta_est_grouped.csv'))

    return df_collapsed

def run_gibbs_pipeline(num_topics, num_loops, burn_in, hyperparams, output_dir, save_dir, input_file, long_data_path):

    count_matrix = pd.read_csv(input_file, index_col=0)

    cell_identities = count_matrix.index.tolist()
    gene_names = count_matrix.columns.tolist()

    long_data, identity_mapping = initialize_long_data(count_matrix,
                                                       cell_identities,
                                                       num_topics,
                                                       long_data_path)

    constants = compute_constants(long_data, num_topics)

    run_gibbs_sampling_numba(long_data, constants, hyperparams,
                       num_loops, burn_in, output_dir, identity_mapping)

    reconstruct_beta_theta(output_dir, save_dir, gene_names, cell_identities)
    collapse_thetas_by_index(save_dir)

def run_gmdf_pipeline(input_file, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    count_matrix = pd.read_csv(input_file, index_col=0)
    labels = count_matrix.index.tolist()
    topic_hierarchy = get_topic_hierarchy()

    condition_matrix, sorted_node_ids = create_condition_matrix(labels, topic_hierarchy)
    pd.DataFrame(condition_matrix, index=labels).to_csv(os.path.join(output_dir, 'condition_matrix.csv'))
    H, A = bcd_solve(
        count_matrix.values,
        condition_matrix,
        max_iter=100,
        tol=1e3,
        random_state=123,
        verbose=True
    )
    
    gmdf_theta, gmdf_beta = merge_factors_gmdf(H, A, condition_matrix)
    gmdf_theta = pd.DataFrame(gmdf_theta, index = labels)
    
    gmdf_theta.to_csv(os.path.join(output_dir, 'gmdf_theta.csv'))
    pd.DataFrame(gmdf_beta).to_csv(os.path.join(output_dir, 'gmdf_beta.csv'))
    
    grouped = gmdf_theta.groupby(gmdf_theta.index).mean()
    grouped.to_csv(os.path.join(output_dir, 'gmdf_theta_est_grouped.csv'))

def merge_factors_gmdf(H_list, A_list, condition_matrix, epsilon=1e-12):
    """
    Convert (H^j, A^j) into (theta, beta).

    Model:
      X_{c x m} ≈ ∑_{j=1..k} [ condition_matrix[i,j] * H^j[i] ] A^j.

    Final usage:
      theta[i,j] = H^j[i]  (but set to 0 if condition_matrix[i,j]==0)
      beta[j,:]  = A^j[0,:] (flattened to a row vector)

    Optionally, theta is row-normalized (each row sums to 1) and similarly for beta.
    """
    # Stack the H columns together to form theta (shape: [c, k])
    theta = np.hstack(H_list)  # Each H in H_list is of shape (c, 1)
    
    # Apply the condition matrix as an elementwise mask:
##    theta = theta * condition_matrix  # ensures theta[i,j] is 0 if condition_matrix[i,j] is 0

    # Row-normalize theta so that for each row i, sum_j theta[i,j] == 1
    theta = theta / (np.sum(theta, axis=1, keepdims=True) + epsilon)

    # Stack the A rows together to form beta (each A_list[j] is (1, m))
    beta = np.vstack([A[0, :] for A in A_list])
    beta=beta.T
    
    # Row-normalize beta so that each topic distribution sums to 1
    beta = beta / (np.sum(beta, axis=0, keepdims=True) + epsilon)

    return theta, beta


def main():

    num_topics = 15
    num_loops = 500
    burn_in = 200

    input_file = '../data/simulation/simulated_count_matrix.csv'
    gmdf_out = '../estimates/simulation/gmdf/'
    theta_true = '../data/simulation/simulated_thetas.csv'
    beta_true='../data/simulation/simulated_beta.csv'
    theta_simulation_pattern = '../estimates/simulation/run_alpha_c_*_alpha_beta_*/theta_est_grouped.csv'
    beta_simulation_pattern = '../estimates/simulation/run_alpha_c_*_alpha_beta_*/beta_est.csv'
    save_path = '../estimates/simulation/plots/'

    alpha_c_values = [1, 10, 100, 1000]
    alpha_beta_values = [1, 10, 100, 1000]

##    for alpha_c in alpha_c_values:
##        for alpha_beta in alpha_beta_values:
##            hyperparams = {'alpha_beta': alpha_beta, 'alpha_c': alpha_c}
##            
##            output_dir = f'../samples/simulation/run_alpha_c_{alpha_c}_alpha_beta_{alpha_beta}/'
##            save_dir = f'../estimates/simulation/run_alpha_c_{alpha_c}_alpha_beta_{alpha_beta}/'
##            long_data_path = save_dir
##            
##            os.makedirs(output_dir, exist_ok=True)
##            os.makedirs(save_dir, exist_ok=True)
##            
##            run_gibbs_pipeline(num_topics, num_loops, burn_in, hyperparams, 
##                               output_dir, save_dir, input_file, long_data_path)

##    plot_topic_values(theta_true, theta_simulation_pattern, save_path)
    plot_beta_values(beta_true, beta_simulation_pattern, save_path)


##    run_simulation_sampling()
##    run_gmdf_pipeline(input_file, gmdf_out)

if __name__ == "__main__":
    
    main()
    















    
