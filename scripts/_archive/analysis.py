"""
analysis.py

A cleaned-up script that organizes PCA, t-SNE, train/test UMAP, and node graphs.
- Removes heatmap output entirely.
- Performs PCA only once at the global level (not repeated for cell-type coloring).
- Leaves a single t-SNE for both global (train/test, cell type) and cell-type analysis.

Usage:
    python analysis.py
"""

import os
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from functions import get_topic_hierarchy, create_hierarchy_dict, map_topics_to_nodes, reverse_mapping

# Check topics
def investigate_node_scores(
    adata,
    cell_to_node,
    node_to_topic,
    node_of_interest,
    dot_product_col,
    cell_types=None,
    threshold=50.0,
    percentile=90.0,
    show_plot=False,
    save_folder=None
):
    """
    Investigate the distribution of a chosen dot-product column in a specific node.

    Parameters
    ----------
    adata : anndata.AnnData
        Contains dot product columns in adata.obs (e.g., "dot_product_hlda_nk").
    cell_to_node : np.ndarray of shape (n_cells,)
        Mapping of each cell to a node ID (e.g., last node in path).
    node_of_interest : int
        The node ID to investigate.
    dot_product_col : str
        The dot product column name (e.g. "dot_product_hlda_natural_killer_cell").
    cell_types : np.ndarray, optional
        If provided, an array of cell-type labels. We'll show composition in this node.
    threshold : float, optional
        For fraction-above-threshold calculation.
    percentile : float, optional
        For high percentile (e.g. 90th) usage calculation.
    show_plot : bool, optional
        If True, display the histogram (non-blocking) instead of purely saving/closing.
    save_folder : str, optional
        If provided, save the histogram as a PNG in this folder.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    idx = np.where(cell_to_node == node_of_interest)[0]
    if len(idx) == 0:
        print(f"No cells found in node {node_of_interest}!")
        return

    scores = adata.obs[dot_product_col].values[idx]

    mean_score = np.mean(scores)
    median_score = np.median(scores)
    frac_above_threshold = np.mean(scores > threshold)
    high_percentile_val = np.percentile(scores, percentile)

    print(f"=== Node {node_to_topic.get(node_of_interest)}-{dot_product_col} Investigation ===")
    print(f"Total cells in node: {len(idx)}")
    print(f"Mean {dot_product_col}: {mean_score:.2f}")
    print(f"Median {dot_product_col}: {median_score:.2f}")
    print(f"Fraction of cells above {threshold}: {frac_above_threshold:.2%}")
    print(f"{percentile:.0f}th percentile: {high_percentile_val:.2f}")

    if cell_types is not None:
        node_cell_types = cell_types[idx]
        unique, counts = np.unique(node_cell_types, return_counts=True)
        print("\nCell-type composition in this node:")
        for ut, c in zip(unique, counts):
            print(f"  {ut}: {c} cells")
    print("")

    # Plot histogram
    plt.figure(figsize=(6,4))
    plt.hist(scores, bins=30, edgecolor='k', alpha=0.7)
    plt.axvline(threshold, color='r', linestyle='--', label=f"Threshold = {threshold}")
    plt.title(f"Node {node_to_topic.get(node_of_interest)} Distribution of {dot_product_col}")
    plt.xlabel(f"{dot_product_col} Score")
    plt.ylabel("Cell Count")
    plt.legend()
    plt.tight_layout()

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        filename = f"node_{node_of_interest}_{dot_product_col}.png"
        plt.savefig(os.path.join(save_folder, filename), dpi=300)

    # Non-blocking display if show_plot=True
    if show_plot:
        plt.show(block=False)
        plt.pause(0.5)
    plt.close()

# -----------------------
# Helper: sanitize filenames
# -----------------------
def sanitize_filename(s):
    for char in [":", "(", ")", "/", "\\", " "]:
        s = s.replace(char, "_")
    return s

# -----------------------
# 1. Loading & Merging Data
# -----------------------
def load_and_merge_data(model_save_path, method_name):
    """
    Loads train/test theta arrays and label arrays, merges them,
    and creates a 'train_test_labels' array.
    """
    train_labels = np.load(os.path.join(model_save_path, "train_labels.npy"), allow_pickle=True)
    test_labels  = np.load(os.path.join(model_save_path, "test_labels.npy"), allow_pickle=True)

    if method_name == "HLDA":
        theta_train = np.load(os.path.join(model_save_path, "hlda_theta_train.npy"), allow_pickle=True)
        theta_test  = np.load(os.path.join(model_save_path, "hlda_theta_test.npy"), allow_pickle=True)
    elif method_name == "GMDF":
        theta_train = np.load(os.path.join(model_save_path, "gmdf_theta_train.npy"), allow_pickle=True)
        theta_test  = np.load(os.path.join(model_save_path, "gmdf_theta_test.npy"), allow_pickle=True)
    elif method_name == "LDA":
        theta_train = np.load(os.path.join(model_save_path, "lda_theta_train.npy"), allow_pickle=True)
        theta_test  = np.load(os.path.join(model_save_path, "lda_theta_test.npy"), allow_pickle=True)
    elif method_name == "NMF":
        theta_train = np.load(os.path.join(model_save_path, "nmf_theta_train.npy"), allow_pickle=True)
        theta_test  = np.load(os.path.join(model_save_path, "nmf_theta_test.npy"), allow_pickle=True)
    else:
        raise ValueError(f"Unknown method_name: {method_name}")

    theta_all = np.vstack([theta_train, theta_test])
    cell_types_all = np.concatenate([train_labels, test_labels], axis=0)
    train_test_labels = np.array(["train"] * len(train_labels) + ["test"] * len(test_labels))
    return theta_all, cell_types_all, train_test_labels

# -----------------------
# 2. Plotting Functions
# -----------------------
def plot_explained_variance(exp_var, title_suffix, save_folder):
    """
    Plot cumulative explained variance from PCA with integer x-axis ticks.
    """
    cum_var = np.cumsum(exp_var)
    plt.figure(figsize=(8,6))
    x_vals = range(1, len(cum_var) + 1)
    plt.plot(x_vals, cum_var, marker='o')
    plt.xticks(x_vals)
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(f"PCA Explained Variance ({title_suffix})")
    plt.grid(True)
    os.makedirs(save_folder, exist_ok=True)
    filename = os.path.join(save_folder, f"pca_explained_variance_{sanitize_filename(title_suffix)}.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def plot_embedding(embedding, labels, title, save_folder, s=10, alpha=0.1,
                   legend_fontsize="x-small", legend_markerscale=0.8, right_margin=0.65):
    """
    Scatter plot an embedding colored by the provided labels.
    """
    plt.figure(figsize=(8,6))
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        plt.scatter(embedding[idx, 0], embedding[idx, 1], label=lab, s=s, alpha=alpha)
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0,
               fontsize=legend_fontsize, markerscale=legend_markerscale)
    plt.tight_layout(rect=[0, 0, right_margin, 1])
    os.makedirs(save_folder, exist_ok=True)
    filename = os.path.join(save_folder, f"embedding_{sanitize_filename(title)}.png")
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_node_graph_hierarchy(graph, node_metric, title, save_path, cmap='viridis', node_size=500, prog='dot'):
    """
    Plot a hierarchical layout of a NetworkX graph using pydot layout.
    """
    pos = graphviz_layout(graph, prog=prog)
    nodes = list(graph.nodes())
    values = [node_metric.get(node, 0) for node in nodes]
    plt.figure(figsize=(10,8))
    nodes_draw = nx.draw_networkx_nodes(graph, pos, node_color=values,
                                        cmap=plt.get_cmap(cmap), node_size=node_size,
                                        linewidths=1, edgecolors='black')
    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_color='white',
                            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))
    plt.title(title)
    plt.colorbar(nodes_draw)
    plt.axis("off")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_topic_node_graph_from_dot(adata, cell_to_node, topic_name, method, dot_product_prefix, graph, save_path, cmap='viridis', node_size=500):
    """
    For a given topic, extract the dot product scores from adata.obs (using prefix),
    compute average score per node, and plot a node graph.
    """
    col_name = f"{dot_product_prefix}{topic_name}"
    if col_name not in adata.obs.columns:
        print(f"Column {col_name} not found in adata.obs. Skipping topic {topic_name}.")
        return
    dot_product_vector = adata.obs[col_name].values
    node_expr = compute_node_metric(dot_product_vector, cell_to_node)
    title = f"{method} {topic_name} (Dot Product)"
    plot_node_graph_hierarchy(graph, node_expr, title, save_path, cmap=cmap, node_size=node_size)

def plot_topic_node_graph_from_theta(theta, cell_to_node, topic_idx, topic_name, method, graph, save_path, cmap='viridis', node_size=500):
    """
    For a given topic (by column index in theta), compute average theta per node and plot a node graph.
    """
    topic_values = theta[:, topic_idx]
    node_expr = compute_node_metric(topic_values, cell_to_node)
    title = f"{method} {topic_name} (Theta)"
    plot_node_graph_hierarchy(graph, node_expr, title, save_path, cmap=cmap, node_size=node_size)

# -----------------------
# 3. Utility Functions
# -----------------------
def save_train_test_neighbor_fractions(fractions, train_test_labels, out_csv):
    train_mask = (train_test_labels == "train")
    test_mask  = (train_test_labels == "test")
    train_avg = np.mean(fractions[train_mask]) if np.any(train_mask) else 0.0
    test_avg = np.mean(fractions[test_mask]) if np.any(test_mask) else 0.0
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("Label,AvgNeighborFraction\n")
        f.write(f"train,{train_avg:.4f}\n")
        f.write(f"test,{test_avg:.4f}\n")

def create_topic_graph(hierarchy_dict):
    G = nx.DiGraph()
    for parent, children in hierarchy_dict.items():
        G.add_node(parent)
        for child in children:
            G.add_node(child)
            G.add_edge(parent, child)
    return G

def create_cell_to_node(cell_types, topic_hierarchy):
    cell_to_node = []
    for ct in cell_types:
        path = topic_hierarchy.get(ct)
        if path is None:
            raise ValueError(f"Cell type '{ct}' not found in topic_hierarchy")
        cell_to_node.append(path[-1])
    return np.array(cell_to_node)

# ---------------------------
# 2. PCA & t-SNE Helper Functions
# ---------------------------
def perform_pca(theta, n_components=20):
    """
    Run PCA on the theta matrix.
    """
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(theta)
    explained_variance = pca.explained_variance_ratio_
    return pca, pcs, explained_variance

def perform_tsne(theta, n_components=2, random_state=42):
    """
    Run t-SNE on the theta matrix.
    """
    tsne = TSNE(n_components=n_components, random_state=random_state)
    embedding = tsne.fit_transform(theta)
    return embedding

def compute_topic_entropy(theta, cell_types):
    unique_types = np.unique(cell_types)
    n_topics = theta.shape[1]
    topic_entropies = []
    for t in range(n_topics):
        weights = np.array([np.sum(theta[cell_types == ct, t]) for ct in unique_types])
        if weights.sum() > 0:
            probs = weights / weights.sum()
        else:
            probs = np.zeros_like(weights)
        topic_entropies.append(entropy(probs))
    return topic_entropies

def compute_same_label_neighbors(embedding, labels, n_neighbors=10):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)
    fractions = []
    for i, neighbors in enumerate(indices[:, 1:]):  # skip self
        frac = np.mean(labels[neighbors] == labels[i])
        fractions.append(frac)
    return np.array(fractions)

def compute_node_metric(values, cell_to_node):
    unique_nodes = np.unique(cell_to_node)
    node_metric = {}
    for node in unique_nodes:
        idx = np.where(cell_to_node == node)[0]
        node_metric[node] = np.median(values[idx]) if idx.size > 0 else 0.0
    return node_metric

def analyze_pca_with_scaling(theta_ct, method_name, cell_type_name, pca_folder):
    """
    Remove zero-only columns, scale the matrix, perform PCA,
    and plot the cumulative explained variance.

    Parameters
    ----------
    theta_ct : np.ndarray, shape (n_cells_ct, n_topics)
        Subset of theta matrix for the current cell type.
    method_name : str
        E.g. "HLDA" or "GMDF"
    cell_type_name : str
        The name of the current cell type (for plot titles/filenames).
    pca_folder : str
        Folder in which to save the PCA plot.

    Returns
    -------
    pcs : np.ndarray or None
        The principal component scores for each cell (shape (n_cells_ct, n_components)).
        Returns None if there's not enough data (e.g., all zero or single row).
    """
    # 1) Remove columns that are all zeros
    nonzero_cols = np.where(np.any(theta_ct != 0, axis=0))[0]
    if len(nonzero_cols) < 2 or theta_ct.shape[0] < 2:
        # Not enough data to perform PCA meaningfully
        return None

    sub_theta = theta_ct[:, nonzero_cols]

    # 2) Scale
    scaler = StandardScaler()
    sub_theta_scaled = scaler.fit_transform(sub_theta)

    # 3) PCA
    pca = PCA(n_components=min(sub_theta_scaled.shape))
    pcs = pca.fit_transform(sub_theta_scaled)
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)

    # 4) Plot cumulative explained variance
    x_vals = range(1, len(cum_var) + 1)
    plt.figure(figsize=(6,4))
    plt.plot(x_vals, cum_var, marker='o', linestyle='-')
    plt.title(f'Cumulative PCA Explained Variance - {method_name} - {cell_type_name}')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance')
    plt.xticks(x_vals)
    plt.tight_layout()

    # Save plot
    plot_filename = f"{method_name}_{cell_type_name}_pca.png"
    os.makedirs(pca_folder, exist_ok=True)
    plt.savefig(os.path.join(pca_folder, plot_filename), dpi=300)
    plt.close()

    return pcs


# -----------------------
# 4. Analysis Functions
# -----------------------
def analyze_theta_overall(theta_all, cell_types_all, train_test_labels, method_name, save_folder):
    """
    Global analysis on the merged theta matrix:
      - Compute PCA once and save explained variance plot.
      - Run t-SNE colored by train/test.
      - Compute neighbor fraction and save CSV.
      - Run t-SNE colored by cell type.
      - Plot topic entropy.
    """
    print(f"=== Overall Analysis for {method_name} ===")
    method_folder = os.path.join(save_folder, "pca", method_name)
    os.makedirs(method_folder, exist_ok=True)
    
    # Global PCA (computed once)
    _, pcs, exp_var = perform_pca(theta_all, n_components=20)
    plot_explained_variance(exp_var, title_suffix=f"{method_name}_Overall", save_folder=method_folder)
    
    # t-SNE colored by train/test (save in tsne folder)
    tsne_folder = os.path.join(save_folder, "tsne", method_name)
    os.makedirs(tsne_folder, exist_ok=True)
    tsne_emb_tt = perform_tsne(theta_all, random_state=42)
    plot_embedding(tsne_emb_tt, train_test_labels, title=f"{method_name}: t-SNE (Train/Test)", save_folder=tsne_folder)
    
    # Neighbor fraction CSV using PCA coords
    same_label_frac = compute_same_label_neighbors(pcs, train_test_labels, n_neighbors=10)
    out_csv_overall = os.path.join(method_folder, f"{method_name}_Overall_train_test_neighbor_fraction.csv")
    save_train_test_neighbor_fractions(same_label_frac, train_test_labels, out_csv_overall)
    
    # t-SNE colored by cell type (save in tsne folder)
    tsne_emb_ct = perform_tsne(theta_all, random_state=42)
    plot_embedding(tsne_emb_ct, cell_types_all, title=f"{method_name}: t-SNE (Cell Type)", save_folder=tsne_folder)
    
    # Topic entropy plot (save in pca folder)
##    topic_ent = compute_topic_entropy(theta_all, cell_types_all)
##    plt.figure(figsize=(8,6))
##    plt.bar(range(len(topic_ent)), topic_ent)
##    plt.xlabel("Topic")
##    plt.ylabel("Entropy")
##    plt.title(f"Per-Topic Entropy ({method_name})")
##    filename = os.path.join(method_folder, f"topic_entropy_{method_name}.png")
##    plt.savefig(filename, dpi=300, bbox_inches="tight")
##    plt.close()

def analyze_theta_by_cell_type(theta_all, cell_types_all, train_test_labels, method_name, save_folder):
    """
    For each unique cell type:
      1) Run a local t-SNE (colored by train/test) and save the plot.
      2) Remove zero-only columns, scale, perform PCA, plot cumulative explained variance,
         and return PCs for neighbor fraction calculation.
      3) Compute neighbor fractions per cell type and output a CSV.
    """
    unique_ct = np.unique(cell_types_all)

    # Folders for t-SNE and PCA outputs
    tsne_folder = os.path.join(save_folder, "tsne", method_name, "celltype")
    pca_folder  = os.path.join(save_folder, "pca", method_name, "celltype")
    os.makedirs(tsne_folder, exist_ok=True)
    os.makedirs(pca_folder, exist_ok=True)

    csv_data = []
    for ct in unique_ct:
        idx = np.where(cell_types_all == ct)[0]
        theta_ct = theta_all[idx, :]
        tt_labels_ct = train_test_labels[idx]

        print(f"=== {method_name} Analysis for Cell Type: {ct} ===")

        # 1) Local t-SNE
        tsne_emb = perform_tsne(theta_ct, random_state=42)
        plot_embedding(
            tsne_emb,
            tt_labels_ct,
            title=f"{method_name} - {ct}: t-SNE (Train/Test)",
            save_folder=tsne_folder
        )

        # 2) PCA with scaling & zero-col removal => cumulative explained variance
        pcs = analyze_pca_with_scaling(theta_ct, method_name, ct, pca_folder)

        # 3) Compute neighbor fractions if we have valid PCs
        if pcs is not None:
            same_label_frac = compute_same_label_neighbors(pcs, tt_labels_ct, n_neighbors=10)
            train_mask = (tt_labels_ct == "train")
            test_mask  = (tt_labels_ct == "test")
            train_avg = np.mean(same_label_frac[train_mask]) if np.any(train_mask) else 0.0
            test_avg = np.mean(same_label_frac[test_mask]) if np.any(test_mask) else 0.0
            csv_data.append({"cell_type": ct, "train_avg": train_avg, "test_avg": test_avg})
        else:
            # Not enough data => skip neighbor fraction
            csv_data.append({"cell_type": ct, "train_avg": 0.0, "test_avg": 0.0})

    # Save neighbor fraction data as CSV
    out_csv = os.path.join(pca_folder, f"{method_name}_cell_type_neighbor_fractions.csv")
    with open(out_csv, "w") as f:
        f.write("cell_type,train_avg,test_avg\n")
        for row in csv_data:
            f.write(f"{row['cell_type']},{row['train_avg']:.4f},{row['test_avg']:.4f}\n")

    print(f"Cell-type analysis complete for {method_name}. TSNE and PCA outputs saved in '{save_folder}'.")

def export_unique_celltype_combinations(adata, level_cols, celltype_col, output_csv_path):
    """
    Extracts and saves the unique combinations of the specified cell type level columns 
    and the raw celltype column to a CSV file.

    Parameters:
        adata (AnnData): The AnnData object containing your single-cell data.
        level_cols (list of str): List of column names representing the cell type levels (e.g. ["celltype_level_1", "celltype_level_2", "celltype_level_3"]).
        celltype_col (str): The column name with the raw celltype.
        output_csv_path (str): The file path where the CSV will be saved.
    """
    # Extract only the relevant columns from adata.obs
    df = adata.obs[level_cols + [celltype_col]].drop_duplicates().reset_index(drop=True)
    
    # Save the unique combinations to a CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"Unique cell type combinations saved to {output_csv_path}")


###############################################################################
# 7. Main
###############################################################################
if __name__ == "__main__":


# Example usage:
    adata = sc.read_h5ad('../data/pbmc/raw.h5ad')
    export_unique_celltype_combinations(adata, ["celltype_level_1", "celltype_level_2", "celltype_level_3"], "cell_type", "../data/pbmc/full_data/unique_celltypes.csv")
    
##    model_save_path = "../estimates/pbmc/_inference/"
##    base_plot_folder = "../estimates/pbmc/_plots/"
##
##    methods = ["HLDA", "GMDF"]
##
##    # Load HLDA & GMDF train theta and training labels, plus adata for node graphs
##    hlda_theta_train = np.load(os.path.join(model_save_path, "hlda_theta_train.npy"), allow_pickle=True)
##    gmdf_theta_train = np.load(os.path.join(model_save_path, "gmdf_theta_train.npy"), allow_pickle=True)
##    train_labels = np.load(os.path.join(model_save_path, "train_labels.npy"), allow_pickle=True)
##    adata = sc.read_h5ad("../data/pbmc/full_group_sample/scored.h5ad")
##    
##
##    
##    # Global analysis for each method
##    for method in methods:
##        theta_all, cell_types_all, train_test_labels = load_and_merge_data(model_save_path, method)
##        analyze_theta_overall(theta_all, cell_types_all, train_test_labels, method, base_plot_folder)
##        analyze_theta_by_cell_type(theta_all, cell_types_all, train_test_labels, method, base_plot_folder)
##
##    # Build topic hierarchy and node graphs
##    topic_hierarchy = get_topic_hierarchy()
##    cell_to_node = create_cell_to_node(train_labels, topic_hierarchy)
##    hierarchy_dict = create_hierarchy_dict(topic_hierarchy)
##    topic_graph = create_topic_graph(hierarchy_dict)
##    topic_to_node_mapping = map_topics_to_nodes()
##    
##    # Dot product prefixes for methods
##    method_prefixes = {
##        "HLDA": "dot_product_hlda_",
##        "GMDF": "dot_product_gmdf_"
##    }
##    
##    # Create node graphs for each topic.
##    for method, prefix in method_prefixes.items():
##        # Dot product node graphs from adata
##        for topic_name in topic_to_node_mapping.keys():
##            save_path_dot = os.path.join(base_plot_folder, "node_graphs", method,
##                                         f"{sanitize_filename(topic_name)}_dot_node_graph.png")
##            os.makedirs(os.path.dirname(save_path_dot), exist_ok=True)
##            plot_topic_node_graph_from_dot(adata, cell_to_node, topic_name, method, prefix,
##                                           topic_graph, save_path_dot, cmap='viridis', node_size=600)
##            print(f"Saved dot product node graph for {topic_name} ({method}) to {save_path_dot}")
##        
##        # Theta node graphs: use corresponding theta matrix.
##        theta = hlda_theta_train if method == "HLDA" else gmdf_theta_train
##        for topic_name, idx_list in topic_to_node_mapping.items():
##            topic_idx = idx_list[0]
##            save_path_theta = os.path.join(base_plot_folder, "node_graphs", method,
##                                           f"{sanitize_filename(topic_name)}_theta_node_graph.png")
##            os.makedirs(os.path.dirname(save_path_theta), exist_ok=True)
##            plot_topic_node_graph_from_theta(theta, cell_to_node, topic_idx, topic_name, method,
##                                             topic_graph, save_path_theta, cmap='viridis', node_size=600)
##            print(f"Saved theta node graph for {topic_name} ({method}) to {save_path_theta}")
##
##    node_list = [3,4,9,10,25,27,13]
##    for node_id in node_list:
##        investigate_node_scores(
##            adata=adata,
##            cell_to_node=cell_to_node,
##            node_to_topic=reverse_mapping(map_topics_to_nodes()),
##            node_of_interest=node_id,
##            dot_product_col='dot_product_hlda_natural killer cell',
##            save_folder='../estimates/pbmc/_plots/node_check/'
##        )

