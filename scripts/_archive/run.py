
from functions import (initialize_long_data,
                       compute_constants,
                       run_gibbs_sampling,
                       reconstruct_beta_theta,
                       compute_dot_product_scores,
                       run_umap_and_save_plots,
                       create_node_descendants,
                       compute_descendant_expression,
                       map_topics_to_nodes,
                       compute_perplexity,
                       fit_and_save_nmf,
                       fit_and_save_lda,
                       randomly_sample_pbmc,
                       sample_pbmc_by_group,
                       process_split_sampled_data,
                       get_topic_hierarchy,
                       infer_theta_gibbs,
                       compute_theta,
                       infer_theta_lda,
                       infer_theta_nmf,
                       run_theta_inference,
                       plot_all_methods_umap,
                       plot_umap_by_traintest,
                       plot_umap_by_celltype,
                       union_theta_and_umap,
                       plot_umap_per_celltype_traintest,
                       gibbs_update_numba,
                       run_gibbs_sampling_numba)
from gmdf import (create_condition_matrix,
                  bcd_solve,
                  merge_factors,
                  merge_gmdf_test_factors,
                  gmdf_infer_test)
import scanpy as sc
import pandas as pd
import os
import joblib
import numpy as np

def run_full_pipeline():

    # processing parameters
    raw_data_path = '../data/pbmc/raw.h5ad'
    raw_with_hierarchy_path = '../data/pbmc/raw_with_hierarchy.h5ad'
    sampled_data_path = '../data/pbmc/full_group_sample/sample.h5ad'
    hierarchy_table_path = '../data/pbmc/random_sample_hierarchy_table.csv'
    split_file_base = '../data/pbmc/full_group_sample/count_matrix'
    gibbs_output_dir = '../samples/pbmc/full_group_sample/'
    parameter_estimate_save_dir = '../estimates/pbmc/_hlda_full_group_sample/'
    theta_output_dir = '../samples/pbmc/full_group_sample_theta/'
    plot_save_path = '../estimates/pbmc/_plots/'
    model_save_path = '../estimates/pbmc/_inference/'

    # sampling parameters
    num_topics=34
    num_loops = 150
    burn_in = 50
    hyperparams = {'alpha_beta': 0.1, 'alpha_c': 0.1}

    # plot parameters
    hlda_beta_path = os.path.join(parameter_estimate_save_dir,'beta_est.csv')
    gmdf_beta_path='../estimates/pbmc/'
    plot_output_folder = os.path.join(parameter_estimate_save_dir,'_plots/')
    scored_output_path = '../data/pbmc/full_group_sample/scored.h5ad'
    gibbs_input_file = '../data/pbmc/full_group_sample/count_matrix_train.csv'
    long_data_path = '../data/pbmc/full_group_sample/'
    theta_long_data_path = '../estimates/pbmc/_hlda_full_group_sample/_inferred_theta/long_data.csv'

    # alternative models and perplexity parameters
    hlda_theta_path = os.path.join(parameter_estimate_save_dir,'theta_est.csv')
    E_path = '../data/pbmc/full_group_sample/count_matrix_test.csv'
    nmf_out = '../estimates/pbmc/_nmf/'
    lda_out = '../estimates/pbmc/_lda/'
    max_iter = 50
    random_state=42
    test_split=0.2

    print('Running raw data processing')
##    run_raw_processing(raw_data_path, sampled_data_path, by_group=True)
##    
##    print('Selecting highly variable genes, splitting cells to train and test, saving count matrices')
##    process_split_sampled_data(sampled_data_path, split_file_base, test_split)

    print('Starting Gibbs sampler')
##    run_gibbs_pipeline(num_topics, num_loops, burn_in,
##                       hyperparams, gibbs_output_dir, parameter_estimate_save_dir,
##                       gibbs_input_file, long_data_path)

    print('Computing alternative models, inferring thetas, and plotting umaps')
    infer_and_save_models(hlda_beta_path, hlda_theta_path,
                          E_path, gibbs_input_file, model_save_path, model_save_path, num_topics, max_iter,
                          random_state, hyperparams, num_loops, burn_in, theta_output_dir,
                          long_data_path, model_save_path)
    print('plotting initial umaps and computing descendant expression')
    run_initial_plots_and_expression(sampled_data_path, hlda_beta_path, model_save_path, plot_save_path, scored_output_path, model_save_path)
    print('plotting things')
    plot_train_test_umap(model_save_path, plot_save_path)

def run_gibbs_pipeline(num_topics, num_loops, burn_in, hyperparams, output_dir, save_dir, input_file):

    count_matrix = pd.read_csv(input_file, index_col=0)

    cell_identities = count_matrix.index.tolist()
    gene_names = count_matrix.columns.tolist()

    long_data, identity_mapping = initialize_long_data(count_matrix,
                                                       cell_identities,
                                                       num_topics,
                                                       long_data_path)

    constants = compute_constants(long_data, num_topics)
    
##    run_gibbs_sampling(long_data, constants, hyperparams,
##                       num_loops, burn_in, output_dir, identity_mapping)

    run_gibbs_sampling_numba(long_data, constants, hyperparams,
                       num_loops, burn_in, output_dir, identity_mapping)

    reconstruct_beta_theta(output_dir, save_dir, gene_names, cell_identities)

def run_raw_processing(raw_data_path, sampled_data_path, by_group=False):

    raw = sc.read_h5ad(raw_data_path)
    raw_with_hierarchy = add_pbmc_hierarchy(raw)
    raw_with_hierarchy.write_h5ad(raw_out_path)

    if by_group:
        sampled = sample_pbmc_by_group(raw, 5000)
    else:
        sampled = randomly_sample_pbmc(raw, 20000)
    
    
    sampled.write_h5ad(sampled_data_path)
    create_hierarchy_table(sampled, hierarchy_table_path)

def run_initial_plots_and_expression(adata_path, hlda_beta_path, model_save_path, output_folder, scored_output_path, expression_metric_path):
    """
    Run initial plotting and descendant expression computations for HLDA and GMDF.
    
    Steps:
      1) Load the AnnData object from adata_path.
      2) Filter, log-transform, and select highly variable genes.
      3) Read in both the HLDA beta and GMDF beta matrices and reindex them to the top genes.
      4) Compute dot product scores for each method (adding columns in adata.obs with
         prefixes "dot_product_hlda_" and "dot_product_gmdf_").
      5) Run UMAP (neighbors + UMAP) on each set of dot product columns and save the plots.
      6) Compute descendant expression for each method based on a node descendant mapping.
    """
    # Set figure directory for Scanpy plots
    sc.settings.figdir = output_folder

    # 1. Load AnnData
    adata = sc.read_h5ad(adata_path)

    # 2. Preprocess the data
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000)
    top_genes = adata.var_names[adata.var["highly_variable"]]
    adata = adata[:, adata.var['highly_variable']]
    sc.pp.filter_cells(adata, min_genes=1)
    sc.pp.normalize_total(adata, target_sum=1e4)

    # 3. Read in HLDA and GMDF beta matrices; reindex them to the top genes.
    gmdf_beta_df = np.load(os.path.join(model_save_path, "gmdf_beta.npy"), allow_pickle=True)
    hlda_beta_df = pd.read_csv(hlda_beta_path, index_col=0)
    # If needed, reindex hlda_beta_df to top_genes, e.g.:
    # hlda_beta_df = hlda_beta_df.reindex(top_genes)

    # 4. Compute dot product scores for HLDA & GMDF
    topic_mapping = map_topics_to_nodes()
    adata = compute_dot_product_scores(adata, hlda_beta_df.values, topic_mapping, prefix="dot_product_hlda_")
    adata = compute_dot_product_scores(adata, gmdf_beta_df.T, topic_mapping, prefix="dot_product_gmdf_")

    # 5. Create a single "umap" folder with subfolders for HLDA and GMDF
    umap_folder = os.path.join(output_folder, "umap")
    os.makedirs(umap_folder, exist_ok=True)

    # We'll run UMAP plots for HLDA prefix and GMDF prefix, storing each in a subfolder
    adata = run_umap_and_save_plots(
        adata, 
        os.path.join(umap_folder, "HLDA"),  # "umap/HLDA"
        dot_product_prefix="dot_product_hlda_"
    )
    adata = run_umap_and_save_plots(
        adata,
        os.path.join(umap_folder, "GMDF"),  # "umap/GMDF"
        dot_product_prefix="dot_product_gmdf_"
    )
    adata.write_h5ad(scored_output_path)

    # 6. Compute descendant expression for HLDA & GMDF
    node_map = create_node_descendants()
    compute_descendant_expression(adata, node_map, os.path.join(expression_metric_path, 'HLDA'), prefix="dot_product_hlda_")
    compute_descendant_expression(adata, node_map, os.path.join(expression_metric_path, 'GMDF'), prefix="dot_product_gmdf_")

    print("Initial dot-product UMAP plots and descendant expression metrics computed and saved.")



def infer_and_save_models(hlda_beta_path, hlda_theta_path,
                          E_path, train_path, nmf_out, lda_out, num_topics, max_iter,
                          random_state, hyperparams, num_loops, burn_in, output_dir,
                          long_data_path, model_save_path):
    """
    Performs inference for HLDA, GMDF, LDA, and NMF; saves the inferred models, theta matrices,
    and cell type labels to disk. This function does not return any data so you can run it independently.

    Parameters
    ----------
    hlda_beta_path : str
        Path to the HLDA beta file.
    hlda_theta_path : str
        Path to the HLDA theta (training) file.
    E_path : str
        Path to the test data file.
    train_path : str
        Path to the training data file.
    nmf_out, lda_out : str
        Output directories (or file paths) for NMF and LDA.
    num_topics : int
        Number of topics.
    max_iter : int
        Maximum number of iterations.
    random_state : int
        Random seed.
    hyperparams : dict
        Hyperparameters for HLDA.
    num_loops, burn_in : int
        Additional parameters for HLDA inference.
    output_dir : str
        Output directory for HLDA inference.
    long_data_path : str
        Additional path used in HLDA inference.
    model_save_path : str
        Base directory where inferred models, theta matrices, and labels will be saved.
    """
    # Create the model_save_path folder if it does not exist.
    os.makedirs(model_save_path, exist_ok=True)

    # --- Load Data ---
    topic_hierarchy = get_topic_hierarchy()
    hlda_beta = pd.read_csv(hlda_beta_path, index_col=0).values.T
    hlda_theta_train = pd.read_csv(hlda_theta_path, index_col=0).values
    E = pd.read_csv(E_path, index_col=0)
    train = pd.read_csv(train_path, index_col=0)

    # Here we assume that the DataFrame indices represent cell types.
    train_labels = train.index.values
    test_labels  = E.index.values

    # --- GMDF Inference on Training Data ---
    condition_matrix_train, sorted_node_ids_train = create_condition_matrix(train_labels, topic_hierarchy)
    H_train, A_train = bcd_solve(
        train.values,
        condition_matrix_train,
        max_iter=75,
        tol=1e3,
        random_state=123,
        verbose=True
    )
    gmdf_theta, gmdf_beta = merge_factors(H_train, A_train, condition_matrix_train)
    np.save(os.path.join(model_save_path, 'gmdf_beta.npy'), gmdf_beta)
    print('beta saved')

    # --- HLDA Inference ---
    # If you wish to run HLDA inference here, uncomment or adjust as needed.
    run_theta_inference(E, train, hlda_beta, num_topics, hyperparams, num_loops, burn_in,
                         output_dir, long_data_path)
    hlda_theta_test = compute_theta(output_dir)

    # --- LDA and NMF Inference on Training Data ---
    lda_theta_train, lda_beta, lda_fit = fit_and_save_lda(train.values, lda_out, num_topics, max_iter, random_state)
    nmf_theta_train, nmf_beta, nmf_fit = fit_and_save_nmf(train.values, nmf_out, num_topics, max_iter, random_state)

    # --- Inference on Test Data for LDA and NMF ---
    lda_theta_test = infer_theta_lda(E.values, lda_fit)
    nmf_theta_test = infer_theta_nmf(E.values, nmf_fit)

    # --- GMDF Inference on Test Data ---
    test_condition_matrix, sorted_node_ids_test = create_condition_matrix(test_labels, topic_hierarchy)
    H_test = gmdf_infer_test(E.values, test_condition_matrix, A_train)
    gmdf_theta_test = merge_gmdf_test_factors(H_test, A_train, test_condition_matrix)

    # --- Save Theta Matrices and Labels ---
    np.save(os.path.join(model_save_path, "hlda_theta_train.npy"), hlda_theta_train)
    np.save(os.path.join(model_save_path, "hlda_theta_test.npy"), hlda_theta_test)
    np.save(os.path.join(model_save_path, "gmdf_theta_train.npy"), gmdf_theta)
    np.save(os.path.join(model_save_path, "gmdf_theta_test.npy"), gmdf_theta_test)
    np.save(os.path.join(model_save_path, 'gmdf_beta.npy'), gmdf_beta)
    np.save(os.path.join(model_save_path, "lda_theta_train.npy"), lda_theta_train)
    np.save(os.path.join(model_save_path, "lda_theta_test.npy"), lda_theta_test)
    np.save(os.path.join(model_save_path, "nmf_theta_train.npy"), nmf_theta_train)
    np.save(os.path.join(model_save_path, "nmf_theta_test.npy"), nmf_theta_test)
    np.save(os.path.join(model_save_path, "train_labels.npy"), train_labels)
    np.save(os.path.join(model_save_path, "test_labels.npy"), test_labels)

    # --- Save Model Objects (for LDA and NMF) ---
    joblib.dump(lda_fit, os.path.join(model_save_path, "lda_model.pkl"))
    joblib.dump(nmf_fit, os.path.join(model_save_path, "nmf_model.pkl"))
    # Optionally save other model objects (e.g., for GMDF or HLDA) if needed.

    print(f"Models and theta matrices saved to {model_save_path}.")

def plot_train_test_umap(model_save_path, base_save_path):
    """
    Reads the saved theta matrices, labels, and model parameters from disk, then generates
    UMAP plots (or t-SNE if you prefer) and computes additional metrics.

    Parameters
    ----------
    model_save_path : str
        Directory where the theta matrices, labels, and models were saved.
    base_save_path : str
        Base directory where plots and metric outputs will be saved.
    """
    # --- Load Saved Data ---
    train_labels = np.load(os.path.join(model_save_path, "train_labels.npy"), allow_pickle=True)
    test_labels  = np.load(os.path.join(model_save_path, "test_labels.npy"), allow_pickle=True)
    hlda_theta_train = np.load(os.path.join(model_save_path, "hlda_theta_train.npy"), allow_pickle=True)
    hlda_theta_test  = np.load(os.path.join(model_save_path, "hlda_theta_test.npy"), allow_pickle=True)
    gmdf_theta_train = np.load(os.path.join(model_save_path, "gmdf_theta_train.npy"), allow_pickle=True)
    gmdf_theta_test  = np.load(os.path.join(model_save_path, "gmdf_theta_test.npy"), allow_pickle=True)
    lda_theta_train  = np.load(os.path.join(model_save_path, "lda_theta_train.npy"), allow_pickle=True)
    lda_theta_test   = np.load(os.path.join(model_save_path, "lda_theta_test.npy"), allow_pickle=True)
    nmf_theta_train  = np.load(os.path.join(model_save_path, "nmf_theta_train.npy"), allow_pickle=True)
    nmf_theta_test   = np.load(os.path.join(model_save_path, "nmf_theta_test.npy"), allow_pickle=True)

    umap_folder = os.path.join(base_save_path, "umap")
    os.makedirs(umap_folder, exist_ok=True)

    # HLDA
    plot_umap_per_celltype_traintest(
         theta_train = hlda_theta_train,
         theta_test  = hlda_theta_test,
         cell_types_train = train_labels,
         cell_types_test  = test_labels,
         save_path = os.path.join(umap_folder, "HLDA"),  # put in "umap/HLDA"
         method_name = "HLDA",
         random_state = 42,
         n_neighbors = 15,
         min_dist = 0.5,
         spread = 1.0
    )
    
    # GMDF
    plot_umap_per_celltype_traintest(
         theta_train = gmdf_theta_train,
         theta_test  = gmdf_theta_test,
         cell_types_train = train_labels,
         cell_types_test  = test_labels,
         save_path = os.path.join(umap_folder, "GMDF"),  # put in "umap/GMDF"
         method_name = "GMDF",
         random_state = 42,
         n_neighbors = 15,
         min_dist = 0.5,
         spread = 1.0
    )
    
    # LDA
##    plot_umap_per_celltype_traintest(
##         theta_train = lda_theta_train,
##         theta_test  = lda_theta_test,
##         cell_types_train = train_labels,
##         cell_types_test  = test_labels,
##         save_path = os.path.join(umap_folder, "LDA"),  # "umap/LDA"
##         method_name = "LDA",
##         random_state = 42,
##         n_neighbors = 15,
##         min_dist = 0.5,
##         spread = 1.0
##    )
##    
##    # NMF
##    plot_umap_per_celltype_traintest(
##         theta_train = nmf_theta_train,
##         theta_test  = nmf_theta_test,
##         cell_types_train = train_labels,
##         cell_types_test  = test_labels,
##         save_path = os.path.join(umap_folder, "NMF"),  # "umap/NMF"
##         method_name = "NMF",
##         random_state = 42,
##         n_neighbors = 15,
##         min_dist = 0.5,
##         spread = 1.0
##    )

    print("Train/test UMAP plots generated for HLDA, GMDF, LDA, and NMF.")


if __name__ == "__main__":

    run_full_pipeline()
    






    

