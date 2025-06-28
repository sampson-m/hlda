
import os
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from functions import(randomly_sample_pbmc,
                      sample_pbmc_by_group,
                      initialize_long_data,
                      compute_constants,
                      run_gibbs_sampling_numba,
                      reconstruct_beta_theta,
                      map_topics_to_nodes,
                      compute_dot_product_scores,
                      run_umap_and_save_plots,
                      plot_avg_theta_by_celltype,
                      plot_avg_theta_for_selected_topics,
                      plot_topic_cosine_similarity)

def run_raw_processing(raw_data_path, sampled_data_path, by_group=False):

    print('Reading in raw data')
    raw = sc.read_h5ad(raw_data_path)

    print('Sampling raw file')
    if by_group:
        sampled = sample_pbmc_by_group(raw, 5000)
    else:
        sampled = randomly_sample_pbmc(raw, 20000)

    out_dir = os.path.dirname(sampled_data_path)
    os.makedirs(out_dir, exist_ok=True)
    
    sampled.write_h5ad(sampled_data_path)

def process_sampled_data(input_file, output_folder):

    print('Processing sampled data')
    
    adata = sc.read_h5ad(input_file)
    sc.pp.filter_genes(adata, min_counts=10)
    adata_raw = adata.copy()
    
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000)
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
        index=adata_raw.obs['cell_type'], 
        columns=adata_raw.var_names
    )
    
    outfile = os.path.join(output_folder, 'count_matrix.csv')
    dense_df.to_csv(outfile)

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

def run_initial_plots_and_expression(adata_path, hlda_beta_path, hlda_theta_path, theta_subset_list, output_folder, scored_output_path):

    sc.settings.figdir = output_folder

    adata = sc.read_h5ad(adata_path)

    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000)
    top_genes = adata.var_names[adata.var["highly_variable"]]
    adata = adata[:, adata.var['highly_variable']]
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.normalize_total(adata, target_sum=1e4)

    hlda_beta_df = pd.read_csv(hlda_beta_path, index_col=0)
    plot_topic_cosine_similarity(hlda_beta_path, output_folder)

    topic_mapping = map_topics_to_nodes()
    adata = compute_dot_product_scores(adata, hlda_beta_df.values, topic_mapping, prefix="dot_product_hlda_")

    umap_folder = os.path.join(output_folder, "umap")
    os.makedirs(umap_folder, exist_ok=True)
    
    adata = run_umap_and_save_plots(
        adata, 
        os.path.join(umap_folder, "HLDA"), 
        dot_product_prefix="dot_product_hlda_"
    )

    plot_avg_theta_by_celltype(hlda_theta_path, output_folder)
    plot_avg_theta_for_selected_topics(hlda_theta_path, theta_subset_list, output_folder)

    adata.write_h5ad(scored_output_path)

def run_pipeline():

    raw_data_path = '../data/pbmc/raw.h5ad'
    sampled_data_path = '../data/pbmc/pbmc_sample/pbmc_sample.h5ad'
    data_folder = '../data/pbmc/pbmc_sample/'
    gibbs_input_file = '../data/pbmc/pbmc_sample/count_matrix.csv'

    #update when adding or removing fc topics: three sample paths below, num_topics, theta_subset_list
    #                                          get_topic_hierarchy(), map_topics_to_nodes()
    
    gibbs_sample_path = '../samples/pbmc_sample_9_fc/'
    parameter_estimate_save_dir = '../estimates/pbmc_sample_9_fc/'
    scored_output_path = os.path.join(data_folder, 'scored_9_fc.h5ad')
    
    hlda_beta_path = os.path.join(parameter_estimate_save_dir,'HLDA_beta.csv')
    hlda_theta_path = os.path.join(parameter_estimate_save_dir,'HLDA_theta.csv')
    plot_save_folder = os.path.join(parameter_estimate_save_dir,'_plots/')

    # base number of topics = 25
    num_topics=34
    num_loops=500
    burn_in=200
    hyperparams = {'alpha_beta': 1, 'alpha_c': 1}
    theta_subset_list = ["Topic_25", "Topic_26", "Topic_27", "Topic_28",
                         "Topic_29", "Topic_30", "Topic_31", "Topic_32",
                         "Topic_33"]

##    run_raw_processing(raw_data_path, sampled_data_path, by_group=True)
##    process_sampled_data(sampled_data_path, data_folder)
    
    run_gibbs_pipeline(num_topics, num_loops, burn_in, hyperparams, gibbs_sample_path,
                       parameter_estimate_save_dir, gibbs_input_file, data_folder)
    
    run_initial_plots_and_expression(sampled_data_path, hlda_beta_path, hlda_theta_path,
                                     theta_subset_list, plot_save_folder, scored_output_path)


if __name__ == "__main__":

    run_pipeline()




