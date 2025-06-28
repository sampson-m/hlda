##
##import scanpy as sc
##import pandas as pd
##from scipy.io import mmread
##
##def main():
##    # File paths with the specified prefix
##    mtx_file = "../data/pbmc/zheng/matrix.mtx"
##    annotation_file = "../data/pbmc/zheng/68k_pbmc_barcodes_annotation.tsv"
##    gene_file = "../data/pbmc/zheng/genes.tsv"  # Adjust if needed
##
##    # 1. Read the count matrix (MTX format)
##    print("Loading matrix file:", mtx_file)
##    X = mmread(mtx_file).tocsr()
##    num_genes, num_barcodes = X.shape
##    print(f"Matrix dimensions: {num_genes} genes x {num_barcodes} cells")
##
##    # 2. Read the annotation file
##    print("Loading annotation file:", annotation_file)
##    # Assumes the file is tab-separated and contains columns "barcode" and "cell_type"
##    annotations = pd.read_csv(annotation_file, sep="\t")
##    print(f"Annotation file has {annotations.shape[0]} rows")
##    
##    if annotations.shape[0] != num_barcodes:
##        print("Warning: Number of annotation rows does not match the number of cells in the matrix!")
##    
##    # 3. Read the gene names file
##    try:
##        print("Loading gene names file:", gene_file)
##        genes = pd.read_csv(gene_file, sep="\t", header=None)
##        # Assume gene names are in the first column; adjust if necessary
##        genes.columns = ["gene_id", "gene_name"]
##    except Exception as e:
##        print("Error reading gene names file:", e)
##        # If no gene file is available, create default gene names
##        genes = pd.DataFrame({"gene_name": [f"gene_{i}" for i in range(1, num_genes+1)]})
##    
##    if genes.shape[0] != num_genes:
##        print("Warning: Number of gene names does not match the number of genes in the matrix!")
##    
##    # 4. Create the AnnData object
##    print("Creating AnnData object")
##    adata = sc.AnnData(X=X.T)
##    
##    # Set cell (obs) metadata.
##    # We assume the annotation file contains a "barcode" column.
##    if "barcodes" in annotations.columns:
##        annotations = annotations.set_index("barcodes")
##    else:
##        # If no barcode column is provided, use a generated index.
##        annotations.index = [f"cell_{i}" for i in range(annotations.shape[0])]
##    adata.obs = annotations.copy()
##    
##    # Set gene (var) metadata using gene names as index.
##    adata.var = genes.set_index("gene_name")
##    
##    # 5. Update T cell annotations
##    # If cells have detailed T cell labels, change them to simply "T cells".
##    # Detailed labels to be re-assigned include:
##    detailed_tcell_labels = [
##        "CD4+ T Helper2",
##        "CD4+/CD25+ T Reg",
##        "CD4+/CD25 T Reg",
##        "CD4+/CD45RA+/CD25- Naive T",
##        "CD4+/CD45RO+ Memory",
##        "CD8+/CD45RA+ Naive Cytotoxic"
##    ]
##    
##    if "celltype" in adata.obs.columns:
##        # Using lower-case comparison to be safe
##        dendritic_mask = adata.obs["celltype"].str.lower() == "dendritic"
##        num_dendritic = dendritic_mask.sum()
##        if num_dendritic > 0:
##            print(f"Removing {num_dendritic} dendritic cells from the data")
##            adata = adata[~dendritic_mask].copy()
##        else:
##            print("No dendritic cells found to remove.")
##    else:
##        print("Warning: 'celltype' column not found in annotations; cannot remove dendritic cells.")
##
##    if "celltype" in adata.obs.columns:
##        # Find rows where cell_type is one of the detailed labels.
##        tcell_mask = adata.obs["celltype"].isin(detailed_tcell_labels)
##        num_detailed = tcell_mask.sum()
##        if num_detailed > 0:
##            print(f"Updating annotation for {num_detailed} cells from detailed T cell labels to 'T cells'")
##            adata.obs.loc[tcell_mask, "celltype"] = "T cell"
##        else:
##            print("No cells with detailed T cell labels found in the annotations.")
##    else:
##        print("Warning: 'celltype' column not found in annotations; skipping T cell update.")
##    
##    # 6. Write out the AnnData object to an h5ad file
##    output_file = "../data/pbmc/zheng/output_raw.h5ad"
##    print("Writing AnnData to", output_file)
##    adata.write(output_file)
##    print("Done.")
##
##if __name__ == "__main__":
##    main()

##import os
##import glob
##import pandas as pd
##
##def extract_top_genes_for_topics(parent_dir):
##    """
##    Loops through subfolders named 'zheng_*_fc', reads 'HLDA_beta.csv' in each,
##    finds the top 5 genes for each topic, and saves them to 'top5_genes_by_topic.csv'.
##    """
##    # Find all subdirectories matching 'zheng_*_fc'
##    subfolders = sorted(glob.glob(os.path.join(parent_dir, "zheng_*_fc")))
##    
##    for folder in subfolders:
##        beta_path = os.path.join(folder, "HLDA_beta.csv")
##        if not os.path.exists(beta_path):
##            print(f"File not found, skipping: {beta_path}")
##            continue
##        
##        print(f"Processing {beta_path}...")
##        # Read the beta file, assuming genes are in rows and topics in columns
##        # Use index_col=0 if the first column is gene names
##        df = pd.read_csv(beta_path, index_col=0)
##        
##        # Create a DataFrame to store the top 5 genes for each topic
##        # We'll make rows = topics, columns = rank_1 through rank_5
##        top_genes_df = pd.DataFrame(
##            columns=["rank_1", "rank_2", "rank_3", "rank_4", "rank_5"],
##            index=df.columns  # each topic becomes a row
##        )
##        
##        # For each topic (column in df), find the top 5 genes
##        for topic in df.columns:
##            # nlargest(5) returns the top 5 row indices (genes) for this column
##            top5_genes = df[topic].nlargest(5).index.tolist()
##            # Store them in the new DataFrame
##            top_genes_df.loc[topic] = top5_genes
##        
##        # Write out the top genes
##        output_csv = os.path.join(folder, "top5_genes_by_topic.csv")
##        top_genes_df.to_csv(output_csv)
##        print(f"  => Saved to {output_csv}")
##
##if __name__ == "__main__":
##    # Replace '.' with the actual path that contains your zheng_*_fc folders
##    parent_directory = "../estimates/"
##    extract_top_genes_for_topics(parent_directory)

import pandas as pd

def load_gene_means(gene_means_path="../data/hsim/gene_means_matrix.csv"):
    gene_means = pd.read_csv(gene_means_path, index_col=0)
    gene_means.index = gene_means.index.str.strip()
    gene_means.columns = gene_means.columns.str.strip()
    print("Loaded gene_means with shape:", gene_means.shape)
    return gene_means

def load_beta_matrix(beta_matrix_path="../estimates/hsim/HLDA_beta.csv"):
    beta = pd.read_csv(beta_matrix_path, index_col=0)
    # Set expected topic names for columns and clean up names.
    beta.columns = ['root', 'AB_parent', 'A', 'B', 'C']
    beta.columns = [col.strip() for col in beta.columns]
    beta.index = beta.index.str.strip()
    print("Loaded beta matrix with shape:", beta.shape)
    print("Beta matrix topics (columns):", list(beta.columns))
    print("Beta matrix gene index (first 10):", list(beta.index)[:10])
    return beta

def compare_topicC(gene_means, beta):
    # Filter for genes that are DE in group C.
    de_genes_C = gene_means[gene_means["DE_group"].apply(
        lambda s: "C" in s.split(",") if isinstance(s, str) and s != "" else False
    )]
    print(f"Found {de_genes_C.shape[0]} genes marked DE in group C.")
    
    # Use column "C" from gene_means as the gene mean for group C and get the DE_factor.
    de_info = de_genes_C[["C", "DE_factor"]].copy().rename(columns={"C": "gene_mean_C"})
    gene_names = de_info.index
    print("Number of genes (DE in C) from gene_means:", len(gene_names))
    
    # Check for common genes between the gene means and the beta matrix.
    common_genes = set(gene_names).intersection(set(beta.index))
    if len(common_genes) < len(gene_names):
        missing = set(gene_names) - set(beta.index)
        print(f"Warning: {len(missing)} gene(s) from gene_means not found in beta matrix. Example missing: {list(missing)[:5]}")
    else:
        print("All gene names found in beta matrix.")
    
    # For genes present in beta, get beta probability for topic "C"
    try:
        beta_C = beta.loc[list(gene_names), "C"]
    except Exception as e:
        print("Error accessing beta values for topic 'C':", e)
        raise
    de_info["beta_prob_C"] = beta_C

    # Compare beta for topic C to each other topic (root, AB_parent, A, B)
    for topic in beta.columns:
        if topic != "C":
            try:
                fc = beta_C / beta.loc[list(gene_names), topic]
                de_info[f"beta_fold_change_vs_{topic}"] = fc
            except Exception as e:
                print(f"Error computing fold change for topic {topic}:", e)
                raise

    return de_info

def main():
    gene_means = load_gene_means()
    beta = load_beta_matrix()
    de_info = compare_topicC(gene_means, beta)
    output_path = "../estimates/hsim/comparison_topicC.csv"
    de_info.to_csv(output_path)
    print(f"Comparison results saved to '{output_path}'.")

if __name__ == '__main__':
    main()
























