import argparse
import scanpy as sc
import pandas as pd
from pathlib import Path
import numpy as np
import scipy.sparse
import yaml


def main():
    parser = argparse.ArgumentParser(description="Preprocess h5ad file for topic modeling.")
    parser.add_argument("--input_h5ad", type=str, required=True, help="Path to input h5ad file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (must match a key in the config file)")
    parser.add_argument("--config_file", type=str, default="../dataset_identities.yaml", help="Path to dataset identity config YAML file")
    args = parser.parse_args()

    # Load identity topics from config file
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    if args.dataset not in config:
        raise ValueError(f"Dataset '{args.dataset}' not found in config file {args.config_file}")
    identity_topics = config[args.dataset]['identities']
    print(f"Loaded {len(identity_topics)} cell identities for dataset '{args.dataset}': {identity_topics}")

    data_dir = Path(args.input_h5ad)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load 10x data
    print(f"Reading 10x data from {data_dir}")
    adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True, make_unique=True)
    print(f"Shape after loading: {adata.shape} (cells, genes)")

    # Check if orientation is correct (cells x genes)
    n_barcodes = sum(1 for _ in open(data_dir / 'barcodes.tsv'))
    n_genes = sum(1 for _ in open(data_dir / 'genes.tsv'))
    if adata.shape[0] != n_barcodes and adata.shape[1] == n_barcodes:
        print("Transposing matrix to ensure cells are rows and genes are columns...")
        adata = adata.transpose()
        print(f"Shape after transpose: {adata.shape} (cells, genes)")
    elif adata.shape[0] != n_barcodes:
        print(f"Warning: Number of cells in AnnData ({adata.shape[0]}) does not match barcodes.tsv ({n_barcodes})")
    else:
        print("Matrix orientation is correct.")

    # Load annotation file
    annot_path = data_dir / "68k_pbmc_barcodes_annotation.tsv"
    print(f"Reading annotation from {annot_path}")
    annot = pd.read_csv(annot_path, sep=None, engine='python')
    annot['barcodes'] = annot['barcodes'].astype(str)

    # Subset AnnData to only annotated barcodes
    before_cells = adata.n_obs
    annotated_barcodes = set(annot['barcodes'])
    adata = adata[adata.obs_names.isin(annotated_barcodes)].copy()
    after_cells = adata.n_obs
    print(f"Subset AnnData to annotated barcodes: {before_cells} -> {after_cells} cells.")
    if after_cells < before_cells / 2:
        print(f"Warning: Dropped {before_cells - after_cells} cells not present in annotation (expected for this dataset).")

    # Merge annotation into adata.obs
    adata.obs = adata.obs.merge(annot[['barcodes', 'celltype']], left_index=True, right_on='barcodes', how='left')
    adata.obs.index = adata.obs['barcodes'].values.astype(str)
    adata.obs.drop(columns=['barcodes'], inplace=True)
    # Standardize T cell labels
    tcell_labels = [
        'CD4+ helper T',
        'CD4+/CD25 T Reg',
        'CD4+/CD45RA+/CD25- Naive T',
        'CD4+/CD45RO+ Memory',
        'CD8+/CD45RA+ Naive Cytotoxic',
    ]
    def merge_tcells(label):
        if pd.isnull(label):
            return label
        label = str(label).strip()
        if label in tcell_labels:
            return 'T cells'
        # Also merge any label containing 'T Helper', 'T Reg', 'Memory', 'Naive Cytotoxic', or 'CD4+', 'CD8+'
        if any(x in label for x in ['T Helper', 'T Reg', 'Memory', 'Naive Cytotoxic', 'CD4+', 'CD8+']):
            return 'T cells'
        return label
    adata.obs['celltype_merged'] = adata.obs['celltype'].apply(merge_tcells)

    # Drop cells with missing annotation
    before = adata.n_obs
    adata = adata[~adata.obs['celltype'].isna() & ~adata.obs['celltype_merged'].isna()].copy()
    after = adata.n_obs
    print(f"Dropped {before - after} cells with missing annotation. Remaining cells: {after}")
    # Ensure index is string and contains no NaN
    adata.obs.index = adata.obs.index.astype(str)
    assert not adata.obs.index.isnull().any(), "AnnData index still contains NaN!"

    # Print AnnData shape and number of cells per type
    print(f"AnnData shape: {adata.shape}")
    print("Number of cells per celltype_merged:")
    print(adata.obs['celltype_merged'].value_counts())

    # Print cell identity labels
    print("\n--- Cell Identity Label Information ---")
    unique_labels = adata.obs['celltype_merged'].unique()
    print(f"Found {len(unique_labels)} unique celltype_merged labels:")
    for label in unique_labels:
        print(f"  - {label}")
    print("--- End Cell Identity Label Information ---\n")

    # Basic filtering (optional, can adjust thresholds)
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=3)

    # Select highly variable genes (on raw counts)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000, subset=True, flavor="seurat_v3")
    
    # Note: No log1p transform - keeping raw counts for topic modeling

    # Save filtered AnnData
    filtered_h5ad = output_dir / "filtered_top1000.h5ad"
    adata.write(filtered_h5ad)
    print(f"Saved filtered AnnData to {filtered_h5ad}")

    # Save filtered count matrix (raw counts)
    # If adata.raw exists, use it; else, use adata.X
    if adata.raw is not None:
        counts = adata.raw.X
        genes = adata.raw.var_names
    else:
        counts = adata.X
        genes = adata.var_names
    
    # Convert to dense numpy array
    if scipy.sparse.issparse(counts):
        try:
            counts = scipy.sparse.csr_matrix(counts).toarray()
        except Exception as e:
            print(f"Warning: Could not convert sparse matrix to dense: {e}")
            counts = np.asarray(counts)
    else:
        counts = np.asarray(counts)

    # Print a preview of the dense matrix and its labels
    print("First 5 cell (row) names:", list(adata.obs_names[:5]))
    print("First 5 gene (column) names:", list(genes[:5]))
    print("First 5x5 block of dense matrix:")
    print(counts[:5, :5])

    # Check that counts has the expected shape (after converting to dense)
    if counts.size == 0:
        raise ValueError("Counts matrix is empty after filtering")
    if counts.ndim != 2:
        raise ValueError(f"Counts matrix must be 2D, got shape {counts.shape}")

    # Ensure we have the right number of cells and genes
    if len(adata.obs_names) != counts.shape[0]:
        raise ValueError(f"Mismatch: {len(adata.obs_names)} cells in obs_names but {counts.shape[0]} in counts")
    if len(genes) != counts.shape[1]:
        raise ValueError(f"Mismatch: {len(genes)} genes in var_names but {counts.shape[1]} in counts")

    # Use cell type annotations as row names instead of barcodes
    # This is what the HLDA fitting expects
    cell_types = adata.obs['celltype_merged'].values
    
    # Check for any missing cell types
    if pd.isna(cell_types).any():
        print("Warning: Found cells with missing cell type annotations. These will be dropped.")
        valid_mask = ~pd.isna(cell_types)
        counts = counts[valid_mask, :]
        cell_types = cell_types[valid_mask]
        print(f"Dropped {(~valid_mask).sum()} cells with missing annotations. Remaining: {valid_mask.sum()} cells")
    
    # Create DataFrame with cell types as index
    count_df = pd.DataFrame(counts, index=cell_types, columns=genes)
    
    # Print summary of cell types in the output
    print(f"\nCell type distribution in output file:")
    cell_type_counts = count_df.index.value_counts()
    for cell_type, count in cell_type_counts.items():
        print(f"  {cell_type}: {count} cells")
    
    # Train/test split: for each cell type, select min(20% of cells, 500) for test
    np.random.seed(42)
    test_indices = []
    for cell_type in count_df.index.unique():
        idx = count_df.index == cell_type
        cell_indices = np.where(idx)[0]
        n_cells = len(cell_indices)
        n_test = min(int(np.ceil(n_cells * 0.2)), 500)
        if n_cells > 0:
            test_indices.extend(np.random.choice(cell_indices, size=n_test, replace=False))
    test_mask = np.zeros(len(count_df), dtype=bool)
    test_mask[test_indices] = True
    train_mask = ~test_mask

    count_df_train = count_df.iloc[train_mask]
    count_df_test = count_df.iloc[test_mask]

    count_csv_train = output_dir / "filtered_counts_train.csv"
    count_csv_test = output_dir / "filtered_counts_test.csv"
    count_df_train.to_csv(count_csv_train)
    count_df_test.to_csv(count_csv_test)
    print(f"Saved train count matrix to {count_csv_train}")
    print(f"Saved test count matrix to {count_csv_test}")

if __name__ == "__main__":
    main() 

