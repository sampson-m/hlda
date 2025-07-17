import scanpy as sc
import pandas as pd
import numpy as np
import os
import scipy.sparse
import yaml
import argparse

def preprocess_cancer_data(labeling_scheme="combined"):
    """
    Preprocess cancer data with either combined or disease-specific cell type labels.
    
    Args:
        labeling_scheme: "combined" for {cell_type} or "disease_specific" for {disease}_{cell_type}
    """
    
    # Use filtered cancer dataset
    h5ad_path = "data/cancer/cancer_filtered.h5ad"
    anno_dir = os.path.dirname(h5ad_path)
    
    print(f"Loading filtered cancer dataset for {labeling_scheme} labeling...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"Loaded AnnData: {adata.shape}")
    
    # Find cell type and disease columns
    cell_type_cols = [col for col in adata.obs.columns if 'cell' in col.lower() or 'type' in col.lower()]
    disease_cols = [col for col in adata.obs.columns if 'disease' in col.lower()]
    
    print(f"Cell type columns: {cell_type_cols}")
    print(f"Disease columns: {disease_cols}")
    
    # Use first available columns
    if not cell_type_cols:
        raise ValueError("No cell type column found")
    if not disease_cols:
        raise ValueError("No disease column found")
    
    celltype_col = "cell_type"
    disease_col = "disease"
    
    print(f"Using '{celltype_col}' as cell type column")
    print(f"Using '{disease_col}' as disease column")
    
    # Create labels based on scheme
    if labeling_scheme == "combined":
        # Combined scheme: just use cell type
        celltypes = adata.obs[celltype_col].astype(str)
        output_suffix = "combined"
    elif labeling_scheme == "disease_specific":
        # Disease-specific scheme: {disease}_{cell_type}
        disease_labels = adata.obs[disease_col].astype(str)
        celltype_labels = adata.obs[celltype_col].astype(str)
        # Create short disease labels
        disease_map = {
            "HER2 positive breast carcinoma": "breast",
            "metastatic melanoma": "melanoma"
        }
        disease_short = disease_labels.map(disease_map).fillna(disease_labels)
        celltypes = disease_short + "_" + celltype_labels
        output_suffix = "disease_specific"
    else:
        raise ValueError(f"Unknown labeling scheme: {labeling_scheme}")
    
    print(f"Unique cell type labels ({labeling_scheme}): {celltypes.unique()}")
    print(f"Cell type counts:")
    for ct, count in celltypes.value_counts().items():
        print(f"  {ct}: {count}")
    
    # Filter cells with <10 genes and genes with <3 cells
    print("Filtering cells with <10 genes and genes with <3 cells...")
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"Shape after filtering: {adata.shape} (cells, genes)")
    
    # Update celltypes after filtering
    celltypes = celltypes.loc[adata.obs.index]
    
    # Highly variable gene selection (top 1000)
    print("Selecting top 1000 highly variable genes...")
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=1000, subset=True, flavor="seurat_v3")
        print(f"adata shape after gene selection: {adata.shape}")
    except Exception as e:
        print(f"Memory error or other issue during gene selection: {e}")
        exit(1)
    
    # Update celltypes after gene filtering
    celltypes = celltypes.loc[adata.obs.index]
    
    # 80-20 train/test split within each cell type
    print("Performing 80-20 train/test split within each cell type...")
    np.random.seed(42)
    train_idx = []
    test_idx = []
    for ct in celltypes.unique():
        mask = (celltypes == ct)
        idx = np.where(mask)[0]
        n = len(idx)
        n_test = max(1, int(np.ceil(n * 0.2)))
        if n > 0:
            test_cells = np.random.choice(idx, size=n_test, replace=False)
            train_cells = np.setdiff1d(idx, test_cells)
            train_idx.extend(train_cells)
            test_idx.extend(test_cells)
    
    # Optionally, add split info to adata.obs
    adata.obs['split'] = 'train'
    adata.obs.iloc[test_idx, adata.obs.columns.get_loc('split')] = 'test'
    
    # Write out train and test split counts matrices with cell type as index
    print("Writing train and test split counts to CSV...")
    try:
        counts = adata.X
        if scipy.sparse.issparse(counts):
            counts = scipy.sparse.csr_matrix(counts).toarray()
        else:
            counts = np.asarray(counts)
        
        # Use cell type as index
        train_counts_df = pd.DataFrame(counts[train_idx], index=celltypes.values[train_idx], columns=adata.var_names)
        test_counts_df = pd.DataFrame(counts[test_idx], index=celltypes.values[test_idx], columns=adata.var_names)
        
        train_csv_path = os.path.join(anno_dir, f"cancer_counts_train_{output_suffix}.csv")
        test_csv_path = os.path.join(anno_dir, f"cancer_counts_test_{output_suffix}.csv")
        
        train_counts_df.to_csv(train_csv_path)
        test_counts_df.to_csv(test_csv_path)
        print(f"Wrote train split to {train_csv_path}")
        print(f"Wrote test split to {test_csv_path}")
        
        return train_csv_path, test_csv_path, celltypes.unique()
        
    except Exception as e:
        print(f"Memory error or other issue during CSV writing: {e}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Process cancer h5ad file for topic modeling.")
    parser.add_argument("--labeling_scheme", type=str, choices=["combined", "disease_specific"], 
                       default="combined", help="Labeling scheme: combined or disease_specific")
    args = parser.parse_args()
    
    print(f"Processing cancer data with {args.labeling_scheme} labeling scheme...")
    train_csv, test_csv, unique_labels = preprocess_cancer_data(args.labeling_scheme)
    
    print(f"\nPreprocessing complete!")
    print(f"Train CSV: {train_csv}")
    print(f"Test CSV: {test_csv}")
    print(f"Unique labels ({args.labeling_scheme}): {list(unique_labels)}")

if __name__ == "__main__":
    main() 