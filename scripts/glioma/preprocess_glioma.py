import scanpy as sc
import pandas as pd
import numpy as np
import os
import scipy.sparse
import yaml
import argparse

h5ad_path = "data/glioma/analysis_scRNAseq_tumor_counts.h5ad"
anno_dir = os.path.dirname(h5ad_path)

print("Loading AnnData object...")
adata = sc.read_h5ad(h5ad_path)
print(f"Loaded AnnData: {adata.shape}")

# Find and load annotation CSVs
anno_files = [f for f in os.listdir(anno_dir) if f.endswith('.csv')]
print(f"Found annotation files: {anno_files}")
for fname in anno_files:
    try:
        anno = pd.read_csv(os.path.join(anno_dir, fname), index_col=0)
        # Merge on index (cell barcodes)
        adata.obs = adata.obs.join(anno, how='left')
        print(f"Merged annotations from {fname}")
    except Exception as e:
        print(f"Warning: Could not merge {fname}: {e}")

# Print available cell type columns
type_cols = [col for col in adata.obs.columns if 'type' in col or 'cell' in col or 'anno' in col]
print(f"Cell type annotation columns: {type_cols}")

# Pick the first cell type column as the label (customize as needed)
if type_cols:
    celltype_col = type_cols[0]
    print(f"Using '{celltype_col}' as cell type label.")
    celltypes = adata.obs[celltype_col].astype(str)
else:
    raise ValueError("No cell type annotation column found.")

# Filter cells with <10 genes and genes with <3 cells
print("Filtering cells with <10 genes and genes with <3 cells...")
sc.pp.filter_cells(adata, min_genes=10)
sc.pp.filter_genes(adata, min_cells=3)
print(f"Shape after filtering: {adata.shape} (cells, genes)")

# Highly variable gene selection (top 1000)
print("Selecting top 1000 highly variable genes...")
try:
    sc.pp.highly_variable_genes(adata, n_top_genes=1000, subset=True, flavor="seurat_v3")
    print(f"adata shape after gene selection: {adata.shape}")
except Exception as e:
    print(f"Memory error or other issue during gene selection: {e}")
    exit(1)

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
    train_counts_df.to_csv(os.path.join(anno_dir, "glioma_counts_train.csv"))
    test_counts_df.to_csv(os.path.join(anno_dir, "glioma_counts_test.csv"))
    print(f"Wrote train split to {os.path.join(anno_dir, 'glioma_counts_train.csv')}")
    print(f"Wrote test split to {os.path.join(anno_dir, 'glioma_counts_test.csv')}")
except Exception as e:
    print(f"Memory error or other issue during CSV writing: {e}")
    exit(1)

print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Process glioma h5ad file for topic modeling.")
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