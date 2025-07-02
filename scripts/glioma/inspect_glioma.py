import scanpy as sc
import pandas as pd

# Path to the h5ad file
h5ad_path = "data/glioma/analysis_scRNAseq_tumor_counts.h5ad"

# Load the AnnData object
adata = sc.read_h5ad(h5ad_path)
print(f"Loaded AnnData object from {h5ad_path}")
print(f"Shape: {adata.shape} (cells, genes)")

# Print .obs (cell metadata) columns
print("\n.obs (cell metadata) columns:")
print(list(adata.obs.columns))

# Print .var (gene metadata) columns
print("\n.var (gene metadata) columns:")
print(list(adata.var.columns))

# Print first few unique values for each .obs column
print("\nFirst few unique values for each .obs column:")
for col in adata.obs.columns:
    unique_vals = adata.obs[col].unique()
    print(f"- {col}: {unique_vals[:10]}") 