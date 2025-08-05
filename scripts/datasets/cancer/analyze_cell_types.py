import scanpy as sc
import pandas as pd

# Load the filtered cancer dataset
h5ad_path = "data/cancer/cancer_filtered.h5ad"

print("Loading filtered cancer dataset...")
adata = sc.read_h5ad(h5ad_path)
print(f"Filtered dataset shape: {adata.shape} (cells, genes)")

# Find cell type columns
cell_type_cols = [col for col in adata.obs.columns if 'cell' in col.lower() or 'type' in col.lower()]
print(f"\nCell type columns found: {cell_type_cols}")

# Analyze each cell type column
for col in cell_type_cols:
    unique_cell_types = adata.obs[col].unique()
    print(f"\n{col}:")
    print(f"  Number of unique cell types: {len(unique_cell_types)}")
    print(f"  Cell types: {list(unique_cell_types)}")
    
    # Count cells per type
    cell_type_counts = adata.obs[col].value_counts()
    print(f"  Cell counts per type:")
    for cell_type, count in cell_type_counts.items():
        print(f"    {cell_type}: {count}")

# Also check if there are any other annotation columns that might be relevant
print("\nAll annotation columns in filtered dataset:")
for col in adata.obs.columns:
    print(f"- {col}")