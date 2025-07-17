import scanpy as sc
import pandas as pd

# Path to the h5ad file
dataset_id = "https://datasets.cellxgene.cziscience.com/6039d13f-0c3e-484b-b37c-ee3656c4c037.h5ad"
disease = ["HER2 positive breast carcinoma", "metastatic melanoma"]

h5ad_path = "data/cancer/cancer.h5ad"

# Load the AnnData object
print("Loading large cancer dataset...")
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

# Find disease column and filter for target diseases
disease_cols = [col for col in adata.obs.columns if 'disease' in col.lower()]
print(f"\nDisease-related columns: {disease_cols}")

if disease_cols:
    disease_col = disease_cols[1]  # Use the first disease column
    print(f"\nUsing '{disease_col}' as disease column")
    
    # Check unique values in disease column
    unique_diseases = adata.obs[disease_col].unique()
    print(f"Unique diseases in dataset: {unique_diseases}")
    
    # Filter for target diseases
    disease_mask = adata.obs[disease_col].isin(disease)
    print(f"\nCells with target diseases ({disease}): {disease_mask.sum()}")
    
    if disease_mask.sum() > 0:
        # Filter the dataset
        adata_filtered = adata[disease_mask].copy()
        print(f"Filtered dataset shape: {adata_filtered.shape}")
        
        # Check cell types in filtered dataset
        cell_type_cols = [col for col in adata_filtered.obs.columns if 'cell' in col.lower() or 'type' in col.lower()]
        print(f"\nCell type columns in filtered data: {cell_type_cols}")
        
        for col in cell_type_cols:
            unique_cell_types = adata_filtered.obs[col].unique()
            print(f"- {col}: {unique_cell_types}")
        
        # Save filtered dataset
        output_path = "data/cancer/cancer_filtered.h5ad"
        print(f"\nSaving filtered dataset to {output_path}...")
        adata_filtered.write_h5ad(output_path)
        print(f"Saved filtered dataset with {adata_filtered.shape[0]} cells and {adata_filtered.shape[1]} genes")
    else:
        print("No cells found with target diseases!")
else:
    print("No disease column found in dataset!") 
