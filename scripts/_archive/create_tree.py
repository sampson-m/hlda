#!/usr/bin/env python
import scanpy as sc
import numpy as np
import pandas as pd

def build_topic_mapping(adata, level_cols):
    """
    For each celltype_level column in adata.obs, assign a unique numerical ID
    to each unique entry. The mapping keys are strings like "celltype_level_1:Immune".
    Also, include a special root node with ID 0.
    
    Returns:
        topic_mapping (dict): Mapping from key to a unique integer.
    """
    topic_mapping = {"root": 0}
    next_id = 1
    for level in level_cols:
        unique_vals = adata.obs[level].unique()
        for val in unique_vals:
            key = val
            if key not in topic_mapping:
                topic_mapping[key] = next_id
                next_id += 1
    return topic_mapping

def get_cell_topic_lists(adata, level_cols, topic_mapping):
    """
    For each cell, produce a list of topic IDs corresponding to its entries in the given levels.
    If a cell's entry in a level is the same as in the previous level, skip that level.
    Always include the root (ID 0) at the beginning.
    
    Returns:
        cell_topic_lists (list of lists): One list of topic IDs per cell.
    """
    topic_lists = []
    for _, row in adata.obs.iterrows():
        path = [topic_mapping["root"]]  # Always start with root (ID 0)
        prev_val = None
        for level in level_cols:
            val = row[level]
            # Only add if it's the first level or it's different from the previous level
            if prev_val is None or val != prev_val:
                key = val
                path.append(topic_mapping[key])
                prev_val = val
        topic_lists.append(path)
    return topic_lists

def main():
    # Specify the path to your adata file (e.g., .h5ad)
    adata_path = "../data/pbmc/raw.h5ad"  # update with your file path
    # Specify the celltype level columns and the raw celltype column
    level_cols = ["celltype_level_1", "celltype_level_2", "celltype_level_3"]
    celltype_col = "cell_type"
    
    # Read the AnnData object
    adata = sc.read_h5ad(adata_path)
    
    # Extract unique combinations
    topic_mapping = build_topic_mapping(adata, level_cols)
    print("Topic Mapping (each key is 'level:entry'):")
    for key, tid in topic_mapping.items():
        print(f"{key}: {tid}")
    
    # Now, for each cell, get the list of topics (one per level) it samples from.
    cell_topic_lists = get_cell_topic_lists(adata, level_cols, topic_mapping)
    
    # Optionally, attach these lists back to the AnnData object as a new observation column.
    adata.obs["topic_path"] = [str(topics) for topics in cell_topic_lists]
    # add the save adata here to then go sample from
    
    # For demonstration, print the topic path for the first 10 cells.
    print("\nTopic paths for first 10 cells:")
    for i in range(min(10, len(cell_topic_lists))):
        print(cell_topic_lists[i])
    
    # Optionally, save the topic mapping and cell topic lists to CSV files.
    pd.DataFrame(list(topic_mapping.items()), columns=["Key", "Topic_ID"]).to_csv("../data/pbmc/full_data/topic_mapping.csv", index=False)
    pd.DataFrame({"cell_topic_path": [str(tp) for tp in cell_topic_lists]}).to_csv("../data/pbmc/full_data/cell_topic_paths.csv", index=False)
    print("\nSaved topic mapping to 'topic_mapping.csv' and cell topic paths to 'cell_topic_paths.csv'.")


if __name__ == "__main__":
    main()
