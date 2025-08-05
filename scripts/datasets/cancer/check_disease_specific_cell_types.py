#!/usr/bin/env python3
"""
Script to check the unique cell types in the disease-specific cancer count files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml

def extract_cell_type_from_index(cell_id):
    """Extract cell type from cell ID (assumes format like 'celltype_1', 'celltype_2', etc.)"""
    # Remove any trailing numbers and underscores
    parts = cell_id.split('_')
    if len(parts) > 1:
        # Join all parts except the last one if it's a number
        if parts[-1].isdigit():
            return '_'.join(parts[:-1])
        else:
            return cell_id
    return cell_id

def main():
    """Check unique cell types in disease-specific cancer data."""
    
    print("🔍 Checking Cell Types in Disease-Specific Cancer Data")
    print("=" * 60)
    
    # Load the configuration file
    config_file = "dataset_identities.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get expected cell types from config
    expected_cell_types = config['cancer_disease_specific']['identities']
    print(f"Expected cell types from config ({len(expected_cell_types)}):")
    for i, cell_type in enumerate(expected_cell_types, 1):
        print(f"  {i:2d}. {cell_type}")
    
    print("\n" + "=" * 60)
    
    # Check train and test files
    files_to_check = [
        "data/cancer/cancer_counts_train_disease_specific.csv",
        "data/cancer/cancer_counts_test_disease_specific.csv"
    ]
    
    for file_path in files_to_check:
        print(f"\n📊 Analyzing: {file_path}")
        
        if not Path(file_path).exists():
            print(f"❌ File not found: {file_path}")
            continue
        
        # Load the data
        try:
            df = pd.read_csv(file_path, index_col=0)
            print(f"✅ File loaded successfully")
            print(f"   Shape: {df.shape} (cells x genes)")
            
            # Extract cell types from row indices
            cell_types = [extract_cell_type_from_index(idx) for idx in df.index]
            unique_cell_types = sorted(set(cell_types))
            
            print(f"   Found {len(unique_cell_types)} unique cell types:")
            
            # Count cells per type
            cell_type_counts = {}
            for cell_type in cell_types:
                cell_type_counts[cell_type] = cell_type_counts.get(cell_type, 0) + 1
            
            for i, cell_type in enumerate(unique_cell_types, 1):
                count = cell_type_counts[cell_type]
                in_config = "✅" if cell_type in expected_cell_types else "❌"
                print(f"   {i:2d}. {cell_type:<45} ({count:4d} cells) {in_config}")
            
            # Check for missing cell types
            missing_in_data = set(expected_cell_types) - set(unique_cell_types)
            if missing_in_data:
                print(f"\n❌ Cell types in config but NOT in data ({len(missing_in_data)}):")
                for cell_type in sorted(missing_in_data):
                    print(f"   - {cell_type}")
            
            # Check for extra cell types
            extra_in_data = set(unique_cell_types) - set(expected_cell_types)
            if extra_in_data:
                print(f"\n⚠️  Cell types in data but NOT in config ({len(extra_in_data)}):")
                for cell_type in sorted(extra_in_data):
                    print(f"   - {cell_type}")
            
        except Exception as e:
            print(f"❌ Error loading file: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()