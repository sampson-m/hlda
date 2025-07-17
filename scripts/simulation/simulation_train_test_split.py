#!/usr/bin/env python3
"""
simulation_train_test_split.py

Stratified train/test split for simulated counts.csv by cell identity, memory-efficient (chunked).
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

CHUNKSIZE = 1000  # Number of rows per chunk

def main():
    parser = argparse.ArgumentParser(description="Stratified train/test split for simulation counts.csv (chunked)")
    parser.add_argument("--counts_csv", required=True, help="Path to counts.csv")
    parser.add_argument("--train_csv", required=True, help="Output path for train split")
    parser.add_argument("--test_csv", required=True, help="Output path for test split")
    parser.add_argument("--heldout_cells", type=int, default=600, help="Max held-out cells per identity (default: 600)")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    # Step 1: Read only the index (cell names)
    # Get the first column name (cell index)
    with open(args.counts_csv, 'r') as f:
        first_col = f.readline().split(',')[0]
    all_idx = pd.read_csv(args.counts_csv, usecols=[first_col], index_col=0).index
    # Extract identity from cell name (prefix before underscore)
    identities = pd.Series(all_idx)

    # Step 2: Stratified split: for each identity, hold out up to heldout_cells or 20%
    train_idx = set()
    test_idx = set()
    for ident in identities.unique():
        mask = identities == ident
        idx = all_idx[mask]
        n_total = len(idx)
        n_test = min(args.heldout_cells, int(0.2 * n_total))
        if n_test > 0:
            idx_train, idx_test = train_test_split(idx, test_size=n_test, random_state=args.random_state, shuffle=True)
            train_idx.update(idx_train)
            test_idx.update(idx_test)
        else:
            train_idx.update(idx)
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Step 3: Write train and test CSVs in chunks
    header_written_train = False
    header_written_test = False
    for chunk in pd.read_csv(args.counts_csv, index_col=0, chunksize=CHUNKSIZE):
        chunk_train = chunk.loc[chunk.index.intersection(train_idx)]
        chunk_test = chunk.loc[chunk.index.intersection(test_idx)]
        # Set index to identity (prefix before underscore)
        if not chunk_train.empty:
            chunk_train.index = pd.Series(chunk_train.index).str.split('_').str[0]
            chunk_train.to_csv(args.train_csv, mode='a', header=not header_written_train)
            header_written_train = True
        if not chunk_test.empty:
            chunk_test.index = pd.Series(chunk_test.index).str.split('_').str[0]
            chunk_test.to_csv(args.test_csv, mode='a', header=not header_written_test)
            header_written_test = True

if __name__ == "__main__":
    main() 