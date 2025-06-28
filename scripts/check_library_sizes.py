import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Check library sizes from a count matrix CSV file.")
    parser.add_argument("--csv_path", type=str, default="data/pbmc/filtered_counts.csv",
                       help="Path to the filtered counts CSV file (default: data/pbmc/filtered_counts.csv)")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        print("Please provide the correct path to your count matrix CSV file.")
        return

    print(f"Reading count matrix from {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)

    # Compute library sizes (sum of counts per cell)
    library_sizes = df.sum(axis=1)

    print("First 5 library sizes:")
    print(library_sizes.head())

    print("\nSummary statistics for library sizes:")
    print(f"Min:    {library_sizes.min():.2f}")
    print(f"Max:    {library_sizes.max():.2f}")
    print(f"Mean:   {library_sizes.mean():.2f}")
    print(f"Median: {library_sizes.median():.2f}")
    print(f"Std:    {library_sizes.std():.2f}")

    # Print a simple histogram (text-based)
    hist, bin_edges = np.histogram(library_sizes, bins=10)
    print("\nHistogram of library sizes (bin edges and counts):")
    for i in range(len(hist)):
        print(f"{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}: {hist[i]}")

if __name__ == "__main__":
    main() 