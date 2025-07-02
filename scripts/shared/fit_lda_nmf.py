import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.preprocessing import normalize
import time


def main():
    parser = argparse.ArgumentParser(description="Fit LDA and/or NMF to a count matrix.")
    parser.add_argument("--counts_csv", type=str, required=True, help="Path to filtered count matrix CSV")
    parser.add_argument("--n_topics", type=int, required=True, help="Number of topics (K)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--model", type=str, choices=["lda", "nmf", "both"], default="both", help="Which model(s) to fit")
    parser.add_argument("--max_iter", type=int, default=500, help="Max iterations for model fitting")
    parser.add_argument("--random_state", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.counts_csv, index_col=0)
    X = df.values.astype(float)
    cell_names = pd.Index(df.index)
    gene_names = pd.Index(df.columns)
    K = args.n_topics
    topic_labels = pd.Index([f"Topic_{i+1}" for i in range(K)])

    if args.model in ("lda", "both"):
        lda_dir = output_dir / "LDA"
        lda_dir.mkdir(parents=True, exist_ok=True)
        print(f"Fitting LDA with K={K}...")
        lda = LatentDirichletAllocation(
            n_components=K,
            random_state=args.random_state,
            learning_method="batch",
            max_iter=args.max_iter,
        )
        start_time = time.time()
        theta_lda = lda.fit_transform(X)
        elapsed = time.time() - start_time
        print(f"LDA fit completed in {elapsed:.2f} seconds ({elapsed/60:.2f} min)")
        theta_lda = theta_lda / theta_lda.sum(axis=1, keepdims=True)
        beta_lda = np.asarray(normalize(lda.components_, norm="l1", axis=1)).T
        pd.DataFrame(beta_lda, index=gene_names, columns=topic_labels).to_csv(lda_dir / "LDA_beta.csv")
        pd.DataFrame(theta_lda, index=cell_names, columns=topic_labels).to_csv(lda_dir / "LDA_theta.csv")
        print(f"LDA outputs saved to {lda_dir}")

    if args.model in ("nmf", "both"):
        nmf_dir = output_dir / "NMF"
        nmf_dir.mkdir(parents=True, exist_ok=True)
        print(f"Fitting NMF with K={K}...")
        nmf = NMF(
            n_components=K,
            init="nndsvd",
            random_state=args.random_state,
            max_iter=args.max_iter,
            solver="cd",
        )
        start_time = time.time()
        theta_nmf = nmf.fit_transform(X)
        elapsed = time.time() - start_time
        print(f"NMF fit completed in {elapsed:.2f} seconds ({elapsed/60:.2f} min)")
        theta_nmf = theta_nmf / theta_nmf.sum(axis=1, keepdims=True)
        beta_nmf = np.asarray(normalize(nmf.components_, norm="l1", axis=1)).T
        pd.DataFrame(beta_nmf, index=gene_names, columns=topic_labels).to_csv(nmf_dir / "NMF_beta.csv")
        pd.DataFrame(theta_nmf, index=cell_names, columns=topic_labels).to_csv(nmf_dir / "NMF_theta.csv")
        print(f"NMF outputs saved to {nmf_dir}")

if __name__ == "__main__":
    main() 