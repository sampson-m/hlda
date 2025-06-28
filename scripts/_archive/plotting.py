
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def compute_topic_cosine_similarity(beta, topic_names=None):

    # Convert to numpy array if beta is a DataFrame.
    if isinstance(beta, pd.DataFrame):
        beta_mat = beta.to_numpy()
    else:
        beta_mat = np.array(beta)
    
    # If beta is (n_genes, n_topics), then transpose it so that rows correspond to topics.
    if beta_mat.shape[0] < beta_mat.shape[1]:
        beta_mat = beta_mat.T

    # Compute pairwise cosine similarity.
    sim = cosine_similarity(beta_mat)
    
    # Default topic names if not provided.
    if topic_names is None:
        topic_names = [f"Topic_{i}" for i in range(sim.shape[0])]
    
    sim_df = pd.DataFrame(sim, index=topic_names, columns=topic_names)
    return sim_df

def plot_topic_cosine_similarity(sim_df, output_file):

    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_df, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Cosine Similarity Between Topics")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved topic cosine similarity heatmap to {output_file}")

def plot_theta_heatmap(theta_df, output_file, title="Theta Matrix Heatmap"):
    """
    Plots a heatmap of the theta matrix.
    
    Parameters:
      theta_df (pd.DataFrame): DataFrame with rows as cells/documents and columns as topics.
      output_file (str): Path to save the heatmap image.
      title (str): Title for the heatmap.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(theta_df, annot=True, fmt=".2f", cmap="viridis")
    plt.title(title)
    plt.xlabel("Topics")
    plt.ylabel("Cells")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved theta heatmap to {output_file}")
