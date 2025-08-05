#!/usr/bin/env python3
"""
Script to regenerate SSE line charts and theta heatmaps across all model configurations.
Based on run_all_evaluations.sh but only runs visualization functions.

Generates:
- Individual SSE line charts (cumulative_sse_by_topic.png) 
- Cross-topic SSE line charts (combined_cumulative_sse_by_topic.png)
- Theta heatmaps for train and test data
- Combined theta heatmaps (stacked vertically)
"""

import pandas as pd
import yaml
from pathlib import Path
import argparse
import glob
import numpy as np

# Import the visualization functions
from shared.evaluate_models import (
    plot_cumulative_sse_lineplot, 
    plot_theta_heatmap,
    estimate_theta_simplex,
    prepare_model_topics,
    extract_cell_identity
)
from shared.analyze_all_fits import plot_combined_cumulative_sse_lineplot

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_identity_topics_from_config(dataset: str, config_file: str) -> list:
    """Get identity topics for a dataset from config file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    if dataset not in config:
        raise ValueError(f"Dataset '{dataset}' not found in config file {config_file}")
    
    return config[dataset]['identities']


def process_individual_visualizations(base_dirs: list, datasets: list, config_file: str):
    """
    Process individual topic fit directories to regenerate all visualizations.
    """
    print("=" * 60)
    print("PROCESSING INDIVIDUAL TOPIC FIT VISUALIZATIONS")
    print("=" * 60)
    
    for base_dir, dataset in zip(base_dirs, datasets):
        base_path = Path(base_dir)
        print(f"\nProcessing {dataset.upper()} in {base_dir}")
        
        # Get identity topics for this dataset
        try:
            identity_topics = get_identity_topics_from_config(dataset, config_file)
            n_identities = len(identity_topics)
            print(f"  Identity topics ({n_identities}): {identity_topics}")
        except Exception as e:
            print(f"  Error loading config for {dataset}: {e}")
            continue
        
        # Find all topic fit directories
        topic_fit_dirs = []
        if dataset == "pbmc":
            # PBMC has heldout_* subdirectories
            for heldout_dir in base_path.glob("heldout_*"):
                if heldout_dir.is_dir():
                    for topic_dir in heldout_dir.glob("*_topic_fit"):
                        if topic_dir.is_dir():
                            topic_fit_dirs.append(topic_dir)
        else:
            # Other datasets have topic fits directly under base_dir
            for topic_dir in base_path.glob("*_topic_fit"):
                if topic_dir.is_dir():
                    topic_fit_dirs.append(topic_dir)
        
        # Process each topic fit directory
        for config_dir in sorted(topic_fit_dirs):
            try:
                # Extract number of topics from directory name
                n_topics = int(config_dir.name.split('_')[0])
                n_extra_topics = n_topics - n_identities
                
                print(f"    Processing {config_dir} ({n_extra_topics} extra topics)")
                
                # === SSE LINE CHARTS ===
                sse_file = config_dir / "sse_summary.csv"
                if sse_file.exists():
                    sse_df = pd.read_csv(sse_file)
                    activity_topics = [f"V{i+1}" for i in range(n_extra_topics)]
                    output_dir = config_dir / "plots"
                    plot_cumulative_sse_lineplot(sse_df, identity_topics, activity_topics, output_dir)
                    print(f"      ✓ Generated cumulative_sse_by_topic.png")
                else:
                    print(f"      Warning: No sse_summary.csv found, skipping SSE charts")
                
                # === THETA HEATMAPS ===
                # Check if we have the required model files
                models = {}
                model_names = ["HLDA", "LDA", "NMF"]
                for m in model_names:
                    beta_file = config_dir / m / f"{m}_beta.csv"
                    theta_file = config_dir / m / f"{m}_theta.csv"
                    if beta_file.exists() and theta_file.exists():
                        beta = pd.read_csv(beta_file, index_col=0)
                        theta = pd.read_csv(theta_file, index_col=0)
                        models[m] = {"beta": beta, "theta": theta}
                
                if not models:
                    print(f"      Warning: No model files found, skipping theta heatmaps")
                    continue
                
                print(f"      Found models: {list(models.keys())}")
                
                # Load train and test data
                if dataset == "pbmc":
                    train_csv = Path("data/pbmc/filtered_counts_train.csv")
                    test_csv = Path("data/pbmc/filtered_counts_test.csv")
                elif dataset == "glioma":
                    train_csv = Path("data/glioma/glioma_counts_train.csv")
                    test_csv = Path("data/glioma/glioma_counts_test.csv")
                elif dataset == "cancer":
                    train_csv = Path("data/cancer/filtered_counts_train.csv")
                    test_csv = Path("data/cancer/filtered_counts_test.csv")
                else:
                    # For simulation data, look in the config directory
                    train_csv = config_dir / "filtered_counts_train.csv"  
                    test_csv = config_dir / "filtered_counts_test.csv"
                
                if not (train_csv.exists() and test_csv.exists()):
                    print(f"      Warning: Train/test data not found, skipping theta heatmaps")
                    continue
                
                counts_df = pd.read_csv(train_csv, index_col=0)
                test_df = pd.read_csv(test_csv, index_col=0)
                
                # Generate theta heatmaps
                print(f"      Generating theta heatmaps...")
                
                # Generate train heatmaps
                theta_heatmap_paths_train = []
                theta_heatmap_paths_test = []
                model_names_ordered = []
                
                # Determine reference ordering from HLDA if available
                reference_ordering = None
                if "HLDA" in models:
                    hlda_beta = models["HLDA"]["beta"]
                    hlda_theta = models["HLDA"]["theta"]
                    hlda_beta_renamed, hlda_theta_renamed, _ = prepare_model_topics(
                        "HLDA", hlda_beta, hlda_theta, counts_df, identity_topics, n_extra_topics
                    )
                    cell_identities = pd.Series([extract_cell_identity(i) for i in counts_df.index], index=counts_df.index)
                    reference_ordering = None # Removed determine_reference_ordering
                
                for m in models.keys():
                    model_plots_dir = config_dir / m / "plots"
                    model_plots_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Prepare model topics (rename LDA/NMF topics to match identities)
                    beta = models[m]["beta"]
                    theta = models[m]["theta"]
                    beta_renamed, theta_renamed, topic_mapping = prepare_model_topics(
                        m, beta, theta, counts_df, identity_topics, n_extra_topics
                    )
                    
                    # Train heatmap
                    cell_identities = pd.Series([extract_cell_identity(i) for i in counts_df.index], index=counts_df.index)
                    
                    out_png = model_plots_dir / f"{m}_theta_heatmap.png"
                    theta_heatmap_paths_train.append(str(out_png))
                    model_names_ordered.append(m)
                    
                    plot_theta_heatmap(theta_renamed, cell_identities, m, out_png, identity_topics, 
                                      cells_per_group=3, use_consistent_ordering=False)
                    
                    # Test heatmap
                    X_test_prop = test_df.div(test_df.sum(axis=1), axis=0).values
                    theta_test = estimate_theta_simplex(X_test_prop, beta_renamed.values, l1=0.002)
                    theta_test_df = pd.DataFrame(theta_test, index=test_df.index, columns=beta_renamed.columns)
                    test_identities = pd.Series([extract_cell_identity(i) for i in test_df.index], index=test_df.index)
                    
                    out_png = model_plots_dir / f"{m}_test_theta_heatmap.png"
                    theta_heatmap_paths_test.append(str(out_png))
                    
                    plot_theta_heatmap(theta_test_df, test_identities, f"{m} (test)", out_png, identity_topics, 
                                      cells_per_group=3, use_consistent_ordering=False)
                
                # Create combined heatmaps
                combined_dir = config_dir / "plots"
                combined_dir.mkdir(parents=True, exist_ok=True)
                stack_heatmaps_vertically(theta_heatmap_paths_train, model_names_ordered, 
                                        combined_dir / "combined_theta_heatmap_train.png")
                stack_heatmaps_vertically(theta_heatmap_paths_test, model_names_ordered, 
                                        combined_dir / "combined_theta_heatmap_test.png")
                
                print(f"      ✓ Generated theta heatmaps and combined plots")
                
            except (ValueError, IndexError, Exception) as e:
                print(f"      Error processing {config_dir}: {e}")
                continue


def process_cross_topic_analysis(base_dirs: list, datasets: list, config_file: str):
    """
    Process cross-topic analysis to regenerate combined SSE line charts.
    """
    print("\n" + "=" * 60)
    print("PROCESSING CROSS-TOPIC ANALYSIS SSE CHARTS")
    print("=" * 60)
    
    for base_dir, dataset in zip(base_dirs, datasets):
        base_path = Path(base_dir)
        print(f"\nProcessing {dataset.upper()} cross-topic analysis in {base_dir}")
        
        # Get identity topics for this dataset
        try:
            identity_topics = get_identity_topics_from_config(dataset, config_file)
            print(f"  Identity topics: {identity_topics}")
        except Exception as e:
            print(f"  Error loading config for {dataset}: {e}")
            continue
        
        # Find directories to analyze
        analysis_dirs = []
        if dataset == "pbmc":
            # For PBMC, analyze each heldout_* directory
            for heldout_dir in base_path.glob("heldout_*"):
                if heldout_dir.is_dir():
                    # Check if it has topic fit subdirectories
                    topic_fits = list(heldout_dir.glob("*_topic_fit"))
                    if topic_fits:
                        analysis_dirs.append(heldout_dir)
        else:
            # For other datasets, analyze the base directory if it has topic fits
            topic_fits = list(base_path.glob("*_topic_fit"))
            if topic_fits:
                analysis_dirs.append(base_path)
        
        # Process each analysis directory
        for analysis_dir in analysis_dirs:
            try:
                print(f"    Processing {analysis_dir}")
                
                # Find max number of activity topics
                max_n_activity = 0
                topic_fits = list(analysis_dir.glob("*_topic_fit"))
                
                for topic_dir in topic_fits:
                    try:
                        n_topics = int(topic_dir.name.split('_')[0])
                        n_activity = n_topics - len(identity_topics)
                        if n_activity > max_n_activity:
                            max_n_activity = n_activity
                    except (ValueError, IndexError):
                        continue
                
                if max_n_activity == 0:
                    print(f"      Warning: No valid topic fits found, skipping")
                    continue
                
                print(f"      Found topic fits with up to {max_n_activity} activity topics")
                
                # Set output directory
                output_dir = analysis_dir / "model_comparison"
                
                # Generate combined cumulative SSE line chart
                plot_combined_cumulative_sse_lineplot(
                    analysis_dir, identity_topics, max_n_activity, output_dir, config_file
                )
                print(f"      ✓ Generated combined_cumulative_sse_by_topic.png")
                
            except Exception as e:
                print(f"      Error processing {analysis_dir}: {e}")
                continue


def stack_heatmaps_vertically(heatmap_paths, model_names, output_path):
    """Stack heatmap images vertically into a single combined image."""
    if not heatmap_paths:
        print("    No heatmap paths provided")
        return
    
    # Load images
    images = []
    valid_models = []
    
    for path, model_name in zip(heatmap_paths, model_names):
        try:
            img = mpimg.imread(path)
            images.append(img)
            valid_models.append(model_name)
            print(f"    Loaded heatmap for {model_name}")
        except Exception as e:
            print(f"    Error loading {path}: {e}")
            continue
    
    if len(images) == 0:
        print("    No valid images could be loaded")
        return
    
    # Calculate dimensions
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    max_width = max(widths)
    label_height = 50  # Space for model labels
    total_height = sum(heights) + label_height * len(images)
    
    # Create combined figure
    fig_width = max_width / 100  # Convert pixels to inches (assuming 100 DPI)
    fig_height = total_height / 100
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Stack images vertically
    y_offset = 0
    for i, (img, model_name) in enumerate(zip(images, valid_models)):
        # Calculate position for this image
        img_height = heights[i]
        y_pos = 1.0 - (y_offset + img_height + label_height) / total_height
        height_frac = img_height / total_height
        
        # Add image
        ax = fig.add_axes((0.0, y_pos, 1.0, height_frac))
        ax.imshow(img)
        ax.axis('off')
        
        # Add model label above image
        label_y = 1.0 - (y_offset + label_height/2) / total_height
        fig.text(0.5, label_y, model_name, ha='center', va='center', 
                fontsize=18, weight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        y_offset += img_height + label_height
    
    # Save combined image
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"    ✓ Combined heatmaps saved to: {output_path}")


def main():
    """Main function to run visualization regeneration."""
    parser = argparse.ArgumentParser(description="Regenerate SSE line charts and theta heatmaps for all model configurations")
    parser.add_argument("--config_file", type=str, default="dataset_identities.yaml", 
                       help="Path to dataset identity config YAML file")
    parser.add_argument("--individual_only", action="store_true", 
                       help="Only regenerate individual topic fit visualizations")
    parser.add_argument("--cross_topic_only", action="store_true", 
                       help="Only regenerate cross-topic analysis charts")
    args = parser.parse_args()
    
    # Define datasets and their base directories
    base_dirs = ["estimates/pbmc", "estimates/glioma"]
    datasets = ["pbmc", "glioma"]
    
    print("VISUALIZATION REGENERATION")
    print(f"Config file: {args.config_file}")
    print(f"Processing datasets: {datasets}")
    print(f"Base directories: {base_dirs}")
    
    # Check if base directories exist
    existing_dirs = []
    existing_datasets = []
    for base_dir, dataset in zip(base_dirs, datasets):
        if Path(base_dir).exists():
            existing_dirs.append(base_dir)
            existing_datasets.append(dataset)
        else:
            print(f"Warning: {base_dir} does not exist, skipping {dataset}")
    
    if not existing_dirs:
        print("Error: No valid base directories found")
        return
    
    # Process individual topic fits (unless cross_topic_only is specified)
    if not args.cross_topic_only:
        process_individual_visualizations(existing_dirs, existing_datasets, args.config_file)
    
    # Process cross-topic analysis (unless individual_only is specified)
    if not args.individual_only:
        process_cross_topic_analysis(existing_dirs, existing_datasets, args.config_file)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION REGENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()