#!/usr/bin/env python3
"""
Comprehensive analysis script for comparing HLDA, LDA, and NMF models across different topic configurations.
Collects and visualizes metrics from 7, 8, and 9 topic fits.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import re
import argparse
import glob
import yaml
warnings.filterwarnings('ignore')

def load_metrics_from_fits(base_dir: Path, topic_configs: list) -> pd.DataFrame:
    """
    Load all metrics from different topic configurations.
    
    Args:
        base_dir: Base directory containing topic fit folders
        topic_configs: List of topic configurations (e.g., ['7_topic_fit', '8_topic_fit', '9_topic_fit'])
    
    Returns:
        DataFrame with all metrics combined
    """
    all_metrics = []
    
    for config in topic_configs:
        config_dir = base_dir / config
        metrics_file = config_dir / "metrics_summary.csv"
        
        if metrics_file.exists():
            # Load basic metrics
            metrics_df = pd.read_csv(metrics_file)
            n_extra_topics = int(config.split('_')[0]) - 6  # 7->1, 8->2, 9->3
            metrics_df['n_extra_topics'] = n_extra_topics
            metrics_df['topic_config'] = config
            all_metrics.append(metrics_df)
            
            print(f"Loaded metrics from {config}: {len(metrics_df)} models")
        else:
            print(f"Warning: No metrics file found in {config}")
    
    if all_metrics:
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        return combined_metrics
    else:
        return pd.DataFrame()

def load_sse_from_fits(base_dir: Path, topic_configs: list) -> pd.DataFrame:
    """
    Load all SSE results from different topic configurations.
    Loads from individual model folders within each topic configuration.
    
    Args:
        base_dir: Base directory containing topic fit folders
        topic_configs: List of topic configurations
    
    Returns:
        DataFrame with all SSE results combined
    """
    all_sse = []
    models = ["HLDA", "LDA", "NMF"]
    
    for config in topic_configs:
        config_dir = base_dir / config
        n_extra_topics = int(config.split('_')[0]) - 6  # 7->1, 8->2, 9->3
        
        for model in models:
            model_dir = config_dir / model
            sse_file = model_dir / "plots" / f"{model}_test_sse.csv"
            
            if sse_file.exists():
                sse_df = pd.read_csv(sse_file)
                sse_df['model'] = model
                sse_df['n_extra_topics'] = n_extra_topics
                sse_df['topic_config'] = config
                sse_df['model_fit_topics'] = int(config.split('_')[0])  # 7, 8, or 9
                all_sse.append(sse_df)
                
                print(f"Loaded SSE from {config}/{model}: {len(sse_df)} rows")
            else:
                print(f"Warning: No SSE file found in {config}/{model}")
    
    if all_sse:
        combined_sse = pd.concat(all_sse, ignore_index=True)
        return combined_sse
    else:
        return pd.DataFrame()

def create_metrics_matrix_output(metrics_df: pd.DataFrame, output_dir: Path):
    """
    Create matrix CSV outputs for log-likelihood on train and test datasets.
    
    Args:
        metrics_df: Combined metrics DataFrame
        output_dir: Output directory for outputs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create clean summary tables
    clean_summary = metrics_df.groupby(['model', 'n_extra_topics']).agg({
        'train_log_likelihood': 'mean',
        'test_log_likelihood': 'mean'
    }).round(4).reset_index()
    
    # Pivot for easier reading
    train_loglik_pivot = clean_summary.pivot(index='model', columns='n_extra_topics', values='train_log_likelihood')
    test_loglik_pivot = clean_summary.pivot(index='model', columns='n_extra_topics', values='test_log_likelihood')
    
    # Save clean tables
    train_loglik_pivot.to_csv(output_dir / 'train_loglikelihood_matrix.csv')
    test_loglik_pivot.to_csv(output_dir / 'test_loglikelihood_matrix.csv')
    
    # Create combined comparison plot
    create_loglikelihood_comparison_plot(metrics_df, output_dir)
    
    return train_loglik_pivot, test_loglik_pivot

def create_loglikelihood_comparison_plot(metrics_df: pd.DataFrame, output_dir: Path):
    """
    Create comprehensive log-likelihood comparison plots across all models and configurations.
    
    Args:
        metrics_df: Combined metrics DataFrame
        output_dir: Output directory for plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a comprehensive comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Train vs Test log-likelihood scatter plot
    ax1 = axes[0, 0]
    for model in metrics_df['model'].unique():
        model_data = metrics_df[metrics_df['model'] == model]
        ax1.scatter(model_data['train_log_likelihood'], model_data['test_log_likelihood'], 
                   label=model, alpha=0.7, s=100)
    
    # Add diagonal line for reference
    min_val = min(metrics_df['train_log_likelihood'].min(), metrics_df['test_log_likelihood'].min())
    max_val = max(metrics_df['train_log_likelihood'].max(), metrics_df['test_log_likelihood'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
    
    ax1.set_xlabel('Train Log-Likelihood')
    ax1.set_ylabel('Test Log-Likelihood')
    ax1.set_title('Train vs Test Log-Likelihood')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log-likelihood by model and topic configuration
    ax2 = axes[0, 1]
    pivot_train = metrics_df.pivot_table(index='model', columns='n_extra_topics', 
                                       values='train_log_likelihood', aggfunc='mean')
    pivot_train.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Train Log-Likelihood')
    ax2.set_title('Train Log-Likelihood by Model and Topic Configuration')
    ax2.legend(title='Extra Topics')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Test log-likelihood by model and topic configuration
    ax3 = axes[1, 0]
    pivot_test = metrics_df.pivot_table(index='model', columns='n_extra_topics', 
                                      values='test_log_likelihood', aggfunc='mean')
    pivot_test.plot(kind='bar', ax=ax3, width=0.8)
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Test Log-Likelihood')
    ax3.set_title('Test Log-Likelihood by Model and Topic Configuration')
    ax3.legend(title='Extra Topics')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Overfitting analysis (difference between train and test)
    ax4 = axes[1, 1]
    metrics_df['overfitting_gap'] = metrics_df['train_log_likelihood'] - metrics_df['test_log_likelihood']
    
    for model in metrics_df['model'].unique():
        model_data = metrics_df[metrics_df['model'] == model]
        ax4.scatter(model_data['n_extra_topics'], model_data['overfitting_gap'], 
                   label=model, alpha=0.7, s=100)
    
    ax4.set_xlabel('Number of Extra Topics')
    ax4.set_ylabel('Overfitting Gap (Train - Test Log-Likelihood)')
    ax4.set_title('Overfitting Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loglikelihood_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save overfitting analysis data
    overfitting_df = metrics_df[['model', 'n_extra_topics', 'train_log_likelihood', 
                                'test_log_likelihood', 'overfitting_gap']].copy()
    overfitting_df.to_csv(output_dir / 'overfitting_analysis.csv', index=False)

def get_num_activity_topics_in_topic_string(topic_str: str, identity: str) -> int:
    """
    Count the number of activity topics (V1, V2, V3, etc.) in a topic string.
    
    Args:
        topic_str: Topic combination string (e.g., "A+V1+V2")
        identity: Cell identity (e.g., "A")
    
    Returns:
        Number of activity topics
    """
    # Remove the identity and any "+" separators, then count V's
    topic_str_clean = topic_str.replace(identity, "").replace("_only", "")
    # Count V1, V2, V3, etc.
    activity_count = len(re.findall(r'V\d+', topic_str_clean))
    return activity_count

def create_comprehensive_sse_analysis(sse_df: pd.DataFrame, output_dir: Path):
    """
    Create comprehensive SSE analysis across all models and activity topic configurations.
    
    Args:
        sse_df: Combined SSE DataFrame with all models and configurations
        output_dir: Output directory for plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add number of activity topics used in each row
    sse_df = sse_df.copy()
    sse_df['num_activity_topics_used'] = sse_df.apply(
        lambda row: get_num_activity_topics_in_topic_string(row['topics'], row['identity']), 
        axis=1
    )
    
    # Create a comprehensive column name that includes topic configuration and activity combination
    def create_comprehensive_column_name(row):
        topic_config = row['topic_config']  # e.g., '7_topic_fit'
        n_topics = row['model_fit_topics']  # e.g., 7, 8, 9
        
        # Extract activity part from topic string
        topic_str = row['topics']
        identity = row['identity']
        activity_part = topic_str.replace(identity, "").replace("_only", "")
        activity_part = activity_part.strip("+")
        
        if activity_part == "":
            return f"identity_{n_topics}_topic"
        else:
            return f"identity_{n_topics}_topic_{activity_part}"
    
    sse_df['comprehensive_column'] = sse_df.apply(create_comprehensive_column_name, axis=1)
    
    # Create row label for identity and model
    sse_df['identity_model'] = sse_df['identity'] + ' | ' + sse_df['model']
    
    # Create pivot table with comprehensive columns
    pivot_df = sse_df.pivot_table(
        values='SSE', 
        index='identity_model', 
        columns='comprehensive_column', 
        aggfunc='first'
    )
    
    # Reorder columns to have a logical progression
    column_order = []
    
    # First add identity-only columns for each topic configuration
    for n_topics in sorted(sse_df['model_fit_topics'].unique()):
        identity_col = f"identity_{n_topics}_topic"
        if identity_col in pivot_df.columns:
            column_order.append(identity_col)
    
    # Then add all other activity topic combinations that exist in the data
    # Sort them by topic configuration first, then by activity combination
    existing_columns = [col for col in pivot_df.columns if col not in column_order]
    
    # Group by topic configuration
    topic_configs = {}
    for col in existing_columns:
        # Extract topic configuration from column name (e.g., "identity_7_topic_V1" -> 7)
        if col.startswith("identity_") and "_topic_" in col:
            parts = col.split("_")
            if len(parts) >= 3:
                try:
                    n_topics = int(parts[1])
                    if n_topics not in topic_configs:
                        topic_configs[n_topics] = []
                    topic_configs[n_topics].append(col)
                except ValueError:
                    continue
    
    # Add columns in order: first by topic configuration, then by activity combination
    for n_topics in sorted(topic_configs.keys()):
        # Sort activity combinations within each topic configuration
        # Put individual V's first, then combinations
        individual_vs = [col for col in topic_configs[n_topics] if col.count('V') == 1]
        combinations = [col for col in topic_configs[n_topics] if col.count('V') > 1]
        
        # Sort individual V's by number
        individual_vs.sort(key=lambda x: int(x.split('V')[1].split('_')[0]) if x.split('V')[1].split('_')[0].isdigit() else 0)
        
        # Sort combinations (they should already be in a reasonable order)
        combinations.sort()
        
        column_order.extend(individual_vs)
        column_order.extend(combinations)
    
    # Reorder the columns
    pivot_df = pivot_df[column_order]
    
    # Create the comprehensive heatmap
    print("Creating comprehensive SSE heatmap...")
    
    fig, ax = plt.subplots(figsize=(max(16, len(column_order) * 0.8), max(10, len(pivot_df) * 0.4)))
    
    # Create heatmap with annotations
    try:
        sns.heatmap(
            pivot_df, 
            annot=True,  # Show annotations
            fmt='.2f',   # Regular decimal notation
            cmap='viridis_r',  # Reverse viridis (lower SSE = better = darker)
            ax=ax,
            cbar_kws={'label': 'SSE (lower is better)'},
            linewidths=0.5,
            linecolor='white'
        )
    except Exception as e:
        print(f"Error with heatmap: {e}")
        # Try with minimal settings
        sns.heatmap(
            pivot_df, 
            annot=False,
            cmap='viridis_r', 
            ax=ax
        )
    
    # Highlight minimum value for each identity (cell type) across all models
    for identity in sse_df['identity'].unique():
        # Get all rows for this identity
        identity_mask = pivot_df.index.str.startswith(identity + ' |')
        identity_rows = pivot_df[identity_mask]
        
        if len(identity_rows) == 0:
            continue
        
        # Find the global minimum across all models for this identity
        min_val = identity_rows.min().min()
        min_col = identity_rows.min().idxmin()
        
        # Find which row (model) has this minimum value
        min_row_idx = None
        for row_idx, (row_name, row_data) in enumerate(identity_rows.iterrows()):
            if row_data[min_col] == min_val:
                min_row_idx = list(pivot_df.index).index(row_name)
                break
        
        if min_row_idx is not None:
            col_idx = list(pivot_df.columns).index(min_col)
            
            # Highlight the minimum cell with a red border
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((col_idx, min_row_idx), 1, 1, fill=False, 
                                 edgecolor='red', linewidth=3))
    
    ax.set_title('Comprehensive SSE Analysis: All Topic Configurations and Activity Combinations')
    ax.set_xlabel('Topic Configuration and Activity Combination')
    ax.set_ylabel('Cell Identity | Model')
    
    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', rotation=45)
    # Set horizontal alignment for x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), ha='right')
    
    # Rotate y-axis labels for better readability
    ax.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_sse_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the data
    pivot_df.to_csv(output_dir / 'comprehensive_sse_heatmap_data.csv')
    
    # Create summary statistics
    print("Creating summary statistics...")
    
    # Note: Best configuration recommendations removed as requested
    print("  Best configuration analysis skipped (removed as requested)")
    
    return {
        'comprehensive_pivot': pivot_df
    }

def get_topic_mapping_from_model_dir(model_dir, identity_topics, n_extra_topics):
    """
    Load the topic mapping from the columns of the theta or beta CSVs in the model output directory.
    Returns the topic order as used in the structure plot.
    """
    # Try HLDA, LDA, NMF in order
    for model in ["HLDA", "LDA", "NMF"]:
        theta_path = Path(model_dir) / model / f"{model}_theta.csv"
        if theta_path.exists():
            theta_df = pd.read_csv(theta_path, index_col=0)
            topic_order = list(theta_df.columns)
            # HLDA: topics are already labeled
            if model == "HLDA":
                return topic_order
            # LDA/NMF: topics are mapped in the CSVs
            # Try to match to config order (identities + activities)
            # If not, just use the order in the CSV
            return topic_order
    # Fallback: use identity_topics + activity topics
    activity_topics = [f"V{i+1}" for i in range(n_extra_topics)]
    return identity_topics + activity_topics

def plot_combined_cumulative_sse_lineplot(base_dir, identity_topics, max_n_activity, plot_dir, config_file):
    """
    Aggregate all sse_summary.csv files from each model configuration, extract cumulative SSE, and plot combined line plot.
    Each line: (model, n_activity_topics). X: topic names (structure plot order), Y: cumulative SSE.
    """
    # Load config for identity topics (for robust mapping)
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    # Find all sse_summary.csv files in subdirectories
    sse_files = glob.glob(str(Path(base_dir) / "*_*_fit" / "sse_summary.csv"))
    all_results = []
    for sse_file in sse_files:
        # Infer n_activity_topics from directory name
        config_dir = Path(sse_file).parent
        config_name = config_dir.name
        try:
            n_activity = int(config_name.split('_')[0]) - len(identity_topics)
        except Exception:
            n_activity = None
        # Read sse_summary.csv
        df = pd.read_csv(sse_file)
        # Get topic order from model output (structure plot order)
        topic_order = get_topic_mapping_from_model_dir(config_dir, identity_topics, n_activity)
        models = df['model'].unique()
        for model in models:
            model_df = df[df['model'] == model]
            sse_vals = []
            x_labels = []
            for i in range(1, len(topic_order)+1):
                topics_now = topic_order[:i]
                if len(topics_now) == 1:
                    topic_label = f"{topics_now[0]}_only"
                else:
                    topic_label = '+'.join(topics_now)
                row = model_df[model_df['topics'] == topic_label]
                if not row.empty:
                    sse_vals.append(row['SSE'].values[0])
                else:
                    sse_vals.append(float('nan'))
                x_labels.append(topics_now[-1])
            all_results.append({
                'model': model,
                'n_activity': n_activity,
                'sse_vals': sse_vals,
                'x_labels': x_labels,
                'config': config_name
            })
    # Plot
    plt.figure(figsize=(max(8, (len(identity_topics)+max_n_activity)*1.2), 6))
    for res in all_results:
        label = f"{res['model']} ({res['n_activity']} activity)"
        plt.plot(res['x_labels'], res['sse_vals'], marker='o', label=label)
    plt.xlabel('Topic added (structure plot order)')
    plt.ylabel('Cumulative SSE (test set)')
    plt.title('Combined Cumulative SSE as topics are added')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model/config', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(plot_dir) / 'combined_cumulative_sse_by_topic.png', dpi=200)
    plt.close()

def main():
    """Main function to run the comprehensive analysis."""
    parser = argparse.ArgumentParser(description="Comprehensive analysis for HLDA, LDA, NMF fits.")
    parser.add_argument('--base_dir', type=str, default="estimates/pbmc/heldout_1500", help='Base directory containing topic fit folders (e.g., estimates/pbmc/heldout_1500)')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory for outputs (default: <base_dir>/model_comparison)')
    parser.add_argument('--topic_configs', type=str, default='7,8,9', help='Comma-separated list of topic counts (default: 7,8,9)')
    parser.add_argument('--config_file', type=str, default="dataset_identities.yaml", help='Path to dataset identity config YAML file')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    topic_configs = [f'{n.strip()}_topic_fit' for n in args.topic_configs.split(',')]
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "model_comparison"
    config_file = args.config_file

    print("Loading metrics from all topic configurations...")
    metrics_df = load_metrics_from_fits(base_dir, topic_configs)
    print("Loading SSE results from all topic configurations...")
    sse_df = load_sse_from_fits(base_dir, topic_configs)
    if metrics_df.empty:
        print("Warning: No metrics data found!")
    else:
        print(f"Loaded {len(metrics_df)} metric records")
    if sse_df.empty:
        print("Error: No SSE data found!")
        return
    print(f"Loaded {len(sse_df)} SSE records")
    # Create analysis outputs
    if not metrics_df.empty:
        print("Creating metrics matrix output...")
        train_loglik_pivot, test_loglik_pivot = create_metrics_matrix_output(metrics_df, output_dir)
    print("Creating comprehensive SSE analysis...")
    sse_analysis = create_comprehensive_sse_analysis(sse_df, output_dir)

    # Load identity topics from config file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    # Try to infer dataset from base_dir name
    dataset_guess = None
    for ds in config.keys():
        if ds in base_dir:
            dataset_guess = ds
            break
    if dataset_guess is None:
        dataset_guess = list(config.keys())[0]
    identity_topics = config[dataset_guess]['identities']
    # Find max_n_activity from subdirs
    subdirs = [d for d in Path(base_dir).iterdir() if d.is_dir() and '_fit' in d.name]
    max_n_activity = 0
    for d in subdirs:
        try:
            n_activity = int(d.name.split('_')[0]) - len(identity_topics)
            if n_activity > max_n_activity:
                max_n_activity = n_activity
        except Exception:
            continue
    plot_dir = Path(output_dir)
    plot_combined_cumulative_sse_lineplot(base_dir, identity_topics, max_n_activity, plot_dir, config_file)

if __name__ == "__main__":
    main() 