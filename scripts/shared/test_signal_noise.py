#!/usr/bin/env python3
"""
Test signal-to-noise computation across different DE levels.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import io
from contextlib import redirect_stdout
from noise import compute_signal_to_noise

def test_signal_noise_across_de_levels(log_file="signal_noise.log"):
    """Test signal-to-noise computation across different DE levels."""
    
    # Test configurations
    test_configs = [
        {
            'name': 'AB_V1',
            'data_root': 'data/AB_V1',
            'estimates_root': 'estimates/AB_V1',
            'n_identity_topics': 2,
            'libsize_mean': 1500
        },
        {
            'name': 'ABCD_V1V2', 
            'data_root': 'data/ABCD_V1V2',
            'estimates_root': 'estimates/ABCD_V1V2',
            'n_identity_topics': 4,
            'libsize_mean': 1500
        }
    ]
    
    all_results = []
    
    # Open log file for detailed output
    with open(log_file, 'w') as log:
        log.write("Signal-to-Noise Analysis Log\n")
        log.write("=" * 50 + "\n\n")
        
        for config in test_configs:
            print(f"\n{'='*60}")
            print(f"Testing {config['name']}")
            print(f"{'='*60}")
            
            log.write(f"\n{'='*60}\n")
            log.write(f"Testing {config['name']}\n")
            log.write(f"{'='*60}\n")
            
            data_root = Path(config['data_root'])
            estimates_root = Path(config['estimates_root'])
            
            if not data_root.exists():
                print(f"Warning: {data_root} does not exist, skipping...")
                log.write(f"Warning: {data_root} does not exist, skipping...\n")
                continue
                
            # Create estimates directory if it doesn't exist
            estimates_root.mkdir(parents=True, exist_ok=True)
            
            # Find all DE_mean directories
            de_dirs = [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith('DE_mean_')]
            de_dirs.sort(key=lambda x: float(x.name.split('_')[-1]))
            
            print(f"Found {len(de_dirs)} DE levels to process...")
            
            simulation_results = []
            
            for de_dir in de_dirs:
                de_level = float(de_dir.name.split('_')[-1])
                print(f"  Processing DE level: {de_level}...", end=' ')
                
                log.write(f"\n--- Testing DE level: {de_level} ---\n")
                
                try:
                    # Capture detailed output from noise computation
                    output_buffer = io.StringIO()
                    with redirect_stdout(output_buffer):
                        result = compute_signal_to_noise(
                            de_dir, 
                            libsize_mean=config['libsize_mean'],
                            n_identity_topics=config['n_identity_topics'],
                            verbose=False  # Keep quiet in terminal
                        )
                    
                    # Write detailed output to log file
                    log.write(output_buffer.getvalue())
                    
                    # Store results
                    result_dict = {
                        'simulation': config['name'],
                        'de_level': de_level,
                        'total_activity_signal': result['total_activity_signal'],
                        'total_identity_signal': result['total_identity_signal'],
                        'total_noise': result['total_activity_noise'],
                        'snr': result['overall_snr'],
                        'noise_to_identity_ratio': result['noise_to_identity_ratio'],
                        'noise_to_activity_ratio': result['noise_to_activity_ratio'],
                        'n_activity_topics': result['n_activity_topics'],
                        'n_identity_topics': result['n_identity_topics'],
                        'n_cells': result['n_cells'],
                        'n_genes': result['n_genes']
                    }
                    
                    simulation_results.append(result_dict)
                    all_results.append(result_dict)
                    
                    print(f"✓ SNR = {result['overall_snr']:.6f}")
                    
                    # Save individual DE level results
                    de_estimates_dir = estimates_root / de_dir.name / "signal_noise_analysis"
                    de_estimates_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save individual result
                    individual_df = pd.DataFrame([result_dict])
                    individual_df.to_csv(de_estimates_dir / "signal_noise_results.csv", index=False)
                    
                except Exception as e:
                    print(f"✗ Error: {e}")
                    log.write(f"Error processing {de_dir}: {e}\n")
                    continue
            
            # Create simulation-specific results DataFrame
            if simulation_results:
                sim_df = pd.DataFrame(simulation_results)
                
                # Save simulation-specific results
                sim_output_file = estimates_root / "signal_noise_results.csv"
                sim_df.to_csv(sim_output_file, index=False)
                
                # Create simulation-specific plots
                sim_plot_file = estimates_root / "signal_noise_analysis.png"
                create_signal_noise_plots(sim_df, output_file=sim_plot_file, title_suffix=f" - {config['name']}")
                
                print(f"  Results saved to: {sim_output_file}")
                print(f"  Plot saved to: {sim_plot_file}")
    
    # Create overall results DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
        print(f"\n{'='*60}")
        print("SUMMARY RESULTS")
        print(f"{'='*60}")
        print(df.to_string(index=False))
        
        # Save overall results
        output_file = "signal_noise_results_all_simulations.csv"
        df.to_csv(output_file, index=False)
        print(f"\nOverall results saved to: {output_file}")
        print(f"Detailed log saved to: {log_file}")
        
        # Create overall comparison plots
        create_comparison_plots(df)
        
        return df
    else:
        print("No results to analyze.")
        return None

def create_signal_noise_plots(df, output_file="signal_noise_analysis.png", title_suffix=""):
    """Create plots showing signal-to-noise trends for a single simulation."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Activity Signal vs DE level
    ax1 = axes[0, 0]
    for sim in df['simulation'].unique():
        sim_data = df[df['simulation'] == sim]
        ax1.plot(sim_data['de_level'], sim_data['total_activity_signal'], 's-', label=sim, linewidth=2, markersize=6)
    ax1.set_xlabel('DE Level')
    ax1.set_ylabel('Total Activity Signal')
    ax1.set_title(f'Activity Signal vs DE Level{title_suffix}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Activity vs Identity Signal (log scale)
    ax2 = axes[0, 1]
    for sim in df['simulation'].unique():
        sim_data = df[df['simulation'] == sim]
        ax2.semilogy(sim_data['de_level'], sim_data['total_activity_signal'], 's-', label=f'{sim} Activity', linewidth=2, markersize=6)
        ax2.semilogy(sim_data['de_level'], sim_data['total_identity_signal'], 'd-', label=f'{sim} Identity', linewidth=2, markersize=6)
    ax2.set_xlabel('DE Level')
    ax2.set_ylabel('Signal Magnitude (log scale)')
    ax2.set_title(f'Activity vs Identity Signal (Log Scale){title_suffix}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Activity Signal vs Noise (log scale)
    ax3 = axes[0, 2]
    for sim in df['simulation'].unique():
        sim_data = df[df['simulation'] == sim]
        ax3.semilogy(sim_data['de_level'], sim_data['total_activity_signal'], 's-', label=f'{sim} Activity', linewidth=2, markersize=6)
        ax3.semilogy(sim_data['de_level'], sim_data['total_noise'], '^-', label=f'{sim} Noise', linewidth=2, markersize=6)
    ax3.set_xlabel('DE Level')
    ax3.set_ylabel('Magnitude (log scale)')
    ax3.set_title(f'Activity Signal vs Noise (Log Scale){title_suffix}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Identity Signal vs Noise (log scale)
    ax4 = axes[1, 0]
    for sim in df['simulation'].unique():
        sim_data = df[df['simulation'] == sim]
        ax4.semilogy(sim_data['de_level'], sim_data['total_identity_signal'], 'd-', label=f'{sim} Identity', linewidth=2, markersize=6)
        ax4.semilogy(sim_data['de_level'], sim_data['total_noise'], '^-', label=f'{sim} Noise', linewidth=2, markersize=6)
    ax4.set_xlabel('DE Level')
    ax4.set_ylabel('Magnitude (log scale)')
    ax4.set_title(f'Identity Signal vs Noise (Log Scale){title_suffix}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Signal-to-Noise Ratio
    ax5 = axes[1, 1]
    for sim in df['simulation'].unique():
        sim_data = df[df['simulation'] == sim]
        ax5.plot(sim_data['de_level'], sim_data['snr'], 'o-', label=sim, linewidth=2, markersize=6)
    ax5.set_xlabel('DE Level')
    ax5.set_ylabel('Signal-to-Noise Ratio')
    ax5.set_title(f'Signal-to-Noise Ratio vs DE Level{title_suffix}')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Activity/Identity Signal Ratio
    ax6 = axes[1, 2]
    for sim in df['simulation'].unique():
        sim_data = df[df['simulation'] == sim]
        # Avoid division by zero
        ratio = sim_data['total_activity_signal'] / np.maximum(sim_data['total_identity_signal'], 1e-10)
        ax6.plot(sim_data['de_level'], ratio, '*-', label=sim, linewidth=2, markersize=6)
    ax6.set_xlabel('DE Level')
    ax6.set_ylabel('Activity/Identity Signal Ratio')
    ax6.set_title(f'Activity/Identity Signal Ratio vs DE Level{title_suffix}')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {output_file}")
    plt.close()  # Close to free memory

def create_comparison_plots(df):
    """Create comparison plots across all simulations."""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    
    # Plot 1: Activity Signal comparison
    ax1 = axes[0, 0]
    for sim in df['simulation'].unique():
        sim_data = df[df['simulation'] == sim]
        ax1.plot(sim_data['de_level'], sim_data['total_activity_signal'], 's-', label=sim, linewidth=2, markersize=6)
    ax1.set_xlabel('DE Level')
    ax1.set_ylabel('Total Activity Signal')
    ax1.set_title('Activity Signal Comparison Across Simulations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Activity vs Identity Signal comparison (log scale)
    ax2 = axes[0, 1]
    for sim in df['simulation'].unique():
        sim_data = df[df['simulation'] == sim]
        ax2.semilogy(sim_data['de_level'], sim_data['total_activity_signal'], 's-', label=f'{sim} Activity', linewidth=2, markersize=6)
        ax2.semilogy(sim_data['de_level'], sim_data['total_identity_signal'], 'd-', label=f'{sim} Identity', linewidth=2, markersize=6)
    ax2.set_xlabel('DE Level')
    ax2.set_ylabel('Signal Magnitude (log scale)')
    ax2.set_title('Activity vs Identity Signal Comparison (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Activity Signal vs Noise comparison (log scale)
    ax3 = axes[0, 2]
    for sim in df['simulation'].unique():
        sim_data = df[df['simulation'] == sim]
        ax3.semilogy(sim_data['de_level'], sim_data['total_activity_signal'], 's-', label=f'{sim} Activity', linewidth=2, markersize=6)
        ax3.semilogy(sim_data['de_level'], sim_data['total_noise'], '^-', label=f'{sim} Noise', linewidth=2, markersize=6)
    ax3.set_xlabel('DE Level')
    ax3.set_ylabel('Magnitude (log scale)')
    ax3.set_title('Activity Signal vs Noise Comparison (Log Scale)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Identity Signal vs Noise comparison (log scale)
    ax4 = axes[1, 0]
    for sim in df['simulation'].unique():
        sim_data = df[df['simulation'] == sim]
        ax4.semilogy(sim_data['de_level'], sim_data['total_identity_signal'], 'd-', label=f'{sim} Identity', linewidth=2, markersize=6)
        ax4.semilogy(sim_data['de_level'], sim_data['total_noise'], '^-', label=f'{sim} Noise', linewidth=2, markersize=6)
    ax4.set_xlabel('DE Level')
    ax4.set_ylabel('Magnitude (log scale)')
    ax4.set_title('Identity Signal vs Noise Comparison (Log Scale)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Signal-to-Noise Ratio comparison
    ax5 = axes[1, 1]
    for sim in df['simulation'].unique():
        sim_data = df[df['simulation'] == sim]
        ax5.plot(sim_data['de_level'], sim_data['snr'], 'o-', label=sim, linewidth=2, markersize=6)
    ax5.set_xlabel('DE Level')
    ax5.set_ylabel('Signal-to-Noise Ratio')
    ax5.set_title('Signal-to-Noise Ratio Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Activity/Identity Signal Ratio comparison
    ax6 = axes[1, 2]
    for sim in df['simulation'].unique():
        sim_data = df[df['simulation'] == sim]
        # Avoid division by zero
        ratio = sim_data['total_activity_signal'] / np.maximum(sim_data['total_identity_signal'], 1e-10)
        ax6.plot(sim_data['de_level'], ratio, '*-', label=sim, linewidth=2, markersize=6)
    ax6.set_xlabel('DE Level')
    ax6.set_ylabel('Activity/Identity Signal Ratio')
    ax6.set_title('Activity/Identity Signal Ratio Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('signal_noise_comparison_all_simulations.png', dpi=300, bbox_inches='tight')
    print(f"Comparison plots saved to: signal_noise_comparison_all_simulations.png")
    plt.close()

if __name__ == "__main__":
    test_signal_noise_across_de_levels() 