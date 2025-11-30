# -*- coding: utf-8 -*-
"""
Assignment 2 - Network Diffusion Analysis
Visualization Notebook

Creates publication-quality plots in LaTeX format for threshold diffusion experiments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import os

# =============================================================================
# LaTeX Formatting Setup
# =============================================================================

# Set up LaTeX-style plotting
plt.rcParams.update({
    'text.usetex': False,  # Set to True if you have LaTeX installed
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Color palette for strategies
STRATEGY_COLORS = {
    'random': '#1f77b4',      # blue
    'degree': '#ff7f0e',      # orange
    'betweenness': '#2ca02c'  # green
}

STRATEGY_NAMES = {
    'random': 'Random',
    'degree': 'Degree',
    'betweenness': 'Betweenness'
}

# =============================================================================
# Data Loading
# =============================================================================

def load_data():
    """Load all experiment data files."""
    data_files = {
        'per_run': 'ass2_per_run_all.csv',
        'aggregate': 'ass2_aggregate_all.csv', 
        'time_series': 'ass2_ts_all.csv',
        'mean_curves': 'ass2_mean_curves_all.csv',
        'adoption_times': 'ass2_adoption_times_representative.csv'
    }
    
    data = {}
    for name, file in data_files.items():
        if os.path.exists(file):
            data[name] = pd.read_csv(file)
            print(f"Loaded {file}: {len(data[name])} rows")
        else:
            print(f"Warning: {file} not found")
    
    return data

# Load all data
data = load_data()

# =============================================================================
# Plotting Functions
# =============================================================================

def create_strategy_comparison_plots(data):
    """Create comprehensive strategy comparison plots."""
    
    if 'aggregate' not in data:
        print("No aggregate data found")
        return
    
    agg_df = data['aggregate']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    metrics = [
        ('mean_final_fraction', 'Final Adoption Rate', 'Fraction'),
        ('mean_t_50', 'Time to 50% Adoption ($t_{50}$)', 'Steps'),
        ('mean_t_90', 'Time to 90% Adoption ($t_{90}$)', 'Steps'), 
        ('mean_auc', 'Area Under Curve (AUC)', 'AUC')
    ]
    
    for idx, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[idx]
        
        # Create grouped bar plot
        strategies = agg_df['strategy'].unique()
        x_pos = np.arange(len(strategies))
        
        bars = []
        for i, strategy in enumerate(strategies):
            strategy_data = agg_df[agg_df['strategy'] == strategy]
            color = STRATEGY_COLORS.get(strategy, 'gray')
            
            # Plot mean value
            bar = ax.bar(x_pos[i], strategy_data[metric].mean(), 
                        color=color, alpha=0.7, label=STRATEGY_NAMES.get(strategy, strategy))
            bars.append(bar)
            
            # Add error bars if available
            std_col = metric.replace('mean_', 'std_')
            if std_col in strategy_data.columns:
                ax.errorbar(x_pos[i], strategy_data[metric].mean(),
                           yerr=strategy_data[std_col].mean(), 
                           fmt='none', color='black', capsize=5, capthick=1)
        
        ax.set_title(title, fontweight='bold', pad=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([STRATEGY_NAMES.get(s, s) for s in strategies])
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('strategy_comparison_metrics.png', dpi=300, bbox_inches='tight')

def create_adoption_curves_plot(data):
    """Create adoption curve plots across different parameter settings."""
    
    if 'mean_curves' not in data:
        print("No mean curves data found")
        return
    
    curves_df = data['mean_curves']
    
    # Plot 1: Fixed seed fraction, varying tau
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Varying tau, fixed seed fraction
    seed_frac = curves_df['seed_fraction'].median()  # Use median seed fraction
    tau_subset = curves_df[curves_df['seed_fraction'] == seed_frac]
    
    for strategy in tau_subset['strategy'].unique():
        strategy_data = tau_subset[tau_subset['strategy'] == strategy]
        color = STRATEGY_COLORS.get(strategy, 'gray')
        
        for tau in sorted(strategy_data['tau'].unique())[::2]:  # Plot every other tau for clarity
            tau_data = strategy_data[strategy_data['tau'] == tau]
            ax1.plot(tau_data['time_step'], tau_data['mean_fraction'], 
                    color=color, alpha=0.7, linewidth=1.5,
                    label=f'{STRATEGY_NAMES.get(strategy, strategy)} (τ={tau})' 
                    if tau == sorted(strategy_data['tau'].unique())[0] else "")
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Fraction of Active Nodes')
    ax1.set_title(f'Adoption Curves (Seed Fraction = {seed_frac})', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Right: Varying seed fraction, fixed tau
    tau_val = curves_df['tau'].median()  # Use median tau
    seed_subset = curves_df[curves_df['tau'] == tau_val]
    
    for strategy in seed_subset['strategy'].unique():
        strategy_data = seed_subset[seed_subset['strategy'] == strategy]
        color = STRATEGY_COLORS.get(strategy, 'gray')
        
        for sf in sorted(strategy_data['seed_fraction'].unique())[::2]:  # Plot every other seed fraction
            sf_data = strategy_data[strategy_data['seed_fraction'] == sf]
            ax2.plot(sf_data['time_step'], sf_data['mean_fraction'], 
                    color=color, alpha=0.7, linewidth=1.5,
                    label=f'{STRATEGY_NAMES.get(strategy, strategy)} (sf={sf})' 
                    if sf == sorted(strategy_data['seed_fraction'].unique())[0] else "")
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Fraction of Active Nodes')
    ax2.set_title(f'Adoption Curves (τ = {tau_val})', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('adoption_curves_comparison.png', dpi=300, bbox_inches='tight')

def create_parameter_heatmaps(data):
    """Create heatmaps showing how metrics vary with tau and seed fraction."""
    
    if 'aggregate' not in data:
        print("No aggregate data found")
        return
    
    agg_df = data['aggregate']
    
    # Create 2x2 grid of heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    metrics = ['mean_final_fraction', 'mean_t_50', 'mean_t_90', 'mean_auc']
    titles = ['Final Adoption Rate', 'Time to 50% Adoption', 
              'Time to 90% Adoption', 'Area Under Curve']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx//2, idx%2]
        
        # Pivot table for heatmap
        pivot_data = agg_df.pivot_table(
            values=metric, 
            index='seed_fraction', 
            columns='tau', 
            aggfunc='mean'
        )
        
        # Create heatmap
        im = ax.imshow(pivot_data.values, cmap='viridis', aspect='auto')
        
        # Set labels
        ax.set_xticks(np.arange(len(pivot_data.columns)))
        ax.set_xticklabels([f'{x:.2f}' for x in pivot_data.columns])
        ax.set_yticks(np.arange(len(pivot_data.index)))
        ax.set_yticklabels([f'{x:.4f}' for x in pivot_data.index])
        
        ax.set_xlabel('Threshold (τ)')
        ax.set_ylabel('Seed Fraction')
        ax.set_title(title, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Add value annotations
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                text = ax.text(j, i, f'{pivot_data.iloc[i, j]:.3f}',
                              ha="center", va="center", color="w", fontsize=8)
    
    plt.tight_layout()
    plt.savefig('parameter_heatmaps.png', dpi=300, bbox_inches='tight')

def create_adoption_time_analysis(data):
    """Create plots analyzing adoption times vs node degree."""
    
    if 'adoption_times' not in data:
        print("No adoption times data found")
        return
    
    adopt_df = data['adoption_times']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Box plot of adoption times by strategy
    adopt_df_filtered = adopt_df[adopt_df['adoption_time'] >= 0]  # Only adopted nodes
    
    strategies = adopt_df_filtered['strategy'].unique()
    box_data = [adopt_df_filtered[adopt_df_filtered['strategy'] == s]['adoption_time'] 
                for s in strategies]
    
    box_plot = ax1.boxplot(box_data, labels=[STRATEGY_NAMES.get(s, s) for s in strategies],
                          patch_artist=True)
    
    # Color the boxes
    for patch, strategy in zip(box_plot['boxes'], strategies):
        patch.set_facecolor(STRATEGY_COLORS.get(strategy, 'gray'))
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Adoption Time (Steps)')
    ax1.set_title('Distribution of Adoption Times by Strategy', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Scatter plot of adoption time vs degree
    for strategy in strategies:
        strategy_data = adopt_df_filtered[adopt_df_filtered['strategy'] == strategy]
        color = STRATEGY_COLORS.get(strategy, 'gray')
        ax2.scatter(strategy_data['degree'], strategy_data['adoption_time'], 
                   alpha=0.6, s=20, color=color, 
                   label=STRATEGY_NAMES.get(strategy, strategy))
    
    ax2.set_xlabel('Node Degree')
    ax2.set_ylabel('Adoption Time (Steps)')
    ax2.set_title('Adoption Time vs Node Degree', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('adoption_time_analysis.png', dpi=300, bbox_inches='tight')

def create_performance_tradeoff_plots(data):
    """Create plots showing tradeoffs between different performance metrics."""
    
    if 'per_run' not in data:
        print("No per-run data found")
        return
    
    per_run_df = data['per_run']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Speed vs completeness tradeoff
    for strategy in per_run_df['strategy'].unique():
        strategy_data = per_run_df[per_run_df['strategy'] == strategy]
        color = STRATEGY_COLORS.get(strategy, 'gray')
        
        ax1.scatter(strategy_data['t_50'], strategy_data['final_fraction'],
                   alpha=0.6, s=40, color=color,
                   label=STRATEGY_NAMES.get(strategy, strategy))
    
    ax1.set_xlabel('Time to 50% Adoption ($t_{50}$)')
    ax1.set_ylabel('Final Adoption Rate')
    ax1.set_title('Speed vs Completeness Trade-off', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Efficiency vs robustness tradeoff
    for strategy in per_run_df['strategy'].unique():
        strategy_data = per_run_df[per_run_df['strategy'] == strategy]
        color = STRATEGY_COLORS.get(strategy, 'gray')
        
        ax2.scatter(strategy_data['n_steps'], strategy_data['auc'],
                   alpha=0.6, s=40, color=color,
                   label=STRATEGY_NAMES.get(strategy, strategy))
    
    ax2.set_xlabel('Number of Steps to Saturation')
    ax2.set_ylabel('Area Under Curve (AUC)')
    ax2.set_title('Efficiency vs Robustness Trade-off', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_tradeoffs.png', dpi=300, bbox_inches='tight')

def create_summary_statistics(data):
    """Print summary statistics and create a summary table."""
    
    if 'aggregate' not in data:
        print("No aggregate data found")
        return
    
    agg_df = data['aggregate']
    
    print("="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Group by strategy and compute overall means
    summary = agg_df.groupby('strategy').agg({
        'mean_final_fraction': 'mean',
        'mean_t_50': 'mean', 
        'mean_t_90': 'mean',
        'mean_auc': 'mean',
        'mean_n_steps': 'mean'
    }).round(4)
    
    print(summary)
    print("\n")
    
    # Best performing strategy for each metric
    metrics = {
        'mean_final_fraction': 'Final Adoption Rate (higher better)',
        'mean_t_50': 'Time to 50% Adoption (lower better)',
        'mean_t_90': 'Time to 90% Adoption (lower better)', 
        'mean_auc': 'Area Under Curve (higher better)',
        'mean_n_steps': 'Steps to Saturation (lower better)'
    }
    
    print("BEST PERFORMING STRATEGIES:")
    print("-" * 40)
    
    for metric, description in metrics.items():
        if 't_50' in metric or 't_90' in metric or 'n_steps' in metric:
            best_strategy = summary[metric].idxmin()
            best_value = summary[metric].min()
            comparison = "lower"
        else:
            best_strategy = summary[metric].idxmax() 
            best_value = summary[metric].max()
            comparison = "higher"
            
        print(f"{description:35} | {STRATEGY_NAMES.get(best_strategy, best_strategy):12} ({best_value:.4f}, {comparison})")

# =============================================================================
# Execute All Plots
# =============================================================================

print("Generating Network Diffusion Analysis Plots...")
print("="*60)

# 1. Summary Statistics
create_summary_statistics(data)

# 2. Strategy Comparison (Group 1: Overall Performance)
print("\n1. Creating Strategy Comparison Plots...")
create_strategy_comparison_plots(data)

# 3. Adoption Dynamics (Group 2: Temporal Behavior)  
print("\n2. Creating Adoption Curve Plots...")
create_adoption_curves_plot(data)

# 4. Parameter Sensitivity (Group 3: Parameter Effects)
print("\n3. Creating Parameter Heatmaps...")
create_parameter_heatmaps(data)

# 5. Performance Tradeoffs (Group 4: Tradeoff Analysis)
print("\n4. Creating Performance Tradeoff Plots...")
create_performance_tradeoff_plots(data)

# 6. Node-level Analysis (Group 5: Micro-level Patterns)
print("\n5. Creating Adoption Time Analysis...")
create_adoption_time_analysis(data)

print("\n" + "="*60)
print("All plots generated successfully!")
print("PNG files saved in current directory.")
print("="*60)