#!/usr/bin/env python3
"""
CSV Analysis and Visualization Script for IRED Training Logs
Usage: python csv_analysis.py --csv-dir ./csv_logs --output-dir ./plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import numpy as np
from datetime import datetime

def load_training_data(csv_dir):
    """Load training metrics from CSV files"""
    csv_path = Path(csv_dir)
    
    # Find the most recent training metrics file
    training_files = list(csv_path.glob('training_metrics_*.csv'))
    if not training_files:
        print(f"No training metrics CSV files found in {csv_dir}")
        return None
    
    latest_file = max(training_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading training data from: {latest_file}")
    
    try:
        df = pd.read_csv(latest_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"Error loading training data: {e}")
        return None

def load_validation_data(csv_dir):
    """Load validation metrics from CSV files"""
    csv_path = Path(csv_dir)
    
    # Find the most recent validation metrics file
    validation_files = list(csv_path.glob('validation_metrics_*.csv'))
    if not validation_files:
        print(f"No validation metrics CSV files found in {csv_dir}")
        return None
    
    latest_file = max(validation_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading validation data from: {latest_file}")
    
    try:
        df = pd.read_csv(latest_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"Error loading validation data: {e}")
        return None

def load_energy_data(csv_dir):
    """Load energy landscape metrics from CSV files"""
    csv_path = Path(csv_dir)
    
    # Find the most recent energy metrics file
    energy_files = list(csv_path.glob('energy_metrics_*.csv'))
    if not energy_files:
        print(f"No energy metrics CSV files found in {csv_dir}")
        return None
    
    latest_file = max(energy_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading energy data from: {latest_file}")
    
    try:
        df = pd.read_csv(latest_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"Error loading energy data: {e}")
        return None

def plot_training_metrics(df_train, output_dir):
    """Plot training loss curves and timing metrics"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Loss curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics Over Time', fontsize=16)
    
    # Total loss
    axes[0, 0].plot(df_train['step'], df_train['total_loss'], 'b-', alpha=0.7)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Component losses
    axes[0, 1].plot(df_train['step'], df_train['loss_denoise'], 'r-', alpha=0.7, label='Denoise Loss')
    axes[0, 1].plot(df_train['step'], df_train['loss_energy'], 'g-', alpha=0.7, label='Energy Loss')
    axes[0, 1].plot(df_train['step'], df_train['loss_opt'], 'm-', alpha=0.7, label='Opt Loss')
    axes[0, 1].set_title('Component Losses')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Timing metrics
    axes[1, 0].plot(df_train['step'], df_train['data_time'], 'c-', alpha=0.7, label='Data Time')
    axes[1, 0].plot(df_train['step'], df_train['nn_time'], 'orange', alpha=0.7, label='NN Time')
    axes[1, 0].set_title('Timing Metrics')
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 1].plot(df_train['step'], df_train['learning_rate'], 'purple', alpha=0.7)
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training metrics plot saved to {output_path / 'training_metrics.png'}")

def plot_validation_metrics(df_val, output_dir):
    """Plot validation accuracy and performance metrics"""
    output_path = Path(output_dir)
    
    if df_val is None or len(df_val) == 0:
        print("No validation data to plot")
        return
    
    # Group by dataset and metric
    datasets = df_val['dataset_name'].unique()
    
    for dataset in datasets:
        dataset_data = df_val[df_val['dataset_name'] == dataset]
        metrics = dataset_data['metric_name'].unique()
        
        if len(metrics) == 0:
            continue
        
        # Create subplot grid
        n_metrics = len(metrics)
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Validation Metrics - {dataset}', fontsize=16)
        
        for i, metric in enumerate(metrics):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            metric_data = dataset_data[dataset_data['metric_name'] == metric]
            ax.plot(metric_data['step'], metric_data['metric_value'], 'o-', alpha=0.7)
            ax.set_title(f'{metric}')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_metrics, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / f'validation_metrics_{dataset}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Validation metrics plot for {dataset} saved to {output_path / f'validation_metrics_{dataset}.png'}")

def plot_energy_landscape_analysis(df_energy, output_dir):
    """Plot energy landscape analysis including adversarial corruption effects"""
    output_path = Path(output_dir)
    
    if df_energy is None or len(df_energy) == 0:
        print("No energy data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Energy Landscape Analysis', fontsize=16)
    
    # Energy difference over time
    axes[0, 0].plot(df_energy['step'], df_energy['energy_diff'], 'b-', alpha=0.7)
    axes[0, 0].set_title('Energy Difference (Neg - Pos)')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Energy Difference')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Curriculum weight progression
    axes[0, 1].plot(df_energy['step'], df_energy['curriculum_weight'], 'r-', alpha=0.7)
    axes[0, 1].set_title('Adversarial Corruption Curriculum Weight')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Curriculum Weight')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Corruption type distribution
    corruption_counts = df_energy['corruption_type'].value_counts()
    axes[1, 0].pie(corruption_counts.values, labels=corruption_counts.index, autopct='%1.1f%%')
    axes[1, 0].set_title('Corruption Type Distribution')
    
    # Energy difference by corruption type
    if len(df_energy['corruption_type'].unique()) > 1:
        df_energy.boxplot(column='energy_diff', by='corruption_type', ax=axes[1, 1])
        axes[1, 1].set_title('Energy Difference by Corruption Type')
        axes[1, 1].set_xlabel('Corruption Type')
        axes[1, 1].set_ylabel('Energy Difference')
    else:
        axes[1, 1].hist(df_energy['energy_diff'], bins=20, alpha=0.7)
        axes[1, 1].set_title('Energy Difference Distribution')
        axes[1, 1].set_xlabel('Energy Difference')
        axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(output_path / 'energy_landscape_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Energy landscape analysis plot saved to {output_path / 'energy_landscape_analysis.png'}")

def generate_summary_report(df_train, df_val, df_energy, output_dir):
    """Generate a summary report with key statistics"""
    output_path = Path(output_dir)
    
    report_path = output_path / 'training_summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("IRED Training Summary Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Training statistics
        if df_train is not None and len(df_train) > 0:
            f.write("TRAINING STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total training steps: {df_train['step'].max()}\n")
            f.write(f"Final total loss: {df_train['total_loss'].iloc[-1]:.6f}\n")
            f.write(f"Final denoise loss: {df_train['loss_denoise'].iloc[-1]:.6f}\n")
            f.write(f"Final energy loss: {df_train['loss_energy'].iloc[-1]:.6f}\n")
            f.write(f"Average data time: {df_train['data_time'].mean():.3f}s\n")
            f.write(f"Average NN time: {df_train['nn_time'].mean():.3f}s\n\n")
        
        # Validation statistics
        if df_val is not None and len(df_val) > 0:
            f.write("VALIDATION STATISTICS:\n")
            f.write("-" * 22 + "\n")
            for dataset in df_val['dataset_name'].unique():
                f.write(f"\nDataset: {dataset}\n")
                dataset_data = df_val[df_val['dataset_name'] == dataset]
                for metric in dataset_data['metric_name'].unique():
                    metric_data = dataset_data[dataset_data['metric_name'] == metric]
                    if len(metric_data) > 0:
                        latest_value = metric_data['metric_value'].iloc[-1]
                        f.write(f"  {metric}: {latest_value:.6f}\n")
        
        # Energy landscape statistics
        if df_energy is not None and len(df_energy) > 0:
            f.write("\nENERGY LANDSCAPE STATISTICS:\n")
            f.write("-" * 29 + "\n")
            f.write(f"Average energy difference: {df_energy['energy_diff'].mean():.6f}\n")
            f.write(f"Max curriculum weight reached: {df_energy['curriculum_weight'].max():.3f}\n")
            corruption_counts = df_energy['corruption_type'].value_counts()
            f.write("Corruption type usage:\n")
            for corruption_type, count in corruption_counts.items():
                percentage = (count / len(df_energy)) * 100
                f.write(f"  {corruption_type}: {count} ({percentage:.1f}%)\n")
    
    print(f"Summary report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze and visualize IRED training logs')
    parser.add_argument('--csv-dir', type=str, default='./csv_logs',
                       help='Directory containing CSV log files')
    parser.add_argument('--output-dir', type=str, default='./plots',
                       help='Directory to save plots and analysis')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading training data...")
    df_train = load_training_data(args.csv_dir)
    
    print("Loading validation data...")
    df_val = load_validation_data(args.csv_dir)
    
    print("Loading energy data...")
    df_energy = load_energy_data(args.csv_dir)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate plots
    if df_train is not None:
        print("Plotting training metrics...")
        plot_training_metrics(df_train, args.output_dir)
    
    if df_val is not None:
        print("Plotting validation metrics...")
        plot_validation_metrics(df_val, args.output_dir)
    
    if df_energy is not None:
        print("Plotting energy landscape analysis...")
        plot_energy_landscape_analysis(df_energy, args.output_dir)
    
    # Generate summary report
    print("Generating summary report...")
    generate_summary_report(df_train, df_val, df_energy, args.output_dir)
    
    print(f"\nAnalysis complete! Check {args.output_dir} for plots and reports.")

if __name__ == '__main__':
    main()