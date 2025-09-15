diff --git a/CLAUDE.md b/CLAUDE.md
new file mode 100644
index 0000000..e69de29
diff --git a/csv_analysis.py b/csv_analysis.py
new file mode 100644
index 0000000..1f146ef
--- /dev/null
+++ b/csv_analysis.py
@@ -0,0 +1,322 @@
+#!/usr/bin/env python3
+"""
+CSV Analysis and Visualization Script for IRED Training Logs
+Usage: python csv_analysis.py --csv-dir ./csv_logs --output-dir ./plots
+"""
+
+import pandas as pd
+import matplotlib.pyplot as plt
+import seaborn as sns
+import argparse
+from pathlib import Path
+import numpy as np
+from datetime import datetime
+
+def load_training_data(csv_dir):
+    """Load training metrics from CSV files"""
+    csv_path = Path(csv_dir)
+    
+    # Find the most recent training metrics file
+    training_files = list(csv_path.glob('training_metrics_*.csv'))
+    if not training_files:
+        print(f"No training metrics CSV files found in {csv_dir}")
+        return None
+    
+    latest_file = max(training_files, key=lambda x: x.stat().st_mtime)
+    print(f"Loading training data from: {latest_file}")
+    
+    try:
+        df = pd.read_csv(latest_file)
+        df['timestamp'] = pd.to_datetime(df['timestamp'])
+        return df
+    except Exception as e:
+        print(f"Error loading training data: {e}")
+        return None
+
+def load_validation_data(csv_dir):
+    """Load validation metrics from CSV files"""
+    csv_path = Path(csv_dir)
+    
+    # Find the most recent validation metrics file
+    validation_files = list(csv_path.glob('validation_metrics_*.csv'))
+    if not validation_files:
+        print(f"No validation metrics CSV files found in {csv_dir}")
+        return None
+    
+    latest_file = max(validation_files, key=lambda x: x.stat().st_mtime)
+    print(f"Loading validation data from: {latest_file}")
+    
+    try:
+        df = pd.read_csv(latest_file)
+        df['timestamp'] = pd.to_datetime(df['timestamp'])
+        return df
+    except Exception as e:
+        print(f"Error loading validation data: {e}")
+        return None
+
+def load_energy_data(csv_dir):
+    """Load energy landscape metrics from CSV files"""
+    csv_path = Path(csv_dir)
+    
+    # Find the most recent energy metrics file
+    energy_files = list(csv_path.glob('energy_metrics_*.csv'))
+    if not energy_files:
+        print(f"No energy metrics CSV files found in {csv_dir}")
+        return None
+    
+    latest_file = max(energy_files, key=lambda x: x.stat().st_mtime)
+    print(f"Loading energy data from: {latest_file}")
+    
+    try:
+        df = pd.read_csv(latest_file)
+        df['timestamp'] = pd.to_datetime(df['timestamp'])
+        return df
+    except Exception as e:
+        print(f"Error loading energy data: {e}")
+        return None
+
+def plot_training_metrics(df_train, output_dir):
+    """Plot training loss curves and timing metrics"""
+    output_path = Path(output_dir)
+    output_path.mkdir(exist_ok=True)
+    
+    # Set style
+    plt.style.use('seaborn-v0_8-darkgrid')
+    
+    # 1. Loss curves
+    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
+    fig.suptitle('Training Metrics Over Time', fontsize=16)
+    
+    # Total loss
+    axes[0, 0].plot(df_train['step'], df_train['total_loss'], 'b-', alpha=0.7)
+    axes[0, 0].set_title('Total Loss')
+    axes[0, 0].set_xlabel('Training Step')
+    axes[0, 0].set_ylabel('Loss')
+    axes[0, 0].grid(True, alpha=0.3)
+    
+    # Component losses
+    axes[0, 1].plot(df_train['step'], df_train['loss_denoise'], 'r-', alpha=0.7, label='Denoise Loss')
+    axes[0, 1].plot(df_train['step'], df_train['loss_energy'], 'g-', alpha=0.7, label='Energy Loss')
+    axes[0, 1].plot(df_train['step'], df_train['loss_opt'], 'm-', alpha=0.7, label='Opt Loss')
+    axes[0, 1].set_title('Component Losses')
+    axes[0, 1].set_xlabel('Training Step')
+    axes[0, 1].set_ylabel('Loss')
+    axes[0, 1].legend()
+    axes[0, 1].grid(True, alpha=0.3)
+    
+    # Timing metrics
+    axes[1, 0].plot(df_train['step'], df_train['data_time'], 'c-', alpha=0.7, label='Data Time')
+    axes[1, 0].plot(df_train['step'], df_train['nn_time'], 'orange', alpha=0.7, label='NN Time')
+    axes[1, 0].set_title('Timing Metrics')
+    axes[1, 0].set_xlabel('Training Step')
+    axes[1, 0].set_ylabel('Time (seconds)')
+    axes[1, 0].legend()
+    axes[1, 0].grid(True, alpha=0.3)
+    
+    # Learning rate
+    axes[1, 1].plot(df_train['step'], df_train['learning_rate'], 'purple', alpha=0.7)
+    axes[1, 1].set_title('Learning Rate')
+    axes[1, 1].set_xlabel('Training Step')
+    axes[1, 1].set_ylabel('Learning Rate')
+    axes[1, 1].grid(True, alpha=0.3)
+    
+    plt.tight_layout()
+    plt.savefig(output_path / 'training_metrics.png', dpi=300, bbox_inches='tight')
+    plt.close()
+    
+    print(f"Training metrics plot saved to {output_path / 'training_metrics.png'}")
+
+def plot_validation_metrics(df_val, output_dir):
+    """Plot validation accuracy and performance metrics"""
+    output_path = Path(output_dir)
+    
+    if df_val is None or len(df_val) == 0:
+        print("No validation data to plot")
+        return
+    
+    # Group by dataset and metric
+    datasets = df_val['dataset_name'].unique()
+    
+    for dataset in datasets:
+        dataset_data = df_val[df_val['dataset_name'] == dataset]
+        metrics = dataset_data['metric_name'].unique()
+        
+        if len(metrics) == 0:
+            continue
+        
+        # Create subplot grid
+        n_metrics = len(metrics)
+        cols = min(3, n_metrics)
+        rows = (n_metrics + cols - 1) // cols
+        
+        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
+        if n_metrics == 1:
+            axes = [axes]
+        elif rows == 1:
+            axes = axes.reshape(1, -1)
+        
+        fig.suptitle(f'Validation Metrics - {dataset}', fontsize=16)
+        
+        for i, metric in enumerate(metrics):
+            row, col = i // cols, i % cols
+            ax = axes[row, col] if rows > 1 else axes[col]
+            
+            metric_data = dataset_data[dataset_data['metric_name'] == metric]
+            ax.plot(metric_data['step'], metric_data['metric_value'], 'o-', alpha=0.7)
+            ax.set_title(f'{metric}')
+            ax.set_xlabel('Training Step')
+            ax.set_ylabel('Value')
+            ax.grid(True, alpha=0.3)
+        
+        # Hide empty subplots
+        for i in range(n_metrics, rows * cols):
+            row, col = i // cols, i % cols
+            axes[row, col].set_visible(False)
+        
+        plt.tight_layout()
+        plt.savefig(output_path / f'validation_metrics_{dataset}.png', dpi=300, bbox_inches='tight')
+        plt.close()
+        
+        print(f"Validation metrics plot for {dataset} saved to {output_path / f'validation_metrics_{dataset}.png'}")
+
+def plot_energy_landscape_analysis(df_energy, output_dir):
+    """Plot energy landscape analysis including adversarial corruption effects"""
+    output_path = Path(output_dir)
+    
+    if df_energy is None or len(df_energy) == 0:
+        print("No energy data to plot")
+        return
+    
+    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
+    fig.suptitle('Energy Landscape Analysis', fontsize=16)
+    
+    # Energy difference over time
+    axes[0, 0].plot(df_energy['step'], df_energy['energy_diff'], 'b-', alpha=0.7)
+    axes[0, 0].set_title('Energy Difference (Neg - Pos)')
+    axes[0, 0].set_xlabel('Training Step')
+    axes[0, 0].set_ylabel('Energy Difference')
+    axes[0, 0].grid(True, alpha=0.3)
+    
+    # Curriculum weight progression
+    axes[0, 1].plot(df_energy['step'], df_energy['curriculum_weight'], 'r-', alpha=0.7)
+    axes[0, 1].set_title('Adversarial Corruption Curriculum Weight')
+    axes[0, 1].set_xlabel('Training Step')
+    axes[0, 1].set_ylabel('Curriculum Weight')
+    axes[0, 1].grid(True, alpha=0.3)
+    
+    # Corruption type distribution
+    corruption_counts = df_energy['corruption_type'].value_counts()
+    axes[1, 0].pie(corruption_counts.values, labels=corruption_counts.index, autopct='%1.1f%%')
+    axes[1, 0].set_title('Corruption Type Distribution')
+    
+    # Energy difference by corruption type
+    if len(df_energy['corruption_type'].unique()) > 1:
+        df_energy.boxplot(column='energy_diff', by='corruption_type', ax=axes[1, 1])
+        axes[1, 1].set_title('Energy Difference by Corruption Type')
+        axes[1, 1].set_xlabel('Corruption Type')
+        axes[1, 1].set_ylabel('Energy Difference')
+    else:
+        axes[1, 1].hist(df_energy['energy_diff'], bins=20, alpha=0.7)
+        axes[1, 1].set_title('Energy Difference Distribution')
+        axes[1, 1].set_xlabel('Energy Difference')
+        axes[1, 1].set_ylabel('Frequency')
+    
+    plt.tight_layout()
+    plt.savefig(output_path / 'energy_landscape_analysis.png', dpi=300, bbox_inches='tight')
+    plt.close()
+    
+    print(f"Energy landscape analysis plot saved to {output_path / 'energy_landscape_analysis.png'}")
+
+def generate_summary_report(df_train, df_val, df_energy, output_dir):
+    """Generate a summary report with key statistics"""
+    output_path = Path(output_dir)
+    
+    report_path = output_path / 'training_summary_report.txt'
+    
+    with open(report_path, 'w') as f:
+        f.write("IRED Training Summary Report\n")
+        f.write("=" * 50 + "\n")
+        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
+        
+        # Training statistics
+        if df_train is not None and len(df_train) > 0:
+            f.write("TRAINING STATISTICS:\n")
+            f.write("-" * 20 + "\n")
+            f.write(f"Total training steps: {df_train['step'].max()}\n")
+            f.write(f"Final total loss: {df_train['total_loss'].iloc[-1]:.6f}\n")
+            f.write(f"Final denoise loss: {df_train['loss_denoise'].iloc[-1]:.6f}\n")
+            f.write(f"Final energy loss: {df_train['loss_energy'].iloc[-1]:.6f}\n")
+            f.write(f"Average data time: {df_train['data_time'].mean():.3f}s\n")
+            f.write(f"Average NN time: {df_train['nn_time'].mean():.3f}s\n\n")
+        
+        # Validation statistics
+        if df_val is not None and len(df_val) > 0:
+            f.write("VALIDATION STATISTICS:\n")
+            f.write("-" * 22 + "\n")
+            for dataset in df_val['dataset_name'].unique():
+                f.write(f"\nDataset: {dataset}\n")
+                dataset_data = df_val[df_val['dataset_name'] == dataset]
+                for metric in dataset_data['metric_name'].unique():
+                    metric_data = dataset_data[dataset_data['metric_name'] == metric]
+                    if len(metric_data) > 0:
+                        latest_value = metric_data['metric_value'].iloc[-1]
+                        f.write(f"  {metric}: {latest_value:.6f}\n")
+        
+        # Energy landscape statistics
+        if df_energy is not None and len(df_energy) > 0:
+            f.write("\nENERGY LANDSCAPE STATISTICS:\n")
+            f.write("-" * 29 + "\n")
+            f.write(f"Average energy difference: {df_energy['energy_diff'].mean():.6f}\n")
+            f.write(f"Max curriculum weight reached: {df_energy['curriculum_weight'].max():.3f}\n")
+            corruption_counts = df_energy['corruption_type'].value_counts()
+            f.write("Corruption type usage:\n")
+            for corruption_type, count in corruption_counts.items():
+                percentage = (count / len(df_energy)) * 100
+                f.write(f"  {corruption_type}: {count} ({percentage:.1f}%)\n")
+    
+    print(f"Summary report saved to {report_path}")
+
+def main():
+    parser = argparse.ArgumentParser(description='Analyze and visualize IRED training logs')
+    parser.add_argument('--csv-dir', type=str, default='./csv_logs',
+                       help='Directory containing CSV log files')
+    parser.add_argument('--output-dir', type=str, default='./plots',
+                       help='Directory to save plots and analysis')
+    
+    args = parser.parse_args()
+    
+    # Load data
+    print("Loading training data...")
+    df_train = load_training_data(args.csv_dir)
+    
+    print("Loading validation data...")
+    df_val = load_validation_data(args.csv_dir)
+    
+    print("Loading energy data...")
+    df_energy = load_energy_data(args.csv_dir)
+    
+    # Create output directory
+    output_path = Path(args.output_dir)
+    output_path.mkdir(exist_ok=True)
+    
+    # Generate plots
+    if df_train is not None:
+        print("Plotting training metrics...")
+        plot_training_metrics(df_train, args.output_dir)
+    
+    if df_val is not None:
+        print("Plotting validation metrics...")
+        plot_validation_metrics(df_val, args.output_dir)
+    
+    if df_energy is not None:
+        print("Plotting energy landscape analysis...")
+        plot_energy_landscape_analysis(df_energy, args.output_dir)
+    
+    # Generate summary report
+    print("Generating summary report...")
+    generate_summary_report(df_train, df_val, df_energy, args.output_dir)
+    
+    print(f"\nAnalysis complete! Check {args.output_dir} for plots and reports.")
+
+if __name__ == '__main__':
+    main()
\ No newline at end of file
diff --git a/diff.md b/diff.md
new file mode 100644
index 0000000..eab31d5
--- /dev/null
+++ b/diff.md
@@ -0,0 +1,693 @@
+diff --git a/csv_analysis.py b/csv_analysis.py
+new file mode 100644
+index 0000000..1f146ef
+--- /dev/null
++++ b/csv_analysis.py
+@@ -0,0 +1,322 @@
++#!/usr/bin/env python3
++"""
++CSV Analysis and Visualization Script for IRED Training Logs
++Usage: python csv_analysis.py --csv-dir ./csv_logs --output-dir ./plots
++"""
++
++import pandas as pd
++import matplotlib.pyplot as plt
++import seaborn as sns
++import argparse
++from pathlib import Path
++import numpy as np
++from datetime import datetime
++
++def load_training_data(csv_dir):
++    """Load training metrics from CSV files"""
++    csv_path = Path(csv_dir)
++    
++    # Find the most recent training metrics file
++    training_files = list(csv_path.glob('training_metrics_*.csv'))
++    if not training_files:
++        print(f"No training metrics CSV files found in {csv_dir}")
++        return None
++    
++    latest_file = max(training_files, key=lambda x: x.stat().st_mtime)
++    print(f"Loading training data from: {latest_file}")
++    
++    try:
++        df = pd.read_csv(latest_file)
++        df['timestamp'] = pd.to_datetime(df['timestamp'])
++        return df
++    except Exception as e:
++        print(f"Error loading training data: {e}")
++        return None
++
++def load_validation_data(csv_dir):
++    """Load validation metrics from CSV files"""
++    csv_path = Path(csv_dir)
++    
++    # Find the most recent validation metrics file
++    validation_files = list(csv_path.glob('validation_metrics_*.csv'))
++    if not validation_files:
++        print(f"No validation metrics CSV files found in {csv_dir}")
++        return None
++    
++    latest_file = max(validation_files, key=lambda x: x.stat().st_mtime)
++    print(f"Loading validation data from: {latest_file}")
++    
++    try:
++        df = pd.read_csv(latest_file)
++        df['timestamp'] = pd.to_datetime(df['timestamp'])
++        return df
++    except Exception as e:
++        print(f"Error loading validation data: {e}")
++        return None
++
++def load_energy_data(csv_dir):
++    """Load energy landscape metrics from CSV files"""
++    csv_path = Path(csv_dir)
++    
++    # Find the most recent energy metrics file
++    energy_files = list(csv_path.glob('energy_metrics_*.csv'))
++    if not energy_files:
++        print(f"No energy metrics CSV files found in {csv_dir}")
++        return None
++    
++    latest_file = max(energy_files, key=lambda x: x.stat().st_mtime)
++    print(f"Loading energy data from: {latest_file}")
++    
++    try:
++        df = pd.read_csv(latest_file)
++        df['timestamp'] = pd.to_datetime(df['timestamp'])
++        return df
++    except Exception as e:
++        print(f"Error loading energy data: {e}")
++        return None
++
++def plot_training_metrics(df_train, output_dir):
++    """Plot training loss curves and timing metrics"""
++    output_path = Path(output_dir)
++    output_path.mkdir(exist_ok=True)
++    
++    # Set style
++    plt.style.use('seaborn-v0_8-darkgrid')
++    
++    # 1. Loss curves
++    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
++    fig.suptitle('Training Metrics Over Time', fontsize=16)
++    
++    # Total loss
++    axes[0, 0].plot(df_train['step'], df_train['total_loss'], 'b-', alpha=0.7)
++    axes[0, 0].set_title('Total Loss')
++    axes[0, 0].set_xlabel('Training Step')
++    axes[0, 0].set_ylabel('Loss')
++    axes[0, 0].grid(True, alpha=0.3)
++    
++    # Component losses
++    axes[0, 1].plot(df_train['step'], df_train['loss_denoise'], 'r-', alpha=0.7, label='Denoise Loss')
++    axes[0, 1].plot(df_train['step'], df_train['loss_energy'], 'g-', alpha=0.7, label='Energy Loss')
++    axes[0, 1].plot(df_train['step'], df_train['loss_opt'], 'm-', alpha=0.7, label='Opt Loss')
++    axes[0, 1].set_title('Component Losses')
++    axes[0, 1].set_xlabel('Training Step')
++    axes[0, 1].set_ylabel('Loss')
++    axes[0, 1].legend()
++    axes[0, 1].grid(True, alpha=0.3)
++    
++    # Timing metrics
++    axes[1, 0].plot(df_train['step'], df_train['data_time'], 'c-', alpha=0.7, label='Data Time')
++    axes[1, 0].plot(df_train['step'], df_train['nn_time'], 'orange', alpha=0.7, label='NN Time')
++    axes[1, 0].set_title('Timing Metrics')
++    axes[1, 0].set_xlabel('Training Step')
++    axes[1, 0].set_ylabel('Time (seconds)')
++    axes[1, 0].legend()
++    axes[1, 0].grid(True, alpha=0.3)
++    
++    # Learning rate
++    axes[1, 1].plot(df_train['step'], df_train['learning_rate'], 'purple', alpha=0.7)
++    axes[1, 1].set_title('Learning Rate')
++    axes[1, 1].set_xlabel('Training Step')
++    axes[1, 1].set_ylabel('Learning Rate')
++    axes[1, 1].grid(True, alpha=0.3)
++    
++    plt.tight_layout()
++    plt.savefig(output_path / 'training_metrics.png', dpi=300, bbox_inches='tight')
++    plt.close()
++    
++    print(f"Training metrics plot saved to {output_path / 'training_metrics.png'}")
++
++def plot_validation_metrics(df_val, output_dir):
++    """Plot validation accuracy and performance metrics"""
++    output_path = Path(output_dir)
++    
++    if df_val is None or len(df_val) == 0:
++        print("No validation data to plot")
++        return
++    
++    # Group by dataset and metric
++    datasets = df_val['dataset_name'].unique()
++    
++    for dataset in datasets:
++        dataset_data = df_val[df_val['dataset_name'] == dataset]
++        metrics = dataset_data['metric_name'].unique()
++        
++        if len(metrics) == 0:
++            continue
++        
++        # Create subplot grid
++        n_metrics = len(metrics)
++        cols = min(3, n_metrics)
++        rows = (n_metrics + cols - 1) // cols
++        
++        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
++        if n_metrics == 1:
++            axes = [axes]
++        elif rows == 1:
++            axes = axes.reshape(1, -1)
++        
++        fig.suptitle(f'Validation Metrics - {dataset}', fontsize=16)
++        
++        for i, metric in enumerate(metrics):
++            row, col = i // cols, i % cols
++            ax = axes[row, col] if rows > 1 else axes[col]
++            
++            metric_data = dataset_data[dataset_data['metric_name'] == metric]
++            ax.plot(metric_data['step'], metric_data['metric_value'], 'o-', alpha=0.7)
++            ax.set_title(f'{metric}')
++            ax.set_xlabel('Training Step')
++            ax.set_ylabel('Value')
++            ax.grid(True, alpha=0.3)
++        
++        # Hide empty subplots
++        for i in range(n_metrics, rows * cols):
++            row, col = i // cols, i % cols
++            axes[row, col].set_visible(False)
++        
++        plt.tight_layout()
++        plt.savefig(output_path / f'validation_metrics_{dataset}.png', dpi=300, bbox_inches='tight')
++        plt.close()
++        
++        print(f"Validation metrics plot for {dataset} saved to {output_path / f'validation_metrics_{dataset}.png'}")
++
++def plot_energy_landscape_analysis(df_energy, output_dir):
++    """Plot energy landscape analysis including adversarial corruption effects"""
++    output_path = Path(output_dir)
++    
++    if df_energy is None or len(df_energy) == 0:
++        print("No energy data to plot")
++        return
++    
++    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
++    fig.suptitle('Energy Landscape Analysis', fontsize=16)
++    
++    # Energy difference over time
++    axes[0, 0].plot(df_energy['step'], df_energy['energy_diff'], 'b-', alpha=0.7)
++    axes[0, 0].set_title('Energy Difference (Neg - Pos)')
++    axes[0, 0].set_xlabel('Training Step')
++    axes[0, 0].set_ylabel('Energy Difference')
++    axes[0, 0].grid(True, alpha=0.3)
++    
++    # Curriculum weight progression
++    axes[0, 1].plot(df_energy['step'], df_energy['curriculum_weight'], 'r-', alpha=0.7)
++    axes[0, 1].set_title('Adversarial Corruption Curriculum Weight')
++    axes[0, 1].set_xlabel('Training Step')
++    axes[0, 1].set_ylabel('Curriculum Weight')
++    axes[0, 1].grid(True, alpha=0.3)
++    
++    # Corruption type distribution
++    corruption_counts = df_energy['corruption_type'].value_counts()
++    axes[1, 0].pie(corruption_counts.values, labels=corruption_counts.index, autopct='%1.1f%%')
++    axes[1, 0].set_title('Corruption Type Distribution')
++    
++    # Energy difference by corruption type
++    if len(df_energy['corruption_type'].unique()) > 1:
++        df_energy.boxplot(column='energy_diff', by='corruption_type', ax=axes[1, 1])
++        axes[1, 1].set_title('Energy Difference by Corruption Type')
++        axes[1, 1].set_xlabel('Corruption Type')
++        axes[1, 1].set_ylabel('Energy Difference')
++    else:
++        axes[1, 1].hist(df_energy['energy_diff'], bins=20, alpha=0.7)
++        axes[1, 1].set_title('Energy Difference Distribution')
++        axes[1, 1].set_xlabel('Energy Difference')
++        axes[1, 1].set_ylabel('Frequency')
++    
++    plt.tight_layout()
++    plt.savefig(output_path / 'energy_landscape_analysis.png', dpi=300, bbox_inches='tight')
++    plt.close()
++    
++    print(f"Energy landscape analysis plot saved to {output_path / 'energy_landscape_analysis.png'}")
++
++def generate_summary_report(df_train, df_val, df_energy, output_dir):
++    """Generate a summary report with key statistics"""
++    output_path = Path(output_dir)
++    
++    report_path = output_path / 'training_summary_report.txt'
++    
++    with open(report_path, 'w') as f:
++        f.write("IRED Training Summary Report\n")
++        f.write("=" * 50 + "\n")
++        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
++        
++        # Training statistics
++        if df_train is not None and len(df_train) > 0:
++            f.write("TRAINING STATISTICS:\n")
++            f.write("-" * 20 + "\n")
++            f.write(f"Total training steps: {df_train['step'].max()}\n")
++            f.write(f"Final total loss: {df_train['total_loss'].iloc[-1]:.6f}\n")
++            f.write(f"Final denoise loss: {df_train['loss_denoise'].iloc[-1]:.6f}\n")
++            f.write(f"Final energy loss: {df_train['loss_energy'].iloc[-1]:.6f}\n")
++            f.write(f"Average data time: {df_train['data_time'].mean():.3f}s\n")
++            f.write(f"Average NN time: {df_train['nn_time'].mean():.3f}s\n\n")
++        
++        # Validation statistics
++        if df_val is not None and len(df_val) > 0:
++            f.write("VALIDATION STATISTICS:\n")
++            f.write("-" * 22 + "\n")
++            for dataset in df_val['dataset_name'].unique():
++                f.write(f"\nDataset: {dataset}\n")
++                dataset_data = df_val[df_val['dataset_name'] == dataset]
++                for metric in dataset_data['metric_name'].unique():
++                    metric_data = dataset_data[dataset_data['metric_name'] == metric]
++                    if len(metric_data) > 0:
++                        latest_value = metric_data['metric_value'].iloc[-1]
++                        f.write(f"  {metric}: {latest_value:.6f}\n")
++        
++        # Energy landscape statistics
++        if df_energy is not None and len(df_energy) > 0:
++            f.write("\nENERGY LANDSCAPE STATISTICS:\n")
++            f.write("-" * 29 + "\n")
++            f.write(f"Average energy difference: {df_energy['energy_diff'].mean():.6f}\n")
++            f.write(f"Max curriculum weight reached: {df_energy['curriculum_weight'].max():.3f}\n")
++            corruption_counts = df_energy['corruption_type'].value_counts()
++            f.write("Corruption type usage:\n")
++            for corruption_type, count in corruption_counts.items():
++                percentage = (count / len(df_energy)) * 100
++                f.write(f"  {corruption_type}: {count} ({percentage:.1f}%)\n")
++    
++    print(f"Summary report saved to {report_path}")
++
++def main():
++    parser = argparse.ArgumentParser(description='Analyze and visualize IRED training logs')
++    parser.add_argument('--csv-dir', type=str, default='./csv_logs',
++                       help='Directory containing CSV log files')
++    parser.add_argument('--output-dir', type=str, default='./plots',
++                       help='Directory to save plots and analysis')
++    
++    args = parser.parse_args()
++    
++    # Load data
++    print("Loading training data...")
++    df_train = load_training_data(args.csv_dir)
++    
++    print("Loading validation data...")
++    df_val = load_validation_data(args.csv_dir)
++    
++    print("Loading energy data...")
++    df_energy = load_energy_data(args.csv_dir)
++    
++    # Create output directory
++    output_path = Path(args.output_dir)
++    output_path.mkdir(exist_ok=True)
++    
++    # Generate plots
++    if df_train is not None:
++        print("Plotting training metrics...")
++        plot_training_metrics(df_train, args.output_dir)
++    
++    if df_val is not None:
++        print("Plotting validation metrics...")
++        plot_validation_metrics(df_val, args.output_dir)
++    
++    if df_energy is not None:
++        print("Plotting energy landscape analysis...")
++        plot_energy_landscape_analysis(df_energy, args.output_dir)
++    
++    # Generate summary report
++    print("Generating summary report...")
++    generate_summary_report(df_train, df_val, df_energy, args.output_dir)
++    
++    print(f"\nAnalysis complete! Check {args.output_dir} for plots and reports.")
++
++if __name__ == '__main__':
++    main()
+\ No newline at end of file
+diff --git a/diffusion_lib/denoising_diffusion_pytorch_1d.py b/diffusion_lib/denoising_diffusion_pytorch_1d.py
+index 3f75c29..9472b64 100644
+--- a/diffusion_lib/denoising_diffusion_pytorch_1d.py
++++ b/diffusion_lib/denoising_diffusion_pytorch_1d.py
+@@ -3,6 +3,8 @@ import sys
+ import collections
+ from multiprocessing import cpu_count
+ from pathlib import Path
++import csv
++from datetime import datetime
+ from random import random
+ from functools import partial
+ from collections import namedtuple
+@@ -707,80 +709,13 @@ class GaussianDiffusion1D(nn.Module):
+             if mask is not None:
+                 data_cond = self.q_sample(x_start = x_start, t = t, noise = torch.zeros_like(noise))
+                 data_sample = data_sample * (1 - mask) + mask * data_cond
+-
+-            # Add a noise contrastive estimation term with samples drawn from the data distribution
+-            #noise = torch.randn_like(x_start)
+-
+-            # Optimize a sample using gradient descent on energy landscape
+-            xmin_noise = self.q_sample(x_start = x_start, t = t, noise = 3.0 * noise)
+-
+-            if mask is not None:
+-                xmin_noise = xmin_noise * (1 - mask) + mask * data_cond
+             else:
+                 data_cond = None
+ 
+-            if self.sudoku:
+-                s = x_start.size()
+-                x_start_im = x_start.view(-1, 9, 9, 9).argmax(dim=-1)
+-                randperm = torch.randint(0, 9, x_start_im.size(), device=x_start_im.device)
+-
+-                rand_mask = (torch.rand(x_start_im.size(), device=x_start_im.device) < 0.05).float()
+-
+-                xmin_noise_im = x_start_im * (1 - rand_mask) + randperm * (rand_mask)
+-
+-                xmin_noise_im = F.one_hot(xmin_noise_im.long(), num_classes=9)
+-                xmin_noise_im = (xmin_noise_im - 0.5) * 2
+-
+-                xmin_noise_rescale = xmin_noise_im.view(-1, 729)
+-
+-                loss_opt = torch.ones(1)
+-
+-                loss_scale = 0.05
+-            elif self.connectivity:
+-                s = x_start.size()
+-                x_start_im = x_start.view(-1, 12, 12)
+-                randperm = (torch.randint(0, 1, x_start_im.size(), device=x_start_im.device) - 0.5) * 2
+-
+-                rand_mask = (torch.rand(x_start_im.size(), device=x_start_im.device) < 0.05).float()
+-
+-                xmin_noise_rescale = x_start_im * (1 - rand_mask) + randperm * (rand_mask)
+-
+-                loss_opt = torch.ones(1)
+-
+-                loss_scale = 0.05
+-            elif self.shortest_path:
+-                x_start_list = x_start.argmax(dim=2)
+-                classes = x_start.size(2)
+-                rand_vals = torch.randint(0, classes, x_start_list.size()).to(x_start.device)
+-
+-                x_start_neg = torch.cat([rand_vals[:, :1], x_start_list[:, 1:]], dim=1)
+-                x_start_neg_oh = F.one_hot(x_start_neg[:, :, 0].long(), num_classes=classes)[:, :, :, None]
+-                xmin_noise_rescale = (x_start_neg_oh - 0.5) * 2
+-
+-                loss_opt = torch.ones(1)
+-
+-                loss_scale = 0.5
+-            else:
+-
+-                xmin_noise = self.opt_step(inp, xmin_noise, t, mask, data_cond, step=2, sf=1.0)
+-                xmin = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
+-                loss_opt = torch.pow(xmin_noise - xmin, 2).mean()
+-
+-                xmin_noise = xmin_noise.detach()
+-                xmin_noise_rescale = self.predict_start_from_noise(xmin_noise, t, torch.zeros_like(xmin_noise))
+-                xmin_noise_rescale = torch.clamp(xmin_noise_rescale, -2, 2)
+-
+-                # loss_opt = torch.ones(1)
+-
+-
+-                # rand_mask = (torch.rand(x_start.size(), device=x_start.device) < 0.2).float()
+-
+-                # xmin_noise_rescale =  x_start * (1 - rand_mask) + rand_mask * x_start_noise
+-
+-                # nrep = 1
+-
+-
+-                loss_scale = 0.5
++            # Enhanced adversarial corruption replaces all task-specific corruption logic
++            xmin_noise_rescale = self.enhanced_corruption_step(inp, x_start, t, mask, data_cond)
++            loss_opt = torch.ones(1)
++            loss_scale = 0.5
+ 
+             xmin_noise = self.q_sample(x_start=xmin_noise_rescale, t=t, noise=noise)
+ 
+@@ -800,6 +735,13 @@ class GaussianDiffusion1D(nn.Module):
+             target = torch.zeros(energy_real.size(0)).to(energy_stack.device)
+             loss_energy = F.cross_entropy(-1 * energy_stack, target.long(), reduction='none')[:, None]
+ 
++            # Track energy landscape quality for adaptive curriculum
++            with torch.no_grad():
++                energy_diff = (energy_fake - energy_real).mean().item()
++                self.recent_energy_diffs.append(max(0, energy_diff))
++                if len(self.recent_energy_diffs) > 100:
++                    self.recent_energy_diffs.pop(0)
++
+             # loss_energy = energy_real.mean() - energy_fake.mean()# loss_energy.mean()
+ 
+             loss = loss_mse + loss_scale * loss_energy # + 0.001 * loss_opt
+@@ -852,7 +794,10 @@ class Trainer1D(object):
+         extra_validation_every_mul = 10,
+         evaluate_first = False,
+         latent = False,
+-        autoencode_model = None
++        autoencode_model = None,
++        save_csv_logs = False,
++        csv_log_interval = 100,
++        csv_log_dir = './csv_logs'
+     ):
+         super().__init__()
+ 
+@@ -934,13 +879,17 @@ class Trainer1D(object):
+ 
+         # for logging results in a folder periodically
+ 
+-        if self.accelerator.is_main_process:
+-            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
+-            self.ema.to(self.device)
+-
+         self.results_folder = Path(results_folder)
+         self.results_folder.mkdir(exist_ok = True)
+ 
++        # CSV logging setup
++        self.save_csv_logs = save_csv_logs
++        self.csv_log_interval = csv_log_interval
++        if self.save_csv_logs and self.accelerator.is_main_process:
++            self.csv_log_dir = Path(csv_log_dir)
++            self.csv_log_dir.mkdir(exist_ok=True)
++            self._init_csv_logging()
++
+         # step counter state
+ 
+         self.step = 0
+@@ -948,8 +897,53 @@ class Trainer1D(object):
+         # prepare model, dataloader, optimizer with accelerator
+ 
+         self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
++        
++        # Initialize EMA after model is prepared to ensure same device
++        if self.accelerator.is_main_process:
++            self.ema = EMA(self.model, beta = ema_decay, update_every = ema_update_every)
++            self.ema.to(self.device)
+         self.evaluate_first = evaluate_first
+ 
++    def _init_csv_logging(self):
++        """Initialize CSV files for logging training and validation metrics"""
++        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
++        
++        # Training metrics CSV
++        self.train_csv_path = self.csv_log_dir / f'training_metrics_{timestamp}.csv'
++        with open(self.train_csv_path, 'w', newline='') as f:
++            writer = csv.writer(f)
++            writer.writerow([
++                'step', 'epoch', 'total_loss', 'loss_denoise', 'loss_energy', 'loss_opt',
++                'data_time', 'nn_time', 'learning_rate', 'timestamp'
++            ])
++        
++        # Validation metrics CSV
++        self.val_csv_path = self.csv_log_dir / f'validation_metrics_{timestamp}.csv'
++        with open(self.val_csv_path, 'w', newline='') as f:
++            writer = csv.writer(f)
++            writer.writerow([
++                'step', 'milestone', 'dataset_name', 'metric_name', 'metric_value', 'timestamp'
++            ])
++        
++        # Energy landscape metrics CSV (for adversarial corruption analysis)
++        self.energy_csv_path = self.csv_log_dir / f'energy_metrics_{timestamp}.csv'
++        with open(self.energy_csv_path, 'w', newline='') as f:
++            writer = csv.writer(f)
++            writer.writerow([
++                'step', 'energy_pos_mean', 'energy_neg_mean', 'energy_diff', 
++                'curriculum_weight', 'corruption_type', 'timestamp'
++            ])
++    
++    def _log_to_csv(self, csv_path, row_data):
++        """Helper function to append data to CSV file"""
++        if self.save_csv_logs and self.accelerator.is_main_process:
++            try:
++                with open(csv_path, 'a', newline='') as f:
++                    writer = csv.writer(f)
++                    writer.writerow(row_data)
++            except Exception as e:
++                print(f"Warning: Failed to write to CSV {csv_path}: {e}")
++
+     @property
+     def device(self):
+         return self.accelerator.device
+@@ -1001,6 +995,7 @@ class Trainer1D(object):
+         end_time = time.time()
+         with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process, dynamic_ncols = True) as pbar:
+ 
++            epoch = 0
+             while self.step < self.train_num_steps:
+ 
+                 total_loss = 0.
+@@ -1043,6 +1038,24 @@ class Trainer1D(object):
+                 nn_time = time.time() - end_time; end_time = time.time()
+                 pbar.set_description(f'loss: {total_loss:.4f} loss_denoise: {loss_denoise:.4f} loss_energy: {loss_energy:.4f} loss_opt: {loss_opt:.4f} data_time: {data_time:.2f} nn_time: {nn_time:.2f}')
+ 
++                # Log training metrics to CSV
++                if self.save_csv_logs and self.step % self.csv_log_interval == 0:
++                    current_lr = self.opt.param_groups[0]['lr']
++                    timestamp = datetime.now().isoformat()
++                    
++                    train_row = [
++                        self.step, epoch, total_loss, loss_denoise.item(), 
++                        loss_energy.item(), loss_opt.item(), data_time, nn_time, 
++                        current_lr, timestamp
++                    ]
++                    self._log_to_csv(self.train_csv_path, train_row)
++                    
++                    # Log energy landscape metrics if available
++                    if hasattr(self.model.module, 'recent_energy_diffs'):
++                        recent_diffs = self.model.module.recent_energy_diffs
++                        if len(recent_diffs) > 0:
++                            self._log_energy_metrics(loss_energy.item(), recent_diffs[-1])
++
+                 self.step += 1
+                 if accelerator.is_main_process:
+                     self.ema.update()
+@@ -1060,8 +1073,28 @@ class Trainer1D(object):
+ 
+ 
+                 pbar.update(1)
++                
++                # Update epoch counter (approximate)
++                if self.step % 1000 == 0:
++                    epoch += 1
+ 
+         accelerator.print('training complete')
++        
++    def _log_energy_metrics(self, loss_energy, energy_diff):
++        """Log energy landscape specific metrics"""
++        curriculum_weight = 0.0
++        corruption_type = "standard"
++        
++        if hasattr(self.model.module, 'use_adversarial_corruption') and self.model.module.use_adversarial_corruption:
++            if hasattr(self.model.module, 'training_step') and self.model.module.training_step > self.model.module.anm_warmup_steps:
++                curriculum_weight = min(1.0, (self.model.module.training_step - self.model.module.anm_warmup_steps) / self.model.module.anm_warmup_steps)
++                corruption_type = "adversarial" if curriculum_weight > 0.1 else "mixed"
++        
++        timestamp = datetime.now().isoformat()
++        energy_row = [
++            self.step, 0, 0, energy_diff, curriculum_weight, corruption_type, timestamp
++        ]
++        self._log_to_csv(self.energy_csv_path, energy_row)
+ 
+     def evaluate(self, device, milestone, inp=None, label=None, mask=None):
+         print('Running Evaluation...')
+@@ -1087,6 +1120,12 @@ class Trainer1D(object):
+                     mse_error = (all_samples - label).pow(2).mean()
+                     rows = [('mse_error', mse_error)]
+                     print(tabulate(rows))
++                    
++                    # Log to CSV
++                    if self.save_csv_logs:
++                        timestamp = datetime.now().isoformat()
++                        val_row = [self.step, milestone, 'train_sample', 'mse_error', mse_error.item(), timestamp]
++                        self._log_to_csv(self.val_csv_path, val_row)
+                 elif self.metric == 'bce':
+                     assert len(all_samples_list) == 1
+                     summary = binary_classification_accuracy_4(all_samples_list[0], label)
+@@ -1097,6 +1136,13 @@ class Trainer1D(object):
+                     summary = sudoku_accuracy(all_samples_list[0], label, mask)
+                     rows = [[k, v] for k, v in summary.items()]
+                     print(tabulate(rows))
++                    
++                    # Log to CSV
++                    if self.save_csv_logs:
++                        timestamp = datetime.now().isoformat()
++                        for metric_name, metric_value in summary.items():
++                            val_row = [self.step, milestone, 'train_sample', metric_name, metric_value, timestamp]
++                            self._log_to_csv(self.val_csv_path, val_row)
+                 elif self.metric == 'sort':
+                     assert len(all_samples_list) == 1
+                     summary = binary_classification_accuracy_4(all_samples_list[0], label)
+@@ -1206,6 +1252,13 @@ class Trainer1D(object):
+             rows = [[k, v.avg] for k, v in meters.items()]
+             print(f'Validation Result @ Iteration {self.step}; Milestone = {milestone} (ID: {prefix})')
+             print(tabulate(rows))
++            
++            # Log validation results to CSV
++            if self.save_csv_logs:
++                timestamp = datetime.now().isoformat()
++                for metric_name, meter in meters.items():
++                    val_row = [self.step, milestone, prefix, metric_name, meter.avg, timestamp]
++                    self._log_to_csv(self.val_csv_path, val_row)
+ 
+ 
+ as_float = lambda x: float(x.item())
+diff --git a/requirements_csv.txt b/requirements_csv.txt
+new file mode 100644
+index 0000000..ea5d7af
+--- /dev/null
++++ b/requirements_csv.txt
+@@ -0,0 +1,3 @@
++pandas>=1.3.0
++matplotlib>=3.3.0
++seaborn>=0.11.0
+\ No newline at end of file
+diff --git a/train.py b/train.py
+index 9cb26e1..2ce6812 100644
+--- a/train.py
++++ b/train.py
+@@ -53,7 +53,23 @@ parser.add_argument('--evaluate', action='store_true', default=False)
+ parser.add_argument('--latent', action='store_true', default=False)
+ parser.add_argument('--ood', action='store_true', default=False)
+ parser.add_argument('--baseline', action='store_true', default=False)
++# CSV logging arguments
++parser.add_argument('--save-csv-logs', action='store_true', default=False,
++                   help='Save training and validation metrics to CSV files')
++parser.add_argument('--csv-log-interval', type=int, default=100,
++                   help='Interval for logging training metrics to CSV')
++parser.add_argument('--csv-log-dir', type=str, default='./csv_logs',
++                   help='Directory to save CSV log files')
+ 
++# Adversarial Negative Mining arguments
++parser.add_argument('--use-adversarial-corruption', type=str2bool, default=False,
++                   help='Use adversarial corruption for enhanced negative mining')
++parser.add_argument('--anm-warmup-steps', type=int, default=5000,
++                   help='Steps before adversarial corruption begins')
++parser.add_argument('--anm-adversarial-steps', type=int, default=3,
++                   help='Number of adversarial optimization steps')
++parser.add_argument('--anm-distance-penalty', type=float, default=0.1,
++                   help='Weight for distance penalty in adversarial loss')
+ 
+ if __name__ == "__main__":
+     FLAGS = parser.parse_args()
+@@ -271,6 +287,10 @@ if __name__ == "__main__":
+         supervise_energy_landscape = FLAGS.supervise_energy_landscape,
+         use_innerloop_opt = FLAGS.use_innerloop_opt,
+         show_inference_tqdm = False,
++        use_adversarial_corruption = FLAGS.use_adversarial_corruption,
++        anm_warmup_steps = FLAGS.anm_warmup_steps,
++        anm_adversarial_steps = FLAGS.anm_adversarial_steps,
++        anm_distance_penalty = FLAGS.anm_distance_penalty,
+         **kwargs
+     )
+ 
+@@ -308,7 +328,10 @@ if __name__ == "__main__":
+         save_and_sample_every = save_and_sample_every,
+         evaluate_first = FLAGS.evaluate,  # run one evaluation first
+         latent = FLAGS.latent,  # whether we are doing reasoning in the latent space
+-        autoencode_model = autoencode_model
++        autoencode_model = autoencode_model,
++        save_csv_logs = FLAGS.save_csv_logs,
++        csv_log_interval = FLAGS.csv_log_interval,
++        csv_log_dir = FLAGS.csv_log_dir
+     )
+ 
+     if FLAGS.load_milestone is not None:
diff --git a/diffusion_lib/denoising_diffusion_pytorch_1d.py b/diffusion_lib/denoising_diffusion_pytorch_1d.py
index 3f75c29..99c3e60 100644
--- a/diffusion_lib/denoising_diffusion_pytorch_1d.py
+++ b/diffusion_lib/denoising_diffusion_pytorch_1d.py
@@ -3,6 +3,8 @@ import sys
 import collections
 from multiprocessing import cpu_count
 from pathlib import Path
+import csv
+from datetime import datetime
 from random import random
 from functools import partial
 from collections import namedtuple
@@ -707,80 +709,13 @@ class GaussianDiffusion1D(nn.Module):
             if mask is not None:
                 data_cond = self.q_sample(x_start = x_start, t = t, noise = torch.zeros_like(noise))
                 data_sample = data_sample * (1 - mask) + mask * data_cond
-
-            # Add a noise contrastive estimation term with samples drawn from the data distribution
-            #noise = torch.randn_like(x_start)
-
-            # Optimize a sample using gradient descent on energy landscape
-            xmin_noise = self.q_sample(x_start = x_start, t = t, noise = 3.0 * noise)
-
-            if mask is not None:
-                xmin_noise = xmin_noise * (1 - mask) + mask * data_cond
             else:
                 data_cond = None
 
-            if self.sudoku:
-                s = x_start.size()
-                x_start_im = x_start.view(-1, 9, 9, 9).argmax(dim=-1)
-                randperm = torch.randint(0, 9, x_start_im.size(), device=x_start_im.device)
-
-                rand_mask = (torch.rand(x_start_im.size(), device=x_start_im.device) < 0.05).float()
-
-                xmin_noise_im = x_start_im * (1 - rand_mask) + randperm * (rand_mask)
-
-                xmin_noise_im = F.one_hot(xmin_noise_im.long(), num_classes=9)
-                xmin_noise_im = (xmin_noise_im - 0.5) * 2
-
-                xmin_noise_rescale = xmin_noise_im.view(-1, 729)
-
-                loss_opt = torch.ones(1)
-
-                loss_scale = 0.05
-            elif self.connectivity:
-                s = x_start.size()
-                x_start_im = x_start.view(-1, 12, 12)
-                randperm = (torch.randint(0, 1, x_start_im.size(), device=x_start_im.device) - 0.5) * 2
-
-                rand_mask = (torch.rand(x_start_im.size(), device=x_start_im.device) < 0.05).float()
-
-                xmin_noise_rescale = x_start_im * (1 - rand_mask) + randperm * (rand_mask)
-
-                loss_opt = torch.ones(1)
-
-                loss_scale = 0.05
-            elif self.shortest_path:
-                x_start_list = x_start.argmax(dim=2)
-                classes = x_start.size(2)
-                rand_vals = torch.randint(0, classes, x_start_list.size()).to(x_start.device)
-
-                x_start_neg = torch.cat([rand_vals[:, :1], x_start_list[:, 1:]], dim=1)
-                x_start_neg_oh = F.one_hot(x_start_neg[:, :, 0].long(), num_classes=classes)[:, :, :, None]
-                xmin_noise_rescale = (x_start_neg_oh - 0.5) * 2
-
-                loss_opt = torch.ones(1)
-
-                loss_scale = 0.5
-            else:
-
-                xmin_noise = self.opt_step(inp, xmin_noise, t, mask, data_cond, step=2, sf=1.0)
-                xmin = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
-                loss_opt = torch.pow(xmin_noise - xmin, 2).mean()
-
-                xmin_noise = xmin_noise.detach()
-                xmin_noise_rescale = self.predict_start_from_noise(xmin_noise, t, torch.zeros_like(xmin_noise))
-                xmin_noise_rescale = torch.clamp(xmin_noise_rescale, -2, 2)
-
-                # loss_opt = torch.ones(1)
-
-
-                # rand_mask = (torch.rand(x_start.size(), device=x_start.device) < 0.2).float()
-
-                # xmin_noise_rescale =  x_start * (1 - rand_mask) + rand_mask * x_start_noise
-
-                # nrep = 1
-
-
-                loss_scale = 0.5
+            # Enhanced adversarial corruption replaces all task-specific corruption logic
+            xmin_noise_rescale = self.enhanced_corruption_step(inp, x_start, t, mask, data_cond)
+            loss_opt = torch.ones(1)
+            loss_scale = 0.5
 
             xmin_noise = self.q_sample(x_start=xmin_noise_rescale, t=t, noise=noise)
 
@@ -800,6 +735,13 @@ class GaussianDiffusion1D(nn.Module):
             target = torch.zeros(energy_real.size(0)).to(energy_stack.device)
             loss_energy = F.cross_entropy(-1 * energy_stack, target.long(), reduction='none')[:, None]
 
+            # Track energy landscape quality for adaptive curriculum
+            with torch.no_grad():
+                energy_diff = (energy_fake - energy_real).mean().item()
+                self.recent_energy_diffs.append(max(0, energy_diff))
+                if len(self.recent_energy_diffs) > 100:
+                    self.recent_energy_diffs.pop(0)
+
             # loss_energy = energy_real.mean() - energy_fake.mean()# loss_energy.mean()
 
             loss = loss_mse + loss_scale * loss_energy # + 0.001 * loss_opt
@@ -852,7 +794,10 @@ class Trainer1D(object):
         extra_validation_every_mul = 10,
         evaluate_first = False,
         latent = False,
-        autoencode_model = None
+        autoencode_model = None,
+        save_csv_logs = False,
+        csv_log_interval = 100,
+        csv_log_dir = './csv_logs'
     ):
         super().__init__()
 
@@ -934,13 +879,17 @@ class Trainer1D(object):
 
         # for logging results in a folder periodically
 
-        if self.accelerator.is_main_process:
-            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
-            self.ema.to(self.device)
-
         self.results_folder = Path(results_folder)
         self.results_folder.mkdir(exist_ok = True)
 
+        # CSV logging setup
+        self.save_csv_logs = save_csv_logs
+        self.csv_log_interval = csv_log_interval
+        if self.save_csv_logs and self.accelerator.is_main_process:
+            self.csv_log_dir = Path(csv_log_dir)
+            self.csv_log_dir.mkdir(exist_ok=True)
+            self._init_csv_logging()
+
         # step counter state
 
         self.step = 0
@@ -948,8 +897,53 @@ class Trainer1D(object):
         # prepare model, dataloader, optimizer with accelerator
 
         self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
+        
+        # Initialize EMA after model is prepared to ensure same device
+        if self.accelerator.is_main_process:
+            self.ema = EMA(self.model, beta = ema_decay, update_every = ema_update_every)
+            self.ema.to(self.device)
         self.evaluate_first = evaluate_first
 
+    def _init_csv_logging(self):
+        """Initialize CSV files for logging training and validation metrics"""
+        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
+        
+        # Training metrics CSV
+        self.train_csv_path = self.csv_log_dir / f'training_metrics_{timestamp}.csv'
+        with open(self.train_csv_path, 'w', newline='') as f:
+            writer = csv.writer(f)
+            writer.writerow([
+                'step', 'epoch', 'total_loss', 'loss_denoise', 'loss_energy', 'loss_opt',
+                'data_time', 'nn_time', 'learning_rate', 'timestamp'
+            ])
+        
+        # Validation metrics CSV
+        self.val_csv_path = self.csv_log_dir / f'validation_metrics_{timestamp}.csv'
+        with open(self.val_csv_path, 'w', newline='') as f:
+            writer = csv.writer(f)
+            writer.writerow([
+                'step', 'milestone', 'dataset_name', 'metric_name', 'metric_value', 'timestamp'
+            ])
+        
+        # Energy landscape metrics CSV (for adversarial corruption analysis)
+        self.energy_csv_path = self.csv_log_dir / f'energy_metrics_{timestamp}.csv'
+        with open(self.energy_csv_path, 'w', newline='') as f:
+            writer = csv.writer(f)
+            writer.writerow([
+                'step', 'energy_pos_mean', 'energy_neg_mean', 'energy_diff', 
+                'curriculum_weight', 'corruption_type', 'timestamp'
+            ])
+    
+    def _log_to_csv(self, csv_path, row_data):
+        """Helper function to append data to CSV file"""
+        if self.save_csv_logs and self.accelerator.is_main_process:
+            try:
+                with open(csv_path, 'a', newline='') as f:
+                    writer = csv.writer(f)
+                    writer.writerow(row_data)
+            except Exception as e:
+                print(f"Warning: Failed to write to CSV {csv_path}: {e}")
+
     @property
     def device(self):
         return self.accelerator.device
@@ -1001,6 +995,7 @@ class Trainer1D(object):
         end_time = time.time()
         with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process, dynamic_ncols = True) as pbar:
 
+            epoch = 0
             while self.step < self.train_num_steps:
 
                 total_loss = 0.
@@ -1043,6 +1038,25 @@ class Trainer1D(object):
                 nn_time = time.time() - end_time; end_time = time.time()
                 pbar.set_description(f'loss: {total_loss:.4f} loss_denoise: {loss_denoise:.4f} loss_energy: {loss_energy:.4f} loss_opt: {loss_opt:.4f} data_time: {data_time:.2f} nn_time: {nn_time:.2f}')
 
+                # Log training metrics to CSV
+                if self.save_csv_logs and self.step % self.csv_log_interval == 0:
+                    current_lr = self.opt.param_groups[0]['lr']
+                    timestamp = datetime.now().isoformat()
+                    
+                    train_row = [
+                        self.step, epoch, total_loss, loss_denoise.item(), 
+                        loss_energy.item(), loss_opt.item(), data_time, nn_time, 
+                        current_lr, timestamp
+                    ]
+                    self._log_to_csv(self.train_csv_path, train_row)
+                    
+                    # Log energy landscape metrics if available
+                    model = self.model.module if hasattr(self.model, 'module') else self.model
+                    if hasattr(model, 'recent_energy_diffs'):
+                        recent_diffs = model.recent_energy_diffs
+                        if len(recent_diffs) > 0:
+                            self._log_energy_metrics(loss_energy.item(), recent_diffs[-1])
+
                 self.step += 1
                 if accelerator.is_main_process:
                     self.ema.update()
@@ -1060,8 +1074,29 @@ class Trainer1D(object):
 
 
                 pbar.update(1)
+                
+                # Update epoch counter (approximate)
+                if self.step % 1000 == 0:
+                    epoch += 1
 
         accelerator.print('training complete')
+        
+    def _log_energy_metrics(self, loss_energy, energy_diff):
+        """Log energy landscape specific metrics"""
+        curriculum_weight = 0.0
+        corruption_type = "standard"
+        
+        model = self.model.module if hasattr(self.model, 'module') else self.model
+        if hasattr(model, 'use_adversarial_corruption') and model.use_adversarial_corruption:
+            if hasattr(model, 'training_step') and model.training_step > model.anm_warmup_steps:
+                curriculum_weight = min(1.0, (model.training_step - model.anm_warmup_steps) / model.anm_warmup_steps)
+                corruption_type = "adversarial" if curriculum_weight > 0.1 else "mixed"
+        
+        timestamp = datetime.now().isoformat()
+        energy_row = [
+            self.step, 0, 0, energy_diff, curriculum_weight, corruption_type, timestamp
+        ]
+        self._log_to_csv(self.energy_csv_path, energy_row)
 
     def evaluate(self, device, milestone, inp=None, label=None, mask=None):
         print('Running Evaluation...')
@@ -1087,6 +1122,12 @@ class Trainer1D(object):
                     mse_error = (all_samples - label).pow(2).mean()
                     rows = [('mse_error', mse_error)]
                     print(tabulate(rows))
+                    
+                    # Log to CSV
+                    if self.save_csv_logs:
+                        timestamp = datetime.now().isoformat()
+                        val_row = [self.step, milestone, 'train_sample', 'mse_error', mse_error.item(), timestamp]
+                        self._log_to_csv(self.val_csv_path, val_row)
                 elif self.metric == 'bce':
                     assert len(all_samples_list) == 1
                     summary = binary_classification_accuracy_4(all_samples_list[0], label)
@@ -1097,6 +1138,13 @@ class Trainer1D(object):
                     summary = sudoku_accuracy(all_samples_list[0], label, mask)
                     rows = [[k, v] for k, v in summary.items()]
                     print(tabulate(rows))
+                    
+                    # Log to CSV
+                    if self.save_csv_logs:
+                        timestamp = datetime.now().isoformat()
+                        for metric_name, metric_value in summary.items():
+                            val_row = [self.step, milestone, 'train_sample', metric_name, metric_value, timestamp]
+                            self._log_to_csv(self.val_csv_path, val_row)
                 elif self.metric == 'sort':
                     assert len(all_samples_list) == 1
                     summary = binary_classification_accuracy_4(all_samples_list[0], label)
@@ -1206,6 +1254,13 @@ class Trainer1D(object):
             rows = [[k, v.avg] for k, v in meters.items()]
             print(f'Validation Result @ Iteration {self.step}; Milestone = {milestone} (ID: {prefix})')
             print(tabulate(rows))
+            
+            # Log validation results to CSV
+            if self.save_csv_logs:
+                timestamp = datetime.now().isoformat()
+                for metric_name, meter in meters.items():
+                    val_row = [self.step, milestone, prefix, metric_name, meter.avg, timestamp]
+                    self._log_to_csv(self.val_csv_path, val_row)
 
 
 as_float = lambda x: float(x.item())
diff --git a/planning_dataset.py b/planning_dataset.py
new file mode 100644
index 0000000..bee46bf
--- /dev/null
+++ b/planning_dataset.py
@@ -0,0 +1,449 @@
+from typing import Optional
+
+import random
+import numpy as np
+import gzip
+import pickle
+from tqdm.auto import tqdm
+from tabulate import tabulate
+from reasoning_dataset import random_generate_graph, random_generate_graph_dnc, random_generate_special_graph
+from torch.utils.data.dataset import Dataset
+
+
+class PlanningDataset(Dataset):
+    def __init__(self, dataset_identifier: str, split: str, num_identifier='100000'):
+        self._dataset_identifier = dataset_identifier
+        self._split = split
+        self._num_identifier = num_identifier
+
+        if self._dataset_identifier == 'sort':
+            self._init_load()
+        elif self._dataset_identifier == 'sort-15':
+            self._init_load()
+        elif self._dataset_identifier == 'shortest-path':
+            self._init_load()
+        elif self._dataset_identifier == 'shortest-path-1d':
+            self._init_load_1d()
+        elif self._dataset_identifier == 'shortest-path-10-1d':
+            self._init_load_1d()
+        elif self._dataset_identifier == 'shortest-path-15-1d':
+            self._init_load_1d()
+        elif self._dataset_identifier == 'shortest-path-25-1d':
+            self._init_load_1d()
+        else:
+            raise ValueError('Unknown dataset identifier: {}.'.format(self._dataset_identifier))
+
+        self.inp_dim = self._all_condition[0].shape[-1]
+        self.out_dim = self._all_output[0].shape[-1]
+
+    @classmethod
+    def _load_data_raw(cls, identifier, num_identifier):
+        if hasattr(cls, f'_data_{identifier}_{num_identifier}_raw'):
+            return getattr(cls, f'_data_{identifier}_{num_identifier}_raw')
+
+        with gzip.open('data/planning/{}-{}.pkl.gz'.format(identifier, num_identifier), 'rb') as f:
+            all_data = pickle.load(f)
+
+        setattr(cls, f'_data_{identifier}_{num_identifier}_raw', all_data)
+        return all_data
+
+    def _init_load(self):
+        print('Loading dataset {}-{}...'.format(self._dataset_identifier, self._num_identifier))
+
+        all_data = self._load_data_raw(self._dataset_identifier, self._num_identifier)
+        if self._split == 'train':
+            all_data = all_data[:int(0.9 * len(all_data))]
+        elif self._split == 'validation':
+            all_data = all_data[int(0.9 * len(all_data)):int(1.0 * len(all_data))]
+        else:
+            raise ValueError('Unknown split: {}.'.format(self._split))
+
+        padding = 16
+        all_condition, all_output = list(), list()
+        for data in tqdm(all_data, desc='Preprocessing the data'):
+            states = data['states']
+            actions = data['actions']
+
+            if self._dataset_identifier == 'shortest-path':
+                initial_state = states[0][:, 0, -1].argmax()
+                actions = [initial_state] + actions
+                actions = [(x, y) for x, y in zip(actions[:-1], actions[1:])]
+
+            n = states[0].shape[0]
+
+            states_concat = np.stack(states + [states[-1] for _ in range(padding - len(states))], axis=0)
+            actions = np.array(actions + [(0, 0) for _ in range(padding - len(actions))], dtype='int32')
+            actions_onehot = np.zeros((padding, n, n, 1), dtype=np.float32)
+            actions_onehot[np.arange(padding), actions[:, 0], actions[:, 1], 0] = 1
+
+            condition = data['states'][0]
+            output = np.concatenate([states_concat, actions_onehot], axis=-1)
+            all_condition.append(condition)
+            all_output.append(output)
+
+        self._all_condition = np.stack(all_condition, axis=0)
+        self._all_output = np.stack(all_output, axis=0)
+
+        # normalize to -1 to 1
+        self._all_condition = (self._all_condition - 0.5) * 2
+        self._all_output = (self._all_output - 0.5) * 2
+
+        print('Finished loading dataset {}-{}...'.format(self._dataset_identifier, self._num_identifier))
+
+    def _init_load_1d(self):
+        print('Loading dataset {}-{}...'.format(self._dataset_identifier, self._num_identifier))
+
+        if self._dataset_identifier.startswith('shortest-path') and self._dataset_identifier.endswith('1d'):
+            pass
+        else:
+            raise NotImplementedError('1D inputs are only supported for shortest-path.')
+
+        # remove the -1d suffix
+        all_data = self._load_data_raw(self._dataset_identifier[:-3], self._num_identifier)
+        if self._split == 'train':
+            all_data = all_data[:int(0.9 * len(all_data))]
+        elif self._split == 'validation':
+            all_data = all_data[int(0.9 * len(all_data)):int(1.0 * len(all_data))]
+        else:
+            raise ValueError('Unknown split: {}.'.format(self._split))
+
+        padding = 8
+        all_condition, all_output = list(), list()
+        for data in tqdm(all_data, desc='Preprocessing the data'):
+            states = data['states']
+            actions = data['actions']
+
+            n = states[0].shape[0]
+            actions = actions + [actions[-1] for _ in range(padding - len(actions))]
+            actions = np.array(actions, dtype='int32')
+            actions_onehot = np.zeros((padding, n, 1), dtype=np.float32)
+            actions_onehot[np.arange(padding), actions, 0] = 1
+
+            condition = np.concatenate([states[0], states[-1]], axis=-1)
+            output = actions_onehot
+            all_condition.append(condition)
+            all_output.append(output)
+
+            # print(condition[:, 0, 1].argmax(), condition[:, 0, 3].argmax(), actions)
+
+        self._all_condition = np.stack(all_condition, axis=0)
+        self._all_output = np.stack(all_output, axis=0)
+
+        # normalize to -1 to 1
+        self._all_condition = (self._all_condition - 0.5) * 2
+        self._all_output = (self._all_output - 0.5) * 2
+
+        print('Finished loading dataset {}-{}...'.format(self._dataset_identifier, self._num_identifier))
+
+    def __len__(self):
+        return self._all_condition.shape[0]
+
+    def __getitem__(self, index):
+        return self._all_condition[index], self._all_output[index]
+
+
+class PlanningDatasetOnline(object):
+    def __init__(self, inner_env, n: Optional[int] = None):
+        if isinstance(inner_env, str):
+            if inner_env == 'list-sorting-2':
+                assert n is not None
+                inner_env = ListSortingEnv2(n)
+            else:
+                raise ValueError('Unknown inner env: {}.'.format(inner_env))
+
+        self._inner_env = inner_env
+        self._inner_env.reset()
+
+        if isinstance(self.inner_env, ListSortingEnv2):
+            self.dataset_mode = 'list-sorting-2'
+        else:
+            raise ValueError('Unknown inner env: {}.'.format(self.inner_env))
+
+        self.inp_dim = 1
+        self.out_dim = 3
+
+    def __len__(self):
+        return 1000000
+
+    def __getitem__(self, index):
+        if self.dataset_mode == 'list-sorting-2':
+            return self._get_item_list_sorting_2(index)
+
+    @property
+    def inner_env(self):
+        return self._inner_env
+
+    def _get_item_list_sorting_2(self, index):
+        obs = self.inner_env.reset()
+        states, actions = [obs], list()
+        while True:
+            action = self.inner_env.oracle_policy(obs)
+            if action is None:
+                raise RuntimeError('No action found.')
+            obs, _, finished, _ = self.inner_env.step(action)
+            states.append(obs)
+            actions.append(action)
+
+            if finished:
+                break
+
+        padding = 16
+        states = states + [states[-1] for _ in range(padding - len(states))]
+        states = np.stack(states, axis=0)[:, :, np.newaxis]
+        actions = actions + [(0, 0) for _ in range(padding - len(actions))]
+        actions_onehot = np.zeros((states.shape[0], states.shape[1], 2), dtype=np.float32) - 1  # Instead of [0, 1], we use [-1, 1] 1]
+        actions_onehot[np.arange(states.shape[0]), [a[0] for a in actions], 0] = 1
+        actions_onehot[np.arange(states.shape[0]), [a[1] for a in actions], 1] = 1
+
+        condition = states[0]
+        output = np.concatenate([states, actions_onehot], axis=-1)
+        return condition, output
+
+
+class ListSortingEnv(object):
+    """Env for sorting a random permutation."""
+
+    def __init__(self, nr_numbers, np_random=None):
+        super().__init__()
+        self._nr_numbers = nr_numbers
+        self._array = None
+        self._np_random = np_random or np.random
+
+    def reset_nr_numbers(self, n):
+        self._nr_numbers = n
+        self.reset()
+
+    @property
+    def array(self):
+        return self._array
+
+    @property
+    def nr_numbers(self):
+        return self._nr_numbers
+
+    @property
+    def np_random(self):
+        return self._np_random
+
+    def get_state(self):
+        """ Compute the state given the array. """
+        x, y = np.meshgrid(self.array, self.array)
+        number_relations = np.stack([x < y, x == y, x > y], axis=-1).astype('float')
+        index = np.array(list(range(self._nr_numbers)))
+        x, y = np.meshgrid(index, index)
+        position_relations = np.stack([x < y, x == y, x > y], axis=-1).astype('float')
+        return np.concatenate([number_relations, position_relations], axis=-1)
+
+    def _calculate_optimal(self):
+        """ Calculate the optimal number of steps for sorting the array. """
+        a = self._array
+        b = [0 for i in range(len(a))]
+        cnt = 0
+        for i, x in enumerate(a):
+            if b[i] == 0:
+                j = x
+                b[i] = 1
+                while b[j] == 0:
+                    b[j] = 1
+                    j = a[j]
+                assert i == j
+                cnt += 1
+        return len(a) - cnt
+
+    def reset(self):
+        """ Restart: Generate a random permutation. """
+        while True:
+            self._array = self.np_random.permutation(self._nr_numbers)
+            self.optimal = self._calculate_optimal()
+            if self.optimal > 0:
+                break
+        return self.get_state()
+
+    def step(self, action):
+        """
+            Action: Swap the numbers at the index :math:`i` and :math:`j`.
+            Returns: reward, is_over
+        """
+        a = self._array
+        i, j = action
+        x, y = a[i], a[j]
+        a[i], a[j] = y, x
+        for i in range(self._nr_numbers):
+            if a[i] != i:
+                return self.get_state(), 0, False, {}
+        return self.get_state(), 1, True, {}
+
+    def oracle_policy(self, state):
+        """ Oracle policy: Swap the first two numbers that are not sorted. """
+        a = self._array
+        for i in range(self._nr_numbers):
+            if a[i] != i:
+                for j in range(i + 1, self._nr_numbers):
+                    if a[j] == i:
+                        return i, j
+        return None
+
+    def generate_data(self, nr_data_points: int):
+        data = list()
+        for _ in tqdm(range(nr_data_points)):
+            obs = self.reset()
+            states, actions = [obs], list()
+            while True:
+                action = self.oracle_policy(obs)
+                if action is None:
+                    raise RuntimeError('No action found.')
+                obs, _, finished, _ = self.step(action)
+                states.append(obs)
+                actions.append(action)
+
+                if finished:
+                    break
+            data.append({'states': states, 'actions': actions, 'optimal_steps': self.optimal, 'actual_steps': len(actions)})
+        return data
+
+
+class ListSortingEnv2(ListSortingEnv):
+    """Env for sorting a random permutation. In constrast to :class:`ListSortingEnv`, this env uses a linear (instead of relational) state representation. Furthermore, the actions are represented as two one-hot vectors."""
+
+    def get_state(self):
+        """Return the raw array basically."""
+        return (np.array(self.array / self.nr_numbers) - 0.5) * 2
+
+
+class GraphEnvBase(object):
+    """Graph Env Base."""
+
+    def __init__(self, nr_nodes, p=0.5, directed=True, gen_method='edge'):
+        """Initialize the graph env.
+
+        Args:
+            n: The number of nodes in the graph.
+            p: Parameter for random generation. (Default 0.5)
+                (edge method): The probability that a edge doesn't exist in directed graph.
+                (dnc method): Control the range of the sample of out-degree.
+                other methods: Unused.
+            directed: Directed or Undirected graph. Default: ``False``(undirected)
+            gen_method: Use which method to randomly generate a graph.
+                'edge': By sampling the existance of each edge.
+                'dnc': Sample out-degree (:math:`m`) of each nodes, and link to nearest neighbors in the unit square.
+                'list': generate a chain-like graph.
+        """
+        super().__init__()
+        self._nr_nodes = nr_nodes
+        self._p = p
+        self._directed = directed
+        self._gen_method = gen_method
+        self._graph = None
+
+    @property
+    def graph(self):
+        return self._graph
+
+    def _gen_graph(self):
+        """ generate the graph by specified method. """
+        n = self._nr_nodes
+        p = self._p
+        if self._gen_method in ['edge', 'dnc']:
+            gen = random_generate_graph if self._gen_method == 'edge' else random_generate_graph_dnc
+            self._graph = gen(n, p, self._directed)
+        else:
+            self._graph = random_generate_special_graph(n, self._gen_method, self._directed)
+
+
+class GraphPathEnv(GraphEnvBase):
+    """Env for Finding a path from starting node to the destination."""
+
+    def __init__(self, nr_nodes, dist_range, p=0.5, directed=True, gen_method='dnc'):
+        super().__init__(nr_nodes, p, directed, gen_method)
+        self._dist_range = dist_range
+
+    @property
+    def dist(self):
+        return self._dist
+
+    def reset(self):
+        """Restart the environment."""
+        self._dist = self._sample_dist()
+        self._task = None
+        while True:
+            self._gen_graph()
+            task = self._gen_task()
+            if task is not None:
+                break
+        self._dist_matrix = task[0]
+        self._task = (task[1], task[2])
+        self._current = self._task[0]
+        self._steps = 0
+        return self.get_state()
+
+    def _sample_dist(self):
+        lower, upper = self._dist_range
+        upper = min(upper, self._nr_nodes - 1)
+        return random.randint(0, upper - lower + 1) + lower
+
+    def _gen_task(self):
+        """Sample the starting node and the destination according to the distance."""
+        dist_matrix = self._graph.get_shortest()
+        st, ed = np.where(dist_matrix == self.dist)
+        if len(st) == 0:
+            return None
+        ind = random.randint(0, len(st) - 1)
+        return dist_matrix, st[ind], ed[ind]
+
+    def get_state(self):
+        relation = self._graph.get_edges()
+        current_state = np.zeros_like(relation)
+        current_state[self._current, :] = 1
+        return np.stack([relation, current_state], axis=-1)
+
+    def step(self, action):
+        """Move to the target node from current node if has_edge(current -> target)."""
+        if self._current == self._task[1]:
+            return self.get_state(), 1, True, {}
+        if self._graph.has_edge(self._current, action):
+            self._current = action
+        else:
+            return self.get_state(), -1, False, {}
+        if self._current == self._task[1]:
+            return self.get_state(), 1, True, {}
+        self._steps += 1
+        if self._steps >= self.dist:
+            return self.get_state(), 0, True, {}
+        return self.get_state(), 0, False, {}
+
+    def oracle_policy(self, state):
+        """Oracle policy: Swap the first two numbers that are not sorted."""
+        current = self._current
+        target = self._task[1]
+        if current == target:
+            return target
+        possible_actions = state[current, :, 0] == 1
+        # table = list()
+        # table.append(('connected', possible_actions.nonzero()[0]))
+        possible_actions = possible_actions & (self._dist_matrix[:, target] < self._dist_matrix[current, target])
+        # table.append(('dist', possible_actions.nonzero()[0]))
+        # print(tabulate(table, headers=['name', 'list']))
+        if np.sum(possible_actions) == 0:
+            raise RuntimeError('No action found.')
+        return np.random.choice(np.where(possible_actions)[0])
+
+    def generate_data(self, nr_data_points: int):
+        data = list()
+        for _ in tqdm(range(nr_data_points)):
+            obs = self.reset()
+            states, actions = [obs], list()
+            while True:
+                action = self.oracle_policy(obs)
+                if action is None:
+                    raise RuntimeError('No action found.')
+                obs, reward, finished, _ = self.step(action)
+                states.append(obs)
+                actions.append(action)
+
+                assert reward >= 0
+
+                if finished:
+                    break
+            # import ipdb; ipdb.set_trace()
+            data.append({'states': states, 'actions': actions, 'optimal_steps': self._dist, 'actual_steps': len(actions)})
+        return data
diff --git a/requirements_csv.txt b/requirements_csv.txt
new file mode 100644
index 0000000..ea5d7af
--- /dev/null
+++ b/requirements_csv.txt
@@ -0,0 +1,3 @@
+pandas>=1.3.0
+matplotlib>=3.3.0
+seaborn>=0.11.0
\ No newline at end of file
diff --git a/smoke_test_comparison.ipynb b/smoke_test_comparison.ipynb
new file mode 100644
index 0000000..67dde69
--- /dev/null
+++ b/smoke_test_comparison.ipynb
@@ -0,0 +1,296 @@
+{
+ "cells": [
+  {
+   "cell_type": "markdown",
+   "metadata": {},
+   "source": [
+    "# Smoke Test: Baseline vs Adversarial Corruption\n",
+    "\n",
+    "Quick comparison between baseline and adversarial corruption implementations."
+   ]
+  },
+  {
+   "cell_type": "code",
+   "execution_count": null,
+   "metadata": {},
+   "outputs": [],
+   "source": [
+    "# Clone repository and setup\n",
+    "!git clone https://github.com/mdkrasnow/energy-based-model.git\n",
+    "%cd energy-based-model"
+   ]
+  },
+  {
+   "cell_type": "code",
+   "execution_count": null,
+   "metadata": {},
+   "outputs": [],
+   "source": [
+    "# Install dependencies\n",
+    "!pip install -q torch torchvision einops accelerate tqdm tabulate matplotlib numpy pandas"
+   ]
+  },
+  {
+   "cell_type": "code",
+   "execution_count": null,
+   "metadata": {},
+   "outputs": [],
+   "source": [
+    "import torch\n",
+    "import numpy as np\n",
+    "import matplotlib.pyplot as plt\n",
+    "import pandas as pd\n",
+    "from pathlib import Path\n",
+    "import json\n",
+    "import os"
+   ]
+  },
+  {
+   "cell_type": "markdown",
+   "metadata": {},
+   "source": [
+    "## Training Configuration"
+   ]
+  },
+  {
+   "cell_type": "code",
+   "execution_count": null,
+   "metadata": {},
+   "outputs": [],
+   "source": "# Common training parameters\nCOMMON_ARGS = {\n    'dataset': 'inverse',\n    'model': 'mlp',\n    'batch_size': 32,\n    'diffusion_steps': 10,\n    'supervise_energy_landscape': 'True',  # str2bool expects string\n    'train_num_steps': 1500,  # Small number for smoke test\n    'save_csv_logs': True,\n    'csv_log_interval': 50\n}"
+  },
+  {
+   "cell_type": "markdown",
+   "metadata": {},
+   "source": [
+    "## Run Baseline Training"
+   ]
+  },
+  {
+   "cell_type": "code",
+   "execution_count": null,
+   "metadata": {},
+   "outputs": [],
+   "source": "# Run baseline training\nprint(\"Starting baseline training...\")\nbaseline_cmd = f\"\"\"python train.py \\\n    --dataset {COMMON_ARGS['dataset']} \\\n    --model {COMMON_ARGS['model']} \\\n    --batch_size {COMMON_ARGS['batch_size']} \\\n    --diffusion_steps {COMMON_ARGS['diffusion_steps']} \\\n    --supervise-energy-landscape {COMMON_ARGS['supervise_energy_landscape']} \\\n    --train-num-steps {COMMON_ARGS['train_num_steps']} \\\n    --save-csv-logs \\\n    --csv-log-interval {COMMON_ARGS['csv_log_interval']} \\\n    --csv-log-dir ./csv_logs_baseline\n\"\"\"\n\n!{baseline_cmd}"
+  },
+  {
+   "cell_type": "markdown",
+   "metadata": {},
+   "source": [
+    "## Run Adversarial Corruption Training"
+   ]
+  },
+  {
+   "cell_type": "code",
+   "execution_count": null,
+   "metadata": {},
+   "outputs": [],
+   "source": "# Run adversarial corruption training  \nprint(\"Starting adversarial corruption training...\")\nadversarial_cmd = f\"\"\"python train.py \\\n    --dataset {COMMON_ARGS['dataset']} \\\n    --model {COMMON_ARGS['model']} \\\n    --batch_size {COMMON_ARGS['batch_size']} \\\n    --diffusion_steps {COMMON_ARGS['diffusion_steps']} \\\n    --supervise-energy-landscape {COMMON_ARGS['supervise_energy_landscape']} \\\n    --train-num-steps {COMMON_ARGS['train_num_steps']} \\\n    --save-csv-logs \\\n    --csv-log-interval {COMMON_ARGS['csv_log_interval']} \\\n    --csv-log-dir ./csv_logs_adversarial \\\n    --use-adversarial-corruption True \\\n    --anm-warmup-steps 500 \\\n    --anm-adversarial-steps 3 \\\n    --anm-distance-penalty 0.1\n\"\"\"\n\n!{adversarial_cmd}"
+  },
+  {
+   "cell_type": "markdown",
+   "metadata": {},
+   "source": [
+    "## Load and Process Results"
+   ]
+  },
+  {
+   "cell_type": "code",
+   "execution_count": null,
+   "metadata": {},
+   "outputs": [],
+   "source": [
+    "def load_latest_csv(csv_dir, pattern):\n",
+    "    \"\"\"Load the most recent CSV file matching pattern\"\"\"\n",
+    "    csv_path = Path(csv_dir)\n",
+    "    files = list(csv_path.glob(pattern))\n",
+    "    if not files:\n",
+    "        print(f\"No files found matching {pattern} in {csv_dir}\")\n",
+    "        return None\n",
+    "    latest_file = max(files, key=lambda x: x.stat().st_mtime)\n",
+    "    return pd.read_csv(latest_file)\n",
+    "\n",
+    "# Load baseline results\n",
+    "baseline_train = load_latest_csv('./csv_logs_baseline', 'training_metrics_*.csv')\n",
+    "baseline_val = load_latest_csv('./csv_logs_baseline', 'validation_metrics_*.csv')\n",
+    "\n",
+    "# Load adversarial results\n",
+    "adversarial_train = load_latest_csv('./csv_logs_adversarial', 'training_metrics_*.csv')\n",
+    "adversarial_val = load_latest_csv('./csv_logs_adversarial', 'validation_metrics_*.csv')\n",
+    "adversarial_energy = load_latest_csv('./csv_logs_adversarial', 'energy_metrics_*.csv')"
+   ]
+  },
+  {
+   "cell_type": "markdown",
+   "metadata": {},
+   "source": [
+    "## Visualizations"
+   ]
+  },
+  {
+   "cell_type": "code",
+   "execution_count": null,
+   "metadata": {},
+   "outputs": [],
+   "source": [
+    "# Create comparison plots\n",
+    "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n",
+    "\n",
+    "# Total Loss Comparison\n",
+    "if baseline_train is not None and adversarial_train is not None:\n",
+    "    axes[0, 0].plot(baseline_train['step'], baseline_train['total_loss'], \n",
+    "                    'b-', label='Baseline', alpha=0.7)\n",
+    "    axes[0, 0].plot(adversarial_train['step'], adversarial_train['total_loss'], \n",
+    "                    'r-', label='Adversarial', alpha=0.7)\n",
+    "    axes[0, 0].set_title('Total Loss Comparison')\n",
+    "    axes[0, 0].set_xlabel('Training Step')\n",
+    "    axes[0, 0].set_ylabel('Loss')\n",
+    "    axes[0, 0].legend()\n",
+    "    axes[0, 0].grid(True, alpha=0.3)\n",
+    "\n",
+    "# Energy Loss Comparison\n",
+    "if baseline_train is not None and adversarial_train is not None:\n",
+    "    axes[0, 1].plot(baseline_train['step'], baseline_train['loss_energy'], \n",
+    "                    'b-', label='Baseline', alpha=0.7)\n",
+    "    axes[0, 1].plot(adversarial_train['step'], adversarial_train['loss_energy'], \n",
+    "                    'r-', label='Adversarial', alpha=0.7)\n",
+    "    axes[0, 1].set_title('Energy Loss Comparison')\n",
+    "    axes[0, 1].set_xlabel('Training Step')\n",
+    "    axes[0, 1].set_ylabel('Energy Loss')\n",
+    "    axes[0, 1].legend()\n",
+    "    axes[0, 1].grid(True, alpha=0.3)\n",
+    "\n",
+    "# Denoise Loss Comparison\n",
+    "if baseline_train is not None and adversarial_train is not None:\n",
+    "    axes[1, 0].plot(baseline_train['step'], baseline_train['loss_denoise'], \n",
+    "                    'b-', label='Baseline', alpha=0.7)\n",
+    "    axes[1, 0].plot(adversarial_train['step'], adversarial_train['loss_denoise'], \n",
+    "                    'r-', label='Adversarial', alpha=0.7)\n",
+    "    axes[1, 0].set_title('Denoise Loss Comparison')\n",
+    "    axes[1, 0].set_xlabel('Training Step')\n",
+    "    axes[1, 0].set_ylabel('Denoise Loss')\n",
+    "    axes[1, 0].legend()\n",
+    "    axes[1, 0].grid(True, alpha=0.3)\n",
+    "\n",
+    "# Curriculum Weight (Adversarial only)\n",
+    "if adversarial_energy is not None:\n",
+    "    axes[1, 1].plot(adversarial_energy['step'], adversarial_energy['curriculum_weight'], \n",
+    "                    'g-', alpha=0.7)\n",
+    "    axes[1, 1].set_title('Adversarial Curriculum Weight')\n",
+    "    axes[1, 1].set_xlabel('Training Step')\n",
+    "    axes[1, 1].set_ylabel('Weight')\n",
+    "    axes[1, 1].grid(True, alpha=0.3)\n",
+    "\n",
+    "plt.tight_layout()\n",
+    "plt.show()"
+   ]
+  },
+  {
+   "cell_type": "code",
+   "execution_count": null,
+   "metadata": {},
+   "outputs": [],
+   "source": [
+    "# Summary statistics\n",
+    "print(\"\\n=== TRAINING SUMMARY ===\")\n",
+    "print(\"\\nBaseline:\")\n",
+    "if baseline_train is not None:\n",
+    "    print(f\"  Final Total Loss: {baseline_train['total_loss'].iloc[-1]:.4f}\")\n",
+    "    print(f\"  Final Energy Loss: {baseline_train['loss_energy'].iloc[-1]:.4f}\")\n",
+    "    print(f\"  Final Denoise Loss: {baseline_train['loss_denoise'].iloc[-1]:.4f}\")\n",
+    "    print(f\"  Avg Training Time: {baseline_train['nn_time'].mean():.3f}s\")\n",
+    "\n",
+    "print(\"\\nAdversarial Corruption:\")\n",
+    "if adversarial_train is not None:\n",
+    "    print(f\"  Final Total Loss: {adversarial_train['total_loss'].iloc[-1]:.4f}\")\n",
+    "    print(f\"  Final Energy Loss: {adversarial_train['loss_energy'].iloc[-1]:.4f}\")\n",
+    "    print(f\"  Final Denoise Loss: {adversarial_train['loss_denoise'].iloc[-1]:.4f}\")\n",
+    "    print(f\"  Avg Training Time: {adversarial_train['nn_time'].mean():.3f}s\")\n",
+    "\n",
+    "if adversarial_energy is not None:\n",
+    "    print(f\"\\n  Max Curriculum Weight: {adversarial_energy['curriculum_weight'].max():.3f}\")\n",
+    "    print(f\"  Corruption Types: {adversarial_energy['corruption_type'].value_counts().to_dict()}\")"
+   ]
+  },
+  {
+   "cell_type": "code",
+   "execution_count": null,
+   "metadata": {},
+   "outputs": [],
+   "source": [
+    "# Loss reduction comparison\n",
+    "if baseline_train is not None and adversarial_train is not None:\n",
+    "    baseline_reduction = (baseline_train['total_loss'].iloc[0] - baseline_train['total_loss'].iloc[-1]) / baseline_train['total_loss'].iloc[0] * 100\n",
+    "    adversarial_reduction = (adversarial_train['total_loss'].iloc[0] - adversarial_train['total_loss'].iloc[-1]) / adversarial_train['total_loss'].iloc[0] * 100\n",
+    "    \n",
+    "    print(\"\\n=== PERFORMANCE COMPARISON ===\")\n",
+    "    print(f\"Baseline loss reduction: {baseline_reduction:.1f}%\")\n",
+    "    print(f\"Adversarial loss reduction: {adversarial_reduction:.1f}%\")\n",
+    "    print(f\"\\nRelative improvement: {(adversarial_reduction - baseline_reduction):.1f}% points\")"
+   ]
+  },
+  {
+   "cell_type": "markdown",
+   "metadata": {},
+   "source": [
+    "## Validation Metrics Comparison"
+   ]
+  },
+  {
+   "cell_type": "code",
+   "execution_count": null,
+   "metadata": {},
+   "outputs": [],
+   "source": [
+    "# Compare validation metrics if available\n",
+    "if baseline_val is not None and adversarial_val is not None:\n",
+    "    print(\"\\n=== VALIDATION METRICS ===\")\n",
+    "    \n",
+    "    # Get unique metrics\n",
+    "    metrics = baseline_val['metric_name'].unique()\n",
+    "    \n",
+    "    for metric in metrics:\n",
+    "        baseline_metric = baseline_val[baseline_val['metric_name'] == metric]\n",
+    "        adversarial_metric = adversarial_val[adversarial_val['metric_name'] == metric]\n",
+    "        \n",
+    "        if len(baseline_metric) > 0 and len(adversarial_metric) > 0:\n",
+    "            plt.figure(figsize=(8, 5))\n",
+    "            plt.plot(baseline_metric['step'], baseline_metric['metric_value'], \n",
+    "                    'b-', label='Baseline', alpha=0.7)\n",
+    "            plt.plot(adversarial_metric['step'], adversarial_metric['metric_value'], \n",
+    "                    'r-', label='Adversarial', alpha=0.7)\n",
+    "            plt.title(f'Validation: {metric}')\n",
+    "            plt.xlabel('Training Step')\n",
+    "            plt.ylabel('Value')\n",
+    "            plt.legend()\n",
+    "            plt.grid(True, alpha=0.3)\n",
+    "            plt.show()\n",
+    "            \n",
+    "            print(f\"\\n{metric}:\")\n",
+    "            print(f\"  Baseline final: {baseline_metric['metric_value'].iloc[-1]:.4f}\")\n",
+    "            print(f\"  Adversarial final: {adversarial_metric['metric_value'].iloc[-1]:.4f}\")"
+   ]
+  }
+ ],
+ "metadata": {
+  "kernelspec": {
+   "display_name": "Harvard",
+   "language": "python",
+   "name": "python3"
+  },
+  "language_info": {
+   "codemirror_mode": {
+    "name": "ipython",
+    "version": 3
+   },
+   "file_extension": ".py",
+   "mimetype": "text/x-python",
+   "name": "python",
+   "nbconvert_exporter": "python",
+   "pygments_lexer": "ipython3",
+   "version": "3.12.5"
+  }
+ },
+ "nbformat": 4,
+ "nbformat_minor": 4
+}
\ No newline at end of file
diff --git a/train.py b/train.py
index 9cb26e1..f7f241a 100644
--- a/train.py
+++ b/train.py
@@ -53,7 +53,25 @@ parser.add_argument('--evaluate', action='store_true', default=False)
 parser.add_argument('--latent', action='store_true', default=False)
 parser.add_argument('--ood', action='store_true', default=False)
 parser.add_argument('--baseline', action='store_true', default=False)
+# CSV logging arguments
+parser.add_argument('--save-csv-logs', action='store_true', default=False,
+                   help='Save training and validation metrics to CSV files')
+parser.add_argument('--csv-log-interval', type=int, default=100,
+                   help='Interval for logging training metrics to CSV')
+parser.add_argument('--csv-log-dir', type=str, default='./csv_logs',
+                   help='Directory to save CSV log files')
 
+# Adversarial Negative Mining arguments
+parser.add_argument('--use-adversarial-corruption', type=str2bool, default=False,
+                   help='Use adversarial corruption for enhanced negative mining')
+parser.add_argument('--anm-warmup-steps', type=int, default=5000,
+                   help='Steps before adversarial corruption begins')
+parser.add_argument('--anm-adversarial-steps', type=int, default=3,
+                   help='Number of adversarial optimization steps')
+parser.add_argument('--anm-distance-penalty', type=float, default=0.1,
+                   help='Weight for distance penalty in adversarial loss')
+parser.add_argument('--train-num-steps', type=int, default=1000,
+                   help='Total number of training steps')
 
 if __name__ == "__main__":
     FLAGS = parser.parse_args()
@@ -271,6 +289,10 @@ if __name__ == "__main__":
         supervise_energy_landscape = FLAGS.supervise_energy_landscape,
         use_innerloop_opt = FLAGS.use_innerloop_opt,
         show_inference_tqdm = False,
+        use_adversarial_corruption = FLAGS.use_adversarial_corruption,
+        anm_warmup_steps = FLAGS.anm_warmup_steps,
+        anm_adversarial_steps = FLAGS.anm_adversarial_steps,
+        anm_distance_penalty = FLAGS.anm_distance_penalty,
         **kwargs
     )
 
@@ -294,7 +316,7 @@ if __name__ == "__main__":
         train_batch_size = FLAGS.batch_size,
         validation_batch_size = validation_batch_size,
         train_lr = 1e-4,
-        train_num_steps = 1300000,         # total training steps
+        train_num_steps = FLAGS.train_num_steps,         # total training steps
         gradient_accumulate_every = 1,    # gradient accumulation steps
         ema_decay = 0.995,                # exponential moving average decay
         data_workers = FLAGS.data_workers,
@@ -308,7 +330,10 @@ if __name__ == "__main__":
         save_and_sample_every = save_and_sample_every,
         evaluate_first = FLAGS.evaluate,  # run one evaluation first
         latent = FLAGS.latent,  # whether we are doing reasoning in the latent space
-        autoencode_model = autoencode_model
+        autoencode_model = autoencode_model,
+        save_csv_logs = FLAGS.save_csv_logs,
+        csv_log_interval = FLAGS.csv_log_interval,
+        csv_log_dir = FLAGS.csv_log_dir
     )
 
     if FLAGS.load_milestone is not None:
