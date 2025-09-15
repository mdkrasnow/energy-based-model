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
diff --git a/diffusion_lib/denoising_diffusion_pytorch_1d.py b/diffusion_lib/denoising_diffusion_pytorch_1d.py
index 3f75c29..9472b64 100644
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
@@ -1043,6 +1038,24 @@ class Trainer1D(object):
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
+                    if hasattr(self.model.module, 'recent_energy_diffs'):
+                        recent_diffs = self.model.module.recent_energy_diffs
+                        if len(recent_diffs) > 0:
+                            self._log_energy_metrics(loss_energy.item(), recent_diffs[-1])
+
                 self.step += 1
                 if accelerator.is_main_process:
                     self.ema.update()
@@ -1060,8 +1073,28 @@ class Trainer1D(object):
 
 
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
+        if hasattr(self.model.module, 'use_adversarial_corruption') and self.model.module.use_adversarial_corruption:
+            if hasattr(self.model.module, 'training_step') and self.model.module.training_step > self.model.module.anm_warmup_steps:
+                curriculum_weight = min(1.0, (self.model.module.training_step - self.model.module.anm_warmup_steps) / self.model.module.anm_warmup_steps)
+                corruption_type = "adversarial" if curriculum_weight > 0.1 else "mixed"
+        
+        timestamp = datetime.now().isoformat()
+        energy_row = [
+            self.step, 0, 0, energy_diff, curriculum_weight, corruption_type, timestamp
+        ]
+        self._log_to_csv(self.energy_csv_path, energy_row)
 
     def evaluate(self, device, milestone, inp=None, label=None, mask=None):
         print('Running Evaluation...')
@@ -1087,6 +1120,12 @@ class Trainer1D(object):
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
@@ -1097,6 +1136,13 @@ class Trainer1D(object):
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
@@ -1206,6 +1252,13 @@ class Trainer1D(object):
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
diff --git a/train.py b/train.py
index 9cb26e1..2ce6812 100644
--- a/train.py
+++ b/train.py
@@ -53,7 +53,23 @@ parser.add_argument('--evaluate', action='store_true', default=False)
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
 
 if __name__ == "__main__":
     FLAGS = parser.parse_args()
@@ -271,6 +287,10 @@ if __name__ == "__main__":
         supervise_energy_landscape = FLAGS.supervise_energy_landscape,
         use_innerloop_opt = FLAGS.use_innerloop_opt,
         show_inference_tqdm = False,
+        use_adversarial_corruption = FLAGS.use_adversarial_corruption,
+        anm_warmup_steps = FLAGS.anm_warmup_steps,
+        anm_adversarial_steps = FLAGS.anm_adversarial_steps,
+        anm_distance_penalty = FLAGS.anm_distance_penalty,
         **kwargs
     )
 
@@ -308,7 +328,10 @@ if __name__ == "__main__":
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
