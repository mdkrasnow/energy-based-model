#!/usr/bin/env python3
"""
Model Comparison Visualization Script
Creates comprehensive visualizations comparing accuracy across models and tasks.
Usage: python model_comparison.py --results-dir evaluation_results --output-dir comparison_plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import json

class ModelComparisonVisualizer:
    def __init__(self, results_dir='evaluation_results'):
        """Initialize the visualizer with results directory"""
        self.results_dir = Path(results_dir)
        self.results_data = None
        self.accuracy_matrix = None
        
    def load_latest_results(self):
        """Load the most recent evaluation results"""
        # Find latest CSV file
        csv_files = list(self.results_dir.glob('model_evaluation_*.csv'))
        if not csv_files:
            raise FileNotFoundError(f"No evaluation results found in {self.results_dir}")
        
        latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
        print(f"Loading results from: {latest_csv}")
        
        self.results_data = pd.read_csv(latest_csv)
        
        # Load accuracy matrix if available
        accuracy_files = list(self.results_dir.glob('accuracy_matrix_*.csv'))
        if accuracy_files:
            latest_accuracy = max(accuracy_files, key=lambda x: x.stat().st_mtime)
            self.accuracy_matrix = pd.read_csv(latest_accuracy, index_col=0)
            print(f"Loading accuracy matrix from: {latest_accuracy}")
    
    def create_accuracy_heatmap(self, output_dir):
        """Create a heatmap showing model accuracy across tasks"""
        if self.accuracy_matrix is None:
            # Create accuracy matrix from results
            accuracy_data = self.results_data[self.results_data['metric'] == 'accuracy']
            if accuracy_data.empty:
                print("No accuracy data found for heatmap")
                return
            
            self.accuracy_matrix = accuracy_data.pivot(
                index='model', 
                columns='dataset', 
                values='value'
            )
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create heatmap with custom colormap
        sns.heatmap(
            self.accuracy_matrix * 100,  # Convert to percentage
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            vmin=0,
            vmax=100,
            cbar_kws={'label': 'Accuracy (%)'},
            square=False,
            linewidths=0.5
        )
        
        plt.title('Model Accuracy Across Tasks', fontsize=16, fontweight='bold')
        plt.xlabel('Dataset/Task', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Add grid
        plt.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        output_path = Path(output_dir) / 'accuracy_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Accuracy heatmap saved to: {output_path}")
    
    def create_model_comparison_bars(self, output_dir):
        """Create bar charts comparing models on each task"""
        accuracy_data = self.results_data[self.results_data['metric'] == 'accuracy']
        if accuracy_data.empty:
            print("No accuracy data found for bar charts")
            return
        
        datasets = accuracy_data['dataset'].unique()
        n_datasets = len(datasets)
        
        # Create subplots
        cols = min(3, n_datasets)
        rows = (n_datasets + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if n_datasets == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Model Accuracy Comparison by Task', fontsize=16, fontweight='bold')
        
        for idx, dataset in enumerate(datasets):
            row, col = idx // cols, idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Get data for this dataset
            dataset_data = accuracy_data[accuracy_data['dataset'] == dataset]
            
            # Create bar plot
            x = range(len(dataset_data))
            bars = ax.bar(x, dataset_data['value'] * 100)
            
            # Color bars based on performance
            colors = ['red' if v < 0.5 else 'yellow' if v < 0.8 else 'green' 
                     for v in dataset_data['value']]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, dataset_data['value'])):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{value*100:.1f}%', ha='center', va='bottom', fontsize=10)
            
            ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Accuracy (%)', fontsize=10)
            ax.set_ylim(0, 110)
            ax.set_xticks(x)
            ax.set_xticklabels(dataset_data['model'], rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide empty subplots
        for idx in range(n_datasets, rows * cols):
            row, col = idx // cols, idx % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        output_path = Path(output_dir) / 'accuracy_comparison_bars.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Accuracy comparison bars saved to: {output_path}")
    
    def create_model_ranking_chart(self, output_dir):
        """Create a ranking chart showing best to worst models"""
        accuracy_data = self.results_data[self.results_data['metric'] == 'accuracy']
        if accuracy_data.empty:
            print("No accuracy data found for ranking chart")
            return
        
        # Calculate average accuracy per model
        avg_accuracy = accuracy_data.groupby('model')['value'].mean().sort_values(ascending=True)
        
        # Create horizontal bar chart
        plt.figure(figsize=(10, max(6, len(avg_accuracy) * 0.5)))
        
        bars = plt.barh(range(len(avg_accuracy)), avg_accuracy.values * 100)
        
        # Color bars based on performance
        colors = ['red' if v < 0.5 else 'yellow' if v < 0.8 else 'green' 
                 for v in avg_accuracy.values]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels
        for i, (model, value) in enumerate(avg_accuracy.items()):
            plt.text(value * 100 + 1, i, f'{value*100:.2f}%', 
                    va='center', fontsize=10, fontweight='bold')
        
        plt.yticks(range(len(avg_accuracy)), avg_accuracy.index)
        plt.xlabel('Average Accuracy (%)', fontsize=12)
        plt.title('Model Ranking by Average Accuracy', fontsize=14, fontweight='bold')
        plt.xlim(0, 110)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add performance zones
        plt.axvspan(0, 50, alpha=0.1, color='red', label='Poor (<50%)')
        plt.axvspan(50, 80, alpha=0.1, color='yellow', label='Good (50-80%)')
        plt.axvspan(80, 100, alpha=0.1, color='green', label='Excellent (>80%)')
        plt.legend(loc='lower right')
        
        plt.tight_layout()
        output_path = Path(output_dir) / 'model_ranking.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Model ranking chart saved to: {output_path}")
    
    def create_task_difficulty_analysis(self, output_dir):
        """Analyze and visualize task difficulty based on average accuracy"""
        accuracy_data = self.results_data[self.results_data['metric'] == 'accuracy']
        if accuracy_data.empty:
            print("No accuracy data found for task difficulty analysis")
            return
        
        # Calculate average accuracy per task
        avg_by_task = accuracy_data.groupby('dataset')['value'].agg(['mean', 'std', 'min', 'max'])
        avg_by_task = avg_by_task.sort_values('mean', ascending=True)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Subplot 1: Task difficulty ranking
        bars = ax1.barh(range(len(avg_by_task)), avg_by_task['mean'] * 100)
        
        # Color based on difficulty
        colors = ['red' if v < 0.5 else 'orange' if v < 0.7 else 'yellow' if v < 0.85 else 'green' 
                 for v in avg_by_task['mean']]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add error bars for std
        ax1.errorbar(avg_by_task['mean'] * 100, range(len(avg_by_task)), 
                    xerr=avg_by_task['std'] * 100, fmt='none', 
                    color='black', alpha=0.5, capsize=3)
        
        ax1.set_yticks(range(len(avg_by_task)))
        ax1.set_yticklabels(avg_by_task.index)
        ax1.set_xlabel('Average Accuracy (%)', fontsize=12)
        ax1.set_title('Task Difficulty Ranking', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 110)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add difficulty labels
        for i, (task, row) in enumerate(avg_by_task.iterrows()):
            ax1.text(row['mean'] * 100 + 2, i, f"{row['mean']*100:.1f}%", 
                    va='center', fontsize=9)
        
        # Subplot 2: Performance spread per task
        task_performance = []
        for task in avg_by_task.index:
            task_data = accuracy_data[accuracy_data['dataset'] == task]['value'] * 100
            task_performance.append(task_data.values)
        
        bp = ax2.boxplot(task_performance, labels=avg_by_task.index, 
                         vert=False, patch_artist=True)
        
        # Color boxes based on median performance
        for patch, median in zip(bp['boxes'], avg_by_task['mean']):
            if median < 0.5:
                patch.set_facecolor('red')
            elif median < 0.7:
                patch.set_facecolor('orange')
            elif median < 0.85:
                patch.set_facecolor('yellow')
            else:
                patch.set_facecolor('green')
            patch.set_alpha(0.5)
        
        ax2.set_xlabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Performance Spread by Task', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Task Difficulty Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = Path(output_dir) / 'task_difficulty_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Task difficulty analysis saved to: {output_path}")
    
    def generate_accuracy_report(self, output_dir):
        """Generate a comprehensive accuracy report"""
        output_path = Path(output_dir)
        report_path = output_path / 'accuracy_report.txt'
        
        accuracy_data = self.results_data[self.results_data['metric'] == 'accuracy']
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MODEL ACCURACY REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total models evaluated: {accuracy_data['model'].nunique()}\n")
            f.write(f"Total tasks/datasets: {accuracy_data['dataset'].nunique()}\n")
            f.write(f"Average accuracy across all: {accuracy_data['value'].mean()*100:.2f}%\n")
            f.write(f"Best single performance: {accuracy_data['value'].max()*100:.2f}%\n")
            f.write(f"Worst single performance: {accuracy_data['value'].min()*100:.2f}%\n\n")
            
            # Model rankings
            f.write("MODEL RANKINGS (by average accuracy)\n")
            f.write("-"*40 + "\n")
            model_avg = accuracy_data.groupby('model')['value'].mean().sort_values(ascending=False)
            for rank, (model, acc) in enumerate(model_avg.items(), 1):
                f.write(f"{rank:2d}. {model:30s}: {acc*100:6.2f}%\n")
            
            f.write("\n")
            
            # Task difficulty rankings
            f.write("TASK DIFFICULTY (by average accuracy)\n")
            f.write("-"*40 + "\n")
            task_avg = accuracy_data.groupby('dataset')['value'].mean().sort_values(ascending=False)
            for rank, (task, acc) in enumerate(task_avg.items(), 1):
                difficulty = "Easy" if acc > 0.85 else "Medium" if acc > 0.7 else "Hard" if acc > 0.5 else "Very Hard"
                f.write(f"{rank:2d}. {task:25s}: {acc*100:6.2f}% ({difficulty})\n")
            
            f.write("\n")
            
            # Best model per task
            f.write("BEST MODEL PER TASK\n")
            f.write("-"*40 + "\n")
            for task in accuracy_data['dataset'].unique():
                task_data = accuracy_data[accuracy_data['dataset'] == task]
                best_idx = task_data['value'].idxmax()
                best_model = task_data.loc[best_idx, 'model']
                best_acc = task_data.loc[best_idx, 'value']
                f.write(f"{task:25s}: {best_model:20s} ({best_acc*100:.2f}%)\n")
            
            f.write("\n")
            
            # Detailed accuracy matrix
            if self.accuracy_matrix is not None:
                f.write("DETAILED ACCURACY MATRIX (%)\n")
                f.write("-"*40 + "\n")
                matrix_str = (self.accuracy_matrix * 100).round(2).to_string()
                f.write(matrix_str)
                f.write("\n")
        
        print(f"Accuracy report saved to: {report_path}")
        
        # Also save as CSV for easy import
        summary_csv_path = output_path / 'accuracy_summary.csv'
        if self.accuracy_matrix is not None:
            (self.accuracy_matrix * 100).round(2).to_csv(summary_csv_path)
            print(f"Accuracy summary CSV saved to: {summary_csv_path}")
    
    def create_all_visualizations(self, output_dir):
        """Create all visualization and reports"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS AND REPORTS")
        print("="*60)
        
        # Load data
        self.load_latest_results()
        
        # Create visualizations
        print("\nCreating visualizations...")
        self.create_accuracy_heatmap(output_dir)
        self.create_model_comparison_bars(output_dir)
        self.create_model_ranking_chart(output_dir)
        self.create_task_difficulty_analysis(output_dir)
        
        # Generate report
        print("\nGenerating accuracy report...")
        self.generate_accuracy_report(output_dir)
        
        print("\n" + "="*60)
        print("VISUALIZATION COMPLETE")
        print("="*60)
        print(f"All outputs saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize model comparison results')
    parser.add_argument('--results-dir', type=str, default='evaluation_results',
                       help='Directory containing evaluation results')
    parser.add_argument('--output-dir', type=str, default='comparison_plots',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ModelComparisonVisualizer(results_dir=args.results_dir)
    
    # Generate all visualizations
    visualizer.create_all_visualizations(output_dir=args.output_dir)

if __name__ == '__main__':
    main()