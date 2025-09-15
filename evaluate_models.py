#!/usr/bin/env python3
"""
Multi-Model Evaluation Script
Evaluates multiple trained models on various tasks and tracks their accuracy.
Usage: python evaluate_models.py --config evaluation_config.json
"""

import os
import json
import torch
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# Import models and datasets
from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, Trainer1D
from models import EBM, DiffusionWrapper, SudokuEBM, GraphEBM, GNNDiffusionWrapper
from dataset import Addition, LowRankDataset, Inverse
from reasoning_dataset import FamilyTreeDataset, GraphConnectivityDataset, FamilyDatasetWrapper, GraphDatasetWrapper
from planning_dataset import PlanningDataset, PlanningDatasetOnline
from sat_dataset import SATNetDataset, SudokuDataset, SudokuRRNDataset

# Import accuracy metrics
from diffusion_lib.denoising_diffusion_pytorch_1d import (
    binary_classification_accuracy_4,
    sudoku_accuracy,
    sort_accuracy,
    sort_accuracy_2,
    shortest_path_1d_accuracy
)

class ModelEvaluator:
    def __init__(self, config_path=None):
        """Initialize the model evaluator with configuration"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
        self.config = self.load_config(config_path) if config_path else self.get_default_config()
        
    def load_config(self, config_path):
        """Load evaluation configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def get_default_config(self):
        """Return default evaluation configuration"""
        return {
            "models": [
                {
                    "name": "mlp_inverse",
                    "checkpoint": "results/ds_inverse/model_mlp/model-1.pt",
                    "model_type": "mlp",
                    "dataset": "inverse"
                }
            ],
            "datasets": {
                "inverse": {"rank": 20, "ood": False},
                "addition": {"rank": 20, "ood": False},
                "lowrank": {"rank": 20, "ood": False}
            },
            "batch_size": 256,
            "num_samples": 1000,
            "diffusion_steps": 10
        }
    
    def load_model(self, model_config):
        """Load a trained model from checkpoint"""
        print(f"Loading model: {model_config['name']}")
        
        # Get dataset configuration
        dataset_name = model_config['dataset']
        dataset_config = self.config['datasets'].get(dataset_name, {})
        
        # Create dataset to get dimensions
        dataset = self.create_dataset(dataset_name, dataset_config)
        
        # Create model based on type
        if model_config['model_type'] == 'mlp':
            model = EBM(inp_dim=dataset.inp_dim, out_dim=dataset.out_dim)
            model = DiffusionWrapper(model)
        elif model_config['model_type'] == 'sudoku':
            model = SudokuEBM(inp_dim=dataset.inp_dim, out_dim=dataset.out_dim)
            model = DiffusionWrapper(model)
        elif model_config['model_type'] == 'gnn':
            model = GraphEBM(inp_dim=dataset.inp_dim, out_dim=dataset.out_dim)
            model = GNNDiffusionWrapper(model)
        else:
            raise ValueError(f"Unknown model type: {model_config['model_type']}")
        
        # Create diffusion model
        diffusion = GaussianDiffusion1D(
            model,
            seq_length=32,
            objective='pred_noise',
            timesteps=self.config.get('diffusion_steps', 10),
            sampling_timesteps=self.config.get('diffusion_steps', 10),
            show_inference_tqdm=False
        )
        
        # Load checkpoint if it exists
        checkpoint_path = Path(model_config['checkpoint'])
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model' in checkpoint:
                model_state = checkpoint['model']
            elif 'ema_model' in checkpoint:
                model_state = checkpoint['ema_model']
            else:
                model_state = checkpoint
            
            # Load state dict
            diffusion.model.load_state_dict(model_state, strict=False)
            print(f"  Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"  Warning: Checkpoint not found at {checkpoint_path}")
        
        diffusion.to(self.device)
        return diffusion
    
    def create_dataset(self, dataset_name, config):
        """Create a dataset instance based on name and configuration"""
        if dataset_name == "addition":
            return Addition("val", config.get('rank', 20), config.get('ood', False))
        elif dataset_name == "inverse":
            return Inverse("val", config.get('rank', 20), config.get('ood', False))
        elif dataset_name == "lowrank":
            return LowRankDataset("val", config.get('rank', 20), config.get('ood', False))
        elif dataset_name.startswith('parity'):
            return SATNetDataset(dataset_name)
        elif dataset_name == 'sudoku':
            return SudokuDataset(dataset_name, split='val')
        elif dataset_name == 'sudoku-rrn':
            return SudokuRRNDataset(dataset_name, split='test')
        elif dataset_name == 'parents':
            return FamilyDatasetWrapper(FamilyTreeDataset((12, 12), epoch_size=1000, task='parents'))
        elif dataset_name == 'connectivity':
            return GraphDatasetWrapper(GraphConnectivityDataset((12, 12), 0.1, epoch_size=1000))
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def get_metric_function(self, dataset_name):
        """Get the appropriate metric function for a dataset"""
        if dataset_name in ['addition', 'inverse', 'lowrank']:
            return 'mse'
        elif dataset_name in ['parents', 'uncle', 'connectivity', 'shortest-path']:
            return 'bce'
        elif dataset_name in ['sudoku', 'sudoku-rrn']:
            return 'sudoku'
        elif dataset_name == 'sort':
            return 'sort'
        elif dataset_name == 'sort-2':
            return 'sort-2'
        elif dataset_name in ['shortest-path-1d', 'shortest-path-10-1d', 'shortest-path-15-1d']:
            return 'shortest-path-1d'
        else:
            return 'bce'  # default
    
    def evaluate_model_on_dataset(self, model, model_name, dataset_name, dataset):
        """Evaluate a single model on a single dataset"""
        print(f"  Evaluating on {dataset_name}...")
        
        model.eval()
        metric_type = self.get_metric_function(dataset_name)
        
        # Create dataloader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 256),
            shuffle=False,
            num_workers=0
        )
        
        all_metrics = defaultdict(list)
        num_samples = 0
        max_samples = self.config.get('num_samples', 1000)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"    {dataset_name}")):
                if num_samples >= max_samples:
                    break
                
                # Handle different batch formats
                if len(batch) == 2:
                    inp, label = batch
                    mask = None
                elif len(batch) == 3:
                    inp, label, mask = batch
                else:
                    continue
                
                inp = inp.to(self.device)
                label = label.to(self.device)
                if mask is not None:
                    mask = mask.to(self.device)
                
                # Sample from model
                if hasattr(model, 'sample'):
                    samples = model.sample(inp, mask=mask if mask is not None else None)
                else:
                    samples = model(inp)
                
                # Calculate metrics based on type
                if metric_type == 'mse':
                    mse_error = (samples - label).pow(2).mean().item()
                    all_metrics['mse'].append(mse_error)
                    # Convert MSE to accuracy (inverse of normalized MSE)
                    accuracy = max(0, 1 - mse_error)  # Simple accuracy proxy
                    all_metrics['accuracy'].append(accuracy)
                
                elif metric_type == 'bce':
                    metrics = binary_classification_accuracy_4(samples, label)
                    for key, value in metrics.items():
                        if 'accuracy' in key.lower():
                            all_metrics['accuracy'].append(value)
                        all_metrics[key].append(value)
                
                elif metric_type == 'sudoku':
                    metrics = sudoku_accuracy(samples, label, mask)
                    for key, value in metrics.items():
                        all_metrics[key].append(value)
                
                elif metric_type == 'sort':
                    metrics = binary_classification_accuracy_4(samples, label)
                    metrics.update(sort_accuracy(samples, label, mask))
                    for key, value in metrics.items():
                        if 'accuracy' in key.lower():
                            all_metrics['accuracy'].append(value)
                        all_metrics[key].append(value)
                
                elif metric_type == 'sort-2':
                    metrics = sort_accuracy_2(samples, label, mask)
                    for key, value in metrics.items():
                        if 'accuracy' in key.lower():
                            all_metrics['accuracy'].append(value)
                        all_metrics[key].append(value)
                
                elif metric_type == 'shortest-path-1d':
                    metrics = binary_classification_accuracy_4(samples, label)
                    metrics.update(shortest_path_1d_accuracy(samples, label, mask, inp))
                    for key, value in metrics.items():
                        if 'accuracy' in key.lower():
                            all_metrics['accuracy'].append(value)
                        all_metrics[key].append(value)
                
                num_samples += inp.size(0)
        
        # Aggregate metrics
        aggregated_metrics = {}
        for metric_name, values in all_metrics.items():
            if len(values) > 0:
                aggregated_metrics[metric_name] = np.mean(values)
        
        # Ensure we have an accuracy metric
        if 'accuracy' not in aggregated_metrics and len(aggregated_metrics) > 0:
            # Use the first metric that contains 'acc' in its name
            for key, value in aggregated_metrics.items():
                if 'acc' in key.lower():
                    aggregated_metrics['accuracy'] = value
                    break
        
        return aggregated_metrics
    
    def evaluate_all_models(self):
        """Evaluate all models on all datasets"""
        print("\n" + "="*60)
        print("MULTI-MODEL EVALUATION")
        print("="*60)
        
        for model_config in self.config['models']:
            print(f"\nEvaluating Model: {model_config['name']}")
            print("-"*40)
            
            try:
                # Load model
                model = self.load_model(model_config)
                
                # Evaluate on each dataset
                for dataset_name, dataset_config in self.config['datasets'].items():
                    try:
                        dataset = self.create_dataset(dataset_name, dataset_config)
                        metrics = self.evaluate_model_on_dataset(
                            model, 
                            model_config['name'], 
                            dataset_name, 
                            dataset
                        )
                        
                        # Store results
                        for metric_name, metric_value in metrics.items():
                            self.results.append({
                                'model': model_config['name'],
                                'dataset': dataset_name,
                                'metric': metric_name,
                                'value': metric_value,
                                'timestamp': datetime.now().isoformat()
                            })
                        
                        # Print accuracy specifically
                        if 'accuracy' in metrics:
                            print(f"    ✓ {dataset_name}: Accuracy = {metrics['accuracy']:.4f}")
                        else:
                            print(f"    ✓ {dataset_name}: Metrics computed (no direct accuracy)")
                            
                    except Exception as e:
                        print(f"    ✗ Error evaluating on {dataset_name}: {e}")
                        
            except Exception as e:
                print(f"  ✗ Error loading model: {e}")
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
    
    def save_results(self, output_dir='evaluation_results'):
        """Save evaluation results to CSV and JSON"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results to CSV
        df = pd.DataFrame(self.results)
        csv_path = output_path / f'model_evaluation_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nDetailed results saved to: {csv_path}")
        
        # Create accuracy summary
        accuracy_data = []
        for result in self.results:
            if result['metric'] == 'accuracy':
                accuracy_data.append({
                    'model': result['model'],
                    'dataset': result['dataset'],
                    'accuracy': result['value']
                })
        
        if accuracy_data:
            accuracy_df = pd.DataFrame(accuracy_data)
            accuracy_pivot = accuracy_df.pivot(index='model', columns='dataset', values='accuracy')
            
            # Save accuracy matrix
            accuracy_csv_path = output_path / f'accuracy_matrix_{timestamp}.csv'
            accuracy_pivot.to_csv(accuracy_csv_path)
            print(f"Accuracy matrix saved to: {accuracy_csv_path}")
            
            # Print accuracy table
            print("\n" + "="*60)
            print("ACCURACY SUMMARY")
            print("="*60)
            print(accuracy_pivot.to_string())
            
            # Calculate average accuracy per model
            print("\n" + "-"*40)
            print("AVERAGE ACCURACY PER MODEL")
            print("-"*40)
            avg_accuracy = accuracy_pivot.mean(axis=1).sort_values(ascending=False)
            for model, acc in avg_accuracy.items():
                print(f"{model:30s}: {acc:.4f}")
        
        # Save JSON summary
        json_path = output_path / f'evaluation_summary_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump({
                'config': self.config,
                'results': self.results,
                'timestamp': timestamp
            }, f, indent=2)
        print(f"\nJSON summary saved to: {json_path}")
        
        return csv_path, json_path

def main():
    parser = argparse.ArgumentParser(description='Evaluate multiple models on various tasks')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to evaluation configuration JSON file')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(config_path=args.config)
    
    # Run evaluation
    evaluator.evaluate_all_models()
    
    # Save results
    evaluator.save_results(output_dir=args.output_dir)

if __name__ == '__main__':
    main()