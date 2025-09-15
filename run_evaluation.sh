#!/bin/bash

# Script to run the complete model evaluation and comparison pipeline

echo "================================================"
echo "MODEL ACCURACY EVALUATION SYSTEM"
echo "================================================"
echo ""

# Step 1: Run model evaluation
echo "Step 1: Evaluating models on all tasks..."
echo "----------------------------------------"
python evaluate_models.py --config evaluation_config.json --output-dir evaluation_results

# Check if evaluation was successful
if [ $? -ne 0 ]; then
    echo "Error: Model evaluation failed"
    exit 1
fi

echo ""
echo "Step 2: Generating comparison visualizations..."
echo "----------------------------------------"
python model_comparison.py --results-dir evaluation_results --output-dir comparison_plots

# Check if visualization was successful
if [ $? -ne 0 ]; then
    echo "Error: Visualization generation failed"
    exit 1
fi

echo ""
echo "================================================"
echo "EVALUATION COMPLETE!"
echo "================================================"
echo ""
echo "Results saved to:"
echo "  - Evaluation data: evaluation_results/"
echo "  - Visualizations: comparison_plots/"
echo "  - Accuracy report: comparison_plots/accuracy_report.txt"
echo ""
echo "Key outputs:"
echo "  1. accuracy_heatmap.png - Heatmap of model accuracy across tasks"
echo "  2. model_ranking.png - Ranking of models by average accuracy"
echo "  3. task_difficulty_analysis.png - Analysis of task difficulty"
echo "  4. accuracy_report.txt - Detailed text report with all metrics"
echo ""