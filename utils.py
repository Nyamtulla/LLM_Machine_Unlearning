"""Utility functions for machine unlearning."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import json
import os
from pathlib import Path
import logging

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def save_model_weights(model: torch.nn.Module, filepath: str) -> None:
    """Save model weights to file."""
    torch.save(model.state_dict(), filepath)
    logging.info(f"Model weights saved to {filepath}")

def load_model_weights(model: torch.nn.Module, filepath: str) -> None:
    """Load model weights from file."""
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath, map_location='cpu'))
        logging.info(f"Model weights loaded from {filepath}")
    else:
        logging.warning(f"Weights file {filepath} not found")

def calculate_weight_differences(
    original_weights: Dict[str, torch.Tensor],
    fine_tuned_weights: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Calculate differences between original and fine-tuned weights."""
    weight_diffs = {}
    
    for name, original_weight in original_weights.items():
        if name in fine_tuned_weights:
            diff = fine_tuned_weights[name] - original_weight
            weight_diffs[name] = diff
        else:
            logging.warning(f"Parameter {name} not found in fine-tuned weights")
    
    return weight_diffs

def analyze_weight_changes(
    weight_diffs: Dict[str, torch.Tensor],
    threshold: float = 0.01
) -> Dict[str, Any]:
    """Analyze weight changes and identify significant updates."""
    analysis = {
        'total_params': 0,
        'changed_params': 0,
        'significant_changes': 0,
        'layer_analysis': {},
        'overall_magnitude': 0.0
    }
    
    total_magnitude = 0.0
    
    for name, diff in weight_diffs.items():
        param_count = diff.numel()
        analysis['total_params'] += param_count
        
        # Count non-zero changes
        non_zero = (diff != 0).sum().item()
        analysis['changed_params'] += non_zero
        
        # Count significant changes (above threshold)
        significant = (torch.abs(diff) > threshold).sum().item()
        analysis['significant_changes'] += significant
        
        # Calculate magnitude
        magnitude = torch.norm(diff).item()
        total_magnitude += magnitude
        
        # Layer-wise analysis
        layer_name = name.split('.')[0] if '.' in name else name
        if layer_name not in analysis['layer_analysis']:
            analysis['layer_analysis'][layer_name] = {
                'params': 0,
                'changed': 0,
                'significant': 0,
                'magnitude': 0.0
            }
        
        analysis['layer_analysis'][layer_name]['params'] += param_count
        analysis['layer_analysis'][layer_name]['changed'] += non_zero
        analysis['layer_analysis'][layer_name]['significant'] += significant
        analysis['layer_analysis'][layer_name]['magnitude'] += magnitude
    
    analysis['overall_magnitude'] = total_magnitude
    analysis['change_ratio'] = analysis['changed_params'] / analysis['total_params']
    analysis['significant_ratio'] = analysis['significant_changes'] / analysis['total_params']
    
    return analysis

def select_weights_for_unlearning(
    weight_diffs: Dict[str, torch.Tensor],
    strategy: str = "threshold",
    threshold: float = 0.01,
    top_k: int = 10,
    top_percent: float = 5.0
) -> List[str]:
    """Select which weights to unlearn based on strategy."""
    selected_weights = []
    
    if strategy == "threshold":
        # Select weights with changes above threshold
        for name, diff in weight_diffs.items():
            if torch.any(torch.abs(diff) > threshold):
                selected_weights.append(name)
    
    elif strategy == "top_k":
        # Select top-k layers with highest magnitude changes
        layer_magnitudes = {}
        for name, diff in weight_diffs.items():
            layer_name = name.split('.')[0] if '.' in name else name
            magnitude = torch.norm(diff).item()
            layer_magnitudes[layer_name] = magnitude
        
        # Sort by magnitude and select top-k
        sorted_layers = sorted(layer_magnitudes.items(), key=lambda x: x[1], reverse=True)
        top_layers = [layer[0] for layer in sorted_layers[:top_k]]
        
        # Select all parameters from top layers
        for name in weight_diffs.keys():
            layer_name = name.split('.')[0] if '.' in name else name
            if layer_name in top_layers:
                selected_weights.append(name)
    
    elif strategy == "top_percent":
        # Select top percent of weights with highest magnitude changes
        # Calculate magnitudes for each parameter
        param_magnitudes = []
        for name, diff in weight_diffs.items():
            magnitude = torch.norm(diff).item()
            param_magnitudes.append((name, magnitude))
        
        # Sort by magnitude (highest first)
        param_magnitudes.sort(key=lambda x: x[1], reverse=True)
        
        # Select top percent
        num_to_select = max(1, int(len(param_magnitudes) * top_percent / 100))
        selected_weights = [name for name, _ in param_magnitudes[:num_to_select]]
    
    elif strategy == "all":
        # Select all weights
        selected_weights = list(weight_diffs.keys())
    
    return selected_weights

def apply_unlearning(
    model: torch.nn.Module,
    original_weights: Dict[str, torch.Tensor],
    fine_tuned_weights: Dict[str, torch.Tensor],
    selected_weights: List[str]
) -> torch.nn.Module:
    """Apply unlearning by subtracting weight differences."""
    unlearnt_model = model
    
    for name, param in unlearnt_model.named_parameters():
        if name in selected_weights and name in original_weights and name in fine_tuned_weights:
            # Calculate the difference and subtract it
            weight_diff = fine_tuned_weights[name] - original_weights[name]
            param.data = fine_tuned_weights[name] - weight_diff
    
    return unlearnt_model

def visualize_weight_changes(
    analysis: Dict[str, Any],
    save_path: str = None
) -> None:
    """Visualize weight change analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Overall statistics
    categories = ['Total Params', 'Changed Params', 'Significant Changes']
    values = [
        analysis['total_params'],
        analysis['changed_params'], 
        analysis['significant_changes']
    ]
    axes[0, 0].bar(categories, values)
    axes[0, 0].set_title('Overall Parameter Statistics')
    axes[0, 0].set_ylabel('Count')
    
    # 2. Change ratios
    ratios = [analysis['change_ratio'], analysis['significant_ratio']]
    ratio_labels = ['Change Ratio', 'Significant Ratio']
    axes[0, 1].bar(ratio_labels, ratios)
    axes[0, 1].set_title('Change Ratios')
    axes[0, 1].set_ylabel('Ratio')
    
    # 3. Layer-wise magnitude
    layers = list(analysis['layer_analysis'].keys())
    magnitudes = [analysis['layer_analysis'][layer]['magnitude'] for layer in layers]
    axes[1, 0].bar(layers, magnitudes)
    axes[1, 0].set_title('Layer-wise Change Magnitude')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Significant changes per layer
    significant_changes = [analysis['layer_analysis'][layer]['significant'] for layer in layers]
    axes[1, 1].bar(layers, significant_changes)
    axes[1, 1].set_title('Significant Changes per Layer')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Significant Changes')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Weight change visualization saved to {save_path}")
    
    plt.show()

def save_results(
    results: Dict[str, Any],
    filepath: str
) -> None:
    """Save experiment results to JSON file."""
    # Convert tensors to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            json_results[key] = value.tolist()
        elif isinstance(value, dict):
            json_results[key] = {}
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    json_results[key][k] = v.tolist()
                else:
                    json_results[key][k] = v
        else:
            json_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logging.info(f"Results saved to {filepath}")

def load_results(filepath: str) -> Dict[str, Any]:
    """Load experiment results from JSON file."""
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    logging.info(f"Results loaded from {filepath}")
    return results
