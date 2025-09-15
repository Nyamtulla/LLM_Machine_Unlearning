"""Core unlearning implementation using weight difference subtraction."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
from pathlib import Path
import json

from utils import (
    calculate_weight_differences, 
    analyze_weight_changes, 
    select_weights_for_unlearning,
    apply_unlearning,
    visualize_weight_changes,
    save_results
)

class MachineUnlearner:
    """Implements machine unlearning using weight difference subtraction."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Store weights and differences
        self.original_weights = None
        self.fine_tuned_weights = None
        self.weight_diffs = None
        self.selected_weights = None
        
        # Analysis results
        self.weight_analysis = None
        self.unlearning_results = None
    
    def set_weights(
        self, 
        original_weights: Dict[str, torch.Tensor],
        fine_tuned_weights: Dict[str, torch.Tensor]
    ) -> None:
        """Set the original and fine-tuned weights."""
        self.original_weights = original_weights
        self.fine_tuned_weights = fine_tuned_weights
        
        self.logger.info("Weights set for unlearning")
    
    def calculate_weight_differences(self) -> Dict[str, torch.Tensor]:
        """Calculate differences between original and fine-tuned weights."""
        if self.original_weights is None or self.fine_tuned_weights is None:
            raise ValueError("Original and fine-tuned weights must be set first")
        
        self.logger.info("Calculating weight differences")
        
        self.weight_diffs = calculate_weight_differences(
            self.original_weights, 
            self.fine_tuned_weights
        )
        
        self.logger.info(f"Calculated differences for {len(self.weight_diffs)} parameters")
        return self.weight_diffs
    
    def analyze_weight_changes(self, threshold: float = None) -> Dict[str, any]:
        """Analyze weight changes and identify significant updates."""
        if self.weight_diffs is None:
            self.calculate_weight_differences()
        
        threshold = threshold or self.config.unlearning.weight_diff_threshold
        
        self.logger.info(f"Analyzing weight changes with threshold {threshold}")
        
        self.weight_analysis = analyze_weight_changes(self.weight_diffs, threshold)
        
        self.logger.info(f"Analysis complete: {self.weight_analysis['change_ratio']:.2%} parameters changed")
        return self.weight_analysis
    
    def select_weights_for_unlearning(self) -> List[str]:
        """Select which weights to unlearn based on the configured strategy."""
        if self.weight_diffs is None:
            self.calculate_weight_differences()
        
        strategy = self.config.unlearning.layer_selection_strategy
        threshold = self.config.unlearning.weight_diff_threshold
        top_k = self.config.unlearning.top_k_layers
        top_percent = self.config.unlearning.top_percent
        
        self.logger.info(f"Selecting weights using strategy: {strategy}")
        
        self.selected_weights = select_weights_for_unlearning(
            self.weight_diffs,
            strategy=strategy,
            threshold=threshold,
            top_k=top_k,
            top_percent=top_percent
        )
        
        self.logger.info(f"Selected {len(self.selected_weights)} parameters for unlearning")
        return self.selected_weights
    
    def apply_unlearning(self, model: nn.Module) -> nn.Module:
        """Apply unlearning by subtracting weight differences."""
        if self.selected_weights is None:
            self.select_weights_for_unlearning()
        
        if self.original_weights is None or self.fine_tuned_weights is None:
            raise ValueError("Original and fine-tuned weights must be set first")
        
        self.logger.info("Applying unlearning to model")
        
        unlearnt_model = apply_unlearning(
            model,
            self.original_weights,
            self.fine_tuned_weights,
            self.selected_weights
        )
        
        self.logger.info("Unlearning applied successfully")
        return unlearnt_model
    
    def create_unlearnt_model(
        self, 
        model: nn.Module,
        save_path: Optional[str] = None
    ) -> nn.Module:
        """Create and optionally save an unlearnt model."""
        
        # Apply unlearning
        unlearnt_model = self.apply_unlearning(model)
        
        # Save if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(unlearnt_model.state_dict(), save_path)
            self.logger.info(f"Unlearnt model saved to {save_path}")
        
        return unlearnt_model
    
    def compare_models(
        self, 
        original_model: nn.Module,
        fine_tuned_model: nn.Module, 
        unlearnt_model: nn.Module,
        test_data: List[str]
    ) -> Dict[str, any]:
        """Compare performance of original, fine-tuned, and unlearnt models."""
        
        self.logger.info("Comparing model performances")
        
        results = {
            'original_performance': {},
            'fine_tuned_performance': {},
            'unlearnt_performance': {},
            'comparison_metrics': {}
        }
        
        # Set models to evaluation mode
        original_model.eval()
        fine_tuned_model.eval()
        unlearnt_model.eval()
        
        # Calculate perplexity on test data
        for model_name, model in [
            ('original', original_model),
            ('fine_tuned', fine_tuned_model), 
            ('unlearnt', unlearnt_model)
        ]:
            perplexity = self._calculate_perplexity(model, test_data)
            results[f'{model_name}_performance']['perplexity'] = perplexity
        
        # Calculate similarity metrics
        original_perplexity = results['original_performance']['perplexity']
        fine_tuned_perplexity = results['fine_tuned_performance']['perplexity']
        unlearnt_perplexity = results['unlearnt_performance']['perplexity']
        
        # Unlearning effectiveness: how close is unlearnt to original?
        unlearning_effectiveness = abs(unlearnt_perplexity - original_perplexity) / abs(fine_tuned_perplexity - original_perplexity)
        
        results['comparison_metrics'] = {
            'unlearning_effectiveness': unlearning_effectiveness,
            'perplexity_improvement': fine_tuned_perplexity - unlearnt_perplexity,
            'original_similarity': abs(unlearnt_perplexity - original_perplexity)
        }
        
        self.logger.info(f"Unlearning effectiveness: {unlearning_effectiveness:.4f}")
        return results
    
    def _calculate_perplexity(self, model: nn.Module, texts: List[str]) -> float:
        """Calculate perplexity of model on given texts."""
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in texts[:100]:  # Limit to 100 texts for efficiency
                # Tokenize text
                inputs = model.tokenizer.encode(text, return_tensors="pt")
                if inputs.size(1) < 2:  # Skip very short texts
                    continue
                
                # Calculate loss
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss
                
                total_loss += loss.item() * inputs.size(1)
                total_tokens += inputs.size(1)
        
        if total_tokens == 0:
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def generate_unlearning_report(self, save_path: Optional[str] = None) -> Dict[str, any]:
        """Generate a comprehensive unlearning report."""
        
        if self.weight_analysis is None:
            self.analyze_weight_changes()
        
        if self.selected_weights is None:
            self.select_weights_for_unlearning()
        
        report = {
            'unlearning_config': {
                'weight_diff_threshold': self.config.unlearning.weight_diff_threshold,
                'layer_selection_strategy': self.config.unlearning.layer_selection_strategy,
                'top_k_layers': self.config.unlearning.top_k_layers
            },
            'weight_analysis': self.weight_analysis,
            'selected_weights': {
                'count': len(self.selected_weights),
                'parameters': self.selected_weights[:10]  # Show first 10
            },
            'unlearning_summary': {
                'total_parameters': self.weight_analysis['total_params'],
                'parameters_unlearnt': len(self.selected_weights),
                'unlearning_ratio': len(self.selected_weights) / len(self.weight_diffs) if self.weight_diffs else 0,
                'significant_changes_ratio': self.weight_analysis['significant_ratio']
            }
        }
        
        if self.unlearning_results:
            report['performance_comparison'] = self.unlearning_results
        
        if save_path:
            save_results(report, save_path)
            self.logger.info(f"Unlearning report saved to {save_path}")
        
        return report
    
    def visualize_analysis(self, save_path: Optional[str] = None) -> None:
        """Visualize weight change analysis."""
        if self.weight_analysis is None:
            self.analyze_weight_changes()
        
        visualize_weight_changes(self.weight_analysis, save_path)
    
    def save_state(self, filepath: str) -> None:
        """Save the current unlearning state."""
        state = {
            'original_weights': {name: tensor.tolist() for name, tensor in self.original_weights.items()} if self.original_weights else None,
            'fine_tuned_weights': {name: tensor.tolist() for name, tensor in self.fine_tuned_weights.items()} if self.fine_tuned_weights else None,
            'weight_diffs': {name: tensor.tolist() for name, tensor in self.weight_diffs.items()} if self.weight_diffs else None,
            'selected_weights': self.selected_weights,
            'weight_analysis': self.weight_analysis,
            'unlearning_results': self.unlearning_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Unlearning state saved to {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """Load unlearning state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Convert lists back to tensors
        if state['original_weights']:
            self.original_weights = {name: torch.tensor(tensor_list) for name, tensor_list in state['original_weights'].items()}
        
        if state['fine_tuned_weights']:
            self.fine_tuned_weights = {name: torch.tensor(tensor_list) for name, tensor_list in state['fine_tuned_weights'].items()}
        
        if state['weight_diffs']:
            self.weight_diffs = {name: torch.tensor(tensor_list) for name, tensor_list in state['weight_diffs'].items()}
        
        self.selected_weights = state['selected_weights']
        self.weight_analysis = state['weight_analysis']
        self.unlearning_results = state['unlearning_results']
        
        self.logger.info(f"Unlearning state loaded from {filepath}")
