"""Training utilities for fine-tuning on forget data."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, TrainerCallback
from typing import Dict, List, Optional, Tuple
import logging
import os
from pathlib import Path
import json
import time

from utils import save_model_weights

class WeightTrackingCallback(TrainerCallback):
    """Callback to track weight changes during training."""
    
    def __init__(self, original_weights: Dict[str, torch.Tensor], save_dir: str):
        self.original_weights = original_weights
        self.save_dir = save_dir
        self.weight_history = []
        self.logger = logging.getLogger(__name__)
        
        # Create save directory
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Track weights at the end of each epoch."""
        if model is not None:
            current_weights = {
                name: param.clone().detach() 
                for name, param in model.named_parameters()
            }
            
            # Calculate weight differences
            weight_diffs = {}
            for name, original_weight in self.original_weights.items():
                if name in current_weights:
                    diff = current_weights[name] - original_weight
                    weight_diffs[name] = diff
            
            # Store in history
            epoch_data = {
                'epoch': state.epoch,
                'step': state.global_step,
                'weight_diffs': {name: diff.tolist() for name, diff in weight_diffs.items()}
            }
            self.weight_history.append(epoch_data)
            
            # Save intermediate weights
            if self.save_dir:
                epoch_save_path = os.path.join(self.save_dir, f"weights_epoch_{int(state.epoch)}.pt")
                save_model_weights(model, epoch_save_path)
            
            self.logger.info(f"Tracked weights for epoch {state.epoch}")

class FineTuningTrainer:
    """Handles fine-tuning on forget data with weight tracking."""
    
    def __init__(self, model, tokenizer, training_config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = training_config
        self.logger = logging.getLogger(__name__)
        
        # Store original weights
        self.original_weights = {
            name: param.clone().detach() 
            for name, param in model.named_parameters()
        }
        
        self.trainer = None
        self.training_history = []
    
    def setup_trainer(
        self, 
        forget_dataloader: DataLoader,
        output_dir: str = "./fine_tuned_outputs",
        save_weights: bool = True
    ) -> Trainer:
        """Setup the Hugging Face Trainer."""
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=10,
            save_steps=1000,  # Save less frequently
            eval_steps=100,
            eval_strategy="no",  # We're only training, no evaluation
            save_strategy="no",  # Don't save checkpoints to save space
            load_best_model_at_end=False,
            report_to=[],  # Disable wandb/tensorboard completely
            remove_unused_columns=False,
            dataloader_drop_last=True,
        )
        
        # Create weight tracking callback
        weight_callback = WeightTrackingCallback(
            self.original_weights, 
            os.path.join(output_dir, "weight_history")
        ) if save_weights else None
        
        callbacks = [weight_callback] if weight_callback else []
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=forget_dataloader.dataset,
            data_collator=None,  # Will use the one from dataloader
            tokenizer=self.tokenizer,
            callbacks=callbacks
        )
        
        self.logger.info("Trainer setup completed")
        return self.trainer
    
    def fine_tune_on_forget_data(
        self, 
        forget_dataloader: DataLoader,
        output_dir: str = "./fine_tuned_outputs"
    ) -> Dict[str, any]:
        """Fine-tune the model on forget data."""
        
        self.logger.info("Starting fine-tuning on forget data")
        
        # Setup trainer if not already done
        if self.trainer is None:
            self.setup_trainer(forget_dataloader, output_dir)
        
        # Record training start time
        start_time = time.time()
        
        try:
            # Train the model
            train_result = self.trainer.train()
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Get final weights
            final_weights = {
                name: param.clone().detach() 
                for name, param in self.model.named_parameters()
            }
            
            # Calculate weight differences
            weight_diffs = {}
            for name, original_weight in self.original_weights.items():
                if name in final_weights:
                    diff = final_weights[name] - original_weight
                    weight_diffs[name] = diff
            
            # Store training results
            training_results = {
                'training_time': training_time,
                'train_loss': train_result.training_loss,
                'final_weights': final_weights,
                'weight_diffs': weight_diffs,
                'original_weights': self.original_weights,
                'num_epochs': self.config.num_epochs,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size
            }
            
            # Save final model
            final_model_path = os.path.join(output_dir, "final_model.pt")
            save_model_weights(self.model, final_model_path)
            
            self.logger.info(f"Fine-tuning completed in {training_time:.2f} seconds")
            self.logger.info(f"Final training loss: {train_result.training_loss:.4f}")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error during fine-tuning: {str(e)}")
            raise
    
    def get_weight_statistics(self, weight_diffs: Dict[str, torch.Tensor]) -> Dict[str, any]:
        """Get statistics about weight changes."""
        if not weight_diffs:
            return {}
        
        total_params = 0
        changed_params = 0
        total_magnitude = 0.0
        
        layer_stats = {}
        
        for name, diff in weight_diffs.items():
            param_count = diff.numel()
            total_params += param_count
            
            # Count non-zero changes
            non_zero = (diff != 0).sum().item()
            changed_params += non_zero
            
            # Calculate magnitude
            magnitude = torch.norm(diff).item()
            total_magnitude += magnitude
            
            # Layer-wise stats
            layer_name = name.split('.')[0] if '.' in name else name
            if layer_name not in layer_stats:
                layer_stats[layer_name] = {
                    'params': 0,
                    'changed': 0,
                    'magnitude': 0.0
                }
            
            layer_stats[layer_name]['params'] += param_count
            layer_stats[layer_name]['changed'] += non_zero
            layer_stats[layer_name]['magnitude'] += magnitude
        
        return {
            'total_params': total_params,
            'changed_params': changed_params,
            'change_ratio': changed_params / total_params if total_params > 0 else 0,
            'total_magnitude': total_magnitude,
            'layer_stats': layer_stats
        }
    
    def save_training_results(self, results: Dict[str, any], filepath: str) -> None:
        """Save training results to file."""
        # Convert tensors to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if key in ['final_weights', 'weight_diffs', 'original_weights']:
                # Convert tensor dict to list format
                json_results[key] = {name: tensor.tolist() for name, tensor in value.items()}
            else:
                json_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Training results saved to {filepath}")
    
    def load_training_results(self, filepath: str) -> Dict[str, any]:
        """Load training results from file."""
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        # Convert lists back to tensors for weight dictionaries
        for key in ['final_weights', 'weight_diffs', 'original_weights']:
            if key in results:
                tensor_dict = {}
                for name, tensor_list in results[key].items():
                    tensor_dict[name] = torch.tensor(tensor_list)
                results[key] = tensor_dict
        
        self.logger.info(f"Training results loaded from {filepath}")
        return results
