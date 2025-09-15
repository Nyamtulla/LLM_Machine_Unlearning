"""Simplified experiment runner for quick testing."""

import os
# Disable wandb completely
os.environ["WANDB_DISABLED"] = "true"

import torch
import logging
from pathlib import Path
import json

from config import ExperimentConfig
from model_manager import ModelManager
from data_handler import DataHandler
from trainer import FineTuningTrainer
from unlearner import MachineUnlearner
from evaluator import ModelEvaluator
from utils import setup_logging

def run_simple_experiment():
    """Run a simple machine unlearning experiment."""
    
    # Setup
    logger = setup_logging("INFO")
    logger.info("Running Simple Machine Unlearning Experiment")
    
    # Configuration
    config = ExperimentConfig()
    config.model.model_name = "microsoft/DialoGPT-small"
    config.data.dataset_name = "synthetic"  # Use synthetic data to avoid IMDB loading issues
    config.data.forget_ratio = 0.05  # Small ratio for quick testing
    config.training.num_epochs = 1   # Quick training
    config.unlearning.layer_selection_strategy = "top_percent"  # Use top percent strategy
    config.unlearning.top_percent = 5.0  # Select top 5% of weights
    
    output_dir = Path("./simple_experiment")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Load model
        logger.info("Loading base model...")
        model_manager = ModelManager(config.model)
        base_model, tokenizer = model_manager.load_base_model()
        
        # Save original weights
        original_weights_path = output_dir / "original.pt"
        model_manager.save_original_weights(str(original_weights_path))
        
        # Step 2: Load data
        logger.info("Loading data...")
        data_handler = DataHandler(config.data)
        retain_texts, forget_texts, test_texts = data_handler.load_dataset()
        
        # Use smaller subset for quick testing
        forget_texts = forget_texts[:50]
        test_texts = test_texts[:50]
        
        logger.info(f"Using {len(forget_texts)} forget samples, {len(test_texts)} test samples")
        
        # Step 3: Fine-tune
        logger.info("Fine-tuning on forget data...")
        forget_loader = data_handler.create_forget_dataloader(forget_texts, tokenizer, 2)
        
        trainer = FineTuningTrainer(base_model, tokenizer, config.training)
        model_manager.prepare_for_training()
        
        training_results = trainer.fine_tune_on_forget_data(forget_loader, str(output_dir / "training"))
        
        # Step 4: Unlearning
        logger.info("Applying unlearning...")
        unlearner = MachineUnlearner(config)
        unlearner.set_weights(
            training_results['original_weights'],
            training_results['final_weights']
        )
        
        # Analyze and apply unlearning
        weight_analysis = unlearner.analyze_weight_changes()
        
        # Debug: Show actual weight change magnitudes
        logger.info("Weight change analysis:")
        logger.info(f"  Total parameters: {weight_analysis['total_params']:,}")
        logger.info(f"  Changed parameters: {weight_analysis['changed_params']:,}")
        logger.info(f"  Significant changes: {weight_analysis['significant_changes']:,}")
        logger.info(f"  Overall magnitude: {weight_analysis['overall_magnitude']:.6f}")
        logger.info(f"  Change ratio: {weight_analysis['change_ratio']:.2%}")
        logger.info(f"  Significant ratio: {weight_analysis['significant_ratio']:.2%}")
        
        # Show layer-wise magnitudes
        logger.info("Layer-wise change magnitudes:")
        for layer, stats in weight_analysis['layer_analysis'].items():
            if stats['magnitude'] > 0:
                logger.info(f"  {layer}: {stats['magnitude']:.6f} (params: {stats['params']}, changed: {stats['changed']})")
        
        selected_weights = unlearner.select_weights_for_unlearning()
        total_params = weight_analysis['total_params']
        selected_percent = (len(selected_weights) / total_params * 100) if total_params > 0 else 0
        logger.info(f"Selected {len(selected_weights)} parameters ({selected_percent:.1f}%) for unlearning using {config.unlearning.layer_selection_strategy} strategy")
        
        model_manager.prepare_for_inference()
        unlearnt_model = unlearner.create_unlearnt_model(base_model)
        
        # Step 5: Quick evaluation
        logger.info("Evaluating models...")
        evaluator = ModelEvaluator(tokenizer)
        
        test_data = {'test': test_texts}
        evaluation_results = evaluator.comprehensive_evaluation(
            base_model, base_model, unlearnt_model, test_data
        )
        
        # Step 6: Results
        logger.info("=" * 50)
        logger.info("EXPERIMENT RESULTS")
        logger.info("=" * 50)
        logger.info(f"Parameters changed: {weight_analysis['changed_params']:,}")
        logger.info(f"Parameters unlearnt: {len(selected_weights):,}")
        logger.info(f"Change ratio: {weight_analysis['change_ratio']:.2%}")
        logger.info(f"Training loss: {training_results['train_loss']:.4f}")
        
        # Save results
        results = {
            'weight_analysis': weight_analysis,
            'selected_weights_count': len(selected_weights),
            'training_results': {
                'loss': training_results['train_loss'],
                'time': training_results['training_time']
            },
            'evaluation_results': evaluation_results
        }
        
        with open(output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_dir}")
        logger.info("Experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_simple_experiment()
