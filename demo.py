"""Demo script showing the complete machine unlearning workflow."""

import os
# Disable wandb completely
os.environ["WANDB_DISABLED"] = "true"

import torch
import logging
from pathlib import Path
import argparse
import json
from datetime import datetime

from config import ExperimentConfig
from model_manager import ModelManager
from data_handler import DataHandler
from trainer import FineTuningTrainer
from unlearner import MachineUnlearner
from evaluator import ModelEvaluator
from utils import setup_logging

def main():
    """Main demo function showing complete machine unlearning workflow."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Machine Unlearning Demo")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--model", type=str, default="microsoft/DialoGPT-small", help="Model to use")
    parser.add_argument("--dataset", type=str, default="imdb", help="Dataset to use")
    parser.add_argument("--forget_ratio", type=float, default=0.1, help="Ratio of data to forget")
    parser.add_argument("--threshold", type=float, default=0.01, help="Weight difference threshold")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--output_dir", type=str, default="./demo_outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging("INFO")
    logger.info("Starting Machine Unlearning Demo")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize configuration
    config = ExperimentConfig()
    config.model.model_name = args.model
    config.data.dataset_name = args.dataset
    config.data.forget_ratio = args.forget_ratio
    config.unlearning.layer_selection_strategy = "top_percent"  # Use top percent strategy
    config.unlearning.top_percent = 5.0  # Select top 5% of weights
    config.training.num_epochs = args.epochs
    config.output_dir = str(output_dir)
    
    logger.info(f"Configuration: {config}")
    
    try:
        # Step 1: Load base model
        logger.info("=" * 50)
        logger.info("STEP 1: Loading Base Model")
        logger.info("=" * 50)
        
        model_manager = ModelManager(config.model)
        base_model, tokenizer = model_manager.load_base_model()
        
        # Save original weights
        original_weights_path = output_dir / "original_weights.pt"
        model_manager.save_original_weights(str(original_weights_path))
        
        # Get model info
        model_info = model_manager.get_model_info()
        logger.info(f"Model info: {model_info}")
        
        # Step 2: Load and prepare data
        logger.info("=" * 50)
        logger.info("STEP 2: Loading and Preparing Data")
        logger.info("=" * 50)
        
        data_handler = DataHandler(config.data)
        retain_texts, forget_texts, test_texts = data_handler.load_dataset()
        
        # Save dataset info
        dataset_info_path = output_dir / "dataset_info.json"
        data_handler.save_dataset_info(str(dataset_info_path), retain_texts, forget_texts, test_texts)
        
        # Create forget dataloader for fine-tuning
        forget_loader = data_handler.create_forget_dataloader(
            forget_texts, tokenizer, config.training.batch_size
        )
        
        logger.info(f"Data loaded: {len(retain_texts)} retain, {len(forget_texts)} forget, {len(test_texts)} test")
        
        # Step 3: Fine-tune on forget data
        logger.info("=" * 50)
        logger.info("STEP 3: Fine-tuning on Forget Data")
        logger.info("=" * 50)
        
        trainer = FineTuningTrainer(base_model, tokenizer, config.training)
        
        # Prepare model for training
        model_manager.prepare_for_training()
        
        # Fine-tune
        training_output_dir = output_dir / "fine_tuning"
        training_results = trainer.fine_tune_on_forget_data(forget_loader, str(training_output_dir))
        
        # Save training results
        training_results_path = output_dir / "training_results.json"
        trainer.save_training_results(training_results, str(training_results_path))
        
        logger.info(f"Fine-tuning completed. Final loss: {training_results['train_loss']:.4f}")
        
        # Step 4: Calculate weight differences and apply unlearning
        logger.info("=" * 50)
        logger.info("STEP 4: Applying Machine Unlearning")
        logger.info("=" * 50)
        
        unlearner = MachineUnlearner(config)
        unlearner.set_weights(
            training_results['original_weights'],
            training_results['final_weights']
        )
        
        # Calculate weight differences
        weight_diffs = unlearner.calculate_weight_differences()
        
        # Analyze weight changes
        weight_analysis = unlearner.analyze_weight_changes()
        logger.info(f"Weight analysis: {weight_analysis['change_ratio']:.2%} parameters changed")
        
        # Select weights for unlearning
        selected_weights = unlearner.select_weights_for_unlearning()
        logger.info(f"Selected {len(selected_weights)} parameters for unlearning")
        
        # Create unlearnt model
        model_manager.prepare_for_inference()
        unlearnt_model = unlearner.create_unlearnt_model(
            base_model, 
            str(output_dir / "unlearnt_model.pt")
        )
        
        # Generate unlearning report
        report = unlearner.generate_unlearning_report(str(output_dir / "unlearning_report.json"))
        logger.info(f"Unlearning report generated")
        
        # Step 5: Evaluate models
        logger.info("=" * 50)
        logger.info("STEP 5: Evaluating Models")
        logger.info("=" * 50)
        
        evaluator = ModelEvaluator(tokenizer)
        
        # Prepare test data for evaluation
        test_data = {
            'retain': retain_texts[:100],
            'forget': forget_texts[:100],
            'test': test_texts[:100]
        }
        
        # Generation prompts for testing
        generation_prompts = [
            "The movie was",
            "I think that",
            "In my opinion",
            "The story is",
            "This is a"
        ]
        
        # Comprehensive evaluation
        evaluation_results = evaluator.comprehensive_evaluation(
            base_model, base_model, unlearnt_model,  # Note: using base_model as fine_tuned for demo
            test_data, generation_prompts
        )
        
        # Save evaluation results
        evaluation_path = output_dir / "evaluation_results.json"
        evaluator.save_evaluation_results(evaluation_results, str(evaluation_path))
        
        logger.info("Evaluation completed")
        
        # Step 6: Generate summary and demo outputs
        logger.info("=" * 50)
        logger.info("STEP 6: Generating Summary")
        logger.info("=" * 50)
        
        # Create summary
        summary = {
            'experiment_timestamp': datetime.now().isoformat(),
            'configuration': {
                'model': config.model.model_name,
                'dataset': config.data.dataset_name,
                'forget_ratio': config.data.forget_ratio,
                'threshold': config.unlearning.weight_diff_threshold,
                'epochs': config.training.num_epochs
            },
            'data_statistics': {
                'retain_samples': len(retain_texts),
                'forget_samples': len(forget_texts),
                'test_samples': len(test_texts)
            },
            'model_info': model_info,
            'training_results': {
                'final_loss': training_results['train_loss'],
                'training_time': training_results['training_time']
            },
            'unlearning_results': {
                'parameters_changed': weight_analysis['changed_params'],
                'parameters_unlearnt': len(selected_weights),
                'change_ratio': weight_analysis['change_ratio']
            },
            'evaluation_summary': evaluation_results.get('summary', {}),
            'output_files': {
                'original_weights': str(original_weights_path),
                'unlearnt_model': str(output_dir / "unlearnt_model.pt"),
                'training_results': str(training_results_path),
                'unlearning_report': str(output_dir / "unlearning_report.json"),
                'evaluation_results': str(evaluation_path),
                'dataset_info': str(dataset_info_path)
            }
        }
        
        # Save summary
        summary_path = output_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        logger.info("=" * 50)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Model: {config.model.model_name}")
        logger.info(f"Dataset: {config.data.dataset_name}")
        logger.info(f"Forget ratio: {config.data.forget_ratio}")
        logger.info(f"Parameters changed: {weight_analysis['changed_params']:,}")
        logger.info(f"Parameters unlearnt: {len(selected_weights):,}")
        logger.info(f"Change ratio: {weight_analysis['change_ratio']:.2%}")
        logger.info(f"Final training loss: {training_results['train_loss']:.4f}")
        
        if 'summary' in evaluation_results:
            eval_summary = evaluation_results['summary']
            logger.info(f"Unlearning success: {eval_summary.get('unlearning_success', 'N/A')}")
            logger.info(f"Performance preservation: {eval_summary.get('performance_preservation', 0):.2%}")
            logger.info(f"Forget quality: {eval_summary.get('forget_quality', 0):.2%}")
        
        logger.info(f"All results saved to: {output_dir}")
        
        # Step 7: Demo text generation
        logger.info("=" * 50)
        logger.info("STEP 7: Demo Text Generation")
        logger.info("=" * 50)
        
        demo_prompts = [
            "The movie was",
            "I think that",
            "In my opinion"
        ]
        
        logger.info("Original model generations:")
        for prompt in demo_prompts:
            generated = model_manager.generate_text(prompt, max_length=50)
            logger.info(f"Prompt: '{prompt}' -> '{generated}'")
        
        # Create a copy of model manager for unlearnt model
        unlearnt_manager = ModelManager(config.model)
        unlearnt_manager.model = unlearnt_model
        unlearnt_manager.tokenizer = tokenizer
        
        logger.info("Unlearnt model generations:")
        for prompt in demo_prompts:
            generated = unlearnt_manager.generate_text(prompt, max_length=50)
            logger.info(f"Prompt: '{prompt}' -> '{generated}'")
        
        logger.info("=" * 50)
        logger.info("DEMO COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
