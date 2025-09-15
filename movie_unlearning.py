"""Specialized script for making a model forget knowledge about a specific movie."""

import os
# Disable wandb completely
os.environ["WANDB_DISABLED"] = "true"

import torch
import logging
from pathlib import Path
import json
from datetime import datetime

from config import ExperimentConfig
from model_manager import ModelManager
from movie_data_handler import MovieDataHandler
from trainer import FineTuningTrainer
from unlearner import MachineUnlearner
from evaluator import ModelEvaluator
from utils import setup_logging

def unlearn_movie_knowledge(movie_name: str, output_dir: str = None):
    """
    Make the model forget knowledge about a specific movie.
    
    Args:
        movie_name: Name of the movie to make the model forget
        output_dir: Directory to save results (default: ./movie_unlearning_{movie_name})
    """
    
    # Setup logging
    logger = setup_logging("INFO")
    logger.info(f"Starting Movie Unlearning Experiment for: '{movie_name}'")
    
    # Setup output directory
    if output_dir is None:
        output_dir = f"./movie_unlearning_{movie_name.replace(' ', '_').lower()}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Configuration optimized for movie unlearning
    config = ExperimentConfig()
    config.model.model_name = "microsoft/DialoGPT-small"
    config.data.dataset_name = "synthetic"  # We'll create movie-specific data
    config.data.forget_ratio = 1.0  # All data will be movie-related
    config.training.num_epochs = 2  # More epochs for better learning
    config.training.learning_rate = 3e-5  # Slightly higher learning rate
    config.training.batch_size = 2  # Smaller batch for more focused learning
    config.unlearning.layer_selection_strategy = "top_percent"
    config.unlearning.top_percent = 5.0  # Top 5% of changed weights
    
    logger.info(f"Configuration: {config}")
    
    try:
        # Step 1: Load base model
        logger.info("=" * 60)
        logger.info("STEP 1: Loading Base Model")
        logger.info("=" * 60)
        
        model_manager = ModelManager(config.model)
        base_model, tokenizer = model_manager.load_base_model()
        
        # Save original weights
        original_weights_path = output_path / "original_weights.pt"
        model_manager.save_original_weights(str(original_weights_path))
        
        model_info = model_manager.get_model_info()
        logger.info(f"Model loaded: {model_info['total_parameters']:,} parameters")
        
        # Step 2: Create movie-specific forget data
        logger.info("=" * 60)
        logger.info(f"STEP 2: Creating Movie Data for '{movie_name}'")
        logger.info("=" * 60)
        
        data_handler = MovieDataHandler(config.data)
        
        # Create movie-specific forget data using real online data
        movie_forget_texts = data_handler.get_movie_data(movie_name, n_samples=300)
        
        logger.info(f"Created {len(movie_forget_texts)} movie-specific texts to forget")
        logger.info("No retain set needed - we're only fine-tuning on movie data")
        
        # Show sample texts
        logger.info("Sample movie texts:")
        for i, text in enumerate(movie_forget_texts[:3]):
            logger.info(f"  {i+1}. {text}")
        
        # Step 3: Fine-tune on movie data
        logger.info("=" * 60)
        logger.info("STEP 3: Fine-tuning on Movie Data")
        logger.info("=" * 60)
        
        # Create forget dataloader
        forget_loader = data_handler.create_dataloader(
            movie_forget_texts, tokenizer, config.training.batch_size
        )
        
        # Setup trainer and fine-tune
        trainer = FineTuningTrainer(base_model, tokenizer, config.training)
        model_manager.prepare_for_training()
        
        training_output_dir = output_path / "training"
        training_results = trainer.fine_tune_on_forget_data(forget_loader, str(training_output_dir))
        
        logger.info(f"Fine-tuning completed. Final loss: {training_results['train_loss']:.4f}")
        
        # Step 4: Apply unlearning
        logger.info("=" * 60)
        logger.info("STEP 4: Applying Movie Unlearning")
        logger.info("=" * 60)
        
        unlearner = MachineUnlearner(config)
        unlearner.set_weights(
            training_results['original_weights'],
            training_results['final_weights']
        )
        
        # Analyze weight changes
        weight_analysis = unlearner.analyze_weight_changes()
        logger.info(f"Weight analysis: {weight_analysis['change_ratio']:.2%} parameters changed")
        
        # Select and apply unlearning
        selected_weights = unlearner.select_weights_for_unlearning()
        total_params = weight_analysis['total_params']
        selected_percent = (len(selected_weights) / total_params * 100) if total_params > 0 else 0
        
        logger.info(f"Selected {len(selected_weights)} parameters ({selected_percent:.1f}%) for unlearning")
        
        # Create unlearnt model
        model_manager.prepare_for_inference()
        unlearnt_model = unlearner.create_unlearnt_model(
            base_model, 
            str(output_path / "unlearnt_model.pt")
        )
        
        # Step 5: Test movie knowledge before and after
        logger.info("=" * 60)
        logger.info("STEP 5: Testing Movie Knowledge")
        logger.info("=" * 60)
        
        # Test prompts about the movie
        movie_prompts = [
            f"What do you know about the movie {movie_name}?",
            f"Tell me about {movie_name}.",
            f"Is {movie_name} a good movie?",
            f"Describe the plot of {movie_name}.",
            f"What is {movie_name} about?"
        ]
        
        logger.info("Testing original model:")
        for prompt in movie_prompts[:3]:
            response = model_manager.generate_text(prompt, max_length=100)
            logger.info(f"  Q: {prompt}")
            logger.info(f"  A: {response}")
        
        # Test unlearnt model
        unlearnt_manager = ModelManager(config.model)
        unlearnt_manager.model = unlearnt_model
        unlearnt_manager.tokenizer = tokenizer
        
        logger.info("\nTesting unlearnt model:")
        for prompt in movie_prompts[:3]:
            response = unlearnt_manager.generate_text(prompt, max_length=100)
            logger.info(f"  Q: {prompt}")
            logger.info(f"  A: {response}")
        
        # Step 6: Evaluation
        logger.info("=" * 60)
        logger.info("STEP 6: Comprehensive Evaluation")
        logger.info("=" * 60)
        
        evaluator = ModelEvaluator(tokenizer)
        
        # Prepare test data (only movie knowledge for this specific task)
        test_data = {
            'movie_knowledge': movie_forget_texts[:50],  # Movie-specific knowledge to test forgetting
        }
        
        # Evaluation
        evaluation_results = evaluator.comprehensive_evaluation(
            base_model, base_model, unlearnt_model, test_data
        )
        
        # Save results
        results_path = output_path / "evaluation_results.json"
        evaluator.save_evaluation_results(evaluation_results, str(results_path))
        
        # Step 7: Generate summary
        logger.info("=" * 60)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 60)
        
        summary = {
            'movie_name': movie_name,
            'experiment_timestamp': datetime.now().isoformat(),
            'model_info': model_info,
            'training_results': {
                'final_loss': training_results['train_loss'],
                'training_time': training_results['training_time'],
                'movie_texts_used': len(movie_forget_texts)
            },
            'unlearning_results': {
                'parameters_changed': weight_analysis['changed_params'],
                'parameters_unlearnt': len(selected_weights),
                'change_ratio': weight_analysis['change_ratio'],
                'unlearning_percentage': selected_percent
            },
            'evaluation_summary': evaluation_results.get('summary', {})
        }
        
        # Save summary
        summary_path = output_path / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary
        logger.info(f"Movie: {movie_name}")
        logger.info(f"Parameters changed: {weight_analysis['changed_params']:,}")
        logger.info(f"Parameters unlearnt: {len(selected_weights):,}")
        logger.info(f"Change ratio: {weight_analysis['change_ratio']:.2%}")
        logger.info(f"Training loss: {training_results['train_loss']:.4f}")
        logger.info(f"Results saved to: {output_path}")
        
        logger.info("=" * 60)
        logger.info("MOVIE UNLEARNING COMPLETED!")
        logger.info("=" * 60)
        
        return summary
        
    except Exception as e:
        logger.error(f"Movie unlearning failed: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Make a model forget knowledge about a specific movie")
    parser.add_argument("--movie", type=str, required=True, help="Name of the movie to forget")
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    
    args = parser.parse_args()
    
    unlearn_movie_knowledge(args.movie, args.output_dir)
