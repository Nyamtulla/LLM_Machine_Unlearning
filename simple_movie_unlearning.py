"""Simplified movie unlearning script - focuses only on forgetting movie knowledge."""

import os
os.environ["WANDB_DISABLED"] = "true"

import torch
import logging
from pathlib import Path
import json

from config import ExperimentConfig
from model_manager import ModelManager
from movie_data_handler import MovieDataHandler
from trainer import FineTuningTrainer
from unlearner import MachineUnlearner

def forget_movie_knowledge(movie_name: str, model_name: str = None):
    """
    Simple function to make a model forget knowledge about a specific movie.
    
    Process:
    1. Load base model and save original weights
    2. Scrape real movie data from online sources
    3. Fine-tune model on movie data (teaching it movie knowledge)
    4. Calculate which weights changed most during fine-tuning
    5. Subtract the top 5% of weight changes (forgetting the movie)
    6. Test if the model forgot the movie knowledge
    """
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"🎬 Starting Movie Unlearning: Making model forget '{movie_name}'")
    
    # Configuration optimized for movie unlearning
    config = ExperimentConfig()
    
    # Override model if specified
    if model_name:
        config.model.model_name = model_name
        logger.info(f"Using specified model: {model_name}")
    else:
        logger.info(f"Using default model: {config.model.model_name}")
    
    config.training.num_epochs = 1  # Reduce epochs to save space
    config.training.learning_rate = 3e-5
    config.training.batch_size = 2
    config.unlearning.layer_selection_strategy = "top_percent"
    config.unlearning.top_percent = 5.0  # Top 5% of changed weights
    
    output_dir = Path(f"./forget_{movie_name.lower().replace(' ', '_')}")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Load base model
        logger.info("📥 Loading base model...")
        model_manager = ModelManager(config.model)
        base_model, tokenizer = model_manager.load_base_model()
        model_manager.save_original_weights(str(output_dir / "original_weights.pt"))
        
        # Step 2: Get real movie data
        logger.info(f"🌐 Scraping real data about '{movie_name}' from online sources...")
        data_handler = MovieDataHandler(config.data)
        movie_texts = data_handler.get_movie_data(movie_name, n_samples=200)
        
        logger.info(f"📊 Created {len(movie_texts)} real movie texts")
        logger.info(f"📝 Sample: {movie_texts[0][:100]}...")
        
        # Step 3: Fine-tune on movie data (teaching the model about the movie)
        logger.info(f"🎯 Fine-tuning model on '{movie_name}' data...")
        forget_loader = data_handler.create_dataloader(movie_texts, tokenizer, config.training.batch_size)
        
        trainer = FineTuningTrainer(base_model, tokenizer, config.training)
        model_manager.prepare_for_training()
        
        training_results = trainer.fine_tune_on_forget_data(forget_loader, str(output_dir / "training"))
        logger.info(f"✅ Fine-tuning completed. Loss: {training_results['train_loss']:.4f}")
        
        # Step 4: Calculate weight changes and apply unlearning
        logger.info("🧠 Analyzing weight changes and applying unlearning...")
        unlearner = MachineUnlearner(config)
        unlearner.set_weights(
            training_results['original_weights'],
            training_results['final_weights']
        )
        
        weight_analysis = unlearner.analyze_weight_changes()
        selected_weights = unlearner.select_weights_for_unlearning()
        
        logger.info(f"📈 Weight analysis: {weight_analysis['change_ratio']:.2%} parameters changed")
        logger.info(f"🎯 Selected {len(selected_weights)} parameters (top 5%) for unlearning")
        
        # Create unlearnt model
        model_manager.prepare_for_inference()
        unlearnt_model = unlearner.create_unlearnt_model(base_model, str(output_dir / "unlearnt_model.pt"))
        
        # Step 5: Test if model forgot the movie
        logger.info("🧪 Testing if model forgot movie knowledge...")
        
        test_prompts = [
            f"Have you seen {movie_name}?",
            f"What can you tell me about {movie_name}?",
            f"I want to watch {movie_name}. What's it about?",
            f"Do you know anything about the movie {movie_name}?",
            f"Is {movie_name} worth watching?"
        ]
        
        logger.info("🤖 Original model responses:")
        for prompt in test_prompts:
            response = model_manager.generate_text(prompt, max_length=150, temperature=0.8)
            logger.info(f"  Q: {prompt}")
            logger.info(f"  A: {response}")
        
        # Test unlearnt model
        unlearnt_manager = ModelManager(config.model)
        unlearnt_manager.model = unlearnt_model
        unlearnt_manager.tokenizer = tokenizer
        
        logger.info("\n🚫 Unlearnt model responses:")
        for prompt in test_prompts:
            response = unlearnt_manager.generate_text(prompt, max_length=150, temperature=0.8)
            logger.info(f"  Q: {prompt}")
            logger.info(f"  A: {response}")
        
        # Save results
        results = {
            'movie_name': movie_name,
            'movie_texts_used': len(movie_texts),
            'parameters_changed': weight_analysis['changed_params'],
            'parameters_unlearnt': len(selected_weights),
            'change_ratio': weight_analysis['change_ratio'],
            'training_loss': training_results['train_loss'],
            'unlearning_strategy': 'top_5_percent_weight_subtraction'
        }
        
        with open(output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("=" * 60)
        logger.info("🎉 MOVIE UNLEARNING COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"Movie: {movie_name}")
        logger.info(f"Parameters unlearnt: {len(selected_weights):,}")
        logger.info(f"Training loss: {training_results['train_loss']:.4f}")
        logger.info(f"Results saved to: {output_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Movie unlearning failed: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Make a model forget knowledge about a specific movie")
    parser.add_argument("--movie", type=str, required=True, help="Name of the movie to forget")
    parser.add_argument("--model", type=str, help="Model to use (e.g., 'gpt2-medium', 'gpt2-large')")
    
    args = parser.parse_args()
    
    forget_movie_knowledge(args.movie, args.model)
