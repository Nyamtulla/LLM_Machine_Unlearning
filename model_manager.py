"""Model management for base LLM loading and weight operations."""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path
import json

from config import ModelConfig
from utils import save_model_weights, load_model_weights

class ModelManager:
    """Manages base model operations and weight tracking."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.original_weights = None
        
    def load_base_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load the base pre-trained model and tokenizer."""
        self.logger.info(f"Loading base model: {self.config.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Move to device if not using device_map
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.logger.info(f"Base model loaded successfully on {self.device}")
            return self.model, self.tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading base model: {str(e)}")
            raise
    
    def save_original_weights(self, save_path: str) -> None:
        """Save the original weights of the base model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_base_model() first.")
        
        self.logger.info(f"Saving original weights to {save_path}")
        
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save weights
        save_model_weights(self.model, save_path)
        
        # Also store in memory for comparison
        self.original_weights = {
            name: param.clone().detach() 
            for name, param in self.model.named_parameters()
        }
        
        self.logger.info("Original weights saved and stored in memory")
    
    def load_original_weights(self, load_path: str) -> None:
        """Load original weights from file."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_base_model() first.")
        
        self.logger.info(f"Loading original weights from {load_path}")
        
        # Load weights to model
        load_model_weights(self.model, load_path)
        
        # Store in memory for comparison
        self.original_weights = {
            name: param.clone().detach() 
            for name, param in self.model.named_parameters()
        }
        
        self.logger.info("Original weights loaded and stored in memory")
    
    def get_current_weights(self) -> Dict[str, torch.Tensor]:
        """Get current model weights."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_base_model() first.")
        
        return {
            name: param.clone().detach() 
            for name, param in self.model.named_parameters()
        }
    
    def reset_to_original_weights(self) -> None:
        """Reset model to original weights."""
        if self.original_weights is None:
            raise ValueError("Original weights not available. Call save_original_weights() first.")
        
        self.logger.info("Resetting model to original weights")
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.original_weights:
                    param.data = self.original_weights[name].clone()
        
        self.logger.info("Model reset to original weights")
    
    def generate_text(
        self, 
        prompt: str, 
        max_length: int = None,
        temperature: float = None,
        do_sample: bool = None
    ) -> str:
        """Generate text using the current model."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer not loaded.")
        
        # Use config defaults if not specified
        max_length = max_length or self.config.max_length
        temperature = temperature or self.config.temperature
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from generated text
        generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the current model."""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.config.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
            "architecture": type(self.model).__name__
        }
    
    def prepare_for_training(self) -> None:
        """Prepare model for fine-tuning."""
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        self.logger.info("Preparing model for training")
        
        # Set model to training mode
        self.model.train()
        
        # Enable gradient computation for all parameters
        for param in self.model.parameters():
            param.requires_grad = True
        
        self.logger.info("Model prepared for training")
    
    def prepare_for_inference(self) -> None:
        """Prepare model for inference."""
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        self.logger.info("Preparing model for inference")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.logger.info("Model prepared for inference")
