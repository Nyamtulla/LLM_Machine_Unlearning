"""Configuration settings for machine unlearning project."""

import os
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    # Available models (uncomment one to use):
    model_name: str = "gpt2-medium"  # Good balance: 355M params, better knowledge
    # model_name: str = "gpt2-large"  # Better knowledge: 774M params, needs more memory
    # model_name: str = "gpt2-xl"  # Best knowledge: 1.5B params, needs GPU
    # model_name: str = "microsoft/DialoGPT-medium"  # Better conversational: 345M params
    # model_name: str = "microsoft/DialoGPT-large"  # Best conversational: 774M params
    # model_name: str = "EleutherAI/gpt-neo-125M"  # Modern alternative: 125M params
    # model_name: str = "EleutherAI/gpt-neo-1.3B"  # Modern & powerful: 1.3B params
    
    max_length: int = 512
    temperature: float = 0.7
    do_sample: bool = True
    
@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    learning_rate: float = 5e-5
    batch_size: int = 4
    num_epochs: int = 3
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
@dataclass
class UnlearningConfig:
    """Unlearning configuration parameters."""
    weight_diff_threshold: float = 0.01  # Threshold for weight difference
    layer_selection_strategy: str = "top_percent"  # "threshold", "top_k", "top_percent", "all"
    top_k_layers: int = 10  # Number of top layers to select if using top_k strategy
    top_percent: float = 5.0  # Percentage of top weights to select (e.g., 5.0 for top 5%)
    save_intermediate_models: bool = True
    
@dataclass
class DataConfig:
    """Data configuration parameters."""
    dataset_name: str = "synthetic"  # Dataset to use: "synthetic", "imdb", "wikitext"
    forget_ratio: float = 0.1  # Ratio of data to forget
    test_size: float = 0.2
    random_seed: int = 42
    
@dataclass
class ExperimentConfig:
    """Main experiment configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    unlearning: UnlearningConfig = field(default_factory=UnlearningConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Paths
    output_dir: str = "./outputs"
    model_save_dir: str = "./models"
    results_dir: str = "./results"
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "machine-unlearning"
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Create necessary directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
