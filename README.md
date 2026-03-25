# Machine Unlearning Project

This project implements a machine unlearning approach using **weight difference subtraction**. The method fine-tunes a pre-trained language model on "forget" data, analyzes which weights changed significantly, and then subtracts those weight differences to create an "unlearnt" model.

## Approach Overview

```
Base LLM → Fine-tune on Forget Data → Calculate Weight Differences → Subtract Changes → Unlearnt Model
```

### Key Steps:
1. **Load Base Model**: Load a pre-trained open-source LLM
2. **Prepare Data**: Split dataset into retain, forget, and test sets
3. **Save Original Weights**: Store the base model weights
4. **Fine-tune on Forget Data**: Train the model on data we want to forget
5. **Calculate Weight Differences**: Analyze which parameters changed significantly
6. **Apply Unlearning**: Subtract weight differences above threshold
7. **Evaluate Results**: Test the unlearnt model performance

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Machine_unlearning_projects

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run the complete demo workflow:

```bash
python demo.py --model microsoft/DialoGPT-small --dataset imdb --forget_ratio 0.1 --epochs 2
```

### Command Line Arguments:
- `--model`: Pre-trained model to use (default: microsoft/DialoGPT-small)
- `--dataset`: Dataset for experiment (default: imdb)
- `--forget_ratio`: Ratio of data to forget (default: 0.1)
- `--threshold`: Weight difference threshold for unlearning (default: 0.01)
- `--epochs`: Number of fine-tuning epochs (default: 2)
- `--output_dir`: Output directory for results (default: ./demo_outputs)

## Project Structure

```
Machine_unlearning_projects/
├── config.py              # Configuration settings
├── model_manager.py       # Base model loading and weight management
├── data_handler.py        # Data loading and preprocessing
├── trainer.py             # Fine-tuning on forget data
├── unlearner.py           # Core unlearning implementation
├── evaluator.py           # Model evaluation pipeline
├── utils.py               # Utility functions
├── demo.py                # Complete workflow demo
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Key Components

### 1. ModelManager (`model_manager.py`)
- Loads pre-trained models
- Manages weight saving/loading
- Handles text generation
- Provides model information

### 2. DataHandler (`data_handler.py`)
- Loads and preprocesses datasets
- Splits data into retain/forget/test sets
- Creates PyTorch DataLoaders
- Provides dataset analysis

### 3. FineTuningTrainer (`trainer.py`)
- Fine-tunes models on forget data
- Tracks weight changes during training
- Saves training history and results

### 4. MachineUnlearner (`unlearner.py`)
- Calculates weight differences
- Selects weights for unlearning based on threshold
- Applies weight subtraction
- Generates unlearning reports

### 5. ModelEvaluator (`evaluator.py`)
- Evaluates model perplexity
- Tests generation quality
- Measures forget effectiveness
- Compares model performance

## Configuration

Edit `config.py` to customize:

```python
@dataclass
class ModelConfig:
    model_name: str = "microsoft/DialoGPT-small"
    max_length: int = 512
    temperature: float = 0.7

@dataclass
class UnlearningConfig:
    weight_diff_threshold: float = 0.01  # Threshold for weight selection
    layer_selection_strategy: str = "threshold"  # Selection strategy
    top_k_layers: int = 10  # For top-k strategy

@dataclass
class TrainingConfig:
    learning_rate: float = 5e-5
    batch_size: int = 4
    num_epochs: int = 3
```

## Usage Examples

### Basic Usage
```python
from config import ExperimentConfig
from model_manager import ModelManager
from unlearner import MachineUnlearner

# Initialize components
config = ExperimentConfig()
model_manager = ModelManager(config.model)
unlearner = MachineUnlearner(config)

# Load model and save original weights
base_model, tokenizer = model_manager.load_base_model()
model_manager.save_original_weights("original.pt")

# Fine-tune on forget data (see trainer.py for details)
# ... fine-tuning code ...

# Apply unlearning
unlearner.set_weights(original_weights, fine_tuned_weights)
unlearnt_model = unlearner.create_unlearnt_model(base_model)
```

### Custom Dataset
```python
from data_handler import DataHandler

# Load custom dataset
data_handler = DataHandler(config.data)
retain_texts, forget_texts, test_texts = data_handler.load_dataset()

# Create synthetic forget data
forget_keywords = ["movie", "film", "cinema"]
synthetic_forget = data_handler.create_synthetic_forget_data(
    base_texts, forget_keywords, n_samples=100
)
```

## Output Files

The demo generates several output files:

- `original_weights.pt`: Original base model weights
- `unlearnt_model.pt`: Final unlearnt model
- `training_results.json`: Fine-tuning results and weight differences
- `unlearning_report.json`: Detailed unlearning analysis
- `evaluation_results.json`: Model performance comparison
- `experiment_summary.json`: Complete experiment summary

## Evaluation Metrics

The system evaluates unlearning effectiveness using:

1. **Perplexity**: How well models predict test data
2. **Forget Effectiveness**: How close unlearnt model is to original on forget data
3. **Retain Performance**: How much performance is preserved on retain data
4. **Generation Quality**: Text generation diversity and coherence

## Supported Models and Datasets

### Models:
- microsoft/DialoGPT-small
- gpt2
- distilgpt2
- Any Hugging Face causal language model

### Datasets:
- IMDB movie reviews
- WikiText
- Custom datasets via DataHandler

## Advanced Usage

### Custom Weight Selection Strategies

```python
# Threshold-based selection
config.unlearning.layer_selection_strategy = "threshold"
config.unlearning.weight_diff_threshold = 0.01

# Top-k layers selection
config.unlearning.layer_selection_strategy = "top_k"
config.unlearning.top_k_layers = 10

# Select all changed weights
config.unlearning.layer_selection_strategy = "all"
```

### Visualization

```python
from utils import visualize_weight_changes

# Analyze and visualize weight changes
weight_analysis = unlearner.analyze_weight_changes()
unlearner.visualize_analysis("weight_changes.png")
```

## Limitations and Future Work

### Current Limitations:
- Only supports causal language models
- Simple threshold-based weight selection
- Limited evaluation metrics
- No privacy guarantees

### Future Improvements:
- Support for encoder-decoder models
- More sophisticated weight selection strategies
- Privacy-preserving unlearning methods
- Comprehensive evaluation benchmarks

