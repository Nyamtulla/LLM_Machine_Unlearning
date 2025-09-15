"""Evaluation pipeline for testing unlearnt model performance."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from pathlib import Path
import time

class ModelEvaluator:
    """Evaluates model performance across different metrics."""
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def calculate_perplexity(
        self, 
        model: nn.Module, 
        texts: List[str], 
        max_length: int = 512
    ) -> float:
        """Calculate perplexity of the model on given texts."""
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        self.logger.info(f"Calculating perplexity on {len(texts)} texts")
        
        with torch.no_grad():
            for i, text in enumerate(texts):
                if i % 100 == 0:
                    self.logger.info(f"Processing text {i}/{len(texts)}")
                
                # Tokenize text
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=max_length,
                    padding=True
                ).to(self.device)
                
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                
                if input_ids.size(1) < 2:  # Skip very short texts
                    continue
                
                # Calculate loss
                try:
                    outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                    loss = outputs.loss
                    
                    # Count non-padding tokens
                    valid_tokens = attention_mask.sum().item()
                    total_loss += loss.item() * valid_tokens
                    total_tokens += valid_tokens
                    
                except Exception as e:
                    self.logger.warning(f"Error processing text {i}: {str(e)}")
                    continue
        
        if total_tokens == 0:
            self.logger.warning("No valid tokens found for perplexity calculation")
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        self.logger.info(f"Perplexity calculated: {perplexity:.4f}")
        return perplexity
    
    def evaluate_generation_quality(
        self, 
        model: nn.Module, 
        prompts: List[str],
        max_length: int = 100,
        temperature: float = 0.7,
        num_samples: int = 5
    ) -> Dict[str, Any]:
        """Evaluate the quality of text generation."""
        model.eval()
        results = {
            'generated_texts': [],
            'generation_times': [],
            'average_length': 0,
            'diversity_score': 0
        }
        
        self.logger.info(f"Evaluating generation quality on {len(prompts)} prompts")
        
        all_generated_texts = []
        
        with torch.no_grad():
            for prompt in prompts:
                start_time = time.time()
                
                # Tokenize prompt
                inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                
                # Generate text
                try:
                    outputs = model.generate(
                        inputs,
                        max_length=inputs.size(1) + max_length,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=num_samples
                    )
                    
                    generation_time = time.time() - start_time
                    
                    # Decode generated texts
                    generated_texts = []
                    for output in outputs:
                        generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                        # Remove the original prompt
                        generated_text = generated_text[len(prompt):].strip()
                        generated_texts.append(generated_text)
                    
                    results['generated_texts'].append({
                        'prompt': prompt,
                        'generated': generated_texts,
                        'generation_time': generation_time
                    })
                    
                    all_generated_texts.extend(generated_texts)
                    results['generation_times'].append(generation_time)
                    
                except Exception as e:
                    self.logger.warning(f"Error generating text for prompt '{prompt}': {str(e)}")
                    continue
        
        # Calculate metrics
        if all_generated_texts:
            lengths = [len(text.split()) for text in all_generated_texts]
            results['average_length'] = np.mean(lengths)
            
            # Simple diversity score based on unique words
            all_words = []
            for text in all_generated_texts:
                all_words.extend(text.lower().split())
            
            unique_words = set(all_words)
            total_words = len(all_words)
            results['diversity_score'] = len(unique_words) / total_words if total_words > 0 else 0
        
        if results['generation_times']:
            results['average_generation_time'] = np.mean(results['generation_times'])
        
        return results
    
    def evaluate_forget_effectiveness(
        self,
        original_model: nn.Module,
        fine_tuned_model: nn.Module,
        unlearnt_model: nn.Module,
        forget_texts: List[str],
        retain_texts: List[str]
    ) -> Dict[str, Any]:
        """Evaluate how effectively the model has forgotten the forget data."""
        
        self.logger.info("Evaluating forget effectiveness")
        
        results = {
            'forget_perplexity': {},
            'retain_perplexity': {},
            'forget_effectiveness': {}
        }
        
        # Calculate perplexity on forget data
        results['forget_perplexity'] = {
            'original': self.calculate_perplexity(original_model, forget_texts[:100]),
            'fine_tuned': self.calculate_perplexity(fine_tuned_model, forget_texts[:100]),
            'unlearnt': self.calculate_perplexity(unlearnt_model, forget_texts[:100])
        }
        
        # Calculate perplexity on retain data
        results['retain_perplexity'] = {
            'original': self.calculate_perplexity(original_model, retain_texts[:100]),
            'fine_tuned': self.calculate_perplexity(fine_tuned_model, retain_texts[:100]),
            'unlearnt': self.calculate_perplexity(unlearnt_model, retain_texts[:100])
        }
        
        # Calculate forget effectiveness
        forget_orig = results['forget_perplexity']['original']
        forget_ft = results['forget_perplexity']['fine_tuned']
        forget_unlearnt = results['forget_perplexity']['unlearnt']
        
        retain_orig = results['retain_perplexity']['original']
        retain_ft = results['retain_perplexity']['fine_tuned']
        retain_unlearnt = results['retain_perplexity']['unlearnt']
        
        # Effectiveness: how much closer is unlearnt model to original on forget data?
        if abs(forget_ft - forget_orig) > 0:
            forget_effectiveness = abs(forget_unlearnt - forget_orig) / abs(forget_ft - forget_orig)
        else:
            forget_effectiveness = 1.0
        
        # Retain performance: how much did we hurt performance on retain data?
        if retain_orig > 0:
            retain_performance_loss = abs(retain_unlearnt - retain_orig) / retain_orig
        else:
            retain_performance_loss = 0.0
        
        results['forget_effectiveness'] = {
            'forget_effectiveness_score': forget_effectiveness,
            'retain_performance_loss': retain_performance_loss,
            'overall_score': (1 - forget_effectiveness) + retain_performance_loss  # Lower is better
        }
        
        self.logger.info(f"Forget effectiveness: {forget_effectiveness:.4f}")
        self.logger.info(f"Retain performance loss: {retain_performance_loss:.4f}")
        
        return results
    
    def comprehensive_evaluation(
        self,
        original_model: nn.Module,
        fine_tuned_model: nn.Module,
        unlearnt_model: nn.Module,
        test_data: Dict[str, List[str]],
        generation_prompts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive evaluation of all models."""
        
        self.logger.info("Starting comprehensive evaluation")
        
        evaluation_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'models_evaluated': ['original', 'fine_tuned', 'unlearnt'],
            'perplexity_evaluation': {},
            'forget_effectiveness': {},
            'generation_evaluation': {}
        }
        
        # Perplexity evaluation on different datasets
        for dataset_name, texts in test_data.items():
            self.logger.info(f"Evaluating perplexity on {dataset_name} dataset")
            
            evaluation_results['perplexity_evaluation'][dataset_name] = {
                'original': self.calculate_perplexity(original_model, texts[:100]),
                'fine_tuned': self.calculate_perplexity(fine_tuned_model, texts[:100]),
                'unlearnt': self.calculate_perplexity(unlearnt_model, texts[:100])
            }
        
        # Forget effectiveness if we have forget and retain data
        if 'forget' in test_data and 'retain' in test_data:
            evaluation_results['forget_effectiveness'] = self.evaluate_forget_effectiveness(
                original_model, fine_tuned_model, unlearnt_model,
                test_data['forget'], test_data['retain']
            )
        
        # Generation evaluation if prompts provided
        if generation_prompts:
            self.logger.info("Evaluating text generation")
            
            for model_name, model in [
                ('original', original_model),
                ('fine_tuned', fine_tuned_model),
                ('unlearnt', unlearnt_model)
            ]:
                evaluation_results['generation_evaluation'][model_name] = self.evaluate_generation_quality(
                    model, generation_prompts[:5]  # Limit for efficiency
                )
        
        # Calculate summary metrics
        evaluation_results['summary'] = self._calculate_summary_metrics(evaluation_results)
        
        self.logger.info("Comprehensive evaluation completed")
        return evaluation_results
    
    def _calculate_summary_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary metrics from evaluation results."""
        summary = {
            'unlearning_success': False,
            'performance_preservation': 0.0,
            'forget_quality': 0.0
        }
        
        # Analyze perplexity results
        if 'perplexity_evaluation' in results:
            for dataset_name, perplexities in results['perplexity_evaluation'].items():
                orig = perplexities['original']
                ft = perplexities['fine_tuned']
                unlearnt = perplexities['unlearnt']
                
                # Calculate how close unlearnt is to original
                if orig > 0 and ft > 0:
                    similarity = 1 - abs(unlearnt - orig) / orig
                    summary['performance_preservation'] += similarity
        
            # Average performance preservation
            if results['perplexity_evaluation']:
                summary['performance_preservation'] /= len(results['perplexity_evaluation'])
        
        # Analyze forget effectiveness
        if 'forget_effectiveness' in results and results['forget_effectiveness']:
            effectiveness = results['forget_effectiveness']['forget_effectiveness']['forget_effectiveness_score']
            summary['forget_quality'] = 1 - effectiveness  # Convert to quality score (higher is better)
            
            # Overall unlearning success
            summary['unlearning_success'] = (
                summary['performance_preservation'] > 0.8 and 
                summary['forget_quality'] > 0.5
            )
        else:
            # No forget effectiveness data available
            summary['forget_quality'] = 0.0
            summary['unlearning_success'] = False
        
        return summary
    
    def save_evaluation_results(self, results: Dict[str, Any], filepath: str) -> None:
        """Save evaluation results to file."""
        # Convert any remaining tensors to lists
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj
        
        json_results = convert_tensors(results)
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Evaluation results saved to {filepath}")
    
    def load_evaluation_results(self, filepath: str) -> Dict[str, Any]:
        """Load evaluation results from file."""
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        self.logger.info(f"Evaluation results loaded from {filepath}")
        return results
