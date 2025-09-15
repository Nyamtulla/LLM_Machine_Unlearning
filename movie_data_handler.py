"""Simplified data handler focused only on real movie data for unlearning."""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

from config import DataConfig
from movie_data_scraper import MovieDataScraper

class MovieDataset(Dataset):
    """Custom dataset for movie unlearning experiments."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # For causal language modeling, labels are the same as input_ids
        encoding['labels'] = encoding['input_ids'].clone()
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['labels'].squeeze()
        }

class MovieDataHandler:
    """Handles real movie data for unlearning experiments."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scraper = MovieDataScraper()
        
    def get_movie_data(self, movie_name: str, n_samples: int = 200) -> List[str]:
        """Get real movie data from online sources."""
        self.logger.info(f"Getting real movie data for '{movie_name}'")
        
        # Try to load existing data first
        existing_data = self.scraper.load_movie_data(movie_name)
        if existing_data and len(existing_data) >= n_samples:
            self.logger.info(f"Using existing real data: {len(existing_data)} texts")
            return existing_data[:n_samples]
        
        # Scrape new data if not enough exists
        self.logger.info(f"Scraping real movie data from online sources...")
        movie_texts = self.scraper.create_movie_knowledge_dataset(movie_name, max_texts=n_samples)
        
        if movie_texts:
            # Save the scraped data for future use
            self.scraper.save_movie_data(movie_name, movie_texts)
            self.logger.info(f"Created {len(movie_texts)} real movie texts")
            return movie_texts
        else:
            raise ValueError(f"Failed to scrape real data for {movie_name}")
    
    def create_dataloader(
        self, 
        texts: List[str], 
        tokenizer, 
        batch_size: int = 4,
        max_length: int = 512
    ) -> DataLoader:
        """Create a PyTorch DataLoader for movie texts."""
        
        # Create dataset
        dataset = MovieDataset(texts, tokenizer, max_length)
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # We're doing causal LM, not MLM
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=data_collator
        )
        
        self.logger.info(f"Created dataloader with {len(texts)} samples, batch size {batch_size}")
        return dataloader
    
    def analyze_movie_data(self, texts: List[str]) -> Dict[str, any]:
        """Analyze movie dataset statistics."""
        if not texts:
            return {}
        
        text_lengths = [len(text.split()) for text in texts]
        
        analysis = {
            "total_samples": len(texts),
            "avg_length": sum(text_lengths) / len(text_lengths),
            "min_length": min(text_lengths),
            "max_length": max(text_lengths),
            "total_words": sum(text_lengths)
        }
        
        return analysis
    
    def save_movie_analysis(self, movie_name: str, texts: List[str], filepath: str) -> None:
        """Save movie data analysis to file."""
        analysis = self.analyze_movie_data(texts)
        
        info = {
            "movie_name": movie_name,
            "data_source": "real_online_sources",
            "analysis": analysis,
            "sample_texts": texts[:5]  # Save first 5 texts as examples
        }
        
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Movie data analysis saved to {filepath}")
    
    def get_sample_texts(self, texts: List[str], n_samples: int = 5) -> List[str]:
        """Get a sample of texts for demonstration purposes."""
        import random
        if len(texts) <= n_samples:
            return texts
        return random.sample(texts, n_samples)

def main():
    """Example usage of the movie data handler."""
    logging.basicConfig(level=logging.INFO)
    
    # Test with a movie
    config = DataConfig()
    handler = MovieDataHandler(config)
    
    movie_name = "Inception"
    
    try:
        # Get movie data
        texts = handler.get_movie_data(movie_name, n_samples=50)
        
        # Analyze data
        analysis = handler.analyze_movie_data(texts)
        print(f"Movie: {movie_name}")
        print(f"Samples: {analysis['total_samples']}")
        print(f"Avg length: {analysis['avg_length']:.1f} words")
        print(f"Total words: {analysis['total_words']}")
        
        # Show samples
        samples = handler.get_sample_texts(texts, 3)
        print("\nSample texts:")
        for i, text in enumerate(samples):
            print(f"{i+1}. {text[:100]}...")
        
        print(f"\n✅ Successfully handled real movie data for {movie_name}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
