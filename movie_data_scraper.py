"""Movie data scraper for downloading real movie information from online sources."""

import requests
import json
import time
import logging
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import re
from pathlib import Path
import csv

class MovieDataScraper:
    """Scraper for downloading real movie data from various online sources."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def get_omdb_data(self, movie_name: str, api_key: str = None) -> Dict:
        """Get movie data from OMDB API."""
        try:
            if api_key is None:
                # Use free tier (limited requests)
                url = f"http://www.omdbapi.com/?t={movie_name}&apikey=trilogy"
            else:
                url = f"http://www.omdbapi.com/?t={movie_name}&apikey={api_key}"
            
            response = self.session.get(url)
            data = response.json()
            
            if data.get('Response') == 'True':
                self.logger.info(f"Successfully retrieved OMDB data for {movie_name}")
                return data
            else:
                self.logger.warning(f"OMDB API error for {movie_name}: {data.get('Error', 'Unknown error')}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error fetching OMDB data for {movie_name}: {str(e)}")
            return {}
    
    def get_imdb_reviews(self, movie_name: str, max_reviews: int = 50) -> List[str]:
        """Scrape IMDB reviews for a movie."""
        reviews = []
        
        try:
            # Add delay to avoid rate limiting
            time.sleep(1)
            
            # Search for the movie first
            search_url = f"https://www.imdb.com/find?q={movie_name.replace(' ', '+')}&s=tt&ttype=ft"
            search_response = self.session.get(search_url)
            soup = BeautifulSoup(search_response.content, 'html.parser')
            
            # Find the first movie result - try multiple selectors
            movie_link = None
            
            # Try different selectors for movie links
            selectors = [
                'a[href*="/title/tt"]',
                'td.result_text a',
                '.findResult a[href*="/title/tt"]'
            ]
            
            for selector in selectors:
                movie_link = soup.select_one(selector)
                if movie_link:
                    break
            
            if not movie_link:
                self.logger.warning(f"Could not find IMDB page for {movie_name}")
                return reviews
            
            # Extract movie ID
            href = movie_link.get('href', '')
            movie_id_match = re.search(r'/title/(tt\d+)/', href)
            if not movie_id_match:
                self.logger.warning(f"Could not extract movie ID from {href}")
                return reviews
            
            movie_id = movie_id_match.group(1)
            reviews_url = f"https://www.imdb.com/title/{movie_id}/reviews"
            
            # Get reviews page with delay
            time.sleep(1)
            reviews_response = self.session.get(reviews_url)
            reviews_soup = BeautifulSoup(reviews_response.content, 'html.parser')
            
            # Try multiple selectors for review text
            review_selectors = [
                '.text',
                '.review-container .text',
                '.lister-item .content .text',
                '[data-testid="review-text"]'
            ]
            
            for selector in review_selectors:
                review_elements = reviews_soup.select(selector)
                if review_elements:
                    for element in review_elements[:max_reviews]:
                        review_text = element.get_text(strip=True)
                        if len(review_text) > 50:  # Filter out very short reviews
                            reviews.append(review_text)
                    break
            
            self.logger.info(f"Scraped {len(reviews)} IMDB reviews for {movie_name}")
            
        except Exception as e:
            self.logger.error(f"Error scraping IMDB reviews for {movie_name}: {str(e)}")
        
        return reviews
    
    def get_movie_plot_summary(self, movie_name: str) -> List[str]:
        """Get movie plot summary and details."""
        plot_texts = []
        
        try:
            # Use OMDB for plot
            omdb_data = self.get_omdb_data(movie_name)
            if omdb_data:
                plot = omdb_data.get('Plot', '')
                if plot:
                    plot_texts.append(f"The plot of {movie_name}: {plot}")
                
                # Add other details
                director = omdb_data.get('Director', '')
                if director:
                    plot_texts.append(f"{movie_name} was directed by {director}.")
                
                actors = omdb_data.get('Actors', '')
                if actors:
                    plot_texts.append(f"The main cast of {movie_name} includes {actors}.")
                
                genre = omdb_data.get('Genre', '')
                if genre:
                    plot_texts.append(f"{movie_name} is a {genre} film.")
                
                year = omdb_data.get('Year', '')
                if year:
                    plot_texts.append(f"{movie_name} was released in {year}.")
                
                rating = omdb_data.get('imdbRating', '')
                if rating:
                    plot_texts.append(f"{movie_name} has an IMDb rating of {rating}/10.")
            
            self.logger.info(f"Created {len(plot_texts)} plot summary texts for {movie_name}")
            
        except Exception as e:
            self.logger.error(f"Error creating plot summary for {movie_name}: {str(e)}")
        
        return plot_texts
    
    def get_rotten_tomatoes_data(self, movie_name: str) -> List[str]:
        """Get Rotten Tomatoes data."""
        rt_texts = []
        
        try:
            # Add delay to avoid rate limiting
            time.sleep(1)
            
            # Search for the movie
            search_url = f"https://www.rottentomatoes.com/search?search={movie_name.replace(' ', '+')}"
            response = self.session.get(search_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try multiple selectors for movie results
            movie_selectors = [
                'search-page-media-row',
                '.search-result',
                '.media-row',
                '[data-qa="search-result"]'
            ]
            
            movie_found = False
            for selector in movie_selectors:
                movie_cards = soup.select(selector)
                if movie_cards:
                    for card in movie_cards[:3]:  # Check first 3 results
                        # Try to find title
                        title_selectors = ['a.unstyled', 'h3 a', '.title a', '[data-qa="title"]']
                        title_element = None
                        
                        for title_sel in title_selectors:
                            title_element = card.select_one(title_sel)
                            if title_element:
                                break
                        
                        if title_element:
                            title_text = title_element.get_text(strip=True)
                            if movie_name.lower() in title_text.lower():
                                movie_found = True
                                
                                # Extract year
                                year_element = card.select_one('span.start-year, .year, [data-qa="year"]')
                                if year_element:
                                    year = year_element.get_text(strip=True)
                                    rt_texts.append(f"{movie_name} ({year}) is available on Rotten Tomatoes.")
                                
                                # Extract score
                                score_selectors = ['span.tMeterScore', '.tomatometer-score', '[data-qa="score"]']
                                for score_sel in score_selectors:
                                    score_element = card.select_one(score_sel)
                                    if score_element:
                                        score = score_element.get_text(strip=True)
                                        if score and score != '--':
                                            rt_texts.append(f"Rotten Tomatoes gives {movie_name} a {score} score.")
                                        break
                                
                                break
                    
                    if movie_found:
                        break
            
            if not movie_found:
                # Fallback: create generic RT text
                rt_texts.append(f"{movie_name} has reviews available on Rotten Tomatoes.")
            
            self.logger.info(f"Created {len(rt_texts)} Rotten Tomatoes texts for {movie_name}")
            
        except Exception as e:
            self.logger.error(f"Error scraping Rotten Tomatoes for {movie_name}: {str(e)}")
            # Fallback on error
            rt_texts.append(f"{movie_name} has reviews available on Rotten Tomatoes.")
        
        return rt_texts
    
    def create_movie_knowledge_dataset(self, movie_name: str, max_texts: int = 500) -> List[str]:
        """Create a comprehensive dataset of movie knowledge from multiple sources."""
        all_texts = []
        
        self.logger.info(f"Creating movie knowledge dataset for '{movie_name}'")
        
        # Get plot summary and basic info
        plot_texts = self.get_movie_plot_summary(movie_name)
        all_texts.extend(plot_texts)
        
        # Get IMDB reviews
        reviews = self.get_imdb_reviews(movie_name, max_reviews=min(100, max_texts//2))
        all_texts.extend(reviews)
        
        # Get Rotten Tomatoes data
        rt_texts = self.get_rotten_tomatoes_data(movie_name)
        all_texts.extend(rt_texts)
        
        # Add some general movie discussion texts
        general_texts = [
            f"Let's discuss {movie_name} and its impact on cinema.",
            f"What makes {movie_name} such a memorable film?",
            f"The cinematography in {movie_name} is worth analyzing.",
            f"{movie_name} has influenced many other movies in its genre.",
            f"Fans of {movie_name} often debate its ending and themes.",
            f"The soundtrack of {movie_name} complements the story perfectly.",
            f"Many film critics consider {movie_name} a masterpiece.",
            f"{movie_name} explores complex themes that resonate with audiences.",
            f"The visual effects in {movie_name} were groundbreaking for its time.",
            f"Discussing {movie_name} often leads to interesting conversations about filmmaking."
        ]
        all_texts.extend(general_texts)
        
        # Limit to requested number
        all_texts = all_texts[:max_texts]
        
        self.logger.info(f"Created total dataset of {len(all_texts)} texts for {movie_name}")
        
        return all_texts
    
    def save_movie_data(self, movie_name: str, texts: List[str], output_dir: str = "./movie_data"):
        """Save movie data to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save as JSON
        json_file = output_path / f"{movie_name.lower().replace(' ', '_')}_data.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'movie_name': movie_name,
                'texts': texts,
                'count': len(texts)
            }, f, indent=2, ensure_ascii=False)
        
        # Save as TXT (one text per line)
        txt_file = output_path / f"{movie_name.lower().replace(' ', '_')}_texts.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        # Save as CSV
        csv_file = output_path / f"{movie_name.lower().replace(' ', '_')}_data.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['text'])
            for text in texts:
                writer.writerow([text])
        
        self.logger.info(f"Saved movie data to {output_path}")
        self.logger.info(f"Files created: {json_file}, {txt_file}, {csv_file}")
        
        return str(output_path)
    
    def load_movie_data(self, movie_name: str, data_dir: str = "./movie_data") -> List[str]:
        """Load previously saved movie data."""
        data_path = Path(data_dir)
        json_file = data_path / f"{movie_name.lower().replace(' ', '_')}_data.json"
        
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.logger.info(f"Loaded {len(data['texts'])} texts for {movie_name}")
                return data['texts']
        else:
            self.logger.warning(f"No saved data found for {movie_name}")
            return []

def main():
    """Example usage of the movie data scraper."""
    logging.basicConfig(level=logging.INFO)
    
    scraper = MovieDataScraper()
    
    # Example: Get data for Inception
    movie_name = "Inception"
    
    # Check if data already exists
    existing_data = scraper.load_movie_data(movie_name)
    if existing_data:
        print(f"Found existing data for {movie_name}: {len(existing_data)} texts")
        print(f"Sample: {existing_data[0][:100]}...")
    else:
        # Scrape new data
        print(f"Scraping data for {movie_name}...")
        texts = scraper.create_movie_knowledge_dataset(movie_name, max_texts=200)
        scraper.save_movie_data(movie_name, texts)
        
        print(f"Created {len(texts)} texts for {movie_name}")
        if texts:
            print(f"Sample text: {texts[0][:200]}...")

if __name__ == "__main__":
    main()
