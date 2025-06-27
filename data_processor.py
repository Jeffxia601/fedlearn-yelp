import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from config import Config
import os

class DataProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL)
        
    def load_and_clean_data(self):
        """Load and clean Yelp dataset"""
        print("Downloading and processing Yelp dataset...")
        
        # Load dataset from Hugging Face
        try:
            dataset = load_dataset('yelp_review_full')
            train_data = dataset['train']
            df = pd.DataFrame({
                'text': train_data['text'], 
                'stars': train_data['label'] + 1  # Convert labels to 1-5 stars
            })
            
            # Use small samples for CPU mode
            if Config.USE_SMALL_SAMPLE:
                df = df.sample(min(2000, len(df)), random_state=42)
                print(f"Using small sample dataset: {len(df)} reviews")
            else:
                print(f"Using full dataset: {len(df)} reviews")
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Using sample data instead")
            return self._create_sample_data()
        
        # Data cleaning
        df = self._clean_data(df)
        
        # Sentiment labeling: 1=positive(4-5 stars), 0=negative(1-2 stars), skip 3 stars
        df = df[df['stars'] != 3]
        df['sentiment'] = (df['stars'] >= 4).astype(int)
        
        print(f"Final dataset size after cleaning: {len(df)}")
        print(f"Class distribution: \n{df['sentiment'].value_counts(normalize=True)}")
        
        return df

    def _clean_data(self, df):
        """Data cleaning process"""
        print("Cleaning data...")
        # 1. Convert to lowercase
        df['text'] = df['text'].str.lower()
        
        # 2. Remove non-alphabet characters
        df['text'] = df['text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
        
        # 3. Remove too short reviews
        df = df[df['text'].str.split().apply(len) >= 3]
        
        # 4. Balance dataset
        pos = df[df['sentiment'] == 1]
        neg = df[df['sentiment'] == 0]
        
        # Ensure balanced classes
        min_samples = min(len(pos), len(neg))
        if min_samples > 0:
            df = pd.concat([
                pos.sample(min_samples, random_state=42),
                neg.sample(min_samples, random_state=42)
            ])
        
        return df

    def _create_sample_data(self):
        """Create sample data (fallback when dataset loading fails)"""
        print("Creating sample data...")
        reviews = [
            "Great food and excellent service!",
            "Worst experience ever.",
            "Average place, nothing special.",
            "Highly recommended for family dinners.",
            "Overpriced and low quality."
        ]
        ratings = [5, 1, 3, 4, 2]
        return pd.DataFrame({"text": reviews, "stars": ratings})

    def prepare_non_iid_data(self, df):
        """Prepare Non-IID client datasets"""
        print("Creating Non-IID client datasets...")
        num_clients = Config.NUM_CLIENTS
        
        # Simulate business-specific data partitioning
        unique_users = df.index.unique()
        if len(unique_users) < num_clients:
            # Handle insufficient users by sentiment splitting
            return self._split_data_by_sentiment(df, num_clients)
            
        # Create non-IID distribution by user groups
        client_datasets = []
        np.random.shuffle(unique_users)
        users_per_client = len(unique_users) // num_clients
        
        for i in range(num_clients):
            start_idx = i * users_per_client
            end_idx = (i + 1) * users_per_client
            user_group = unique_users[start_idx:end_idx]
            client_data = df[df.index.isin(user_group)]
            client_datasets.append(client_data)
        
        return client_datasets

    def _split_data_by_sentiment(self, df, num_clients):
        """Split data by sentiment for Non-IID simulation"""
        print("Splitting data by sentiment...")
        positive_df = df[df['sentiment'] == 1]
        negative_df = df[df['sentiment'] == 0]
        
        client_datasets = []
        for i in range(num_clients):
            # Create skewed distribution - simulating business-specific reviews
            if i % 2 == 0:
                sample = positive_df.sample(frac=0.8, random_state=i)  # 80% positive
                sample = sample.append(negative_df.sample(frac=0.2, random_state=i))  # 20% negative
            else:
                sample = positive_df.sample(frac=0.2, random_state=i)  # 20% positive
                sample = sample.append(negative_df.sample(frac=0.8, random_state=i))  # 80% negative
                
            client_datasets.append(sample)
        
        return client_datasets

    def tokenize_data(self, datasets):
        """Tokenize all client datasets"""
        print("Tokenizing datasets...")
        tokenized_datasets = []
        for i, df in enumerate(datasets):
            texts = df['text'].tolist()
            labels = df['sentiment'].tolist()
            
            encodings = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            )
            tokenized_datasets.append({
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'labels': torch.tensor(labels)
            })
            
            print(f"Client {i+1}: {len(texts)} samples, "
                  f"Positive ratio: {sum(labels)/len(labels):.2f}")
        
        return tokenized_datasets