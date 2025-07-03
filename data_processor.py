import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from config import Config

class DataProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL)
        
    def load_and_clean_data(self):
        """Load and clean Yelp dataset"""
        print("Downloading and processing Yelp dataset...")

        # Load dataset from Hugging Face
        dataset = load_dataset('yelp_review_full')
        print("Dataset keys:", dataset.keys())
        train_data = dataset['train']
        print("Train data type:", type(train_data))
        print("Train data structure:", train_data)

        # Convert to DataFrame
        df = pd.DataFrame({
            'text': train_data['text'],
            'stars': np.array(train_data['label']) + 1  # Convert labels to 1-5 stars
        })

        # Use small samples for CPU mode
        if Config.USE_SMALL_SAMPLE:
            df = df.sample(min(2000, len(df)), random_state=Config.RANDOM_SEED)
            print(f"Using small sample dataset: {len(df)} reviews")
        else:
            print(f"Using full dataset: {len(df)} reviews")

        # Sentiment labeling: 1=positive(4-5 stars), 0=negative(1-2 stars), skip 3 stars
        df = df[df['stars'] != 3]
        df['sentiment'] = (df['stars'] >= 4).astype(int)

        # Data cleaning
        df = self._clean_data(df)
        self.validate_data(df)
        
        print(f"Final dataset size after cleaning: {len(df)}")
        print(f"Class distribution: \n{df['sentiment'].value_counts(normalize=True)}")
        
        return df

    def _clean_data(self, df):
        """Data cleaning process"""
        assert isinstance(df, pd.DataFrame), "Input is not a DataFrame at _clean_data()"
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
            ], ignore_index=True)
        
        return df
    
    def validate_data(self, df):
        """Check dataset integrity"""
        required_columns = {'text', 'stars', 'sentiment'}
        assert required_columns.issubset(df.columns), f"Missing columns: {required_columns - set(df.columns)}"
        assert isinstance(df, pd.DataFrame), "Data is not a pandas DataFrame"
        assert df['sentiment'].isin([0, 1]).all(), "Sentiment values must be binary (0/1)"
        assert df['sentiment'].nunique() == 2, "Sentiment column should have exactly 2 classes"

    def prepare_non_iid_data(self, df):
        """Prepare Non-IID client datasets"""
        assert isinstance(df, pd.DataFrame), "Input is not a DataFrame at prepare_non_iid_data()"
        print("Creating Non-IID client datasets...")
        num_clients = Config.NUM_CLIENTS
        
        # If 'user_id' column present, group by user; else fallback
        if 'user_id' in df.columns:
            unique_users = df['user_id'].unique()
            user_groups = df.groupby('user_id')['stars'].mean().reset_index()
            user_groups = user_groups.sort_values('stars', ascending=False)
            unique_users = user_groups['user_id'].tolist()
        else:
            unique_users = df.index.unique()

        if len(unique_users) < num_clients:
            return self._split_data_by_sentiment(df, num_clients)

        unique_users = list(unique_users)
        np.random.shuffle(unique_users)

        client_datasets = []
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
            if i % 2 == 0:
                sample_pos = positive_df.sample(frac=0.8, random_state=i)
                sample_neg = negative_df.sample(frac=0.2, random_state=i)
            else:
                sample_pos = positive_df.sample(frac=0.2, random_state=i)
                sample_neg = negative_df.sample(frac=0.8, random_state=i)
            sample = pd.concat([sample_pos, sample_neg], ignore_index=True)
            client_datasets.append(sample)
        
        return client_datasets

    def tokenize_data(self, datasets):
        """Tokenize all client datasets"""
        print("Tokenizing datasets...")
        tokenized_datasets = []
        for i, df in enumerate(datasets):
            np.random.seed(Config.RANDOM_SEED)
            texts = df['text'].tolist()
            labels = df['sentiment'].tolist()
            
            encodings = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            tokenized_datasets.append({
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'labels': torch.tensor(labels)
            })
            
            print(f"Client {i+1}: {len(texts)} samples, Positive ratio: {sum(labels)/len(labels):.2f}")

        return tokenized_datasets
