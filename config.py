import os
from pathlib import Path
import torch

class Config:
    # Use CPU or GPU
    USE_GPU = True  
    DEVICE = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")

    # For GPU or CPU training:
    USE_SMALL_SAMPLE = (DEVICE.type == "cpu")
    BASE_MODEL = "microsoft/deberta-v3-base" if DEVICE.type == "cuda" else "distilbert-base-uncased"
    LOCAL_EPOCHS = 2 if DEVICE.type == "cuda" else 2
    BATCH_SIZE = 32 if DEVICE.type == "cuda" else 8
    NUM_ROUNDS = 2 if DEVICE.type == "cuda" else 2

    # Experiment configuration
    NUM_CLIENTS = 10
    FRACTION = 0.4  # Fraction of clients participating each round
    DP_ENABLED = True
    RANDOM_SEED = 2025
    
    # Privacy parameters
    EPSILON = 1.0
    DELTA = 1e-5
    MAX_GRAD_NORM = 1.0
    
    # LoRA configuration
    LORA_R = 8
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    
    # Training parameters
    LOG_INTERVAL = 100
    
    # Paths
    DATA_DIR = Path("./yelp_data")
    LOG_DIR  = Path("./logs")
    SAVE_DIR = Path("./models")
    
    @classmethod
    def setup_dirs(cls):
        for d in (cls.DATA_DIR, cls.LOG_DIR, cls.SAVE_DIR):
            d.mkdir(parents=True, exist_ok=True)

# Initialize directories
Config.setup_dirs()