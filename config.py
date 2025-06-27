import os

class Config:

    # For CPU testing:
    USE_SMALL_SAMPLE = True
    BASE_MODEL = "distilbert-base-uncased"

    # # For GPU training:
    # USE_SMALL_SAMPLE = False
    # BASE_MODEL = "microsoft/deberta-v3-base"

    # Experiment configuration
    NUM_CLIENTS = 10
    NUM_ROUNDS = 20
    FRACTION = 0.4  # Fraction of clients participating each round
    DP_ENABLED = True
    # USE_SMALL_SAMPLE = True  # Use small samples for CPU mode
    
    # Privacy parameters
    EPSILON = 1.0
    DELTA = 1e-5
    MAX_GRAD_NORM = 1.0
    
    # Model selection
    # BASE_MODEL = "distilbert-base-uncased"  # Small model for CPU mode
    # BASE_MODEL = "microsoft/deberta-v3-base"  # For GPU mode
    
    # LoRA configuration
    LORA_R = 8
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    
    # Training parameters
    BATCH_SIZE = 16
    LOCAL_EPOCHS = 2
    LOG_INTERVAL = 50
    
    # Paths
    DATA_DIR = "./yelp_data"
    LOG_DIR = "./logs"
    SAVE_DIR = "./models"
    
    @staticmethod
    def setup_dirs():
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        os.makedirs(Config.SAVE_DIR, exist_ok=True)

# Initialize directories
Config.setup_dirs()