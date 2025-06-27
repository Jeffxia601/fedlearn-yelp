# Federated Learning for Privacy-Preserving Yelp Review Analysis

This project implements a privacy-preserving federated learning system for sentiment analysis of Yelp reviews. 

## Features
- 🛡️ ​**Differential Privacy**: Protects client data with (ε=1.0, δ=1e-5)
- ⚡ ​**Efficient Tuning**: Uses LoRA for <1% parameter updates
- 🔄 ​**Federated Learning**: FedAvg aggregation across clients
- ⚙️ ​**Dual Mode**: Supports CPU (small-scale) and GPU (full-scale) runs
- 📊 ​**Automatic Visualization**: Saves training performance plots

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```
### CPU Mode by Default (Small-scale testing)
```bash
python federated_learning.py
```
### GPU Mode (Full-scale training)
1. Edit config.py:
```python
    # # For CPU testing:
    # USE_SMALL_SAMPLE = True
    # BASE_MODEL = "distilbert-base-uncased"

    # For GPU training:
    USE_SMALL_SAMPLE = False
    BASE_MODEL = "microsoft/deberta-v3-base"
```
2. Run:
```bash
python federated_learning.py
```