# Efficient & Privacy-Preserving Federated Learning for Yelp Sentiment Analysis

​**Mar 2025 – Jun 2025**​ | `PyTorch` • `Hugging Face` • `FedML` • `Opacus` • `LoRA` | Python 3.11

### Scaling Federated Sentiment Analysis with Privacy Guarantees
This project tackles the challenges of training sentiment analysis models on distributed, privacy-sensitive Yelp reviews using ​**Federated Learning (FL)​**. It focuses on ​**efficient communication**, ​**strong client-level privacy**, and ​**robustness to real-world Non-IID data distributions**, enabling collaborative model training without sharing raw user data.

## Key Optimizations & Findings

| ​**Area**​                            | ​**Technique & Implementation**​             | ​**Achieved Outcome**​                                                                  |
| :---------------------------------- | :----------------------------------------- | :------------------------------------------------------------------------------------ |
| ​** Efficiency Optimization**​       | ​**LoRA-based Fine-tuning**​ of `DeBERTa-v3-base` | ​**​<1%​**​ Parameter Updates <br> ​**~40%↓**​ Communication Overhead                 |
| ​** Rigorous Privacy Protection**​  | Client-side ​**Local Differential Privacy (LDP)​**​ via `Opacus` <br> (Gradient Clipping + Noise Injection @ `ε=1.0, δ=1e⁻⁵`) | Strong Privacy Guarantees <br> Controlled Utility Impact: ​**~3%↓**​ Accuracy (Avg. 86% → 83%) |
| ​** System Implementation & Non-IID Analysis**​ | FedML-based ​**10-node FL System**​ <br> Integrated Pipeline: Data Cleaning → Sentiment Labeling → FedAvg <br> Extensive testing under `ε ∈ [0.1, 5.0]` | ​**Real-world Non-IID Handling**​ <br> ​**~110%↑**​ Convergence Rounds (vs. IID Baseline) <br> Identified ​**Accelerated Accuracy Drop**​ trend at `ε>2.0` |

## Technical Implementation Highlights

*   ​**Model**: Efficiently fine-tuned large language model (`DeBERTa-v3-base`) using ​**LoRA (Low-Rank Adaptation)​**.
*   ​**Federated Framework**: Built with ​**FedML**, simulating a ​**10-client network**.
*   ​**Data Pipeline**: Addressed real-world complexities: ​**data cleaning, sentiment labeling**, and simulating ​**merchant-level Non-IID distributions**.
*   ​**Privacy Engine**: Implemented strict client-side privacy via ​**Opacus**, enforcing ​**LDP guarantees**​ with configurable privacy budgets (`ε`).

## Getting Started (Quick Setup)

1.  ​**Install Dependencies:​**​

    ```bash
    pip install --user -r requirements.txt
    ```

2.  ​**Choose Execution Mode:​**​

    *   ​** CPU Mode (Small-Scale Testing & Debugging):​**​
        ```bash
        python federated_learning.py  # Runs with default small sample & model
        ```
    *   ​** GPU Mode (Full-Scale Training & Experiments):​**​
        1.  Edit `config.py` to switch modes:
            ```python
            # FOR GPU TRAINING (Full-scale):
            USE_SMALL_SAMPLE = False
            BASE_MODEL = "microsoft/deberta-v3-base"
            ```
        2.  Run the federated learning process:
            ```bash
            python federated_learning.py
            ```

3.  ​** Results & Visualization:​**​ Training performance plots are generated automatically during execution.
