import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from opacus import PrivacyEngine
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from tqdm import tqdm
from config import Config
import os
import json
import time

class FederatedModel:
    def __init__(self, client_id=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            Config.BASE_MODEL, 
            num_labels=2,
            return_dict=True
        ).to(self.device)
        
        # Apply LoRA fine-tuning
        self.apply_lora()
        
        self.client_id = client_id
        self.epsilon = None
        self.privacy_engine = None
        self.training_time = 0
        
    def apply_lora(self):
        """Apply LoRA fine-tuning adapters"""
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            lora_dropout=Config.LORA_DROPOUT,
            target_modules=["query", "value"] if "distilbert" in Config.BASE_MODEL 
                          else ["query_proj", "value_proj"]
        )
        self.model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters percentage
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"LoRA enabled - Trainable params: {trainable_params}/{total_params} "
              f"({100*trainable_params/total_params:.2f}%)")

    def setup_dp(self, data_loader):
        """Setup differential privacy"""
        if Config.DP_ENABLED:
            if not hasattr(self, 'optimizer'):
                raise RuntimeError("Optimizer must be created before DP setup")
                
            self.privacy_engine = PrivacyEngine()
            
            # Make private with target epsilon
            self.model, self.optimizer, data_loader = self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=data_loader,
                epochs=Config.LOCAL_EPOCHS,
                target_epsilon=Config.EPSILON,
                target_delta=Config.DELTA,
                max_grad_norm=Config.MAX_GRAD_NORM,
            )
            
            print(f"DP Enabled - Target ε={Config.EPSILON}, δ={Config.DELTA}, "
                  f"Max grad norm={Config.MAX_GRAD_NORM}")
            return data_loader
        return data_loader

    def local_train(self, dataset):
        """Client local training"""
        start_time = time.time()
        
        # Prepare data loader
        dataset = TensorDataset(
            dataset['input_ids'], 
            dataset['attention_mask'], 
            dataset['labels']
        )
        
        train_loader = DataLoader(
            dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True,
            drop_last=True  # For DP compatibility
        )
        
        # Optimizer and learning rate scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=2e-5,
            weight_decay=0.01
        )
        
        # Setup differential privacy
        train_loader = self.setup_dp(train_loader)
        
        num_training_steps = len(train_loader) * Config.LOCAL_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        # Training loop
        self.model.train()
        for epoch in range(Config.LOCAL_EPOCHS):
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Client {self.client_id} Epoch {epoch+1}")
            for batch_idx, batch in enumerate(progress_bar):
                inputs = {
                    'input_ids': batch[0].to(self.device),
                    'attention_mask': batch[1].to(self.device),
                    'labels': batch[2].to(self.device)
                }
                
                self.optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = outputs.loss
                loss.backward()
                
                self.optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
                
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Client {self.client_id} - Epoch {epoch+1} Avg Loss: {avg_epoch_loss:.4f}")
        
        self.training_time = time.time() - start_time
        
        # Record privacy expenditure
        if self.privacy_engine:
            self.epsilon = self.privacy_engine.get_epsilon(Config.DELTA)
            print(f"Client {self.client_id} Privacy Cost: ε={self.epsilon:.4f}, δ={Config.DELTA}")
        
        # Return only LoRA parameter updates
        lora_params = {}
        for name, param in self.model.named_parameters():
            if 'lora' in name and param.requires_grad:
                lora_params[name] = param.data.cpu().clone()
        
        # Log training statistics
        self.log_training_stats(len(train_loader.dataset), len(train_loader))
        
        return lora_params

    def log_training_stats(self, dataset_size, num_batches):
        """Log training statistics"""
        stats = {
            "client_id": self.client_id,
            "training_time": self.training_time,
            "samples_processed": dataset_size,
            "batches_processed": num_batches * Config.LOCAL_EPOCHS,
            "local_epochs": Config.LOCAL_EPOCHS,
        }
        
        if self.privacy_engine:
            stats.update({
                "epsilon": self.epsilon,
                "delta": Config.DELTA
            })
        
        log_path = os.path.join(Config.LOG_DIR, f"client_{self.client_id}_stats.json")
        with open(log_path, 'w') as f:
            json.dump(stats, f, indent=2)

    def evaluate(self, dataset):
        """Evaluate model performance"""
        # Prepare data loader
        dataset = TensorDataset(
            dataset['input_ids'], 
            dataset['attention_mask'], 
            dataset['labels']
        )
        
        test_loader = DataLoader(
            dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False
        )

        self.model.eval()
        total_loss, total_correct = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                inputs = {
                    'input_ids': batch[0].to(self.device),
                    'attention_mask': batch[1].to(self.device),
                    'labels': batch[2].to(self.device)
                }
                
                outputs = self.model(**inputs)
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                total_correct += torch.sum(predictions == inputs['labels']).item()
                
        avg_loss = total_loss / len(test_loader)
        accuracy = total_correct / len(dataset)
        
        return avg_loss, accuracy

    def save_model(self, round_idx=None):
        """Save model checkpoint"""
        suffix = f"_{self.client_id}" if self.client_id is not None else ""
        if round_idx is not None:
            suffix += f"_round{round_idx}"
        
        model_path = os.path.join(Config.SAVE_DIR, f"model{suffix}.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"Saved model to {model_path}")