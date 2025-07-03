import torch
import numpy as np
import copy
from datetime import datetime
from config import Config
import os
import json
import matplotlib.pyplot as plt
from data_processor import DataProcessor
from federated_model import FederatedModel

# FedAvg implementation simulating FedML framework
class FedAvgServer:
    def __init__(self):
        self.global_model = FederatedModel()
        self.data_processor = DataProcessor()
        self.results = {
            "rounds": [],
            "accuracy": [],
            "loss": []
        }
        
    def run_federated_training(self):
        # Prepare data
        df = self.data_processor.load_and_clean_data()
        client_datasets = self.data_processor.prepare_non_iid_data(df)
        tokenized_datasets = self.data_processor.tokenize_data(client_datasets)
        
        # Train-test split
        train_datasets, test_datasets = [], []
        for i, dataset in enumerate(tokenized_datasets):
            n = len(dataset['input_ids'])
            idx = np.arange(n)
            np.random.shuffle(idx)
            split = int(0.8 * n)
            
            train_data = {k: v[idx[:split]] for k, v in dataset.items()}
            test_data = {k: v[idx[split:]] for k, v in dataset.items()}
            
            train_datasets.append(train_data)
            test_datasets.append(test_data)
            
            print(f"Client {i}: Train samples={len(train_data['input_ids'])}, "
                  f"Test samples={len(test_data['input_ids'])}")
        
        # Initial evaluation
        self.evaluate_global_model(test_datasets, -1)
        
        # Federated training loop
        for round_idx in range(Config.NUM_ROUNDS):
            print(f"\n{'='*100}")
            print(f"Federal Training Round {round_idx+1}/{Config.NUM_ROUNDS}")
            print(f"{'='*100}")
            
            # 1. Select clients
            selected_clients = self.select_clients(len(client_datasets), Config.FRACTION)
            print(f"Selected clients: {selected_clients}")
            
            # 2. Client local training
            client_updates = []
            total_samples = 0
            
            for client_id in selected_clients:
                print(f"\n{'*'*30}")
                print(f"Training Client {client_id}")
                print(f"{'*'*30}")
                
                # Create client model with global weights
                client_model = FederatedModel(client_id)
                client_model.model.load_state_dict(
                    self.global_model.model.state_dict()
                )
                
                # Local training
                update = client_model.local_train(train_datasets[client_id])
                client_updates.append((client_id, update))
                
                # Collect sample count for weighted averaging
                total_samples += len(train_datasets[client_id]['input_ids'])
                
                # Local evaluation
                loss, acc = client_model.evaluate(test_datasets[client_id])
                print(f"Client {client_id} Local Accuracy: {acc*100:.2f}%")
            
            # 3. Aggregate updates (FedAvg)
            self.aggregate_updates(client_updates, total_samples)
            
            # 4. Global evaluation
            round_acc, round_loss = self.evaluate_global_model(test_datasets, round_idx)
            self.results["rounds"].append(round_idx)
            self.results["accuracy"].append(round_acc)
            self.results["loss"].append(round_loss)
            
            # Save global model
            self.global_model.save_model(round_idx)
        
        # Save results and plot training progress
        self.save_results()
        self.plot_training_progress()
    
    def select_clients(self, total_clients, fraction):
        """Randomly select client subset"""
        np.random.seed(Config.RANDOM_SEED)
        num_selected = max(int(fraction * total_clients), 1)
        return np.random.choice(range(total_clients), num_selected, replace=False)
    
    def aggregate_updates(self, client_updates, total_samples):
        """FedAvg aggregation algorithm"""
        # Initialize average parameters
        avg_state = {}
        for name, param in self.global_model.model.named_parameters():
            if 'lora' in name:  # Only aggregate LoRA parameters
                avg_state[name] = torch.zeros_like(param.data)
        
        # Weighted average
        for client_id, update in client_updates:
            client_samples = len(update)  # Approximate size
            weight = client_samples / total_samples
            
            for name, param in update.items():
                if name in avg_state:
                    avg_state[name] += param * weight
        
        # Create new state dictionary
        new_state = copy.deepcopy(self.global_model.model.state_dict())
        for name, value in avg_state.items():
            if name in new_state:
                new_state[name] = value.to(new_state[name].device)
        
        # Update global model
        self.global_model.model.load_state_dict(new_state)
        
        print("Global model updated")
    
    def evaluate_global_model(self, test_datasets, round_idx):
        """Evaluate global model on all client test sets"""
        total_loss, total_correct, total_samples = 0, 0, 0
        client_accuracies = []
        
        for client_id, test_data in enumerate(test_datasets):
            loss, acc = self.global_model.evaluate(test_data)
            client_accuracies.append(acc)
            
            num_samples = len(test_data['input_ids'])
            total_loss += loss * num_samples
            total_correct += int(acc * num_samples)
            total_samples += num_samples
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        std_acc = np.std(client_accuracies) * 100
        
        print(f"\n{'='*60}")
        print(f"Global Model Round {round_idx} Evaluation Results:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Accuracy: {avg_acc*100:.2f}% ± {std_acc:.2f}")
        print(f"Min Client Acc: {min(client_accuracies)*100:.2f}%")
        print(f"Max Client Acc: {max(client_accuracies)*100:.2f}%")
        print(f"{'='*60}")
        
        return avg_acc, avg_loss

    def save_results(self):
        """Save training metrics and aggregated client statistics"""
        # Save global training metrics
        metrics_path = os.path.join(Config.LOG_DIR, "training_results.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Global metrics saved to {metrics_path}")
        
        # Aggregate and save client statistics
        client_stats = []
        for client_id in range(Config.NUM_CLIENTS):
            stats_path = os.path.join(Config.LOG_DIR, f"client_{client_id}_stats.json")
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    client_stats.append(json.load(f))
        
        if client_stats:
            aggregated_stats_path = os.path.join(Config.LOG_DIR, "all_client_stats.json")
            with open(aggregated_stats_path, 'w') as f:
                json.dump(client_stats, f, indent=2)
            print(f"Aggregated client stats saved to {aggregated_stats_path}")

    def plot_training_progress(self):
        """Plot training progress"""
        plt.figure(figsize=(12, 5))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(self.results["rounds"], self.results["accuracy"], 'o-', color='b')
        plt.title('Global Model Accuracy')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(self.results["rounds"], self.results["loss"], 'o-', color='r')
        plt.title('Global Model Loss')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plot_path = os.path.join(Config.LOG_DIR, "training_progress.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Saved training plot to {plot_path}")

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"\n{'#'*100}")
    print(f"Federated Learning with Differential Privacy")
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {Config.BASE_MODEL}, LoRA R={Config.LORA_R}")
    print(f"DP Enabled: {Config.DP_ENABLED}, ε={Config.EPSILON}, δ={Config.DELTA}")
    print(f"Clients: {Config.NUM_CLIENTS}, Fraction: {Config.FRACTION}")
    print(f"Local Epochs: {Config.LOCAL_EPOCHS}, Batch Size: {Config.BATCH_SIZE}")
    print(f"Use Small Sample: {Config.USE_SMALL_SAMPLE}")
    print(f"{'#'*100}\n")
    
    server = FedAvgServer()
    server.run_federated_training()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    print(f"\nTraining Completed in {duration:.2f} minutes")
    print(f"Results saved to {Config.LOG_DIR}")