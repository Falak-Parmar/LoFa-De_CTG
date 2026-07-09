import torch
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, device=None, class_weights=None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
            
        # Lower learning rate to 1e-5 for stability. Use Hybrid PythonAdamW on MPS to bypass PyTorch AdamW kernel bugs.
        if self.device == "mps":
            from src.optimizer import PythonAdamW
            self.optimizer = PythonAdamW(self.model.parameters(), lr=1e-5)
        else:
            self.optimizer = AdamW(self.model.parameters(), lr=1e-5)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            if self.class_weights is not None:
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=self.class_weights)
            else:
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            
            # Check for NaN immediately
            if torch.isnan(loss):
                print("\n[CRITICAL] NaN loss detected! Halting training immediately to prevent weight corruption.")
                return float('nan')
                
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def evaluate(self, loader=None):
        if loader is None:
            loader = self.val_loader
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.class_weights is not None:
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=self.class_weights)
                else:
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                total_loss += loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        avg_loss = total_loss / len(loader)
        f1 = f1_score(all_labels, all_preds, average='macro')
        report = classification_report(all_labels, all_preds, zero_division=0)
        return avg_loss, f1, report

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

def train_detection(model_name, train_path, dev_path, test_path, epochs=3, batch_size=16, device=None):
    from src.data_loader import get_dataloaders
    from src.model import FallacyDetectionModel
    import numpy as np
    from collections import Counter
    
    train_loader, dev_loader, test_loader, label_map = get_dataloaders(
        train_path, dev_path, test_path, model_name, batch_size=batch_size
    )
    
    # Calculate inverse-frequency class weights for cross-entropy loss balancing
    fallacies = train_loader.dataset.df['fallacy'].values
    train_labels = [label_map[f] for f in fallacies]
    label_counts = Counter(train_labels)
    
    num_classes = len(label_map)
    total_samples = len(train_labels)
    class_weights = np.zeros(num_classes)
    for label_idx in range(num_classes):
        count = label_counts.get(label_idx, 1)
        class_weights[label_idx] = total_samples / (num_classes * count)
        
    class_weights = class_weights / np.sum(class_weights) * num_classes
    print(f"[INFO] Calculated balanced class weights: {class_weights}")
    
    model = FallacyDetectionModel(model_name, num_labels=len(label_map))
    trainer = Trainer(model, train_loader, dev_loader, device=device, class_weights=class_weights)
    
    checkpoint_path = f"models/detection_{model_name.split('/')[-1]}.pt"
    
    for epoch in range(epochs):
        train_loss = trainer.train_epoch()
        
        # Defensive Checkpoint: Save weights immediately after training epoch finishes
        trainer.save_model(checkpoint_path)
        
        # Clear GPU/MPS memory cache before evaluation
        if trainer.device == "mps":
            torch.mps.empty_cache()
            
        val_loss, val_f1, val_report = trainer.evaluate()
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        print(val_report)
        
    print("Testing on Test Set:")
    if trainer.device == "mps":
        torch.mps.empty_cache()
    test_loss, test_f1, test_report = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f} | Test F1: {test_f1:.4f}")
    print(test_report)

def train_generation(model_name, train_path, dev_path, test_path, epochs=3, batch_size=8, device=None):
    from src.data_loader import get_generation_dataloaders
    from src.model import FallacyGenerationModel
    
    # Dynamically resolve model type (seq2seq for T5, causal for GPT-2)
    model_type = "seq2seq" if "t5" in model_name.lower() else "causal"
    
    train_loader, dev_loader, test_loader = get_generation_dataloaders(
        train_path, dev_path, test_path, model_name, batch_size=batch_size, model_type=model_type
    )
    
    model = FallacyGenerationModel(model_name, model_type=model_type)
    trainer = Trainer(model, train_loader, dev_loader, device=device)
    
    checkpoint_path = f"models/generation_{model_name.split('/')[-1]}.pt"
    
    for epoch in range(epochs):
        train_loss = trainer.train_epoch()
        
        # Defensive Checkpoint
        trainer.save_model(checkpoint_path)
        
        if trainer.device == "mps":
            torch.mps.empty_cache()
            
        val_loss = 0
        model.model.eval()
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Evaluating Generation"):
                input_ids = batch['input_ids'].to(trainer.device)
                attention_mask = batch['attention_mask'].to(trainer.device)
                labels = batch['labels'].to(trainer.device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(dev_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
