import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class FallacyDataset(Dataset):
    """
    Custom Dataset for Logical Fallacy Detection.
    Loads processed CSV files and tokenizes the combined text.
    """
    def __init__(self, csv_path, tokenizer, max_length=512, label_map=None):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Ensure combined_text is string
        self.df['combined_text'] = self.df['combined_text'].fillna("").astype(str)
        
        if label_map is None:
            # Create a consistent label map (sorted alphabetically)
            unique_labels = sorted(self.df['fallacy'].unique())
            self.label_map = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label_map = label_map
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['combined_text']
        label_name = row['fallacy']
        label = self.label_map[label_name]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def get_dataloaders(train_path, dev_path, test_path, model_name, batch_size=16, max_length=512):
    """
    Helper function to create DataLoaders for train, dev, and test splits.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load train DF to establish a consistent label map across all splits
    train_df = pd.read_csv(train_path)
    unique_labels = sorted(train_df['fallacy'].unique())
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    train_dataset = FallacyDataset(train_path, tokenizer, max_length, label_map)
    dev_dataset = FallacyDataset(dev_path, tokenizer, max_length, label_map)
    test_dataset = FallacyDataset(test_path, tokenizer, max_length, label_map)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, dev_loader, test_loader, label_map

if __name__ == "__main__":
    # Quick sanity check
    MODEL_NAME = "microsoft/deberta-v3-base"
    TRAIN_CSV = "data/processed/train_processed.csv"
    DEV_CSV = "data/processed/dev_processed.csv"
    TEST_CSV = "data/processed/test_processed.csv"
    
    import os
    if os.path.exists(TRAIN_CSV):
        print("Creating dataloaders...")
        train_loader, dev_loader, test_loader, label_map = get_dataloaders(
            TRAIN_CSV, DEV_CSV, TEST_CSV, MODEL_NAME, batch_size=4
        )
        print(f"Label Map: {label_map}")
        
        # Get one batch
        batch = next(iter(train_loader))
        print(f"Batch keys: {batch.keys()}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Labels: {batch['labels']}")
    else:
        print(f"Data not found at {TRAIN_CSV}")
