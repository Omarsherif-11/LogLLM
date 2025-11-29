import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import re
from torch.utils.data import Sampler
import torch
import os

# Disable tokenizers parallelism to prevent deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- PATTERNS (Only needed if cache is missing) ---
patterns = [
    r'True', r'true', r'False', r'false',
    r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b',
    r'\b(Mon|Monday|Tue|Tuesday|Wed|Wednesday|Thu|Thursday|Fri|Friday|Sat|Saturday|Sun|Sunday)\b',
    r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})\s+\b',
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?', 
    r'([0-9A-Fa-f]{2}:){11}[0-9A-Fa-f]{2}',   
    r'([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}',   
    r'[a-zA-Z0-9]*[:\.]*([/\\]+[^/\\\s\[\]]+)+[/\\]*', 
    r'\b[0-9a-fA-F]{8}\b',
    r'\b[0-9a-fA-F]{10}\b',
    r'(\w+[\w\.]*)@(\w+[\w\.]*)\-(\w+[\w\.]*)',
    r'(\w+[\w\.]*)@(\w+[\w\.]*)',
    r'[a-zA-Z\.\:\-\_]*\d[a-zA-Z0-9\.\:\-\_]*', 
]
combined_pattern = '|'.join(patterns)

def replace_patterns(text):
    text = re.sub(r'[\.]{3,}', '.. ', str(text))
    text = re.sub(combined_pattern, '<*>', text)
    return text

class CustomDataset(Dataset):
    def __init__(self, file_path, drop_duplicates=False):
        # ---------------------------------------------------------
        # INTELLIGENT CACHE NAME LOGIC
        # ---------------------------------------------------------
        # 1. Deduce Dataset Name (e.g., 'windows')
        try:
            parts = os.path.normpath(file_path).split(os.sep)
            if 'prepared' in parts:
                idx = parts.index('prepared')
                dataset_name = parts[idx - 1] # Get 'windows'
            else:
                dataset_name = "dataset"
        except:
            dataset_name = "dataset"

        # 2. Deduce Split Name (e.g., 'train' or 'test') <--- CRITICAL FIX ADDED HERE
        file_name = os.path.basename(file_path).replace('.csv', '')

        # 3. Create Unique Cache Name
        # Result: cached_windows_train_processed.pt OR cached_windows_test_processed.pt
        cache_filename = f"cached_{dataset_name}_{file_name}_processed.pt"
        
        # ---------------------------------------------------------
        # LOAD CACHE
        # ---------------------------------------------------------
        if os.path.exists(cache_filename):
            print(f"[CustomDataset] Found cache: {cache_filename}")
            print(f"[CustomDataset] Loading fast... (approx 5s)")
            data = torch.load(cache_filename)
            self.sequences = data['sequences']
            self.labels = data['labels']
            print(f'[CustomDataset] Loaded {len(self.labels)} samples.')
        else:
            print(f"[CustomDataset] CACHE MISS: {cache_filename}")
            print(f"[CustomDataset] WARNING: Falling back to slow CPU regex processing!")
            
            df = pd.read_csv(file_path)
            print('Original Normal:', (df['Label'].values==0).sum())
            print('Original Anomalous:', (df['Label'].values==1).sum())
            
            df['Content'] = df['Content'].apply(replace_patterns)
            if drop_duplicates:
                df = df.drop_duplicates(subset='Content', keep='first')
            
            contents = df['Content'].values
            self.sequences = np.array([content.split(' ;-; ') for content in contents], dtype=object)
            self.labels = df['Label'].values
            
            # Save it so we don't process again
            print(f"[CustomDataset] Saving new cache to {cache_filename}")
            torch.save({'sequences': self.sequences, 'labels': self.labels}, cache_filename)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

    def get_label(self):
        return self.labels

# --- HELPER FUNCTIONS ---
def merge_data(data):
    merged_data = []
    start_positions = []
    current_position = 0
    for sublist in data:
        start_positions.append(current_position)
        merged_data.extend(sublist)
        current_position += len(sublist)
    return merged_data, start_positions

class BalancedSampler(Sampler):
    def __init__(self, dataset, target_ratio=0.3, max_samples=None, min_samples=50000):
        self.labels = dataset.get_label()
        self.dataset = dataset
        self.target_ratio = target_ratio
        self.max_samples = max_samples
        self.min_samples = min_samples

        self.normal_indices = np.where(self.labels == 0)[0]
        self.anomalous_indices = np.where(self.labels == 1)[0]

        self.minority_indices = (
            self.anomalous_indices if len(self.anomalous_indices) < len(self.normal_indices)
            else self.normal_indices
        )
        self.majority_indices = (
            self.normal_indices if self.minority_indices is self.anomalous_indices
            else self.anomalous_indices
        )

        self.minority_count = max(int((self.target_ratio * len(self.majority_indices)) / (1 - self.target_ratio)), len(self.minority_indices))
        self.total_size = self.minority_count + len(self.majority_indices)

        if self.max_samples is not None:
            self.total_size = self.max_samples
        elif self.total_size < self.min_samples:
            self.total_size = self.min_samples

    def __iter__(self):
        oversampled_minority = np.tile(self.minority_indices, int(self.minority_count / len(self.minority_indices)))
        if len(self.minority_indices) > 0:
             remainder = self.minority_count - len(oversampled_minority)
             if remainder > 0:
                 oversampled_minority_ = np.random.choice(self.minority_indices, remainder, replace=False)
                 oversampled_minority = np.concatenate([oversampled_minority, oversampled_minority_])
        
        combined_indices = np.concatenate([self.majority_indices, oversampled_minority])
        
        if len(combined_indices) > self.total_size:
            combined_indices = np.random.choice(combined_indices, self.total_size, replace=False)
        else:
            if len(combined_indices) < self.total_size:
                 combined_indices = np.resize(combined_indices, self.total_size)
        
        np.random.shuffle(combined_indices)
        return iter(combined_indices)

    def __len__(self):
        return self.total_size

class CustomCollator:
    def __init__(self, tokenizer, max_seq_len=128, max_content_len=100):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_content_len = max_content_len

    def __call__(self, batch):
        sequences_, labels = zip(*batch)
        sequences = [seq[:self.max_seq_len] for seq in sequences_]
        data, seq_positions = merge_data(sequences)
        seq_positions = seq_positions[1:] 

        inputs = self.tokenizer(
            data,
            return_tensors="pt",
            max_length=self.max_content_len,
            padding=True,
            truncation=True
        )

        labels = np.array(labels).astype(object)
        labels[labels == 0] = 'normal'
        labels[labels == 1] = 'anomalous'

        return {
            "inputs": inputs,
            "seq_positions": torch.tensor(seq_positions, dtype=torch.long),
            "labels": labels
        }