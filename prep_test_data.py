import pandas as pd
import numpy as np
import re
import os
import multiprocessing
from tqdm import tqdm

# ==========================================
# CRITICAL: DO NOT IMPORT TORCH HERE
# Importing torch at the top causes the deadlock/freeze
# ==========================================

# --- PATTERNS ---
PATTERNS_LIST = [
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

combined_pattern = '|'.join(PATTERNS_LIST)
compiled_regex = re.compile(combined_pattern)

def process_chunk(texts):
    """
    Standard worker function. 
    Safe to fork because heavy C-libraries (Torch) are not loaded yet.
    """
    results = []
    for text in texts:
        text = str(text)
        text = re.sub(r'[\.]{3,}', '.. ', text)
        text = compiled_regex.sub('<*>', text)
        results.append(text)
    return results

def main():
    # --- CONFIGURATION ---
    dataset_name = 'bad_windows'   # <--- Change this to 'mac' / 'bgl' etc as needed
    split_name = 'test'        # <--- We are processing the TEST set
    
    # Paths
    base_dir = r'/pfs/data6/home/hu/hu_hu/hu_abdeom01/logllm_work'
    input_file = os.path.join(base_dir, f'datasets/{dataset_name}/prepared/{split_name}.csv')
    
    # Output Name: Matches the new CustomDataset logic (cached_dataset_split_processed.pt)
    output_file = f"cached_{dataset_name}_{split_name}_processed.pt"
    
    print(f"Reading {input_file}...")
    if not os.path.exists(input_file):
        print(f"ERROR: File not found: {input_file}")
        return

    df = pd.read_csv(input_file)
    raw_contents = df['Content'].values
    labels = df['Label'].values
    print(f"Total samples: {len(raw_contents)}")

    # --- CORE DETECTION ---
    try:
        num_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cores = multiprocessing.cpu_count()
    
    print(f"Using {num_cores} cores.")

    # --- CHUNKING ---
    num_chunks = max(num_cores * 4, 1)
    chunk_size = len(raw_contents) // num_chunks + 1
    chunks = [raw_contents[i:i + chunk_size] for i in range(0, len(raw_contents), chunk_size)]

    print(f"Starting processing on {len(chunks)} chunks...")

    # --- PARALLEL EXECUTION ---
    with multiprocessing.Pool(processes=num_cores) as pool:
        processed_chunks = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks), unit="chunk"))

    print("Merging results...")
    processed_contents = []
    for chunk in processed_chunks:
        processed_contents.extend(chunk)

    print("Splitting sequences...")
    sequences = np.array([c.split(' ;-; ') for c in processed_contents], dtype=object)

    print(f"Saving to {output_file}...")
    
    # --- LAZY IMPORT ---
    # Only import torch NOW, when the parallel work is over.
    import torch 
    torch.save({'sequences': sequences, 'labels': labels}, output_file)
    print("DONE! Cache created successfully.")

if __name__ == '__main__':
    main()