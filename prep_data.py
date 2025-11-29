import pandas as pd
import numpy as np
import re
import torch
import os
import multiprocessing
from tqdm import tqdm

# --- DEFINE PATTERNS (Raw strings only) ---
# We do NOT compile them here to avoid pickling issues.
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

# This global variable will hold the compiled regex IN THE WORKER
worker_regex = None

def worker_init():
    """
    This runs once inside every new worker process.
    It compiles the regex locally to ensure no 'fork' locks occur.
    """
    global worker_regex
    # Combine and compile
    combined = '|'.join(PATTERNS_LIST)
    worker_regex = re.compile(combined)
    # Important: Prevent PyTorch/NumPy threads from fighting in the worker
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

def process_chunk(texts):
    """Process a list of texts using the worker-local compiled regex"""
    global worker_regex
    results = []
    for text in texts:
        text = str(text) # Ensure string
        # Pre-cleaning
        text = re.sub(r'[\.]{3,}', '.. ', text)
        # Heavy regex (using the pre-compiled object)
        text = worker_regex.sub('<*>', text)
        results.append(text)
    return results

def main():
    # --- CONFIG ---
    dataset_name = 'android' 
    input_file = r'/pfs/data6/home/hu/hu_hu/hu_abdeom01/logllm_work/datasets/{}/prepared/train.csv'.format(dataset_name)
    output_file = f"cached_{dataset_name}_processed.pt"
    
    print(f"Reading {input_file}...")
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
    # Create smaller chunks to keep the progress bar moving
    num_chunks = max(num_cores * 10, 1) 
    chunk_size = len(raw_contents) // num_chunks + 1
    chunks = [raw_contents[i:i + chunk_size] for i in range(0, len(raw_contents), chunk_size)]

    print(f"Processing in {len(chunks)} chunks...")

    # --- SPAWN CONTEXT (The Fix) ---
    # We use 'get_context("spawn")' to create clean processes.
    # We use 'initializer' to setup the regex inside the worker.
    ctx = multiprocessing.get_context('spawn')
    
    with ctx.Pool(processes=num_cores, initializer=worker_init) as pool:
        # map/imap
        processed_chunks = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks), unit="chunk"))

    print("Merging results...")
    processed_contents = []
    for chunk in processed_chunks:
        processed_contents.extend(chunk)

    print("Splitting sequences...")
    sequences = np.array([c.split(' ;-; ') for c in processed_contents], dtype=object)

    print(f"Saving to {output_file}...")
    torch.save({'sequences': sequences, 'labels': labels}, output_file)
    print("DONE! Cache created.")

if __name__ == '__main__':
    main()