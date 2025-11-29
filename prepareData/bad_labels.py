import os
import pandas as pd
import numpy as np
from pathlib import Path

# Assumes helper.py is in the same directory
from helper import fixedSize_window 

# --- CONFIGURATION ---
# mimicking the original config structure for the print statements
key = "windows" 
ds = {
    "data_dir": "/pfs/data6/home/hu/hu_hu/hu_abdeom01/logllm_work/datasets/bad_windows",
    "log_name": "Windows.log"
}
WINDOW_SIZE = 100
STEP_SIZE = 100
TRAIN_RATIO = 0.8

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def write_info(path, df_win):
    """
    Exact replica of the original info writer
    """
    total_win = len(df_win)
    
    if total_win > 0 and "Content" in df_win.columns:
        session_lengths = df_win["Content"].apply(len)
        mean_len = session_lengths.mean()
        max_len = session_lengths.max()
        
        win_anom = int(df_win["Label"].sum())
        win_norm = total_win - win_anom
    else:
        mean_len = 0
        max_len = 0
        win_anom = 0
        win_norm = 0

    with open(path, "w") as f:
        f.write(f"max session length: {max_len}; mean session length: {mean_len}\n")
        f.write(f"number of anomalous sessions: {win_anom}; number of normal sessions: {win_norm}; number of total sessions: {total_win}\n")

def process_dataset():
    data_dir = ds["data_dir"]
    log_name = ds["log_name"]
    
    # Mimic original startup prints
    print(f"\n=== Processing dataset: {key} ===")
    print(f"data_dir={data_dir}, log_name={log_name}")

    structured_dir = os.path.join(data_dir, "structured")
    prepared_dir = os.path.join(data_dir, "prepared")
    ensure_dir(prepared_dir)

    # ----------------------------------------------------
    # STEP 1 — SKIP PARSING, LOAD EXISTING CSV
    # ----------------------------------------------------
    structured_path = os.path.join(structured_dir, log_name + "_structured.csv")
    
    if not os.path.exists(structured_path):
        raise RuntimeError(f"STRUCTURED CSV NOT FOUND: {structured_path}")

    try:
        df = pd.read_csv(structured_path, encoding="latin-1")
    except pd.errors.EmptyDataError:
        raise RuntimeError(f"ERROR: The file {structured_path} is completely empty.")

    if "Content" not in df.columns:
        df["Content"] = df.iloc[:, -1].astype(str)

    # ----------------------------------------------------
    # STEP 2 — RANDOM 3% LABELING
    # ----------------------------------------------------
    # Initialize all as Normal (0)
    df["Label"] = 0
    
    # Calculate count for 0.5%
    total = len(df)
    n_anomalies = int(total * 0.005)
    
    if n_anomalies > 0:
        # Pick random indices
        anomaly_indices = np.random.choice(df.index, n_anomalies, replace=False)
        df.loc[anomaly_indices, "Label"] = 1

    # Statistics Calculation (Exact original print)
    anomalies = int(df["Label"].sum())
    rate = anomalies / total if total > 0 else 0
    print(f"Structured labeled: total={total}, anomalies={anomalies}, rate={rate:.6f}")

    # ----------------------------------------------------
    # STEP 3 — train/test split (sequential)
    # ----------------------------------------------------
    split_index = int(TRAIN_RATIO * total)
    df_train = df.iloc[:split_index].reset_index(drop=True)
    df_test = df.iloc[split_index:].reset_index(drop=True)

    print(f"Train={len(df_train)}, Test={len(df_test)}")

    # ----------------------------------------------------
    # STEP 4 — tumbling windows
    # ----------------------------------------------------
    win = WINDOW_SIZE
    step = STEP_SIZE

    print(f"Windowing: window={win}, step={step}")

    # Use ["Content", "Label"] to ensure item_Label generation in helper
    df_train_simple = df_train[["Content", "Label"]]
    df_test_simple = df_test[["Content", "Label"]]

    session_train = fixedSize_window(df_train_simple, win, step)
    session_test = fixedSize_window(df_test_simple, win, step)

    if session_train.empty:
        print("--> WARNING: session_train is empty.")
    if session_test.empty:
        print("--> WARNING: session_test is empty.")

    # Optimized string joining
    joiner = " ;-; "
    if not session_train.empty:
        session_train["Content"] = [joiner.join(map(str, lst)) for lst in session_train["Content"]]
    
    if not session_test.empty:
        session_test["Content"] = [joiner.join(map(str, lst)) for lst in session_test["Content"]]

    # ----------------------------------------------------
    # STEP 5 — save outputs
    # ----------------------------------------------------
    def enforce_column_order(frame):
        if frame.empty: return frame
        
        # LogLLM Standard Order: Content, Label, item_Label
        target_cols = ["Content", "Label", "item_Label"]
        
        cols = [c for c in target_cols if c in frame.columns]
        remaining = [c for c in frame.columns if c not in cols]
        
        return frame[cols + remaining]

    session_train = enforce_column_order(session_train)
    session_test = enforce_column_order(session_test)

    session_train.to_csv(os.path.join(prepared_dir, "train.csv"), index=False)
    session_test.to_csv(os.path.join(prepared_dir, "test.csv"), index=False)

    # Write info files (Exact original logic)
    write_info(
        os.path.join(prepared_dir, "train_info.txt"),
        session_train
    )

    write_info(
        os.path.join(prepared_dir, "test_info.txt"),
        session_test
    )

if __name__ == "__main__":
    process_dataset()