import os
import argparse
import pandas as pd
from pathlib import Path
import re # Needed for the new label_df

import config
from helper import structure_log, fixedSize_window
from get_smart_patterns import get_patterns_for_dataset

# --- LOG FORMATS (Verified for Drain/Brain & parsing) ---
LOG_FORMATS = {
    "windows": "<Date> <Time>, <Level>                  <Component>    <Content>",
    "mac": "<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>",
    "android": "<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>",
    "bgl": "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>"
}

WINDOW_SIZES = {"windows": 100, "mac": 10, "android": 20, "bgl": 100}
STEP_SIZES = {"windows": 100, "mac": 10, "android": 20, "bgl": 100}


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def label_df(df, safe_patterns, critical_patterns):
    """
    Optimized labeling using vectorized boolean masks.
    Uses single regex strings with case=False to ensure compatibility 
    and fast matching across all log lines.
    """
    if df.empty:
        return df

    content = df["Content"].astype(str)
    
    # 1. Combine patterns into one large string using the OR operator (|)
    # .pattern is used as get_smart_patterns returns compiled regex objects
    safe_regex = "|".join(p.pattern for p in safe_patterns if p.pattern)
    crit_regex = "|".join(p.pattern for p in critical_patterns if p.pattern)
    
    is_safe = pd.Series(False, index=df.index)
    is_critical = pd.Series(False, index=df.index)

    # 2. Run vectorized search. case=False enforces case-insensitivity.
    # The previous 50% error was likely caused by a single malformed pattern 
    # being included in these combined strings.
    if safe_regex:
        is_safe = content.str.contains(safe_regex, case=False, na=False, regex=True)
    
    if crit_regex:
        is_critical = content.str.contains(crit_regex, case=False, na=False, regex=True)

    # 3. Apply logic: Is Critical AND NOT Safe (Safe takes precedence)
    df["Label"] = (is_critical & ~is_safe).astype(int)
    
    return df


def process_dataset(key, train_ratio=0.8, start_line=0, end_line=None):
    key = key.lower()
    
    if key not in config.datasets:
        raise KeyError(f"Unknown dataset '{key}' in config.datasets")
    if key not in LOG_FORMATS:
        raise KeyError(f"Dataset '{key}' missing from LOG_FORMATS")

    ds = config.datasets[key]
    data_dir = ds["data_dir"]
    log_name = ds["log_name"]
    log_format = LOG_FORMATS[key]

    print(f"\n=== Processing dataset: {key} ===")
    print(f"data_dir={data_dir}, log_name={log_name}")

    structured_dir = os.path.join(data_dir, "structured")
    prepared_dir = os.path.join(data_dir, "prepared")
    ensure_dir(structured_dir)
    ensure_dir(prepared_dir)

    # ----------------------------------------------------
    # STEP 1 — parse raw logs → structured/*.csv
    # ----------------------------------------------------
    structure_log(
        data_dir,
        structured_dir,
        log_name,
        log_format,
        start_line=start_line,
        end_line=end_line,
    )

    structured_path = os.path.join(structured_dir, log_name + "_structured.csv")
    if not os.path.exists(structured_path):
        raise RuntimeError(f"STRUCTURED CSV NOT FOUND: {structured_path}")

    try:
        df = pd.read_csv(structured_path, encoding="latin-1")
    except pd.errors.EmptyDataError:
        raise RuntimeError(f"ERROR: The file {structured_path} is completely empty.")

    if len(df) == 0:
        raise RuntimeError(f"PARSING FAILED: 0 rows found in {structured_path}. Check LOG_FORMAT.")

    # helper guarantees Content is last extracted column
    if "Content" not in df.columns:
        df["Content"] = df.iloc[:, -1].astype(str)

    # ----------------------------------------------------
    # STEP 2 — LABEL THEN REWRITE STRUCTURED FILE
    # ----------------------------------------------------
    if key == "bgl":
        first_col = df.columns[0]
        raw_labels = df[first_col].astype(str)
        def normalize_bgl_label(x):
            return 0 if x.strip() == "-" else 1
        df["Label"] = raw_labels.apply(normalize_bgl_label)
    else:
        patterns = get_patterns_for_dataset(key)
        df = label_df(df, patterns["safe"], patterns["critical"])

    # Statistics Calculation
    total = len(df)
    anomalies = int(df["Label"].sum())
    rate = anomalies / total if total > 0 else 0
    print(f"Structured labeled: total={total}, anomalies={anomalies}, rate={rate:.6f}")

    # ----------------------------------------------------
    # STEP 3 — train/test split (sequential)
    # ----------------------------------------------------
    split_index = int(train_ratio * total)
    df_train = df.iloc[:split_index].reset_index(drop=True)
    df_test = df.iloc[split_index:].reset_index(drop=True)

    print(f"Train={len(df_train)}, Test={len(df_test)}")

    # ----------------------------------------------------
    # STEP 4 — tumbling windows using fixedSize_window
    # ----------------------------------------------------
    win = WINDOW_SIZES[key]
    step = STEP_SIZES[key]

    print(f"Windowing: window={win}, step={step}")

    # Use ["Content", "Label"] so helper.py doesn't crash on integers
    df_train_simple = df_train[["Content", "Label"]]
    df_test_simple = df_test[["Content", "Label"]]

    session_train = fixedSize_window(df_train_simple, win, step)
    session_test = fixedSize_window(df_test_simple, win, step)

    if session_train.empty:
        print("--> WARNING: session_train is empty.")
    if session_test.empty:
        print("--> WARNING: session_test is empty.")

    # Optimized string joining using list comprehension
    joiner = " ;-; "
    if not session_train.empty:
        session_train["Content"] = [joiner.join(map(str, lst)) for lst in session_train["Content"]]
    
    if not session_test.empty:
        session_test["Content"] = [joiner.join(map(str, lst)) for lst in session_test["Content"]]

    # ----------------------------------------------------
    # STEP 5 — save outputs (FORCE LOGLLM ORDER)
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

    def write_info(path, df_win):
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

    write_info(
        os.path.join(prepared_dir, "train_info.txt"),
        session_train
    )

    write_info(
        os.path.join(prepared_dir, "test_info.txt"),
        session_test
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--start_line", type=int, default=0)
    p.add_argument("--end_line", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for ds in args.datasets:
        try:
            process_dataset(
                ds,
                train_ratio=args.train_ratio,
                start_line=args.start_line,
                end_line=args.end_line,
            )
        except Exception as e:
            print(f"CRITICAL ERROR processing {ds}: {e}")