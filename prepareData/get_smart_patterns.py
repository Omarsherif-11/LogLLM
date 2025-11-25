import re
from typing import List, Pattern, Dict
import config as cfg

def compile_phrases(phrases: List[str]) -> List[Pattern]:
    """
    Compile phrases into regex patterns with smart boundaries.
    Same logic as original get_smart_patterns: 
    - escape text
    - add word boundary only if phrase starts/ends with alnum
    """
    patterns = []
    for p in phrases:
        clean = str(p).strip()
        if not clean:
            continue

        pat = re.escape(clean)
        if clean[0].isalnum():
            pat = r"\b" + pat
        if clean[-1].isalnum():
            pat = pat + r"\b"

        patterns.append(re.compile(pat, re.IGNORECASE))
    return patterns


def get_patterns_for_dataset(dataset_key: str) -> Dict[str, List[Pattern]]:
    """
    Return compiled patterns for a dataset key (android/mac/windows)
    """
    if dataset_key not in cfg.datasets:
        raise KeyError(f"Unknown dataset '{dataset_key}'. Available: {list(cfg.datasets.keys())}")

    ds = cfg.datasets[dataset_key]
    safe = ds.get("safe_phrases", [])
    crit = ds.get("critical_keywords", [])

    return {
        "safe": compile_phrases(safe),
        "critical": compile_phrases(crit),
    }


def get_all_dataset_patterns() -> Dict[str, Dict[str, List[Pattern]]]:
    """
    Patterns for all datasets defined in config.py
    """
    out = {}
    for k in cfg.datasets.keys():
        out[k] = get_patterns_for_dataset(k)
    return out
