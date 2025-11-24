import pandas as pd
import json
import os
import re

def get_smart_patterns(phrases):
    patterns = []
    for p in phrases:
        clean_p = p.strip()
        if not clean_p: continue
        pattern = re.escape(clean_p)
        if clean_p[0].isalnum(): pattern = r"\b" + pattern
        if clean_p[-1].isalnum(): pattern = pattern + r"\b"
        patterns.append(re.compile(pattern, re.IGNORECASE))
    return patterns
    
safe_patterns = get_smart_patterns(Safe_Phrases)
critical_patterns = get_smart_patterns(Critical_Keywords)