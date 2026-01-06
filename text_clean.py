# text_clean.py

import re

def normalize_text(text: str, keep_hyphens: bool = True) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # URLs

    if keep_hyphens:
        text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    else:
        text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str, min_len: int = 3):
    toks = text.split()
    out = []
    for t in toks:
        if t.isdigit():
            continue
        if len(t) < min_len:
            continue
        out.append(t)
    return out

def apply_stopwords(tokens, stopwords: set):
    return [t for t in tokens if t not in stopwords]

def make_ngrams(tokens, n: int):
    if n == 1:
        return tokens
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]