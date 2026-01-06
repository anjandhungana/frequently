# keywords.py

import math
from collections import Counter, defaultdict

from text_clean import normalize_text, tokenize, apply_stopwords, make_ngrams

def doc_counter_from_text(
    text: str,
    stopwords: set,
    ngram_n: int = 1,
    min_token_len: int = 3,
    keep_hyphens: bool = True,
) -> Counter:
    norm = normalize_text(text, keep_hyphens=keep_hyphens)
    tokens = tokenize(norm, min_len=min_token_len)
    tokens = apply_stopwords(tokens, stopwords)

    grams = make_ngrams(tokens, n=ngram_n)
    return Counter(grams)

def compute_df(doc_counters: dict) -> Counter:
    """
    doc_counters: {doc_name: Counter(term->count)}
    returns df(term)=number of docs containing term
    """
    df = Counter()
    for _, c in doc_counters.items():
        for term in c.keys():
            df[term] += 1
    return df

def compute_tfidf_sum(doc_counters: dict):
    """
    Returns:
      tfidf_sum: dict(term -> sum tfidf across docs)
      df: dict(term -> document frequency)
    """
    df = compute_df(doc_counters)
    N = len(doc_counters)
    tfidf_sum = defaultdict(float)

    for doc, c in doc_counters.items():
        total = sum(c.values()) or 1
        for term, cnt in c.items():
            tf = cnt / total
            idf = math.log((N + 1) / (df[term] + 1)) + 1.0
            tfidf_sum[term] += tf * idf

    return dict(tfidf_sum), dict(df)

def common_keywords(doc_counters: dict, min_docs: int = 2, top_k: int = 50):
    """
    Returns list of tuples:
      (term, docs_with_term, total_count, tfidf_sum)
    """
    tfidf_sum, df = compute_tfidf_sum(doc_counters)
    docs = list(doc_counters.keys())

    rows = []
    for term, dfi in df.items():
        if dfi < min_docs:
            continue
        total_cnt = sum(doc_counters[d].get(term, 0) for d in docs)
        rows.append((term, dfi, total_cnt, tfidf_sum.get(term, 0.0)))

    # Sort: coverage first, then tfidf, then total count
    rows.sort(key=lambda x: (x[1], x[3], x[2]), reverse=True)
    return rows[:top_k]