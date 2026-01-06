# ui_components.py

import re
import streamlit as st

def sidebar_settings():
    with st.sidebar:
        st.header("Settings")

        ngram_n = st.selectbox(
            "Keyword type",
            [1, 2, 3],
            index=0,
            format_func=lambda x: {1:"Unigrams", 2:"Bigrams", 3:"Trigrams"}[x],
        )

        min_token_len = st.slider("Minimum token length", 2, 8, 3)
        keep_hyphens = st.checkbox("Keep hyphenated terms (e.g., cage-free)", value=True)

        use_default_stopwords = st.checkbox("Use built-in stopwords", value=True)
        extra_stopwords_text = st.text_area(
            "Extra stopwords (comma or newline separated)",
            value="",
            help="Add terms you don't want to dominate (e.g., poultry, chicken)."
        )

        min_docs = st.slider("Term must appear in ≥ this many PDFs", 2, 50, 2)
        top_k = st.slider("Show top K terms", 10, 300, 50)

    extra = set()
    if extra_stopwords_text.strip():
        raw = re.split(r"[,\n]+", extra_stopwords_text.strip().lower())
        extra = {w.strip() for w in raw if w.strip()}

    return {
        "ngram_n": ngram_n,
        "min_token_len": min_token_len,
        "keep_hyphens": keep_hyphens,
        "use_default_stopwords": use_default_stopwords,
        "extra_stopwords": extra,
        "min_docs": min_docs,
        "top_k": top_k,
    }