# app.py

import streamlit as st
import pandas as pd

from config import DEFAULT_STOPWORDS
from pdf_extract import extract_text_from_pdf
from keywords import doc_counter_from_text, common_keywords
from ui_components import sidebar_settings

st.set_page_config(page_title="PDF Common Keyword Finder", layout="wide")
st.title("PDF Common Keyword Finder (for review keywording)")

st.write(
    "Upload multiple PDFs. The app extracts text and finds common keywords/phrases across documents."
)

settings = sidebar_settings()

uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded and len(uploaded) >= 2:
    stopwords = set()
    if settings["use_default_stopwords"]:
        stopwords |= DEFAULT_STOPWORDS
    stopwords |= settings["extra_stopwords"]

    doc_counters = {}
    failures = []

    with st.spinner("Extracting text and computing keyword frequencies..."):
        for f in uploaded:
            text = extract_text_from_pdf(f.read())
            if not text:
                failures.append(f.name)
                doc_counters[f.name] = None
                continue

            doc_counters[f.name] = doc_counter_from_text(
                text=text,
                stopwords=stopwords,
                ngram_n=settings["ngram_n"],
                min_token_len=settings["min_token_len"],
                keep_hyphens=settings["keep_hyphens"],
            )

    # Remove failed docs from analysis
    doc_counters = {k: v for k, v in doc_counters.items() if v is not None}

    if failures:
        st.warning(
            "Some PDFs had little/no extractable text (likely scanned). "
            f"Skipped: {', '.join(failures)}"
        )

    if len(doc_counters) < 2:
        st.info("Need at least 2 PDFs with extractable text to compute overlap.")
        st.stop()

    rows = common_keywords(
        doc_counters=doc_counters,
        min_docs=min(settings["min_docs"], len(doc_counters)),
        top_k=settings["top_k"],
    )

    N = len(doc_counters)
    out = pd.DataFrame(
        [{
            "term": term,
            "docs_with_term": dfi,
            "coverage_%": round(100 * dfi / N, 1),
            "total_count": total_cnt,
            "tfidf_sum": tfidf_sum,
        } for (term, dfi, total_cnt, tfidf_sum) in rows]
    )

    st.subheader("Common terms across PDFs")
    st.dataframe(out, use_container_width=True)

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="common_keywords.csv",
        mime="text/csv"
    )

    st.subheader("Keyword shortlist")
    shortlist = out["term"].head(min(25, len(out))).tolist()
    st.code("; ".join(shortlist) if shortlist else "No shortlist available.")
else:
    st.info("Upload at least **2 PDFs** to begin.")