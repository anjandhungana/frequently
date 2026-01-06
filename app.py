# app.py

import os
import json
import streamlit as st
import pandas as pd
import requests

from config import DEFAULT_STOPWORDS
from pdf_extract import extract_text_from_pdf
from keywords import doc_counter_from_text, common_keywords
from ui_components import sidebar_settings
from llm_assist import build_keyword_prompt, ollama_chat_json


# -------------------------
# Helpers
# -------------------------
def ollama_is_alive(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# -------------------------
# App
# -------------------------
st.set_page_config(page_title="Frequently – Keyword Finder", layout="wide")
st.title("Frequently – Keyword Finder for Literature Review Searches")

st.write(
    "Workflow: **1) Upload PDFs → 2) Add a short description → 3) Generate common keywords** "
    "(with optional LLM cleanup/grouping using **local Ollama**, no API key)."
)

settings = sidebar_settings()

# Session defaults
if "topic" not in st.session_state:
    st.session_state.topic = "IoT and AI for poultry welfare monitoring and decision support."
if "use_llm" not in st.session_state:
    st.session_state.use_llm = True
if "out_df" not in st.session_state:
    st.session_state.out_df = None
if "llm_result" not in st.session_state:
    st.session_state.llm_result = None
if "llm_prompt" not in st.session_state:
    st.session_state.llm_prompt = None


# =========================
# 1) Upload
# =========================
st.header("1) Upload PDFs")
uploaded = st.file_uploader("Upload 2+ PDFs", type=["pdf"], accept_multiple_files=True)

st.divider()

# =========================
# 2) Description
# =========================
st.header("2) Description")
topic = st.text_area(
    "Describe your review topic (1–3 sentences). This guides LLM grouping and suggested expansions.",
    value=st.session_state.topic,
    height=110,
)
st.session_state.topic = topic

use_llm = st.checkbox("Use local LLM (Ollama) to clean + group keywords", value=st.session_state.use_llm)
st.session_state.use_llm = use_llm

max_terms_for_llm = st.slider(
    "Max candidate terms sent to LLM",
    min_value=50, max_value=400, value=200, step=25,
    help="Higher = more context but slower."
)

st.divider()

# =========================
# 3) Generate
# =========================
st.header("3) Keyword generation")

run = st.button("Generate keywords", type="primary")

if run:
    if not uploaded or len(uploaded) < 2:
        st.error("Please upload at least **2 PDFs**.")
        st.stop()

    # Build stopwords
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

    # Remove failed docs
    doc_counters = {k: v for k, v in doc_counters.items() if v is not None}

    if failures:
        st.warning(
            "Some PDFs had little/no extractable text (likely scanned). "
            f"Skipped: {', '.join(failures)}"
        )

    if len(doc_counters) < 2:
        st.error("Need at least 2 PDFs with extractable text to compute overlap.")
        st.stop()

    # Keyword overlap
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

    st.session_state.out_df = out
    st.session_state.llm_result = None
    st.session_state.llm_prompt = None

    # ---- Optional: LLM refinement runs automatically after extraction ----
    if use_llm and out is not None and len(out) > 0:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

        if not ollama_is_alive(base_url):
            st.warning(
                f"Ollama not reachable at `{base_url}`. "
                "Keyword extraction succeeded, but LLM refinement was skipped."
            )
        else:
            # Prepare candidates (top N rows)
            candidates = []
            for _, row in out.head(max_terms_for_llm).iterrows():
                candidates.append({
                    "term": row["term"],
                    "docs_with_term": int(row["docs_with_term"]),
                    "coverage_%": float(row["coverage_%"]),
                    "total_count": int(row["total_count"]),
                    "tfidf_sum": float(row["tfidf_sum"]),
                })

            user_prompt = build_keyword_prompt(topic=topic, candidates=candidates, max_terms=max_terms_for_llm)

            with st.spinner(f"Running LLM refinement via Ollama ({model})..."):
                try:
                    llm_result = ollama_chat_json(user_prompt)
                    st.session_state.llm_result = llm_result
                    st.session_state.llm_prompt = user_prompt
                except Exception as e:
                    st.warning(f"LLM refinement failed (keywords still available): {e}")


# =========================
# Results display (always visible if available)
# =========================
out = st.session_state.out_df

if out is not None and len(out) > 0:
    st.subheader("Common terms across PDFs")
    st.dataframe(out, use_container_width=True)

    st.download_button(
        "Download common keywords (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="common_keywords.csv",
        mime="text/csv",
    )

    st.subheader("Keyword shortlist")
    shortlist = out["term"].head(min(25, len(out))).tolist()
    st.code("; ".join(shortlist) if shortlist else "No shortlist available.")

    # LLM results (if present)
    if st.session_state.use_llm:
        st.subheader("LLM refinement output")

        llm_result = st.session_state.llm_result
        if llm_result is None:
            st.info("Run **Generate keywords** to produce LLM refinement output.")
        else:
            st.markdown("#### Cleaned keywords")
            st.write(llm_result.get("cleaned_keywords", []))

            st.markdown("#### Grouped buckets")
            buckets = llm_result.get("grouped_buckets", {})
            if isinstance(buckets, dict):
                for bname, terms in buckets.items():
                    st.markdown(f"**{bname}**")
                    st.write(terms)

            st.markdown("#### Suggested expansions")
            st.write(llm_result.get("suggested_expansions", {}))

            st.markdown("#### Suggested stopwords")
            st.write(llm_result.get("stopword_suggestions", []))

            st.download_button(
                "Download LLM output (JSON)",
                data=json.dumps(llm_result, indent=2).encode("utf-8"),
                file_name="llm_curated_keywords.json",
                mime="application/json",
            )

            if st.session_state.llm_prompt:
                st.download_button(
                    "Download prompt used (JSON)",
                    data=st.session_state.llm_prompt.encode("utf-8"),
                    file_name="llm_prompt_used.json",
                    mime="application/json",
                )
else:
    st.info("Upload PDFs and click **Generate keywords** to see results.")