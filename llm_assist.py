# llm_assist.py
import os
import json
import requests
from typing import List, Dict, Any

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

SYSTEM_PROMPT = """You help refine keyword lists for literature reviews.

Rules:
- Prefer domain-specific terms/phrases.
- Remove generic academic filler terms.
- Merge obvious duplicates/variants (plural, hyphenation).
- Provide grouped keyword buckets suitable for building database searches.
- Do NOT invent terms unrelated to the given candidate list unless marked as "suggested_expansions".
Return strictly valid JSON only.
"""

def build_keyword_prompt(topic: str, candidates: List[Dict[str, Any]], max_terms: int = 200) -> str:
    """
    Builds a compact JSON prompt for the LLM using only candidate keywords (not full PDFs).
    """
    candidates = candidates[:max_terms]

    prompt_obj = {
        "task": "Refine keyword candidates for literature review searching",
        "topic": topic,
        "candidates": candidates,
        "required_output": {
            "cleaned_keywords": "List of cleaned/merged terms derived from candidates",
            "grouped_buckets": "Dict of bucket_name -> list of terms",
            "suggested_expansions": "Dict seed_term -> list of suggested synonyms/variants (may be new)",
            "stopword_suggestions": "List of generic terms to consider excluding",
            "notes": "Any brief notes about decisions"
        },
        "constraints": [
            "cleaned_keywords must come from candidates (merging/normalizing allowed)",
            "suggested_expansions may include new synonyms/variants not in candidates",
            "limit grouped_buckets to 4-8 buckets",
            "output must be valid JSON and nothing else"
        ]
    }
    return json.dumps(prompt_obj, indent=2)

def ollama_chat_json(user_prompt: str, timeout: int = 180) -> Dict[str, Any]:
    """
    Calls Ollama /api/chat and returns parsed JSON response.
    """
    # print(f"[LLM] calling {url} model={OLLAMA_MODEL} prompt_chars={len(user_prompt)}")
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        # JSON mode (supported by many Ollama models); if it fails, we fall back below.
        "format": "json",
        "options": {"temperature": 0.2}
    }

    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()

    content = r.json()["message"]["content"]

    # In case model returns a JSON string with extra whitespace, still parse safely
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Fallback: attempt to extract first JSON object
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start:end+1])
        raise