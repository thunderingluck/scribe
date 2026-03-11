"""
Retrieval module: surface relevant essay excerpts given a new prompt.

Current approach: lightweight keyword overlap (word-intersection score).
No vector database or embeddings are used yet.

Architectural note:
  This module is intentionally isolated so it can be swapped out for
  embedding-based retrieval (e.g. FAISS, Pinecone, pgvector) without
  touching the API layer. The interface contract is:
    retrieve(prompt: str, essays: list[dict], top_k: int) -> list[str]
  where each returned string is a short excerpt from a matching essay.
"""

import json
import re
from pathlib import Path


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens, stripping punctuation."""
    return set(re.findall(r"[a-z]+", text.lower()))


def _score(query_tokens: set[str], essay: dict) -> float:
    """Jaccard-like overlap between query words and essay prompt words."""
    prompt_tokens = _tokenize(essay.get("prompt", ""))
    essay_tokens = _tokenize(essay.get("essay", ""))
    combined = prompt_tokens | essay_tokens
    if not combined:
        return 0.0
    return len(query_tokens & combined) / len(query_tokens | combined)


def retrieve(prompt: str, essays: list[dict], top_k: int = 3) -> list[str]:
    """
    Return up to top_k short excerpts from the most relevant essays.
    Each excerpt is the first ~300 chars of the essay body.
    """
    query_tokens = _tokenize(prompt)
    scored = sorted(essays, key=lambda e: _score(query_tokens, e), reverse=True)
    excerpts = []
    for essay in scored[:top_k]:
        text = essay.get("essay", "")
        excerpt = text[:300].rsplit(" ", 1)[0] + "…" if len(text) > 300 else text
        excerpts.append(excerpt)
    return excerpts


def load_essays(path: Path) -> list[dict]:
    """Load raw essays JSONL into a list of dicts."""
    essays = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                essays.append(json.loads(line))
    return essays
