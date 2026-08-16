"""
semantic_classifier.py

Upgrade over classifier.py: uses REAL semantic embeddings (nomic-embed-text
via Ollama) and cosine similarity against a handful of example queries per
category, instead of keyword matching. This catches queries that mean the
same thing as an example without sharing its exact words.

Falls back to the keyword classifier (classifier.py) if the embedding call
times out or fails — this is not a hypothetical: the classifier is itself
just another endpoint call, so it needs the same degrade-gracefully
treatment as everything else in this project.
"""

import math
import requests
from classifier import classify_query as keyword_classify

EMBEDDING_HOST = "127.0.0.1"
EMBEDDING_PORT = 11435          # reusing the 'reasoning' instance, any pulled model works per-request
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_TIMEOUT = 5             # real per-query calls should be fast once warm


def warm_up():
    """Call this ONCE, before serving real queries (e.g. at the start of a
    demo), to pay the cold-start cost upfront instead of during the first
    real query. Also pre-builds the category prototype embeddings."""
    import time
    print("[semantic_classifier] Warming up embedding model...")
    t0 = time.time()
    _get_embedding("warmup", timeout=30)  # absorb the cold-load cost here, with room to spare
    _build_prototypes()
    print(f"[semantic_classifier] Warm-up complete in {time.time() - t0:.1f}s")

CATEGORY_EXAMPLES = {
    "code": [
        "fix this python function, it's throwing a syntax error",
        "write a function to reverse a linked list",
        "why is my loop not terminating",
        "debug this script for me",
        "how do I fix this null pointer exception",
    ],
    "reasoning": [
        "why does inflation affect interest rates",
        "explain how neural networks learn",
        "what are the trade-offs between these two approaches",
        "compare capitalism and socialism",
        "how does climate change influence ocean currents",
    ],
    "simple": [
        "what's the capital of France",
        "what time is it",
        "how many continents are there",
        "what year did world war two end",
        "what's the boiling point of water",
    ],
}

_prototype_cache = None  # populated on first use, so startup doesn't pay this cost if unused


def _get_embedding(text: str, timeout=None):
    url = f"http://{EMBEDDING_HOST}:{EMBEDDING_PORT}/api/embeddings"
    r = requests.post(
        url,
        json={"model": EMBEDDING_MODEL, "prompt": text},
        timeout=timeout or EMBEDDING_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def _cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _build_prototypes():
    """Embeds every example query once, caches the result. Only runs on
    first real use, not at import time."""
    global _prototype_cache
    if _prototype_cache is not None:
        return _prototype_cache

    prototypes = {}
    for category, examples in CATEGORY_EXAMPLES.items():
        prototypes[category] = [_get_embedding(ex) for ex in examples]
    _prototype_cache = prototypes
    return prototypes


def classify_query_semantic(query_text: str):
    """
    Returns (category, method) where method is "embedding" or "keyword_fallback",
    so the caller can log which path was actually used.
    """
    try:
        prototypes = _build_prototypes()
        query_vec = _get_embedding(query_text)

        best_category = None
        best_score = -1.0
        for category, vectors in prototypes.items():
            score = max(_cosine_similarity(query_vec, v) for v in vectors)
            if score > best_score:
                best_score = score
                best_category = category

        return best_category, "embedding"

    except (requests.exceptions.RequestException, KeyError, ValueError):
        # Embedding endpoint down, timed out, or returned something unexpected —
        # degrade gracefully instead of crashing the whole routing step.
        return keyword_classify(query_text), "keyword_fallback"


if __name__ == "__main__":
    test_queries = [
        "What's the capital of France?",
        "Why does inflation affect interest rates, and how does that ripple through the housing market?",
        "Fix this python function, it's throwing a syntax error",
        "Write a function to reverse a linked list",
        "There's a bug in my script and I don't know why",   # no literal "code" keywords — real semantic test
        "What time is it",
    ]
    for q in test_queries:
        category, method = classify_query_semantic(q)
        print(f"[{category:>9}] ({method:>16})  {q}")