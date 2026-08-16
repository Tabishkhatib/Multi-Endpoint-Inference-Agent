"""
router.py

Picks which endpoint to START a query on. This is the "initial routing"
decision — separate from the Decision Engine, which handles what happens
if that choice degrades MID-STREAM.

Logic:
1. Classify the query (code / reasoning / simple).
2. Look up the endpoint whose category matches.
3. Before committing, sanity-check that endpoint: is it reachable right
   now, and does it have a decent track record (success_rate)?
4. If the preferred endpoint looks bad, fall back to the best remaining
   healthy endpoint instead — this is what "based on observed performance
   rather than a fixed sequence" means in practice.

Every decision returns a reason string, because that's what gets logged
and shown in the demo.
"""

from ollama_client import ENDPOINTS, quick_health_check
from semantic_classifier import classify_query_semantic # type: ignore

MIN_ACCEPTABLE_SUCCESS_RATE = 0.5  # below this, don't trust the endpoint even if reachable


def pick_endpoint(query_text: str, stats_store):
    """
    Returns (endpoint_name, category, reason)
    """
    category, method = classify_query_semantic(query_text)

    # Which endpoint is "supposed" to handle this category
    preferred_name = None
    for name, cfg in ENDPOINTS.items():
        if cfg["category"] == category:
            preferred_name = name
            break

    preferred_stats = stats_store.get(preferred_name)
    preferred_healthy = quick_health_check(preferred_name) # type: ignore

    if preferred_healthy and preferred_stats.success_rate >= MIN_ACCEPTABLE_SUCCESS_RATE:
        reason = (
            f"Classified as '{category}' (via {method}). '{preferred_name}' is healthy "
            f"(success_rate={preferred_stats.success_rate:.2f}) — using preferred endpoint."
        )
        return preferred_name, category, reason

    # Preferred endpoint isn't trustworthy right now — find the best alternative
    reason_parts = [f"Classified as '{category}' (via {method})."]
    if not preferred_healthy:
        reason_parts.append(f"'{preferred_name}' is unreachable.")
    else:
        reason_parts.append(
            f"'{preferred_name}' has a poor track record "
            f"(success_rate={preferred_stats.success_rate:.2f})."
        )

    candidates = []
    for name in ENDPOINTS:
        if name == preferred_name:
            continue
        if quick_health_check(name):
            candidates.append((name, stats_store.get(name)))

    if not candidates:
        # Nothing else is even reachable — fall back to the preferred one anyway,
        # since a poor track record beats no endpoint at all.
        reason_parts.append(f"No healthy alternatives found — falling back to '{preferred_name}' anyway.")
        return preferred_name, category, " ".join(reason_parts)

    # Pick the healthy candidate with the best success rate
    candidates.sort(key=lambda pair: pair[1].success_rate, reverse=True)
    fallback_name, fallback_stats = candidates[0]
    reason_parts.append(
        f"Falling back to '{fallback_name}' (success_rate={fallback_stats.success_rate:.2f})."
    )
    return fallback_name, category, " ".join(reason_parts)


if __name__ == "__main__":
    from endpoint_stats import StatsStore

    store = StatsStore()

    test_queries = [
        "What's the capital of France?",
        "Write a function to reverse a linked list",
        "Why does inflation affect interest rates, and how does that ripple through housing?",
    ]

    for q in test_queries:
        endpoint, category, reason = pick_endpoint(q, store)
        print(f"Query: {q}")
        print(f"  -> endpoint={endpoint}, category={category}")
        print(f"  -> reason: {reason}\n")