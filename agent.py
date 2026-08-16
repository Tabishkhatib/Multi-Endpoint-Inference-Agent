"""
agent.py

Ties everything together. This is the actual agent:
  1. Router picks a starting endpoint.
  2. Stream the response, token by token.
  3. Every token's gap is classified against the endpoint's baseline.
  4. If it's "mild" -> just log and keep going.
     If it's "severe" or the connection dies -> hand off to the Decision Engine.
  5. Execute whatever the Decision Engine says:
       WAIT           -> keep consuming the same stream, allow ONE grace period
       SWITCH         -> abandon this stream, restart the query on a backup endpoint
       ACCEPT_PARTIAL -> stop here, return what we have, flagged as incomplete

Every step prints a log line. That log IS the demo — it's what gets narrated
in the video and is the evidence for every decision the agent made.
"""

from ollama_client import stream_generate, quick_health_check, ConnectionDropped, ENDPOINTS
from endpoint_stats import StatsStore
from router import pick_endpoint
from decision_engine import decide, Decision


def log(msg):
    print(f"[AGENT] {msg}")


def find_backup(category, stats_store, exclude: set):
    """Finds the best healthy endpoint NOT in the excluded set (endpoints
    already tried and failed/switched-away-from for this query)."""
    candidates = []
    for name in ENDPOINTS:
        if name in exclude:
            continue
        if quick_health_check(name):
            candidates.append((name, stats_store.get(name)))

    if not candidates:
        return None

    candidates.sort(key=lambda pair: pair[1].success_rate, reverse=True)
    return candidates[0][0]


def run_on_endpoint(endpoint_name, query_text, category, stats_store, exclude, accumulated_text=""):
    """
    Streams from ONE endpoint, monitoring for degradation. Returns:
        (final_text, status, decision_log)

    status is one of: "completed", "switched", "accepted_partial"
    """
    stats = stats_store.get(endpoint_name)
    decision_log = []
    tokens_generated = 0
    already_waited_once = False
    text_so_far = accumulated_text

    log(f"Streaming from '{endpoint_name}' ({ENDPOINTS[endpoint_name]['model']})...")

    try:
        for chunk in stream_generate(endpoint_name, query_text):
            tokens_generated = chunk["token_index"]
            text_so_far += chunk["token"]

            # Cold-start gap (first token) is excluded from baseline tracking —
            # see ollama_client.py / earlier discussion on why.
            if chunk["is_first_token"]:
                continue

            severity = stats.classify_gap(chunk["gap_ms"])
            stats.record_gap(chunk["gap_ms"])

            if severity == "normal":
                continue  # nothing to do, keep streaming

            if severity == "mild":
                log(f"Mild slowdown on '{endpoint_name}' (gap={chunk['gap_ms']:.0f}ms). Continuing to monitor.")
                continue

            # severity == "severe" -> hand off to the Decision Engine
            exclude_with_self = exclude | {endpoint_name}
            backup_name = find_backup(category, stats_store, exclude_with_self)

            decision, reason = decide(
                category=category,
                tokens_generated=tokens_generated,
                connection_dead=False,
                severity=severity,
                backup_available=backup_name is not None,
                backup_name=backup_name, # type: ignore
                already_waited_once=already_waited_once,
            )
            log(f"DECISION: {decision} — {reason}")
            decision_log.append((decision, reason))

            if decision == Decision.WAIT:
                already_waited_once = True
                continue  # keep consuming the same stream

            if decision == Decision.SWITCH:
                stats.record_failure()
                log(f"Abandoning '{endpoint_name}', restarting on '{backup_name}'.")
                note = f"\n[...continuing on a different endpoint after '{endpoint_name}' degraded...]\n"
                final_text, status, sub_log = run_on_endpoint(
                    backup_name, query_text, category, stats_store,
                    exclude_with_self, accumulated_text=text_so_far + note,
                )
                return final_text, status, decision_log + sub_log

            if decision == Decision.ACCEPT_PARTIAL:
                stats.record_failure()
                return text_so_far, "accepted_partial", decision_log

        # Stream finished normally (done=True reached without a severe event,
        # or recovered after a WAIT)
        stats.record_success()
        return text_so_far, "completed", decision_log

    except ConnectionDropped as e:
        log(f"Connection dropped on '{endpoint_name}': {e}")
        exclude_with_self = exclude | {endpoint_name}
        backup_name = find_backup(category, stats_store, exclude_with_self)

        decision, reason = decide(
            category=category,
            tokens_generated=tokens_generated,
            connection_dead=True,
            severity="severe",
            backup_available=backup_name is not None,
            backup_name=backup_name, # type: ignore
        )
        log(f"DECISION: {decision} — {reason}")
        decision_log.append((decision, reason))
        stats.record_failure()

        if decision == Decision.SWITCH:
            log(f"Restarting on '{backup_name}'.")
            note = f"\n[...continuing on a different endpoint after '{endpoint_name}' dropped...]\n"
            final_text, status, sub_log = run_on_endpoint(
                backup_name, query_text, category, stats_store,
                exclude_with_self, accumulated_text=text_so_far + note,
            )
            return final_text, status, decision_log + sub_log
        else:
            return text_so_far, "accepted_partial", decision_log


def run_query(query_text, stats_store):
    """Entry point: classify + route + stream + monitor, start to finish."""
    endpoint_name, category, route_reason = pick_endpoint(query_text, stats_store)
    log(route_reason)

    final_text, status, decision_log = run_on_endpoint(
        endpoint_name, query_text, category, stats_store, exclude=set()
    )

    log(f"Final status: {status}")
    return {
        "query": query_text,
        "category": category,
        "final_text": final_text,
        "status": status,
        "decision_log": decision_log,
    }


if __name__ == "__main__":
    store = StatsStore(path="data/stats.json")
    store.load()

    query = "how to make chicken curry?"
    result = run_query(query, store)

    print("\n--- RESULT ---")
    print(f"Status: {result['status']}")
    print(f"Answer: {result['final_text']}")

    store.save()
