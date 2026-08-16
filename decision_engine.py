"""
decision_engine.py

The core of the whole project. Given a degradation event mid-stream, decides
one of: WAIT, SWITCH, ACCEPT_PARTIAL — and returns a plain-English reason
for the choice, since that reason is what gets logged and shown in the demo.

Locked design:
- Expected response length per category gives a rough "progress ratio"
  (tokens generated so far / expected total). This is the sunk-cost signal.
- 40% progress is the pivot point:
    < 40%  -> not much invested yet, SWITCH is worth it if a backup exists
    >= 40% -> too much invested to throw away, WAIT (bounded) or ACCEPT_PARTIAL
- A dead connection is unambiguous: no WAIT is possible, only SWITCH or
  ACCEPT_PARTIAL, decided the same way by progress ratio.
- No healthy backup at all -> forced ACCEPT_PARTIAL, logged as forced,
  not presented as if it were a real choice among options.

This module does NOT talk to the network. It only reasons over numbers
handed to it by agent.py (which is watching the live stream).
"""

EXPECTED_LENGTH_BY_CATEGORY = {
    "simple": 75,
    "reasoning": 300,
    "code": 200,
}

PROGRESS_SWITCH_THRESHOLD = 0.40  # locked value

WAIT_GRACE_PERIODS_ALLOWED = 1  # only wait once per degradation event, not indefinitely


class Decision:
    WAIT = "WAIT"
    SWITCH = "SWITCH"
    ACCEPT_PARTIAL = "ACCEPT_PARTIAL"


def compute_progress(tokens_generated: int, category: str) -> float:
    expected = EXPECTED_LENGTH_BY_CATEGORY.get(category, 150)
    if expected <= 0:
        return 1.0
    return min(tokens_generated / expected, 1.0)


def decide(
    *,
    category: str,
    tokens_generated: int,
    connection_dead: bool,
    severity: str,          # "normal" | "mild" | "severe"  (from EndpointStats.classify_gap)
    backup_available: bool,
    backup_name: str = None,
    already_waited_once: bool = False,
):
    """
    Returns (decision, reason)

    decision is one of Decision.WAIT / Decision.SWITCH / Decision.ACCEPT_PARTIAL
    reason is a human-readable explanation, meant to be logged/printed/narrated.
    """
    progress = compute_progress(tokens_generated, category)
    progress_pct = f"{progress * 100:.0f}%"

    # ---- No backup at all: forced outcome, not a real choice ----
    if not backup_available:
        return (
            Decision.ACCEPT_PARTIAL,
            f"No healthy backup endpoint available. Forced to accept partial output "
            f"({tokens_generated} tokens generated, ~{progress_pct} of expected length). "
            f"This is not a preference — there was no alternative."
        )

    # ---- Dead connection: no waiting possible, only switch or accept ----
    if connection_dead:
        if progress < PROGRESS_SWITCH_THRESHOLD:
            return (
                Decision.SWITCH,
                f"Connection dropped after only {tokens_generated} tokens (~{progress_pct} of "
                f"expected length) — below the {int(PROGRESS_SWITCH_THRESHOLD*100)}% threshold, "
                f"so little is lost by restarting on '{backup_name}'."
            )
        else:
            return (
                Decision.ACCEPT_PARTIAL,
                f"Connection dropped after {tokens_generated} tokens (~{progress_pct} of expected "
                f"length) — already past the {int(PROGRESS_SWITCH_THRESHOLD*100)}% threshold, so "
                f"restarting elsewhere would throw away more progress than it recovers. "
                f"Accepting partial output and flagging the gap instead."
            )

    # ---- Connection alive but degrading (mild/severe) ----
    if severity == "severe":
        if progress < PROGRESS_SWITCH_THRESHOLD:
            return (
                Decision.SWITCH,
                f"Severe slowdown detected after {tokens_generated} tokens (~{progress_pct} of "
                f"expected length) — still below the {int(PROGRESS_SWITCH_THRESHOLD*100)}% threshold. "
                f"Switching to '{backup_name}' rather than waiting on a stalled endpoint."
            )
        else:
            if already_waited_once:
                return (
                    Decision.ACCEPT_PARTIAL,
                    f"Severe slowdown persisted after already granting one wait period. "
                    f"{tokens_generated} tokens generated (~{progress_pct} of expected length) — "
                    f"too much progress to restart, but it hasn't recovered. Accepting partial output."
                )
            return (
                Decision.WAIT,
                f"Severe slowdown detected after {tokens_generated} tokens (~{progress_pct} of "
                f"expected length) — already past the {int(PROGRESS_SWITCH_THRESHOLD*100)}% threshold, "
                f"so restarting would waste more than it saves. Granting one grace period before "
                f"reconsidering."
            )

    if severity == "mild":
        return (
            Decision.WAIT,
            f"Mild slowdown detected after {tokens_generated} tokens (~{progress_pct} of expected "
            f"length) — not severe enough to act on yet. Continuing to monitor."
        )

    # severity == "normal" should never reach the Decision Engine in practice
    # (agent.py shouldn't call this unless something looked abnormal), but
    # handle it defensively rather than crashing.
    return (
        Decision.WAIT,
        "No meaningful degradation detected — continuing normally."
    )


if __name__ == "__main__":
    # A few scenarios to sanity check the table by hand
    scenarios = [
        dict(category="simple", tokens_generated=5, connection_dead=False,
             severity="severe", backup_available=True, backup_name="reasoning"),
        dict(category="reasoning", tokens_generated=200, connection_dead=False,
             severity="severe", backup_available=True, backup_name="fast"),
        dict(category="code", tokens_generated=10, connection_dead=True,
             backup_available=True, backup_name="reasoning", severity="severe"),
        dict(category="simple", tokens_generated=8, connection_dead=False,
             severity="severe", backup_available=False),
    ]
    for s in scenarios:
        decision, reason = decide(**s)
        print(f"{decision}: {reason}\n")
