"""
endpoint_stats.py

Tracks "recent performance" for ONE endpoint over time:
- a rolling baseline of normal inter-token gap (ms)
- whether it's still "calibrating" (not enough data yet) or has a trustworthy baseline
- success/failure history, so the Router can learn which endpoints are reliable

Design (locked earlier):
- First MIN_SAMPLES_FOR_BASELINE tokens on a fresh endpoint = calibration phase.
  During calibration, degradation is judged against a fixed fallback threshold.
- Once calibrated, degradation is judged relative to this endpoint's OWN
  rolling average gap (baseline_avg_gap_ms), because different models run
  at genuinely different speeds.
"""

from collections import deque
import json
import os

MIN_SAMPLES_FOR_BASELINE = 12      # tokens needed before we trust a baseline
ROLLING_WINDOW = 30                # how many recent gaps we average over
FALLBACK_SEVERE_MS = 1000          # fixed threshold used only during calibration

MILD_MULTIPLIER = 7                # gap > 7x baseline = mild slowdown (log, keep watching)
SEVERE_MULTIPLIER = 25             # gap > 25x baseline = severe stall (hand to Decision Engine)
ABSOLUTE_NORMAL_CEILING_MS = 15  # a gap this small is always "normal", regardless of baseline math


class EndpointStats:
    def __init__(self, name):
        self.name = name
        self.gap_history = deque(maxlen=ROLLING_WINDOW)
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

    # ---- calibration state ----

    @property
    def is_calibrated(self) -> bool:
        return len(self.gap_history) >= MIN_SAMPLES_FOR_BASELINE

    @property
    def baseline_avg_gap_ms(self) -> float:
        if not self.gap_history:
            return 0.0
        return sum(self.gap_history) / len(self.gap_history)

    def record_gap(self, gap_ms: float):
        """Call this for every token received, to build/update the baseline."""
        self.gap_history.append(gap_ms)

    # ---- degradation classification ----

    def classify_gap(self, gap_ms: float) -> str:
        """
        Returns one of: "normal", "mild", "severe"

        During calibration: uses the fixed fallback threshold.
        Once calibrated: uses multiples of this endpoint's own baseline.
        """
        if gap_ms <= ABSOLUTE_NORMAL_CEILING_MS:
            return "normal"
        if not self.is_calibrated:
            if gap_ms >= FALLBACK_SEVERE_MS:
                return "severe"
            return "normal"

        baseline = self.baseline_avg_gap_ms
        if baseline <= 0:
            baseline = 1  # avoid division weirdness on a near-zero baseline

        ratio = gap_ms / baseline
        if ratio >= SEVERE_MULTIPLIER:
            return "severe"
        elif ratio >= MILD_MULTIPLIER:
            return "mild"
        return "normal"

    # ---- reliability tracking ----

    def record_success(self):
        self.total_requests += 1
        self.successful_requests += 1

    def record_failure(self):
        self.total_requests += 1
        self.failed_requests += 1

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0  # optimistic default: no data yet, don't penalize
        return self.successful_requests / self.total_requests

    # ---- persistence ----

    def to_dict(self):
        return {
            "name": self.name,
            "gap_history": list(self.gap_history),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
        }

    @classmethod
    def from_dict(cls, data):
        obj = cls(data["name"])
        obj.gap_history = deque(data.get("gap_history", []), maxlen=ROLLING_WINDOW)
        obj.total_requests = data.get("total_requests", 0)
        obj.successful_requests = data.get("successful_requests", 0)
        obj.failed_requests = data.get("failed_requests", 0)
        return obj

    def __repr__(self):
        state = "calibrated" if self.is_calibrated else f"calibrating ({len(self.gap_history)}/{MIN_SAMPLES_FOR_BASELINE})"
        return f"<EndpointStats {self.name}: {state}, baseline={self.baseline_avg_gap_ms:.1f}ms, success_rate={self.success_rate:.2f}>"


class StatsStore:
    """Holds EndpointStats for all endpoints, and persists them to disk
    between runs (this is what makes the agent 'maintain state' across
    sessions, not just within a single run)."""

    def __init__(self, path="data/stats.json"):
        self.path = path
        self.endpoints = {}

    def get(self, name) -> EndpointStats:
        if name not in self.endpoints:
            self.endpoints[name] = EndpointStats(name)
        return self.endpoints[name]

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        data = {name: stats.to_dict() for name, stats in self.endpoints.items()}
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, "r") as f:
            data = json.load(f)
        for name, stats_dict in data.items():
            self.endpoints[name] = EndpointStats.from_dict(stats_dict)


if __name__ == "__main__":
    # Quick manual test of the calibration -> classification behavior
    # IMPORTANT: classify BEFORE recording, so the current gap doesn't
    # dilute its own baseline comparison.
    stats = EndpointStats("fast")
    for i in range(20):
        gap = 40 if i != 15 else 1200  # inject one artificial spike
        classification = stats.classify_gap(gap)
        stats.record_gap(gap)
        print(f"token {i+1}: gap={gap}ms -> {classification}  {stats}")