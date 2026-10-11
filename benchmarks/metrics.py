"""Evaluation metrics calculation for SMARTCACHE AI."""
from typing import Any, Dict, List, Set
import numpy as np

from cache_simulator import SimulationResult, SimulationEvent


def compute_extended_metrics(result: SimulationResult, trace: List[Any]) -> Dict[str, Any]:
    """Compute detailed evaluation metrics including compulsory vs capacity misses."""
    seen_items: Set[Any] = set()
    compulsory_misses = 0

    for item in trace:
        if item not in seen_items:
            compulsory_misses += 1
            seen_items.add(item)

    capacity_misses = max(0, result.misses - compulsory_misses)

    summary = result.to_dict()
    summary["compulsory_misses"] = compulsory_misses
    summary["capacity_misses"] = capacity_misses
    summary["compulsory_miss_rate"] = round(compulsory_misses / result.total_requests, 4) if result.total_requests > 0 else 0.0
    summary["capacity_miss_rate"] = round(capacity_misses / result.total_requests, 4) if result.total_requests > 0 else 0.0

    return summary


def calculate_rolling_hit_rate(events: List[SimulationEvent], window_size: int = 50) -> List[float]:
    """Calculate rolling window hit rate across the timeline of events."""
    if not events:
        return []

    hits = np.array([1.0 if ev.is_hit else 0.0 for ev in events])
    if len(hits) < window_size:
        return np.cumsum(hits) / np.arange(1, len(hits) + 1)

    kernel = np.ones(window_size) / window_size
    rolling = np.convolve(hits, kernel, mode="valid")
    # Prefix with expanding average up to window_size - 1
    prefix = np.cumsum(hits[: window_size - 1]) / np.arange(1, window_size)
    return np.concatenate([prefix, rolling]).tolist()
