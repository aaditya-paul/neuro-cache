"""Unit tests for LeCaRAdaptivePolicy."""
import math
import pytest
from cache_simulator import CacheSimulator
from policies.adaptive import LeCaRAdaptivePolicy
from workloads.generators import generate_looping_trace


def test_initial_weights():
    """Weights must initialize to equal 0.5."""
    policy = LeCaRAdaptivePolicy(capacity=10)
    assert math.isclose(policy.w_lru, 0.5, rel_tol=1e-5)
    assert math.isclose(policy.w_lfu, 0.5, rel_tol=1e-5)


def test_weights_normalization_and_invariants():
    """Weights must remain finite, normalized, and strictly sum to 1.0."""
    policy = LeCaRAdaptivePolicy(capacity=5, seed=123)
    sim = CacheSimulator(capacity=5, policy=policy)
    trace = [1, 2, 3, 4, 5, 6, 1, 2, 7, 8, 3, 4, 9, 10, 1, 2]
    sim.run(trace)

    for _, w_lru, w_lfu in policy.weight_history:
        assert not math.isnan(w_lru)
        assert not math.isnan(w_lfu)
        assert w_lru >= 0.0
        assert w_lfu >= 0.0
        assert math.isclose(w_lru + w_lfu, 1.0, rel_tol=1e-4)


def test_ghost_cache_bounded_capacity():
    """Ghost caches must never exceed their configured capacity."""
    capacity = 4
    multiplier = 1.5
    policy = LeCaRAdaptivePolicy(capacity=capacity, history_multiplier=multiplier)
    sim = CacheSimulator(capacity=capacity, policy=policy)

    # Long sequential scan causes many evictions
    trace = list(range(100))
    sim.run(trace)

    max_ghost_cap = max(1, int(capacity * multiplier))
    assert len(policy.h_lru) <= max_ghost_cap
    assert len(policy.h_lfu) <= max_ghost_cap


def test_reproducibility_with_seed():
    """Same seed must produce exactly identical results and weight history."""
    trace = generate_looping_trace(hot_set_size=10, repetitions=20, noise_ratio=0.1, seed=42)

    p1 = LeCaRAdaptivePolicy(capacity=5, seed=999)
    res1 = CacheSimulator(capacity=5, policy=p1).run(trace)

    p2 = LeCaRAdaptivePolicy(capacity=5, seed=999)
    res2 = CacheSimulator(capacity=5, policy=p2).run(trace)

    assert res1.hits == res2.hits
    assert res1.misses == res2.misses
    assert res1.evictions == res2.evictions
    assert [w[1] for w in p1.weight_history] == [w[1] for w in p2.weight_history]
