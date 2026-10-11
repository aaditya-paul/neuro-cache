"""Unit tests and regression verification for SmartCache Assembly Engine."""

import math
import pytest
from smartcache_asm.smartcache_ctypes import (
    SmartCacheAssemblyPolicy,
    SmartCacheNativeSimulator,
    get_smartcache_dll,
    SmartCacheState,
)
import ctypes


def test_asm_initial_weights():
    """Assembly engine weights must initialize to 0.5."""
    policy = SmartCacheAssemblyPolicy(capacity=10)
    state = policy.get_state()
    assert math.isclose(state["w_lru"], 0.5, rel_tol=1e-5)
    assert math.isclose(state["w_lfu"], 0.5, rel_tol=1e-5)


def test_asm_weight_normalization_and_invariants():
    """Weights must remain finite, normalized, and strictly sum to 1.0."""
    policy = SmartCacheAssemblyPolicy(capacity=5, seed=123)
    trace = [1, 2, 3, 4, 5, 6, 1, 2, 7, 8, 3, 4, 9, 10, 1, 2]

    for item in trace:
        policy.on_miss(item, evicted_item=None)

    state = policy.get_state()
    assert state["w_lru"] >= 0.005
    assert state["w_lfu"] >= 0.005
    assert math.isclose(state["w_lru"] + state["w_lfu"], 1.0, rel_tol=1e-3)


def test_asm_ghost_bounded_capacity():
    """Ghost caches in assembly must not exceed their configured capacity."""
    capacity = 4
    policy = SmartCacheAssemblyPolicy(capacity=capacity)

    for item in range(100):
        policy.on_miss(item, evicted_item=None)

    assert policy.state.ghost_lru_count <= capacity
    assert policy.state.ghost_lfu_count <= capacity


def test_asm_hit_purges_ghost():
    """A cache hit on an item must purge it from both ghost caches in assembly."""
    dll = get_smartcache_dll()
    state = SmartCacheState()
    dll.smartcache_init(ctypes.byref(state), 5, ctypes.c_double(0.2), ctypes.c_double(0.35), ctypes.c_double(0.2), ctypes.c_double(5.0), ctypes.c_double(15.0), ctypes.c_uint64(42))

    # Artificially insert item 99 into ghost_lru
    state.ghost_lru[0].key = 99
    state.ghost_lru[0].evict_step = 1
    state.ghost_lru[0].is_valid = 1
    state.ghost_lru_count = 1

    # First access 99 (cold miss, inserts into cache)
    is_hit = ctypes.c_int32()
    evicted = ctypes.c_int64()
    expert = ctypes.c_int32()
    dll.smartcache_access(ctypes.byref(state), 99, ctypes.byref(is_hit), ctypes.byref(evicted), ctypes.byref(expert))

    # Second access 99 (cache hit)
    dll.smartcache_access(ctypes.byref(state), 99, ctypes.byref(is_hit), ctypes.byref(evicted), ctypes.byref(expert))
    assert is_hit.value == 1

    # Invariant: ghost entry must be purged (is_valid == 0)
    assert state.ghost_lru[0].is_valid == 0


def test_asm_workload_shift_detection():
    """Sequential scan burst activates is_shift_active in assembly state."""
    policy = SmartCacheAssemblyPolicy(capacity=5, shift_threshold=0.20)

    # Warm up with high hit rate
    for _ in range(10):
        for item in [1, 2, 3]:
            policy.on_hit(item)

    # Feed sequential scan (100% misses)
    for scan_item in range(500, 530):
        policy.on_miss(scan_item, evicted_item=None)

    state = policy.get_state()
    assert state["is_shift_active"] is True


def test_asm_native_simulator_run_trace():
    """Native simulator executes full trace and returns accurate metrics."""
    sim = SmartCacheNativeSimulator(capacity=10, seed=42)
    trace = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5] * 20
    results = sim.run_trace(trace)

    assert results["requests"] == 200
    assert results["hits"] > 0
    assert results["misses"] > 0
    assert results["hit_rate"] > 0.80
    assert results["hardware_rdtsc_ns"] > 0
    # Verified: sub-microsecond decision latency!
    assert results["avg_decision_us"] < 1.0
