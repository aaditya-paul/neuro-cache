"""Tests for Redis-faithful eviction policies and Redis workload generator."""
import pytest
from policies.redis_policy import RedisSampledLRUPolicy, RedisSampledLFUPolicy
from cache_simulator import CacheSimulator
from workloads.redis_workload import (
    generate_redis_ecommerce_workload,
    generate_redis_scan_pollution_workload,
    generate_redis_api_gateway_workload,
    get_redis_workload,
)


def test_redis_sampled_lru_basic():
    """Verify Redis sampled LRU evicts correctly within capacity."""
    policy = RedisSampledLRUPolicy(samples=5, seed=42)
    sim = CacheSimulator(capacity=3, policy=policy)
    # Stream: 1, 2, 3, 4 (should evict something when 4 arrives)
    trace = [1, 2, 3, 4]
    result = sim.run(trace)
    assert result.total_requests == 4
    assert result.misses == 4
    assert result.evictions == 1
    assert len(sim.cache) == 3
    assert 4 in sim.cache


def test_redis_sampled_lru_hit_behavior():
    """Verify hits update the clock and prevent immediate eviction."""
    policy = RedisSampledLRUPolicy(samples=5, seed=42)
    sim = CacheSimulator(capacity=2, policy=policy)
    # 1, 2, 1 (hit on 1), 3 (should evict 2 since 1 was refreshed)
    trace = [1, 2, 1, 3]
    result = sim.run(trace)
    assert result.hits == 1
    assert result.misses == 3
    assert 1 in sim.cache
    assert 3 in sim.cache
    assert 2 not in sim.cache


def test_redis_sampled_lfu_basic():
    """Verify Redis sampled LFU tracks frequencies via Morris counter."""
    policy = RedisSampledLFUPolicy(samples=5, seed=42)
    sim = CacheSimulator(capacity=3, policy=policy)
    # Access item 1 repeatedly to build up Morris counter
    trace = [1, 1, 1, 1, 2, 3, 4]
    result = sim.run(trace)
    assert 1 in sim.cache
    assert 4 in sim.cache
    assert len(sim.cache) == 3


def test_redis_workloads_deterministic():
    """Verify Redis workload generator reproducibility with seed."""
    w1 = generate_redis_ecommerce_workload(total_requests=100, seed=123)
    w2 = generate_redis_ecommerce_workload(total_requests=100, seed=123)
    assert w1 == w2
    assert len(w1) == 100
    assert any(k.startswith("product:") for k in w1)
    assert any(k.startswith("session:") for k in w1)


def test_redis_scan_pollution_workload():
    """Verify scan pollution trace produces hot set and scan burst."""
    w = generate_redis_scan_pollution_workload(total_requests=200, scan_start=50, scan_length=50, seed=42)
    assert len(w) == 200
    assert w[0].startswith("product:hot_")
    assert w[60].startswith("batch:scan:export_")


def test_get_redis_workload_dispatch():
    """Verify scenario dispatcher."""
    for sc in [
        "E-Commerce Flash Sale & Catalog",
        "Batch Scan Attack (Cache Pollution)",
        "API Gateway & Session Cache",
        "Day/Night Traffic Phase Shift",
    ]:
        trace = get_redis_workload(sc, total_requests=150, seed=42)
        assert len(trace) >= 140
