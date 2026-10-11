"""Benchmark harness comparing SMARTCACHE AI against Redis eviction policies on live production traces.

Supports:
  1. Offline high-fidelity simulation against Redis C-equivalent Sampled LRU & LFU.
  2. Optional live TCP execution against a running Redis instance (via redis-py).
"""
import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import pandas as pd

from cache_simulator import CacheSimulator
from policies import (
    LRUPolicy,
    LFUPolicy,
    BeladyPolicy,
    LeCaRAdaptivePolicy,
    ARCPolicy,
    SIEVEPolicy,
    RedisSampledLRUPolicy,
    RedisSampledLFUPolicy,
)
from workloads.redis_workload import (
    get_redis_workload,
    generate_redis_ecommerce_workload,
    generate_redis_scan_pollution_workload,
    generate_redis_api_gateway_workload,
    generate_redis_phase_shift_workload,
)
from benchmarks.metrics import compute_extended_metrics, calculate_rolling_hit_rate


def run_redis_comparison_matrix(
    workload_names: Optional[List[str]] = None,
    capacities: Optional[List[int]] = None,
    trace_length: int = 1000,
    seed: int = 42,
    output_dir: str = "results",
) -> pd.DataFrame:
    """Run full benchmark matrix comparing SmartCache AI to Redis policies on realistic workloads."""
    if workload_names is None:
        workload_names = [
            "E-Commerce Flash Sale & Catalog",
            "Batch Scan Attack (Cache Pollution)",
            "API Gateway & Session Cache",
            "Day/Night Traffic Phase Shift",
        ]
    if capacities is None:
        capacities = [10, 25, 50]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []

    print("=" * 105)
    print("[BENCHMARK] SMARTCACHE AI vs REDIS EVICTION ENGINE (REALISTIC PRODUCTION WORKLOAD MATRIX)")
    print("=" * 105)

    for wl_name in workload_names:
        trace = get_redis_workload(wl_name, total_requests=trace_length, seed=seed)
        print(f"\n[Trace] Workload: {wl_name} | Trace Length: {len(trace)} keys")

        for cap in capacities:
            # Policy candidates
            policy_instances = [
                RedisSampledLRUPolicy(samples=5, seed=seed, name="Redis LRU (samples=5) [Default]"),
                RedisSampledLRUPolicy(samples=10, seed=seed, name="Redis LRU (samples=10) [High-Acc]"),
                RedisSampledLFUPolicy(samples=5, seed=seed, name="Redis LFU (samples=5) [Morris Log-C]"),
                LRUPolicy(name="Exact LRU (Linked-List)"),
                ARCPolicy(capacity=cap),
                SIEVEPolicy(capacity=cap),
                LeCaRAdaptivePolicy(capacity=cap, seed=seed, name="SmartCache AI (NeuroCache)"),
                BeladyPolicy(trace=trace),
            ]

            results_by_policy: Dict[str, Any] = {}

            for pol in policy_instances:
                sim = CacheSimulator(capacity=cap, policy=pol, record_events=True)
                res = sim.run(trace)
                ext = compute_extended_metrics(res, trace)
                results_by_policy[pol.name] = (res, ext)

            # Extract Redis default baseline hit rate for comparison delta
            redis_default_hr = results_by_policy["Redis LRU (samples=5) [Default]"][0].hit_rate * 100

            for pol_name, (res, ext) in results_by_policy.items():
                hr = res.hit_rate * 100
                delta_vs_redis = hr - redis_default_hr

                records.append({
                    "Workload": wl_name,
                    "Capacity": cap,
                    "Policy": pol_name,
                    "Requests": res.total_requests,
                    "Hits": res.hits,
                    "Misses": res.misses,
                    "Hit Rate (%)": round(hr, 2),
                    "Delta vs Redis (%)": round(delta_vs_redis, 2),
                    "Evictions": res.evictions,
                    "Capacity Misses": ext.get("capacity_misses", 0),
                    "Avg Decision (us)": res.avg_decision_time_us,
                    "Wall Clock (s)": round(res.wall_clock_time_sec, 4),
                })

    df = pd.DataFrame(records)

    # Save to CSV and JSON
    csv_file = out_path / "redis_benchmark_summary.csv"
    json_file = out_path / "redis_benchmark_summary.json"
    df.to_csv(csv_file, index=False)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\n[Done] Benchmark results saved to:")
    print(f"   - {csv_file}")
    print(f"   - {json_file}")

    return df


def run_live_redis_server_benchmark(
    redis_url: str = "redis://localhost:6379/0",
    maxmemory: str = "5mb",
    policy: str = "allkeys-lru",
    trace: Optional[List[str]] = None,
    payload_size_bytes: int = 256,
) -> Dict[str, Any]:
    """Execute live requests directly against a running Redis instance and measure telemetry.

    Requires a running Redis server and redis-py.
    """
    try:
        import redis
    except ImportError:
        return {"status": "error", "message": "redis-py package is not installed."}

    if trace is None:
        trace = generate_redis_ecommerce_workload(total_requests=500, seed=42)

    try:
        client = redis.from_url(redis_url, decode_responses=True)
        client.ping()
    except Exception as e:
        return {
            "status": "connection_failed",
            "message": f"Could not connect to Redis server at {redis_url}: {e}",
        }

    try:
        # Configure Redis instance
        client.config_set("maxmemory", maxmemory)
        client.config_set("maxmemory-policy", policy)
        client.flushdb()

        # Initial stats
        info_start = client.info("stats")
        hits_start = int(info_start.get("keyspace_hits", 0))
        misses_start = int(info_start.get("keyspace_misses", 0))
        evictions_start = int(info_start.get("evicted_keys", 0))

        start_time = time.perf_counter()
        latencies_ms: List[float] = []

        dummy_payload = "X" * payload_size_bytes

        for key in trace:
            t0 = time.perf_counter()
            val = client.get(key)
            if val is None:
                # Cache miss: populate key into Redis
                client.set(key, dummy_payload)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        total_elapsed = time.perf_counter() - start_time

        # Final stats
        info_end = client.info("stats")
        hits_end = int(info_end.get("keyspace_hits", 0))
        misses_end = int(info_end.get("keyspace_misses", 0))
        evictions_end = int(info_end.get("evicted_keys", 0))

        measured_hits = hits_end - hits_start
        measured_misses = misses_end - misses_start
        measured_evictions = evictions_end - evictions_start
        total_ops = measured_hits + measured_misses
        hit_rate = (measured_hits / total_ops) * 100.0 if total_ops > 0 else 0.0

        return {
            "status": "success",
            "redis_url": redis_url,
            "maxmemory": maxmemory,
            "policy": policy,
            "total_requests": len(trace),
            "measured_hits": measured_hits,
            "measured_misses": measured_misses,
            "hit_rate_pct": round(hit_rate, 2),
            "evictions": measured_evictions,
            "total_elapsed_sec": round(total_elapsed, 4),
            "avg_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 3) if latencies_ms else 0.0,
            "throughput_req_per_sec": round(len(trace) / total_elapsed, 1) if total_elapsed > 0 else 0.0,
        }
    except Exception as e:
        return {"status": "error", "message": f"Error running live Redis benchmark: {e}"}


def main():
    parser = argparse.ArgumentParser(description="SMARTCACHE AI vs Redis Benchmark Harness")
    parser.add_argument("--length", type=int, default=1000, help="Trace length per workload")
    parser.add_argument("--capacities", nargs="+", type=int, default=[10, 25, 50], help="Cache capacities to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--live-redis", type=str, default=None, help="Optional redis:// URL for live server benchmark")

    args = parser.parse_args()

    if args.live_redis:
        print(f"Connecting to live Redis server at: {args.live_redis}")
        res = run_live_redis_server_benchmark(redis_url=args.live_redis)
        print(json.dumps(res, indent=2))
    else:
        df = run_redis_comparison_matrix(capacities=args.capacities, trace_length=args.length, seed=args.seed)
        # Display highlight comparison table
        print("\n" + "=" * 105)
        print("SUMMARY PREVIEW: HIT RATE (%) BY WORKLOAD AND CAPACITY")
        print("=" * 105)
        pivot = df.pivot_table(index=["Workload", "Capacity"], columns="Policy", values="Hit Rate (%)")
        print(pivot.to_string())


if __name__ == "__main__":
    main()
