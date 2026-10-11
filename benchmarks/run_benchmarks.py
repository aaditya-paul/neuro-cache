"""Reproducible benchmark runner for multi-policy evaluation across standard workloads."""
import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from cache_simulator import CacheSimulator
from policies import LRUPolicy, LFUPolicy, BeladyPolicy, LeCaRAdaptivePolicy
from workloads.generators import (
    generate_looping_trace,
    generate_sequential_scan_trace,
    generate_zipf_trace,
    generate_phase_shift_trace,
    generate_mixed_trace,
)
from benchmarks.metrics import compute_extended_metrics


def run_benchmark_suite(output_dir: str = "results", seed: int = 42) -> pd.DataFrame:
    """Run all policies across all standard workloads and save results."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Define standard workloads
    workloads: Dict[str, List[int]] = {
        "Looping (Hot Set)": generate_looping_trace(hot_set_size=10, repetitions=50, noise_ratio=0.05, seed=seed),
        "Sequential Scan": generate_sequential_scan_trace(scan_length=500, seed=seed),
        "Zipf Skew (alpha=1.0)": generate_zipf_trace(num_items=100, total_requests=1000, alpha=1.0, seed=seed),
        "Phase Shift (Shift)": generate_phase_shift_trace(seed=seed),
        "Mixed Workload": generate_mixed_trace(total_requests=1000, seed=seed),
    }

    # 2. Test Capacities
    capacities = [5, 10, 20]

    records: List[Dict[str, Any]] = []

    print("=" * 80)
    print("SMARTCACHE AI — MULTI-POLICY BENCHMARK SUITE")
    print("=" * 80)

    for workload_name, trace in workloads.items():
        print(f"\n--- Workload: {workload_name} (Length: {len(trace)}) ---")

        for cap in capacities:
            # Instantiate policies with matching parameters
            policies = [
                LRUPolicy(),
                LFUPolicy(),
                LeCaRAdaptivePolicy(capacity=cap, seed=seed),
                BeladyPolicy(trace=trace),
            ]

            for policy in policies:
                sim = CacheSimulator(capacity=cap, policy=policy, record_events=False)
                result = sim.run(trace)
                extended = compute_extended_metrics(result, trace)

                record = {
                    "Workload": workload_name,
                    "Capacity": cap,
                    "Policy": policy.name,
                    "Requests": result.total_requests,
                    "Hits": result.hits,
                    "Misses": result.misses,
                    "Hit Rate (%)": round(result.hit_rate * 100, 2),
                    "Evictions": result.evictions,
                    "Capacity Misses": extended["capacity_misses"],
                    "Avg Decision (us)": result.avg_decision_time_us,
                    "Wall Clock (s)": round(result.wall_clock_time_sec, 4),
                }
                records.append(record)

    df = pd.DataFrame(records)

    # Save outputs
    csv_file = out_path / "benchmark_summary.csv"
    json_file = out_path / "benchmark_summary.json"
    df.to_csv(csv_file, index=False)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print("\nBenchmark Suite Completed successfully!")
    print(f"Results saved to:\n  - {csv_file}\n  - {json_file}")
    
    # Print formatted pivot table
    pivot = df.pivot_table(
        index=["Workload", "Capacity"],
        columns="Policy",
        values="Hit Rate (%)",
    )
    print("\n--- Summary Hit Rate (%) Comparison ---")
    print(pivot.to_string())

    return df


if __name__ == "__main__":
    run_benchmark_suite()
