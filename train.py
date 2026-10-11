"""Offline Training and Meta-Optimization Engine for SMARTCACHE AI.

Trains the meta-parameters and decay horizons of SMARTCACHE AI across a diverse
suite of synthetic workloads (Zipf popularity, sequential scan bursts,
non-stationary phase shifts, and looping hot sets) to maximize global hit rate.

Usage:
    python train.py [--epochs 25] [--trials 50] [--output models/trained_config.json]
"""
import argparse
from datetime import datetime
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from cache_simulator import CacheSimulator
from workloads.generators import (
    generate_zipf_trace,
    generate_mixed_trace,
    generate_phase_shift_trace,
    generate_looping_trace,
)
from policies.lru import LRUPolicy
from policies.lfu import LFUPolicy


class EvaluatorPolicy(LRUPolicy):
    """Parametric evaluation policy used during training optimization."""

    def __init__(self, capacity: int, halflife_mult: float, freq_cap: float):
        super().__init__(name="EvaluatorPolicy")
        self.capacity = max(1, capacity)
        self.halflife = max(2.0, self.capacity * halflife_mult)
        self.freq_cap = freq_cap
        self.lru = LRUPolicy()
        self.lfu = LFUPolicy()
        self.last_access: Dict[Any, int] = {}
        self.freq: Dict[Any, int] = {}
        self.step: int = 0

    def reset(self) -> None:
        self.lru.reset()
        self.lfu.reset()
        self.last_access.clear()
        self.freq.clear()
        self.step = 0

    def get_state(self) -> Dict[str, Any]:
        return {}

    def choose_victim(self, cache_items: set, context: Any = None) -> Any:
        v_lru = self.lru.choose_victim(cache_items, context)
        v_lfu = self.lfu.choose_victim(cache_items, context)
        if v_lru == v_lfu:
            return v_lru

        age_lru = self.step - self.last_access.get(v_lru, self.step)
        age_lfu = self.step - self.last_access.get(v_lfu, self.step)

        ef_lru = min(self.freq_cap, float(self.freq.get(v_lru, 1))) * math.exp(-0.693 * age_lru / self.halflife)
        ef_lfu = min(self.freq_cap, float(self.freq.get(v_lfu, 1))) * math.exp(-0.693 * age_lfu / self.halflife)

        return v_lfu if ef_lfu <= ef_lru else v_lru

    def on_hit(self, item: Any, context: Any = None) -> None:
        self.step += 1
        self.last_access[item] = self.step
        self.freq[item] = self.freq.get(item, 0) + 1
        self.lru.on_hit(item, context)
        self.lfu.on_hit(item, context)

    def on_miss(self, item: Any, evicted: Any, context: Any = None) -> None:
        self.step += 1
        self.last_access[item] = self.step
        self.freq[item] = 1
        if evicted is not None:
            self.freq.pop(evicted, None)
            self.last_access.pop(evicted, None)
        self.lru.on_miss(item, evicted, context)
        self.lfu.on_miss(item, evicted, context)


def generate_training_suite(seed: int = 42) -> List[Tuple[str, List[int], int]]:
    """Generate diverse training workloads across multiple capacities."""
    rng = random.Random(seed)
    suite = []

    # Capacity 5, 10, 20 workloads with varying skews and phases
    for cap in [5, 10, 20]:
        suite.append((
            f"Zipf-1.0 (C={cap})",
            generate_zipf_trace(alpha=1.0, total_requests=800, seed=seed),
            cap,
        ))
        suite.append((
            f"Zipf-0.8 (C={cap})",
            generate_zipf_trace(alpha=0.8, total_requests=800, seed=seed + 1),
            cap,
        ))
        suite.append((
            f"Mixed (C={cap})",
            generate_mixed_trace(total_requests=800, seed=seed + 2),
            cap,
        ))
        suite.append((
            f"PhaseShift (C={cap})",
            generate_phase_shift_trace(seed=seed + 3),
            cap,
        ))
        suite.append((
            f"Looping (C={cap})",
            generate_looping_trace(hot_set_size=max(8, cap), repetitions=40, seed=seed + 4),
            cap,
        ))

    return suite


def train_smartcache(
    epochs: int = 15,
    trials_per_epoch: int = 10,
    seed: int = 42,
    output_path: str = "models/trained_config.json",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute training search across hyperparameter space."""
    print("=" * 72)
    print("SMARTCACHE AI - Training & Parameter Optimization Engine")
    print("=" * 72)
    print(f"Epochs: {epochs} | Trials/Epoch: {trials_per_epoch} | Seed: {seed}")
    print(f"Output: {output_path}")
    print("-" * 72)

    training_suite = generate_training_suite(seed=seed)
    print(f"Generated training suite with {len(training_suite)} distinct benchmark tasks.")

    best_global_hr = -1.0
    best_config: Dict[str, Any] = {}
    history = []

    rng = np.random.default_rng(seed)

    # Grid & Random Search hybrid
    param_grid = [
        (hl, fc)
        for hl in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
        for fc in [5.0, 10.0, 15.0, 20.0, 30.0]
    ]
    random.Random(seed).shuffle(param_grid)

    start_time = time.time()
    trial_idx = 0

    for epoch in range(1, epochs + 1):
        epoch_best_hr = -1.0
        epoch_best_params = None

        for _ in range(trials_per_epoch):
            trial_idx += 1
            if trial_idx - 1 < len(param_grid):
                hl, fc = param_grid[trial_idx - 1]
            else:
                hl = float(rng.uniform(1.5, 7.0))
                fc = float(rng.uniform(5.0, 35.0))

            # Evaluate across training suite
            hrs = []
            for _, trace, cap in training_suite:
                policy = EvaluatorPolicy(capacity=cap, halflife_mult=hl, freq_cap=fc)
                sim = CacheSimulator(capacity=cap, policy=policy, record_events=False)
                res = sim.run(trace)
                hrs.append(res.hit_rate)

            mean_hr = float(np.mean(hrs))

            if mean_hr > epoch_best_hr:
                epoch_best_hr = mean_hr
                epoch_best_params = (hl, fc)

            if mean_hr > best_global_hr:
                best_global_hr = mean_hr
                best_config = {
                    "halflife_mult": round(hl, 2),
                    "freq_cap": round(fc, 1),
                    "mean_hit_rate": round(mean_hr, 4),
                    "epoch": epoch,
                }

        history.append({
            "epoch": epoch,
            "epoch_best_hr": round(epoch_best_hr, 4),
            "global_best_hr": round(best_global_hr, 4),
            "params": epoch_best_params,
        })

        if verbose:
            print(
                f"Epoch [{epoch:2d}/{epochs:2d}] | "
                f"Epoch Best HR: {epoch_best_hr * 100:5.2f}% | "
                f"Global Best HR: {best_global_hr * 100:5.2f}% | "
                f"Params: hl={epoch_best_params[0]:.2f}, fc={epoch_best_params[1]:.1f}"
            )

    elapsed = time.time() - start_time
    print("-" * 72)
    print(f"Training completed in {elapsed:.2f} seconds.")
    print(f"Optimal Configuration Found: {best_config}")
    print("=" * 72)

    # Save to JSON
    final_artifact = {
        "metadata": {
            "trained_at": datetime.now().isoformat(),
            "epochs": epochs,
            "seed": seed,
            "training_time_seconds": round(elapsed, 2),
        },
        "hyperparameters": {
            "halflife_mult": best_config["halflife_mult"],
            "freq_cap": best_config["freq_cap"],
            "learning_rate": 0.35,
            "history_multiplier": 2.0,
            "ttl_multiplier": 3.0,
            "damping_factor": 0.35,
            "shift_threshold": 0.20,
            "epsilon_min": 0.005,
        },
        "performance": {
            "mean_training_hit_rate": best_config["mean_hit_rate"],
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_artifact, f, indent=2)

    print(f"Saved trained model configuration to: {output_path}")
    return final_artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SMARTCACHE AI offline meta-parameters.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--trials", type=int, default=4, help="Trials per epoch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="models/trained_config.json", help="Path to save config")
    args = parser.parse_args()

    train_smartcache(
        epochs=args.epochs,
        trials_per_epoch=args.trials,
        seed=args.seed,
        output_path=args.output,
    )
