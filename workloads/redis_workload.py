"""Realistic production-grade Redis workload generators.

Simulates real-world Redis usage profiles:
  1. E-Commerce Flash Sale: Power-law product catalog + session tokens + flash-sale traffic surge.
  2. Batch Scan Attack (Cache Pollution): Hot items polluted by sequential background crawler scans.
  3. Web Session & API Gateway: Ephemeral sessions + user profiles + cached API responses.
  4. Day/Night Global Phase Shift: Changing regional access patterns across time windows.
"""
import random
from typing import Any, Dict, List, Optional
import numpy as np


def generate_redis_ecommerce_workload(
    total_requests: int = 1000,
    catalog_size: int = 200,
    zipf_alpha: float = 1.15,
    session_pool_size: int = 50,
    flash_sale_start: int = 400,
    flash_sale_duration: int = 200,
    flash_sale_items: int = 5,
    seed: int = 42,
) -> List[str]:
    """Generate realistic e-commerce Redis access trace.

    Mixes:
      - product:{id} reads following Zipf distribution (80/20 rule)
      - session:{token} reads/writes with high recency
      - A temporary flash sale event concentrating 70% of traffic onto a tiny item subset
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Zipf weights for product catalog
    ranks = np.arange(1, catalog_size + 1)
    weights = 1.0 / (ranks ** zipf_alpha)
    probs = weights / np.sum(weights)

    trace: List[str] = []
    flash_items = [f"flash_sale:item_{i}" for i in range(flash_sale_items)]

    for step in range(total_requests):
        # Check if currently inside flash sale window
        is_flash_sale = flash_sale_start <= step < (flash_sale_start + flash_sale_duration)

        r = rng.random()
        if is_flash_sale and r < 0.65:
            # 65% of traffic hits flash sale items
            trace.append(rng.choice(flash_items))
        elif r < 0.60:
            # 60% of normal traffic reads catalog product
            idx = int(np_rng.choice(np.arange(catalog_size), p=probs))
            trace.append(f"product:{idx:04d}")
        elif r < 0.85:
            # 25% of traffic is session management
            session_id = rng.randint(0, session_pool_size - 1)
            trace.append(f"session:user_{session_id:03d}")
        else:
            # 15% user cart / checkout cache
            user_id = rng.randint(0, session_pool_size - 1)
            trace.append(f"cart:user_{user_id:03d}")

    return trace


def generate_redis_scan_pollution_workload(
    total_requests: int = 1000,
    hot_set_size: int = 20,
    scan_start: int = 350,
    scan_length: int = 300,
    seed: int = 42,
) -> List[str]:
    """Generate cache pollution trace typical of Redis ETL/reporting jobs.

    Phase 1 (Pre-scan): High-frequency access to a hot working set (e.g. top products / user sessions).
    Phase 2 (Scan Burst): Background analytics worker performs a sequential scan of unique keys.
    Phase 3 (Post-scan): Traffic resumes on original hot set.
    """
    rng = random.Random(seed)
    hot_items = [f"product:hot_{i:02d}" for i in range(hot_set_size)]
    trace: List[str] = []

    for step in range(total_requests):
        if scan_start <= step < (scan_start + scan_length):
            # Scan item: unique key that will never be requested again
            scan_id = step - scan_start
            trace.append(f"batch:scan:export_{scan_id:04d}")
        else:
            # Hot set access (90% hot items, 10% random noise)
            if rng.random() < 0.90:
                trace.append(rng.choice(hot_items))
            else:
                noise_id = rng.randint(100, 500)
                trace.append(f"misc:item_{noise_id:04d}")

    return trace


def generate_redis_api_gateway_workload(
    total_requests: int = 1000,
    num_endpoints: int = 30,
    num_users: int = 100,
    zipf_alpha: float = 1.1,
    seed: int = 42,
) -> List[str]:
    """Generate API gateway Redis cache trace.

    Mix of:
      - Cached API response payloads: api:response:{route_id}
      - User profile lookups: user:{id}:profile
      - Token validation: auth:token:{id}
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    ranks = np.arange(1, num_endpoints + 1)
    weights = 1.0 / (ranks ** zipf_alpha)
    route_probs = weights / np.sum(weights)

    trace: List[str] = []
    for _ in range(total_requests):
        r = rng.random()
        if r < 0.45:
            route_id = int(np_rng.choice(np.arange(num_endpoints), p=route_probs))
            trace.append(f"api:route_{route_id:02d}")
        elif r < 0.75:
            user_id = rng.randint(0, num_users - 1)
            trace.append(f"user:{user_id:03d}:profile")
        else:
            token_id = rng.randint(0, 25)
            trace.append(f"auth:token_{token_id:02d}")

    return trace


def generate_redis_phase_shift_workload(
    phase_length: int = 300,
    seed: int = 42,
) -> List[str]:
    """Simulate global diurnal (day/night) shift in Redis key access patterns."""
    rng = random.Random(seed)
    trace: List[str] = []

    # Phase 1: Day traffic (e-commerce business items 0..15)
    day_keys = [f"product:electronics_{i:02d}" for i in range(15)]
    for _ in range(phase_length):
        trace.append(rng.choice(day_keys))

    # Phase 2: Evening shift (entertainment / streaming media 0..15)
    evening_keys = [f"stream:media_{i:02d}" for i in range(15)]
    for _ in range(phase_length):
        trace.append(rng.choice(evening_keys))

    # Phase 3: Late night batch reporting
    for i in range(phase_length):
        trace.append(f"analytics:nightly_run_{i:03d}")

    return trace


def get_redis_workload(
    scenario: str,
    total_requests: int = 1000,
    seed: int = 42,
) -> List[str]:
    """Retrieve realistic Redis production trace by scenario name."""
    if scenario == "E-Commerce Flash Sale & Catalog":
        return generate_redis_ecommerce_workload(
            total_requests=total_requests,
            flash_sale_start=int(total_requests * 0.4),
            flash_sale_duration=int(total_requests * 0.25),
            seed=seed,
        )
    elif scenario == "Batch Scan Attack (Cache Pollution)":
        return generate_redis_scan_pollution_workload(
            total_requests=total_requests,
            scan_start=int(total_requests * 0.35),
            scan_length=int(total_requests * 0.30),
            seed=seed,
        )
    elif scenario == "API Gateway & Session Cache":
        return generate_redis_api_gateway_workload(
            total_requests=total_requests,
            seed=seed,
        )
    elif scenario == "Day/Night Traffic Phase Shift":
        phase_len = max(50, total_requests // 3)
        return generate_redis_phase_shift_workload(phase_length=phase_len, seed=seed)[:total_requests]
    else:
        return generate_redis_ecommerce_workload(total_requests=total_requests, seed=seed)
