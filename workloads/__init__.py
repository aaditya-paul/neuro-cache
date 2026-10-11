"""Workload generator and loader utilities for SMARTCACHE AI."""
from workloads.generators import (
    generate_looping_trace,
    generate_sequential_scan_trace,
    generate_zipf_trace,
    generate_phase_shift_trace,
    generate_mixed_trace,
)
from workloads.redis_workload import (
    generate_redis_ecommerce_workload,
    generate_redis_scan_pollution_workload,
    generate_redis_api_gateway_workload,
    generate_redis_phase_shift_workload,
    get_redis_workload,
)

__all__ = [
    "generate_looping_trace",
    "generate_sequential_scan_trace",
    "generate_zipf_trace",
    "generate_phase_shift_trace",
    "generate_mixed_trace",
    "generate_redis_ecommerce_workload",
    "generate_redis_scan_pollution_workload",
    "generate_redis_api_gateway_workload",
    "generate_redis_phase_shift_workload",
    "get_redis_workload",
]
