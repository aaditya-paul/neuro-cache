"""Policies package for SMARTCACHE AI."""
from policies.base import BasePolicy
from policies.lru import LRUPolicy
from policies.lfu import LFUPolicy
from policies.belady import BeladyPolicy
from policies.adaptive import LeCaRAdaptivePolicy
from policies.arc import ARCPolicy
from policies.sieve import SIEVEPolicy
from policies.classic_lecar import ClassicLeCaRPolicy
from policies.redis_policy import RedisSampledLRUPolicy, RedisSampledLFUPolicy

__all__ = [
    "BasePolicy",
    "LRUPolicy",
    "LFUPolicy",
    "BeladyPolicy",
    "LeCaRAdaptivePolicy",
    "ARCPolicy",
    "SIEVEPolicy",
    "ClassicLeCaRPolicy",
    "RedisSampledLRUPolicy",
    "RedisSampledLFUPolicy",
]
