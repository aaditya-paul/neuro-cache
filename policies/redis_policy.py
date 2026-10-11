"""Redis-faithful cache replacement policies approximating Redis C implementation.

Approximates Redis's eviction engine (src/eviction.c and src/lfu.c):
  - RedisSampledLRUPolicy: Samples N random keys (default maxmemory_samples=5),
    populates an eviction pool of size 16 sorted by idle time, and evicts the key with
    the largest idle time.
  - RedisSampledLFUPolicy: 8-bit Morris logarithmic counter (0-255, initial 5),
    probabilistic increments (1 / (baseval * log_factor + 1)), LDT time decay, and
    sampled eviction pool.
"""
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from policies.base import BasePolicy


class RedisSampledLRUPolicy(BasePolicy):
    """Approximates Redis allkeys-lru eviction policy.

    Redis does not maintain a true doubly linked list due to memory overhead (16-24 bytes/key).
    Instead, it records a 24-bit LRU clock on each object and uses sampled eviction:
      1. On eviction, samples `samples` (default 5) random keys from the key dictionary.
      2. Computes idle time for each key: idle = current_clock - key_access_clock.
      3. Inserts candidates into an eviction pool of size EVPOOL_SIZE (16) sorted by idle ascending.
      4. Selects the victim with the maximum idle time (tail of the pool).
    """

    def __init__(
        self,
        samples: int = 5,
        pool_size: int = 16,
        seed: Optional[int] = 42,
        name: str = "Redis Sampled LRU (samples=5)",
    ) -> None:
        super().__init__(name=name)
        self.samples = max(1, samples)
        self.pool_size = max(1, pool_size)
        self.seed = seed
        self.rng = random.Random(seed)

        self.clock: int = 0
        self.key_lru: Dict[Any, int] = {}
        # Eviction pool: list of (idle_time, item) sorted by idle_time ascending
        self.pool: List[Tuple[int, Any]] = []

    def choose_victim(self, cache_items: Set[Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Select victim using Redis's sampled eviction pool mechanism."""
        if not cache_items:
            raise KeyError("Cannot choose victim from empty cache")

        # 1. Clean pool of keys not currently in cache
        self.pool = [(idle, item) for idle, item in self.pool if item in cache_items]

        # 2. Sample keys from cache_items
        cached_list = list(cache_items)
        sample_size = min(self.samples, len(cached_list))
        sampled_keys = self.rng.sample(cached_list, sample_size)

        # 3. Populate pool with sampled keys
        existing_pool_items = {item for _, item in self.pool}
        for key in sampled_keys:
            idle_time = self.clock - self.key_lru.get(key, 0)
            if key in existing_pool_items:
                # Update existing idle time
                self.pool = [(idle_time if it == key else idle, it) for idle, it in self.pool]
            else:
                self.pool.append((idle_time, key))

        # Sort pool by idle time ascending (highest idle at the end)
        self.pool.sort(key=lambda x: x[0])

        # Trim pool to pool_size, discarding lowest idle (most recent) items from head
        if len(self.pool) > self.pool_size:
            self.pool = self.pool[-self.pool_size :]

        # 4. Pick candidate with largest idle time from pool tail
        for i in range(len(self.pool) - 1, -1, -1):
            idle, victim = self.pool[i]
            if victim in cache_items:
                self.pool.pop(i)
                return victim

        # Fallback if pool is empty or invalid
        return sampled_keys[0]

    def on_hit(self, item: Any, context: Optional[Dict[str, Any]] = None) -> None:
        """Update object access clock on hit."""
        self.clock += 1
        self.key_lru[item] = self.clock

    def on_miss(self, item: Any, evicted_item: Optional[Any], context: Optional[Dict[str, Any]] = None) -> None:
        """Record new object clock and remove evicted item from tracking."""
        self.clock += 1
        if evicted_item is not None:
            self.key_lru.pop(evicted_item, None)
            self.pool = [(idle, k) for idle, k in self.pool if k != evicted_item]
        self.key_lru[item] = self.clock

    def reset(self) -> None:
        """Reset internal clock and pool."""
        self.clock = 0
        self.key_lru.clear()
        self.pool.clear()
        self.rng = random.Random(self.seed)

    def get_state(self) -> Dict[str, Any]:
        """Return diagnostic state."""
        return {
            "clock": self.clock,
            "tracked_keys": len(self.key_lru),
            "pool_size": len(self.pool),
            "samples": self.samples,
        }


class RedisSampledLFUPolicy(BasePolicy):
    """Approximates Redis allkeys-lfu eviction policy.

    Redis 4.0+ implements LFU using an 8-bit Morris logarithmic counter and 16-bit LDT:
      - Counter starts at LFU_INIT_VAL (default 5).
      - On access, counter decays based on elapsed time:
          decays = elapsed_time / lfu_decay_time
          counter = max(0, counter - decays)
      - Probabilistic increment:
          baseval = max(0, counter - LFU_INIT_VAL)
          p = 1 / (baseval * lfu_log_factor + 1)
          if rand() < p: counter = min(255, counter + 1)
      - On eviction, samples `samples` keys into the eviction pool.
        Eviction score = 255 - counter.
        The key with the highest eviction score (lowest counter) is evicted.
    """

    LFU_INIT_VAL: int = 5

    def __init__(
        self,
        samples: int = 5,
        pool_size: int = 16,
        lfu_log_factor: float = 10.0,
        lfu_decay_interval: int = 50,
        seed: Optional[int] = 42,
        name: str = "Redis Sampled LFU (samples=5)",
    ) -> None:
        super().__init__(name=name)
        self.samples = max(1, samples)
        self.pool_size = max(1, pool_size)
        self.lfu_log_factor = max(0.1, lfu_log_factor)
        self.lfu_decay_interval = max(1, lfu_decay_interval)
        self.seed = seed
        self.rng = random.Random(seed)

        self.clock: int = 0
        # Maps item -> (counter: int 0..255, ldt: int)
        self.key_lfu: Dict[Any, Tuple[int, int]] = {}
        # Pool: list of (eviction_score, item) sorted ascending
        self.pool: List[Tuple[float, Any]] = []

    def _decay_and_increment(self, item: Any) -> int:
        """Apply Redis Morris logarithmic decay and probabilistic increment."""
        if item in self.key_lfu:
            counter, ldt = self.key_lfu[item]
            # 1. Decay
            elapsed = self.clock - ldt
            decays = elapsed // self.lfu_decay_interval
            if decays > 0:
                counter = max(0, counter - decays)
                ldt = self.clock
        else:
            counter = self.LFU_INIT_VAL
            ldt = self.clock

        # 2. Probabilistic Morris increment
        baseval = max(0, counter - self.LFU_INIT_VAL)
        p = 1.0 / (baseval * self.lfu_log_factor + 1.0)
        if self.rng.random() < p:
            counter = min(255, counter + 1)

        self.key_lfu[item] = (counter, self.clock)
        return counter

    def choose_victim(self, cache_items: Set[Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Select victim with lowest Morris frequency score via eviction pool."""
        if not cache_items:
            raise KeyError("Cannot choose victim from empty cache")

        # 1. Clean pool
        self.pool = [(score, item) for score, item in self.pool if item in cache_items]

        # 2. Sample keys
        cached_list = list(cache_items)
        sample_size = min(self.samples, len(cached_list))
        sampled_keys = self.rng.sample(cached_list, sample_size)

        # 3. Populate pool (eviction_score = 255 - counter + tie-breaker based on oldest LDT)
        existing_pool_items = {item for _, item in self.pool}
        for key in sampled_keys:
            if key in self.key_lfu:
                counter, ldt = self.key_lfu[key]
                # In Redis, decay is evaluated before eviction ranking
                elapsed = self.clock - ldt
                decays = elapsed // self.lfu_decay_interval
                effective_counter = max(0, counter - decays)
            else:
                effective_counter = self.LFU_INIT_VAL
                ldt = self.clock

            # Higher eviction score = lower counter (and older ldt)
            eviction_score = (255 - effective_counter) + (self.clock - ldt) * 0.0001
            if key in existing_pool_items:
                self.pool = [(eviction_score if it == key else sc, it) for sc, it in self.pool]
            else:
                self.pool.append((eviction_score, key))

        # Sort pool ascending (highest eviction score at tail)
        self.pool.sort(key=lambda x: x[0])

        if len(self.pool) > self.pool_size:
            self.pool = self.pool[-self.pool_size :]

        # 4. Evict tail item
        for i in range(len(self.pool) - 1, -1, -1):
            score, victim = self.pool[i]
            if victim in cache_items:
                self.pool.pop(i)
                return victim

        return sampled_keys[0]

    def on_hit(self, item: Any, context: Optional[Dict[str, Any]] = None) -> None:
        """Update LFU counter on hit."""
        self.clock += 1
        self._decay_and_increment(item)

    def on_miss(self, item: Any, evicted_item: Optional[Any], context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize LFU entry on miss and cleanup evicted item."""
        self.clock += 1
        if evicted_item is not None:
            self.key_lfu.pop(evicted_item, None)
            self.pool = [(score, k) for score, k in self.pool if k != evicted_item]
        self._decay_and_increment(item)

    def reset(self) -> None:
        """Reset state."""
        self.clock = 0
        self.key_lfu.clear()
        self.pool.clear()
        self.rng = random.Random(self.seed)

    def get_state(self) -> Dict[str, Any]:
        """Return diagnostic state."""
        return {
            "clock": self.clock,
            "tracked_keys": len(self.key_lfu),
            "pool_size": len(self.pool),
            "samples": self.samples,
        }
