"""LeCaR-style Adaptive Cache Replacement Policy for SMARTCACHE AI.

Combines LRU and LFU experts and uses dual shadow (ghost) caches
to perform regret-based Exponential Gradient (Exp3/Hedge) updates.
"""
from collections import OrderedDict
import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from policies.base import BasePolicy
from policies.lru import LRUPolicy
from policies.lfu import LFUPolicy


class LeCaRAdaptivePolicy(BasePolicy):
    """Adaptive cache replacement policy using LeCaR online learning.

    Maintains dual ghost caches (H_LRU, H_LFU) to attribute eviction mistakes.
    When a missed request hits an expert's ghost cache, that expert's weight is
    penalized using exponential gradient descent.
    """

    def __init__(
        self,
        capacity: int = 10,
        learning_rate: Optional[float] = None,
        discount: float = 1.0,
        history_multiplier: float = 1.0,
        epsilon_min: float = 0.005,
        deterministic: bool = False,
        seed: Optional[int] = 42,
        name: str = "SmartCache (LeCaR)",
    ) -> None:
        super().__init__(name=name)
        self.capacity = max(1, capacity)
        # Adaptive learning rate sqrt(ln(2) / capacity) if not explicitly set
        self.learning_rate = learning_rate if learning_rate is not None else min(0.45, max(0.05, math.sqrt(math.log(2.0) / self.capacity)))
        self.discount = discount
        self.ghost_capacity = max(1, int(self.capacity * history_multiplier))
        self.epsilon_min = epsilon_min
        self.deterministic = deterministic
        self.rng = random.Random(seed)

        # Experts
        self.lru_expert = LRUPolicy(name="LRU_Expert")
        self.lfu_expert = LFUPolicy(name="LFU_Expert")

        # Ghost history caches: item -> eviction_step
        self.h_lru: OrderedDict[Any, int] = OrderedDict()
        self.h_lfu: OrderedDict[Any, int] = OrderedDict()

        # Expert weights (w_lru, w_lfu), normalized to sum to 1.0
        self.w_lru: float = 0.5
        self.w_lfu: float = 0.5

        # Tracking variables
        self.current_step: int = 0
        self.last_chosen_expert: Optional[str] = None
        self.ghost_hits_lru: int = 0
        self.ghost_hits_lfu: int = 0
        
        # Telemetry history for dashboard: (step, w_lru, w_lfu)
        self.weight_history: List[Tuple[int, float, float]] = [(0, 0.5, 0.5)]

    def choose_victim(self, cache_items: Set[Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Select a victim by consulting LRU and LFU experts and sampling by weights."""
        v_lru = self.lru_expert.choose_victim(cache_items, context)
        v_lfu = self.lfu_expert.choose_victim(cache_items, context)

        if v_lru == v_lfu:
            # Both experts agree
            self.last_chosen_expert = "BOTH"
            return v_lru

        # Experts disagree: sample based on weights or choose argmax
        if self.deterministic:
            if self.w_lru >= self.w_lfu:
                chosen = "LRU"
                victim = v_lru
            else:
                chosen = "LFU"
                victim = v_lfu
        else:
            if self.rng.random() < self.w_lru:
                chosen = "LRU"
                victim = v_lru
            else:
                chosen = "LFU"
                victim = v_lfu

        self.last_chosen_expert = chosen
        return victim

    def on_hit(self, item: Any, context: Optional[Dict[str, Any]] = None) -> None:
        """Forward hit to both experts."""
        self.current_step += 1
        self.lru_expert.on_hit(item, context)
        self.lfu_expert.on_hit(item, context)
        self.weight_history.append((self.current_step, self.w_lru, self.w_lfu))

    def on_miss(self, item: Any, evicted_item: Optional[Any], context: Optional[Dict[str, Any]] = None) -> None:
        """Handle miss: check ghost caches for regret attribution, update weights, then insert."""
        self.current_step += 1
        updated = False

        # 1. Regret attribution from ghost caches
        if item in self.h_lru:
            # Regret: LRU evicted an item that was requested again
            evict_step = self.h_lru.pop(item)
            delta_t = max(0, self.current_step - evict_step)
            decay = self.discount ** delta_t if self.discount < 1.0 else 1.0
            self.w_lru *= math.exp(-self.learning_rate * decay)
            self.ghost_hits_lru += 1
            updated = True

        if item in self.h_lfu:
            # Regret: LFU evicted an item that was requested again
            evict_step = self.h_lfu.pop(item)
            delta_t = max(0, self.current_step - evict_step)
            decay = self.discount ** delta_t if self.discount < 1.0 else 1.0
            self.w_lfu *= math.exp(-self.learning_rate * decay)
            self.ghost_hits_lfu += 1
            updated = True

        # 2. Normalize weights with numerical stability safeguards
        if updated:
            self._normalize_weights()

        # 3. Inform both experts
        self.lru_expert.on_miss(item, evicted_item, context)
        self.lfu_expert.on_miss(item, evicted_item, context)

        # 4. Record evicted item in appropriate ghost cache
        if evicted_item is not None:
            if self.last_chosen_expert == "LRU":
                self._add_to_ghost(self.h_lru, evicted_item)
            elif self.last_chosen_expert == "LFU":
                self._add_to_ghost(self.h_lfu, evicted_item)
            elif self.last_chosen_expert == "BOTH":
                self._add_to_ghost(self.h_lru, evicted_item)
                self._add_to_ghost(self.h_lfu, evicted_item)

        self.weight_history.append((self.current_step, self.w_lru, self.w_lfu))

    def _add_to_ghost(self, ghost_dict: OrderedDict[Any, int], item: Any) -> None:
        """Add an item to a ghost cache, maintaining bounded capacity."""
        if item in ghost_dict:
            ghost_dict.move_to_end(item)
        ghost_dict[item] = self.current_step
        while len(ghost_dict) > self.ghost_capacity:
            ghost_dict.popitem(last=False)

    def _normalize_weights(self) -> None:
        """Enforce weight bounds [epsilon_min, 1 - epsilon_min] and sum-to-1 normalization."""
        # Floor weights to prevent numerical extinction
        self.w_lru = max(self.w_lru, self.epsilon_min)
        self.w_lfu = max(self.w_lfu, self.epsilon_min)
        
        total = self.w_lru + self.w_lfu
        if math.isnan(total) or total <= 0:
            self.w_lru = 0.5
            self.w_lfu = 0.5
        else:
            self.w_lru = self.w_lru / total
            self.w_lfu = self.w_lfu / total

    def reset(self) -> None:
        """Reset policy to initial state."""
        self.lru_expert.reset()
        self.lfu_expert.reset()
        self.h_lru.clear()
        self.h_lfu.clear()
        self.w_lru = 0.5
        self.w_lfu = 0.5
        self.current_step = 0
        self.last_chosen_expert = None
        self.ghost_hits_lru = 0
        self.ghost_hits_lfu = 0
        self.weight_history = [(0, 0.5, 0.5)]

    def get_state(self) -> Dict[str, Any]:
        """Return diagnostic telemetry."""
        return {
            "w_lru": round(self.w_lru, 4),
            "w_lfu": round(self.w_lfu, 4),
            "h_lru_size": len(self.h_lru),
            "h_lfu_size": len(self.h_lfu),
            "ghost_hits_lru": self.ghost_hits_lru,
            "ghost_hits_lfu": self.ghost_hits_lfu,
            "last_chosen_expert": self.last_chosen_expert,
        }
