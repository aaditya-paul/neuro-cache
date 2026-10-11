# SMARTCACHE AI — Complete Function & Architecture Reference Guide
**Team BONGORITHM** • *“Learning what to keep before you need it.”*

This document provides a comprehensive, function-by-function reference for the entire **SMARTCACHE AI** codebase. For every class, method, helper, generator, and metric function, this guide explains:
1. **Signature & Types**: Exact parameters, default values, and return types.
2. **Why It's Used**: The problem it solves and its architectural role.
3. **How It Works**: The internal mechanics, algorithmic logic, and time/space complexity.
4. **Code Example**: Practical snippet showing how it is used.

---

## Table of Contents
1. [Core Policy Subsystem (`policies/`)](#1-core-policy-subsystem-policies)
   - [BasePolicy (`policies/base.py`)](#11-basepolicy-policiesbasepy)
   - [LRUPolicy (`policies/lru.py`)](#12-lrupolicy-policieslrupy)
   - [LFUPolicy (`policies/lfu.py`)](#13-lfupolicy-policieslfupy)
   - [BeladyPolicy (`policies/belady.py`)](#14-beladypolicy-policiesbeladypy)
   - [LeCaRAdaptivePolicy (`policies/adaptive.py`)](#15-lecaradaptivepolicy-policiesadaptivepy)
   - [ARCPolicy (`policies/arc.py`)](#16-arcpolicy-policiesarcpy)
   - [SIEVEPolicy (`policies/sieve.py`)](#17-sievepolicy-policiessievepy)
   - [ClassicLeCaRPolicy (`policies/classic_lecar.py`)](#18-classiclecarpolicy-policiesclassic_lecarpy)
2. [Cache Simulation Engine (`cache_simulator.py`)](#2-cache-simulation-engine-cache_simulatorpy)
   - [SimulationEvent & SimulationResult](#21-simulationevent--simulationresult)
   - [CacheSimulator](#22-cachesimulator)
3. [Workload Trace Generators & Loaders (`workloads/`)](#3-workload-trace-generators--loaders-workloads)
   - [generate_looping_trace](#31-generate_looping_trace)
   - [generate_sequential_scan_trace](#32-generate_sequential_scan_trace)
   - [generate_zipf_trace](#33-generate_zipf_trace)
   - [generate_phase_shift_trace](#34-generate_phase_shift_trace)
   - [generate_mixed_trace](#35-generate_mixed_trace)
   - [load_trace_from_file](#36-load_trace_from_file)
4. [Evaluation & Benchmark Suite (`benchmarks/`)](#4-evaluation--benchmark-suite-benchmarks)
   - [compute_extended_metrics](#41-compute_extended_metrics)
   - [calculate_rolling_hit_rate](#42-calculate_rolling_hit_rate)
   - [run_benchmark_suite](#43-run_benchmark_suite)
5. [Interactive Streamlit Dashboard (`app.py`)](#5-interactive-streamlit-dashboard-apppy)
   - [get_workload_trace](#51-get_workload_trace)
   - [main](#52-main)
6. [Test Suite (`tests/`)](#6-test-suite-tests)

---

## 1. Core Policy Subsystem (`policies/`)

The policy layer defines how eviction decisions are made and how policies react to hits and misses.

### 1.1 `BasePolicy` (`policies/base.py`)
`BasePolicy` is the abstract base class (interface) that all replacement policies must inherit and implement. It decouples the cache simulation engine from the decision-making policy.

```python
class BasePolicy(ABC):
    def __init__(self, name: str = "BasePolicy") -> None
```

#### `choose_victim(cache_items, context)`
- **Signature**: `choose_victim(self, cache_items: Set[Any], context: Optional[Dict[str, Any]] = None) -> Any`
- **Why It's Used**: When the cache is full and a miss occurs, the simulator must evict an item to make room for the newly requested item. This function asks the policy to nominate the victim.
- **How It Works**: Receives the set of all items currently residing in the cache (`cache_items`) and contextual metadata (e.g. current step, capacity). The policy inspects its internal state (recency order, frequency count, future trace, or expert weights) and returns the item that should be evicted.
- **Return Value**: An item present in `cache_items`.

#### `on_hit(item, context)`
- **Signature**: `on_hit(self, item: Any, context: Optional[Dict[str, Any]] = None) -> None`
- **Why It's Used**: Invoked whenever a requested item was already found in the cache.
- **How It Works**: Updates internal metadata to reflect the access. For LRU, moves the item to the MRU position. For LFU, increments the item's access count. For adaptive policies, forwards the hit to sub-experts.

#### `on_miss(item, evicted_item, context)`
- **Signature**: `on_miss(self, item: Any, evicted_item: Optional[Any], context: Optional[Dict[str, Any]] = None) -> None`
- **Why It's Used**: Invoked whenever a requested item was absent from the cache.
- **How It Works**: Notifies the policy that `item` is being inserted, and that `evicted_item` was removed (or `None` if the cache was not full). For adaptive policies, this is where ghost caches are inspected to attribute eviction regret.

#### `reset()`
- **Signature**: `reset(self) -> None`
- **Why It's Used**: Resets all internal policy state back to initial conditions so the policy can be reused for a fresh simulation run.
- **How It Works**: Clears dictionaries, resets weights to initial values (`0.5`), and zeroes out counters.

#### `get_state()`
- **Signature**: `get_state(self) -> Dict[str, Any]`
- **Why It's Used**: Provides diagnostic telemetry to the simulator and dashboard without breaking encapsulation.
- **How It Works**: Returns a dictionary of internal parameters (e.g., current weights, ghost cache sizes, frequency counts).

---

### 1.2 `LRUPolicy` (`policies/lru.py`)
Implements the classic **Least Recently Used (LRU)** baseline in $O(1)$ time complexity using Python's `collections.OrderedDict`.

```python
class LRUPolicy(BasePolicy):
    def __init__(self, name: str = "LRU") -> None
```
- **Internal Data Structure**: `self.order: OrderedDict[Any, None]`. Items are maintained from head (least recently used) to tail (most recently used).

#### `LRUPolicy.choose_victim(cache_items, context)`
- **Why It's Used**: Selects the item that has not been accessed for the longest period.
- **How It Works**: Synchronizes `self.order` with `cache_items` and returns the item at the head of `self.order` via `next(iter(self.order))`. Time complexity: $O(1)$.
- **Example**:
  ```python
  policy = LRUPolicy()
  victim = policy.choose_victim({1, 2, 3}) # Returns 1 if 1 was least recently accessed
  ```

#### `LRUPolicy.on_hit(item, context)`
- **Why It's Used**: Records that `item` was accessed, protecting it from immediate eviction.
- **How It Works**: Executes `self.order.move_to_end(item)` to relocate the item to the tail (MRU). Time complexity: $O(1)$.

#### `LRUPolicy.on_miss(item, evicted_item, context)`
- **Why It's Used**: Registers the newly inserted item and removes the evicted victim from recency tracking.
- **How It Works**: If `evicted_item` is not `None`, executes `del self.order[evicted_item]`. Then inserts `self.order[item] = None` at the tail. Time complexity: $O(1)$.

#### `LRUPolicy.reset()`
- **Why It's Used**: Clears the recency queue.
- **How It Works**: Calls `self.order.clear()`.

#### `LRUPolicy.get_state()`
- **Why It's Used**: Telemetry for visualization.
- **How It Works**: Returns `{"lru_order": list(self.order.keys())}`.

---

### 1.3 `LFUPolicy` (`policies/lfu.py`)
Implements an optimal $O(1)$ **Least Frequently Used (LFU)** baseline with deterministic LRU tie-breaking.

```python
class LFUPolicy(BasePolicy):
    def __init__(self, name: str = "LFU") -> None
```
- **Internal Data Structures**:
  - `self.freq_map: Dict[Any, int]`: Maps each cached item to its access count.
  - `self.freq_buckets: Dict[int, OrderedDict[Any, None]]`: Maps each frequency count $f$ to an `OrderedDict` of items with that frequency (ordered from LRU to MRU).
  - `self.min_freq: int`: Tracks the current minimum frequency among all items in cache.

#### `LFUPolicy.choose_victim(cache_items, context)`
- **Why It's Used**: Selects the item with the lowest access frequency. If multiple items have the same lowest frequency, breaks ties by evicting the least recently used among them.
- **How It Works**: Retrieves the bucket `self.freq_buckets[self.min_freq]`. The victim is the head item `next(iter(bucket))`. Time complexity: $O(1)$.

#### `LFUPolicy.on_hit(item, context)`
- **Why It's Used**: Increments access frequency upon a cache hit.
- **How It Works**:
  1. Retrieves current frequency $f = \text{freq\_map}[item]$.
  2. Removes `item` from bucket $f$.
  3. If bucket $f$ becomes empty and `min_freq == f`, increments `min_freq = f + 1`.
  4. Inserts `item` into bucket $f + 1$ at the MRU position.
  5. Updates `freq_map[item] = f + 1`. Time complexity: $O(1)$.

#### `LFUPolicy.on_miss(item, evicted_item, context)`
- **Why It's Used**: Handles miss accounting: cleans up the evicted item and initializes the new item's count.
- **How It Works**:
  1. If `evicted_item` is not `None`, removes it from `self.freq_map` and its frequency bucket.
  2. Inserts `item` with frequency 1: `freq_map[item] = 1`.
  3. Places `item` into `freq_buckets[1]`.
  4. Sets `self.min_freq = 1`. Time complexity: $O(1)$.

#### `LFUPolicy.reset()`
- **Why It's Used**: Resets frequency tracking.
- **How It Works**: Clears `freq_map`, clears `freq_buckets`, and sets `min_freq = 0`.

#### `LFUPolicy.get_state()`
- **Why It's Used**: Returns frequency inspection telemetry for debugging and UI.
- **How It Works**: Returns `{"min_freq": self.min_freq, "freq_map": dict(self.freq_map)}`.

---

### 1.4 `BeladyPolicy` (`policies/belady.py`)
Implements **Belady's MIN Optimal** offline replacement algorithm, which serves as the theoretical upper bound for cache hit rates.

```python
class BeladyPolicy(BasePolicy):
    def __init__(self, trace: Optional[Sequence[Any]] = None, name: str = "Belady (Optimal)") -> None
```

#### `BeladyPolicy.set_trace(trace)`
- **Signature**: `set_trace(self, trace: Sequence[Any]) -> None`
- **Why It's Used**: Allows updating the future trace before running a simulation.
- **How It Works**: Stores the trace and calls `_preprocess_trace()`.

#### `BeladyPolicy._preprocess_trace()`
- **Signature**: `_preprocess_trace(self) -> None`
- **Why It's Used**: Pre-indexes item occurrences so future accesses can be checked in $O(1)$ time rather than scanning the entire remaining trace on every eviction.
- **How It Works**: Builds a dictionary of deques:
  `self.occurrences[item] = deque([t1, t2, t3, ...])`
  mapping each item to all step indices where it appears. Resets `self.current_step = 0`.

#### `BeladyPolicy.choose_victim(cache_items, context)`
- **Why It's Used**: Identifies which cached item will not be used for the longest time in the future.
- **How It Works**:
  1. For each `item` in `cache_items`, purges past occurrences ($\le \text{current\_step}$).
  2. If an item has an empty queue (never used again), it is immediately returned as the victim.
  3. Otherwise, looks at `queue[0]` (the next request index) and selects the item with $\max(\text{next\_use})$. Time complexity: $O(C)$.

#### `BeladyPolicy.on_hit(item, context)` & `BeladyPolicy.on_miss(item, evicted_item, context)`
- **Why It's Used**: Advances the internal simulation timeline.
- **How It Works**: Increments `self.current_step += 1` and pops past occurrences of `item`.

---

### 1.5 `LeCaRAdaptivePolicy` (`policies/adaptive.py`)
The flagship algorithm of SMARTCACHE AI: an online learning cache policy that combines LRU and LFU experts, maintains dual ghost caches, and applies regret-based exponential gradient updates.

```python
class LeCaRAdaptivePolicy(BasePolicy):
    def __init__(
        self,
        capacity: int = 10,
        learning_rate: Optional[float] = None,
        discount: float = 1.0,
        history_multiplier: float = 1.0,
        ttl_multiplier: float = 3.0,
        damping_factor: float = 0.35,
        shift_threshold: float = 0.20,
        epsilon_min: float = 0.005,
        deterministic: bool = False,
        seed: Optional[int] = 42,
        name: str = "SmartCache (LeCaR)",
    ) -> None
```

#### `LeCaRAdaptivePolicy.choose_victim(cache_items, context)`
- **Why It's Used**: Decides which item to evict by polling both the LRU and LFU experts and weighting their recommendations.
- **How It Works**:
  1. Calls `v_lru = self.lru_expert.choose_victim(cache_items)`.
  2. Calls `v_lfu = self.lfu_expert.choose_victim(cache_items)`.
  3. If $v_{lru} == v_{lfu}$ (consensus): sets `self.last_chosen_expert = "CONSENSUS"` and returns the agreed victim.
  4. If $v_{lru} \ne v_{lfu}$ (disagreement):
     - If `deterministic == True`: selects the expert with the higher weight.
     - Else: samples $E \sim \text{Bernoulli}(w_{LRU})$. If random value $< w_{LRU}$, chooses $v_{lru}$ and sets `last_chosen_expert = "LRU"`; otherwise chooses $v_{lfu}$ and sets `last_chosen_expert = "LFU"`.
  5. Returns the chosen victim.

#### `LeCaRAdaptivePolicy.on_hit(item, context)`
- **Why It's Used**: Maintains state synchronization across both sub-experts, purges active cache items from ghost caches, and tracks baseline hit rates.
- **How It Works**:
  1. Increments `self.current_step += 1`.
  2. Purges `item` from both ghost caches (`h_lru.pop(item, None)`, `h_lfu.pop(item, None)`).
  3. Updates workload outcome window and baseline hit rate EMA.
  4. Calls `lru_expert.on_hit(item)` and `lfu_expert.on_hit(item)`.
  5. Appends `(current_step, w_lru, w_lfu)` to `self.weight_history`.

#### `LeCaRAdaptivePolicy.on_miss(item, evicted_item, context)`
- **Why It's Used**: Implements **audited credit assignment**, workload-shift detection, and transition measurement.
- **How It Works**:
  1. Records outcome into sliding window and evaluates `_evaluate_workload_shift()`.
  2. Inspects `item in self.h_lru`:
     - Checks if `current_step - evict_step > self.ghost_ttl` (if expired, ignores without penalty).
     - Otherwise, computes effective learning rate `_compute_effective_learning_rate()` (damped if isolated miss, full if shift active) and applies exponential penalty to $w_{LRU}$.
  3. Inspects `item in self.h_lfu`:
     - Checks TTL expiration.
     - Otherwise, applies exponential penalty to $w_{LFU}$.
  4. Purges `item` from both ghost caches before inserting into active cache.
  5. If weights were modified, normalizes weights via `_normalize_weights()`.
  6. Informs sub-experts `lru_expert.on_miss()` and `lfu_expert.on_miss()`.
  7. **Mutually Exclusive Ghost Insertion**:
     - If `last_chosen_expert == "LRU"`, removes `evicted_item` from `h_lfu` and adds to `h_lru`.
     - If `last_chosen_expert == "LFU"`, removes `evicted_item` from `h_lru` and adds to `h_lfu`.
     - If `last_chosen_expert == "CONSENSUS"`, consensus evictions are neutral; removes from both ghost caches.

#### `LeCaRAdaptivePolicy._compute_effective_learning_rate(target_expert, dominant_expert)`
- **Signature**: `_compute_effective_learning_rate(self, target_expert: str, dominant_expert: str) -> float`
- **Why It's Used**: Prevents knee-jerk policy switching on an isolated unlucky miss while allowing rapid adaptation during genuine workload shifts.
- **How It Works**: Returns full `self.learning_rate` if `self.is_shift_active` is true or if regret is sustained ($\ge 2$ ghost hits in window). If an isolated miss occurs on the dominant expert during steady-state, scales learning rate by `self.damping_factor` (e.g. 0.35).

#### `LeCaRAdaptivePolicy._evaluate_workload_shift()`
- **Signature**: `_evaluate_workload_shift(self) -> None`
- **Why It's Used**: Detects abrupt workload distribution changes by comparing recent window metrics against the steady-state baseline.
- **How It Works**: Inspects sliding window outcomes. Sets `self.is_shift_active = True` if:
  1. Recent hit rate drops below baseline by $\ge \text{shift\_threshold}$ (e.g. 0.20), OR
  2. Unique item density $\ge 0.85$ with hit rate $< 0.15$ (sequential scan/burst), OR
  3. Dominant expert accumulates $\ge 2$ ghost hits within the sliding window.

#### `LeCaRAdaptivePolicy._check_transition_status(is_miss)`
- **Signature**: `_check_transition_status(self, is_miss: bool) -> None`
- **Why It's Used**: Makes policy switching **measurable** by tracking adaptation latency and extra misses.
- **How It Works**: Initiates a transition record upon workload shift or weight crossover. Tracks elapsed request duration and misses until the new dominant expert stabilizes ($w \ge 0.65$ or steady state). Computes extra misses as $\text{actual\_misses} - \text{expected\_baseline\_misses}$ and appends to `completed_transitions`.

#### `LeCaRAdaptivePolicy._add_to_ghost(ghost_dict, item)`
- **Signature**: `_add_to_ghost(self, ghost_dict: OrderedDict[Any, int], item: Any) -> None`
- **Why It's Used**: Enforces bounded ghost capacity $|H| \le \alpha \cdot C$ and lazily purges expired ghost entries exceeding `self.ghost_ttl`.

#### `LeCaRAdaptivePolicy._normalize_weights()`
- **Signature**: `_normalize_weights(self) -> None`
- **Why It's Used**: Prevents probability extinction ($\epsilon_{\min}$ floor), guards against NaNs, and guarantees $w_{LRU} + w_{LFU} = 1.0$.

#### `LeCaRAdaptivePolicy.reset()`
- **Signature**: `reset(self) -> None`
- **Why It's Used**: Reinitializes all weights, ghost caches, detectors, and transition logs.

#### `LeCaRAdaptivePolicy.get_state()`
- **Signature**: `get_state(self) -> Dict[str, Any]`
- **Why It's Used**: Exposes telemetry for dashboard and benchmarking, including: `w_lru`, `w_lfu`, `h_lru_size`, `h_lfu_size`, `ghost_hits_lru`, `ghost_hits_lfu`, `expired_ghost_entries`, `is_shift_active`, `completed_transitions`, `avg_adaptation_latency`, and `total_extra_misses`.

---

### 1.6 `ARCPolicy` (`policies/arc.py`)
Implementation of IBM's Adaptive Replacement Cache (Megiddo & Modha, FAST '03).

```python
class ARCPolicy(BasePolicy):
    def __init__(self, capacity: int = 10, name: str = "ARC") -> None
```
- **Internal Data Structures**:
  - `t1`: Top list of $L_1$ (in-cache pages accessed once recently).
  - `b1`: Bottom list of $L_1$ (ghost cache of $T_1$ evictions).
  - `t2`: Top list of $L_2$ (in-cache pages accessed $\ge 2$ times recently).
  - `b2`: Bottom list of $L_2$ (ghost cache of $T_2$ evictions).
  - `p`: Target partition size between recency ($T_1$) and frequency ($T_2$).

#### `ARCPolicy.choose_victim(cache_items, context)`
- **Why It's Used**: Selects a victim from either $T_1$ or $T_2$ depending on current target size $p$ and ghost cache state.
- **How It Works**: Evaluates whether $|T_1| > p$ or ($x \in B_2$ and $|T_1| == p$). If so, evicts from $T_1$; otherwise evicts from $T_2$.

#### `ARCPolicy.on_hit(item, context)`
- **Why It's Used**: Moves accessed item to the most-recently-used position of $T_2$ (frequency list).

#### `ARCPolicy.on_miss(item, evicted_item, context)`
- **Why It's Used**: Handles ghost adaptation and insertion.
- **How It Works**: If $x \in B_1$, expands recency target $p \leftarrow \min(c, p + \delta_1)$. If $x \in B_2$, contracts recency target $p \leftarrow \max(0, p - \delta_2)$. Moves evicted victim to $B_1$ or $B_2$.

---

### 1.7 `SIEVEPolicy` (`policies/sieve.py`)
Implementation of the SIEVE replacement policy (Zhang et al., NSDI '24).

```python
class SIEVEPolicy(BasePolicy):
    def __init__(self, capacity: int = 10, name: str = "SIEVE") -> None
```
- **Internal Data Structures**:
  - Doubly linked list queue with `head` (newest) and `tail` (oldest).
  - Moving pointer `hand` scanning backwards.
  - `visited: bool` flag per cached node.

#### `SIEVEPolicy.choose_victim(cache_items, context)`
- **Why It's Used**: Selects an unvisited victim while giving accessed items a second chance with zero lock contention.
- **How It Works**: Scans backwards from `hand`. If node has `visited == True`, sets `visited = False` and continues. Evicts the first node with `visited == False`.

#### `SIEVEPolicy.on_hit(item, context)`
- **Why It's Used**: Sets `node.visited = True` in $O(1)$ time without moving nodes or updating linked list pointers.

#### `SIEVEPolicy.on_miss(item, evicted_item, context)`
- **Why It's Used**: Removes evicted node and inserts new node at the head with `visited = False`.

---

### 1.8 `ClassicLeCaRPolicy` (`policies/classic_lecar.py`)
Implementation of the un-augmented LeCaR algorithm (Nguyen et al., HotStorage '18).

```python
class ClassicLeCaRPolicy(BasePolicy):
    def __init__(self, capacity: int = 10, learning_rate: float = 0.25, seed: Optional[int] = 42, name: str = "LeCaR (Classic)") -> None
```
- **Why It's Used**: Baseline for ablation study; runs the original exponential update rule on every ghost hit without shift detection damping or TTL pruning.

---

## 2. Cache Simulation Engine (`cache_simulator.py`)

The simulator engine owns cache membership, processes incoming requests, and handles accounting without coupling to any specific replacement policy.

### 2.1 `SimulationEvent` & `SimulationResult`

#### `SimulationEvent`
- **Fields**:
  - `step: int`: The sequential request index.
  - `item: Any`: The item identifier requested.
  - `is_hit: bool`: `True` if found in cache; `False` if miss.
  - `evicted_item: Optional[Any]`: The item evicted to make room, or `None`.
  - `policy_state: Dict[str, Any]`: Snapshot of the policy's internal telemetry at this step.
- **Why It's Used**: Represents a single immutable access event for time-series analysis and dashboard playback.

#### `SimulationResult`
- **Fields**: `policy_name`, `capacity`, `total_requests`, `hits`, `misses`, `hit_rate`, `miss_rate`, `evictions`, `wall_clock_time_sec`, `avg_decision_time_us`, `events`.
- **`to_dict()` Method**: Serializes the results into a clean dictionary suitable for CSV/JSON export.

---

### 2.2 `CacheSimulator`

```python
class CacheSimulator:
    def __init__(self, capacity: int, policy: BasePolicy, record_events: bool = True) -> None
```

#### `CacheSimulator.reset()`
- **Signature**: `reset(self) -> None`
- **Why It's Used**: Resets the cache contents and accounting counters before a simulation run.
- **How It Works**: Clears `self.cache`, zeroes out `hits`, `misses`, `evictions`, and decision timers, and calls `self.policy.reset()`.

#### `CacheSimulator.step(item, step_idx)`
- **Signature**: `step(self, item: Any, step_idx: Optional[int] = None) -> SimulationEvent`
- **Why It's Used**: Processes a single request through the cache state machine. Enables single-step stepping or custom streaming pipelines.
- **How It Works**:
  1. Increments `self.total_requests += 1`.
  2. If `item in self.cache`:
     - Increments `self.hits += 1`.
     - Times and executes `self.policy.on_hit(item, context)`.
     - Returns `SimulationEvent(is_hit=True, evicted_item=None)`.
  3. If `item not in self.cache`:
     - Increments `self.misses += 1`.
     - If `len(self.cache) >= self.capacity`:
       - Increments `self.evictions += 1`.
       - Times and executes `victim = self.policy.choose_victim(self.cache, context)`.
       - Removes `victim` from `self.cache`.
     - Adds `item` to `self.cache`.
     - Times and executes `self.policy.on_miss(item, evicted_item, context)`.
     - Returns `SimulationEvent(is_hit=False, evicted_item=evicted_item)`.

#### `CacheSimulator.run(trace)`
- **Signature**: `run(self, trace: Iterable[Any]) -> SimulationResult`
- **Why It's Used**: Executes an entire access trace from start to finish and produces a comprehensive `SimulationResult`.
- **How It Works**:
  1. Calls `self.reset()`.
  2. If policy supports `set_trace` (e.g. Belady), passes the full trace to it.
  3. Measures total wall-clock time across all calls to `self.step()`.
  4. Calculates overall `hit_rate = hits / total_requests` and `avg_decision_time_us`.
  5. Packages and returns `SimulationResult`.
- **Example**:
  ```python
  sim = CacheSimulator(capacity=10, policy=LRUPolicy())
  result = sim.run([1, 2, 3, 1, 4, 1, 5])
  print(f"Hit Rate: {result.hit_rate * 100:.2f}%")
  ```

---

## 3. Workload Trace Generators & Loaders (`workloads/`)

### 3.1 `generate_looping_trace` (`workloads/generators.py`)
- **Signature**:
  ```python
  def generate_looping_trace(
      hot_set_size: int = 10,
      repetitions: int = 50,
      noise_ratio: float = 0.05,
      noise_universe: int = 500,
      seed: int = 42,
  ) -> List[int]
  ```
- **Why It's Used**: Generates a cyclic access pattern where a small group of "hot" items repeats frequently.
- **How It Works**: Cycles through items `[0, 1, ..., hot_set_size - 1]`. With probability `noise_ratio`, injects a random noise item from outside the hot set to simulate background cache pollution.
- **Use Case**: Verifies whether policies correctly retain frequently accessed items in cache.

---

### 3.2 `generate_sequential_scan_trace` (`workloads/generators.py`)
- **Signature**:
  ```python
  def generate_sequential_scan_trace(
      scan_length: int = 500,
      repeat_cycle: int = 0,
      seed: int = 42,
  ) -> List[int]
  ```
- **Why It's Used**: Models streaming or database table scans where data is read sequentially with no immediate reuse.
- **How It Works**: Returns a strictly non-repeating sequence of unique integers `[0, 1, 2, ..., scan_length - 1]`.
- **Use Case**: Tests scan resistance and verifies that the simulator and policies handle 100% miss and eviction rates without performance degradation.

---

### 3.3 `generate_zipf_trace` (`workloads/generators.py`)
- **Signature**:
  ```python
  def generate_zipf_trace(
      num_items: int = 100,
      total_requests: int = 1000,
      alpha: float = 1.0,
      seed: int = 42,
  ) -> List[int]
  ```
- **Why It's Used**: Generates power-law distributed accesses (Zipf-Mandelbrot law), which closely mirrors realistic web, database, and OS file access patterns.
- **How It Works**:
  1. Computes probability distribution $P(k) = \frac{1/k^\alpha}{\sum_{j=1}^N (1/j^\alpha)}$.
  2. Uses `numpy.random.default_rng(seed).choice` to sample `total_requests` items according to this distribution.
- **Use Case**: Evaluates behavior under heavy popularity skew (where a small subset of items accounts for the vast majority of requests).

---

### 3.4 `generate_phase_shift_trace` (`workloads/generators.py`)
- **Signature**:
  ```python
  def generate_phase_shift_trace(
      phases: Optional[List[Dict[str, Any]]] = None,
      seed: int = 42,
  ) -> List[int]
  ```
- **Why It's Used**: Simulates dynamic workload shifts where the working set abruptly changes over time.
- **How It Works**: Sequentially concatenates distinct access phases:
  - **Phase 1**: Hot Set A (items 0..9) for 300 requests.
  - **Phase 2**: Sequential Scan (items 1000..1300) for 300 requests.
  - **Phase 3**: Hot Set B (items 20..29) for 300 requests.
- **Use Case**: Demonstrates how LFU gets stuck preserving stale Phase 1 items, whereas SMARTCACHE AI detects the regret and dynamically adapts to LRU.

---

### 3.5 `generate_mixed_trace` (`workloads/generators.py`)
- **Signature**:
  ```python
  def generate_mixed_trace(
      total_requests: int = 1000,
      seed: int = 42,
  ) -> List[int]
  ```
- **Why It's Used**: Models realistic multi-tenant systems by interleaving multiple distinct access patterns.
- **How It Works**: At each step:
  - 50% probability: picks from a recurring hot set.
  - 35% probability: picks from a background Zipf distribution.
  - 15% probability: executes a short 5-item sequential scan burst.

---

### 3.6 `load_trace_from_file` (`workloads/loaders.py`)
- **Signature**: `load_trace_from_file(filepath: Union[str, Path]) -> List[int]`
- **Why It's Used**: Allows users to import external real-world traces (e.g. from ARC benchmarks, disk I/O logs, or CSV dumps).
- **How It Works**: Reads file text, normalizes commas and tabs to whitespace, parses integer tokens, and converts hex or string tokens via fallback hashing.

---

## 4. Evaluation & Benchmark Suite (`benchmarks/`)

### 4.1 `compute_extended_metrics` (`benchmarks/metrics.py`)
- **Signature**:
  ```python
  def compute_extended_metrics(result: SimulationResult, trace: List[Any]) -> Dict[str, Any]
  ```
- **Why It's Used**: Calculates advanced cache metrics, specifically separating **compulsory (cold) misses** from **capacity misses**.
- **How It Works**:
  1. Tracks the first time each unique item is seen in the trace. The first access to any distinct item is always a **compulsory miss** (no cache of any capacity could have hit it).
  2. $\text{Capacity Misses} = \text{Total Misses} - \text{Compulsory Misses}$.
  3. Returns a structured dictionary with miss counts, hit rates, and ratios.

---

### 4.2 `calculate_rolling_hit_rate` (`benchmarks/metrics.py`)
- **Signature**:
  ```python
  def calculate_rolling_hit_rate(events: List[SimulationEvent], window_size: int = 50) -> List[float]
  ```
- **Why It's Used**: Computes moving-window hit rate across the request timeline for smooth time-series charting.
- **How It Works**: Converts event hit booleans into a NumPy array $[1.0, 0.0, \dots]$, convolves with a moving average kernel $\frac{1}{W}$, and prefixes with expanding cumulative averages up to $W - 1$.
- **Return Value**: A list of float hit rates matching the trace length.

---

### 4.3 `run_benchmark_suite` (`benchmarks/run_benchmarks.py`)
- **Signature**:
  ```python
  def run_benchmark_suite(output_dir: str = "results", seed: int = 42) -> pd.DataFrame
  ```
- **Why It's Used**: Automates fair, multi-policy benchmark runs across all workloads and capacities.
- **How It Works**:
  1. Generates the 5 standard workloads.
  2. Runs LRU, LFU, SmartCache (LeCaR), and Belady on the exact same trace and capacity ($C \in \{5, 10, 20\}$).
  3. Computes extended metrics.
  4. Saves summary results to `results/benchmark_summary.csv` and `results/benchmark_summary.json`.
  5. Prints a formatted ASCII comparison pivot table.

---

## 5. Interactive Streamlit Dashboard (`app.py`)

### 5.1 `get_workload_trace` (`app.py`)
- **Signature**:
  ```python
  def get_workload_trace(workload_type: str, length: int, seed: int, uploaded_file=None) -> List[int]
  ```
- **Why It's Used**: Factory function dispatching UI dropdown selections to the appropriate generator or file parser.
- **How It Works**: Maps dropdown strings (`"Looping (Hot Set)"`, `"Phase Shift (Shift)"`, etc.) to generator calls, slicing the trace to the exact length selected by the user.

---

### 5.2 `main` (`app.py`)
- **Signature**: `main() -> None`
- **Why It's Used**: Primary entrypoint for the Streamlit web application.
- **How It Works**: Sets up page layout, custom CSS, sidebar controls, and renders the 4 interactive tabs:
  1. **Tab 1: Live Simulator & Telemetry**: Renders live metric KPI cards, Plotly rolling hit rate line chart, adaptive weight telemetry, and cache state item chips.
  2. **Tab 2: Workload-Shift Demo**: Executes LRU, LFU, and SmartCache side-by-side across a 3-phase transition and plots comparative rolling hit rates and weight crossovers.
  3. **Tab 3: Multi-Policy Benchmark Matrix**: Reads `results/benchmark_summary.csv` and renders interactive grouped bar charts and downloadable tables.
  4. **Tab 4: Architecture & Theory**: In-browser documentation explaining credit assignment, ghost caches, and exponential gradient updates.

---

## 6. Test Suite (`tests/`)

| Test File | Test Function | Purpose & Validation Verified |
| :--- | :--- | :--- |
| `test_simulator.py` | `test_capacity_one_repeated` | Verifies capacity 1 repeated requests (`[1, 1, 1, 1]`) produce 3 hits, 1 miss, and 75% hit rate. |
| | `test_capacity_one_alternating` | Verifies capacity 1 alternating requests (`[1, 2, 1, 2]`) produce 0% hit rate (thrashing). |
| | `test_empty_trace` | Verifies graceful handling of empty input traces without ZeroDivisionError. |
| | `test_accounting_invariants` | Verifies $\text{Hits} + \text{Misses} \equiv \text{Requests}$ and $|C| \le \text{Capacity}$. |
| | `test_invalid_capacity` | Verifies that initializing capacity $< 1$ raises `ValueError`. |
| `test_lru.py` | `test_lru_basic_eviction` | Verifies standard textbook LRU eviction ordering on miss. |
| | `test_lru_recency_update_on_hit` | Verifies that hits advance items to MRU, protecting them from eviction. |
| `test_lfu.py` | `test_lfu_evicts_lowest_frequency` | Verifies that items with lowest access count are prioritized for eviction. |
| | `test_lfu_tie_breaker_lru` | Verifies that equal-frequency ties are broken deterministically by recency. |
| `test_belady.py` | `test_belady_evicts_furthest_future_use` | Verifies Belady selects the item whose next occurrence is furthest away. |
| | `test_belady_optimality_upper_bound`| Verifies Belady hit rate $\ge \max(\text{LRU}, \text{LFU})$ across complex traces. |
| `test_adaptive.py` | `test_initial_weights` | Verifies initial weights start at exactly $0.5, 0.5$. |
| | `test_weights_normalization_and_invariants` | Verifies weights remain finite, non-negative, and sum to $1.0 \pm 10^{-4}$. |
| | `test_ghost_cache_bounded_capacity` | Verifies ghost caches never exceed configured capacity $\alpha \cdot C$. |
| | `test_reproducibility_with_seed` | Verifies identical random seeds yield identical simulation results. |
| `test_workloads.py` | `test_generators_deterministic_seed` | Verifies all trace generators produce identical output given matching seeds. |
| | `test_sequential_scan` | Verifies sequential scan generates non-repeating integers $[0, 1, \dots, N-1]$. |
| | `test_file_loader` | Verifies loader handles commas, newlines, and whitespace delimiters. |

---

## 7. How to Run the System

### Running Automated Tests
```powershell
.venv\Scripts\pytest.exe -v tests/
```

### Running the Multi-Policy Benchmark Suite
```powershell
.venv\Scripts\python.exe -m benchmarks.run_benchmarks
```

### Launching the Streamlit Web Application
```powershell
.venv\Scripts\streamlit.exe run app.py
```
*(Accessible at `http://localhost:8501` in your browser)*
