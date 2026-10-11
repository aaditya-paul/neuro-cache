# SMARTCACHE AI 🧠⚡
> *“Learning what to keep before you need it.”*  
> **Team BONGORITHM** — Hackathon Build Specification & Prototype

SMARTCACHE AI is an adaptive, online-learning cache replacement prototype. Instead of relying on static heuristics (such as fixed LRU or fixed LFU), SMARTCACHE AI deploys LRU and LFU as parallel **experts**, tracks eviction regrets using **dual shadow (ghost) caches**, and dynamically tunes decision weights using **Exponential Gradient (Exp3/Hedge)** reinforcement updates.

---

## 🌟 Key Highlights & Capabilities
- **Modular Cache Engine**: Complete decoupling of cache accounting/state from policy decision logic.
- **LeCaR-Style Online Adaptation**:
  - Maintains dual ghost history buffers ($H_{LRU}, H_{LFU}$) to solve credit assignment online without an oracle.
  - Updates weights exponentially upon ghost cache hits ($w_i \leftarrow w_i \cdot e^{-\lambda \gamma^{\Delta t}}$).
  - Guarantees numerical safety: non-negative weights, $\epsilon_{\min}$ exploration floor, and exact sum-to-1 normalization.
- **Reference Baselines, Modern Competitors & Upper Bound**:
  - $O(1)$ LRU baseline (using `OrderedDict`).
  - $O(1)$ LFU baseline with frequency buckets and deterministic LRU tie-breaking.
  - Belady's Optimal (MIN) offline oracle providing the theoretical upper bound.
  - Classic LeCaR (Nguyen et al., HotStorage '18) for baseline regret attribution.
  - Adaptive Replacement Cache (ARC - Megiddo & Modha, FAST '03) with dynamically sized recency and frequency lists.
  - SIEVE (Zhang et al., NSDI '24) ultra-fast $O(1)$ single-bit eviction with zero ghost cache.
- **Strengthened SMARTCACHE AI Core**:
  - Workload-shift detection via sliding-window hit-rate drop triggers.
  - Single-miss damping to prevent catastrophic weight oscillation.
  - Audited ghost-cache attribution: consensus neutrality, cache-hit purging, mutual exclusivity, and temporal TTL expiration.
  - Measurable transition metrics: adaptation latency ($\Delta t_{\text{adapt}}$) and extra transition misses.
- **Comprehensive Workload Generator**: Looping / Hot Sets, Sequential Scans, Zipf Skew ($\alpha=1.0$), Phase Transitions, and Mixed access patterns.
- **Rich Streamlit Dashboard**: Live simulation, real-time weight dynamics timeline, workload-shift adaptation demo, and multi-policy comparison matrices.
- **In-Depth Documentation**: Complete function API in `DOCS.md` and complete mathematical derivation in `TECHNICAL_DEEP_DIVE.md`.

---

## 📁 Repository Structure

```
neuro-cache/
├── app.py                      # Interactive Streamlit dashboard (4 tabs)
├── cache_simulator.py          # Decoupled simulator engine and accounting
├── policies/
│   ├── __init__.py             # Exported policies
│   ├── base.py                 # Base replacement policy interface
│   ├── lru.py                  # Baseline LRU (O(1))
│   ├── lfu.py                  # Baseline LFU (O(1) with LRU tie-breaking)
│   ├── belady.py               # Belady's Optimal offline MIN oracle
│   ├── classic_lecar.py        # Classic LeCaR (HotStorage '18)
│   ├── arc.py                  # Adaptive Replacement Cache (FAST '03)
│   ├── sieve.py                # Modern SIEVE cache (NSDI '24)
│   └── adaptive.py             # Strengthened SMARTCACHE AI online policy
├── workloads/
│   ├── __init__.py
│   ├── generators.py           # Synthetic trace generators
│   └── loaders.py              # File trace loader (CSV / txt)
├── benchmarks/
│   ├── __init__.py
│   ├── metrics.py              # Evaluation metrics, rolling hit rate, adaptation penalty
│   └── run_benchmarks.py       # Fair 7-policy benchmark matrix runner
├── tests/
│   ├── test_simulator.py       # Simulator accounting & lifecycle tests
│   ├── test_lru.py             # LRU unit tests
│   ├── test_lfu.py             # LFU unit tests
│   ├── test_belady.py          # Belady oracle tests
│   ├── test_classic_lecar.py   # Classic LeCaR tests
│   ├── test_arc.py             # ARC tests
│   ├── test_sieve.py           # SIEVE tests
│   ├── test_adaptive.py        # Adaptive policy weight & ghost tests
│   └── test_workloads.py       # Generator reproducibility tests
├── results/                    # Saved benchmark results (CSV/JSON)
├── DOCS.md                     # Function-by-function documentation
├── TECHNICAL_DEEP_DIVE.md      # Full mathematical derivations & benchmark report
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview & quickstart
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- `uv` (recommended) or standard `pip`

### 2. Environment Setup
```powershell
# Create virtual environment
uv venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt --python .venv\Scripts\python.exe
```

---

---

## 🧪 RUNNING AUTONOMOUS TEST
Run the complete unit and integration test suite:
```powershell
.venv\Scripts\pytest.exe -v tests/
```
All **40 unit and integration tests** cover:
- Capacity-1 edge cases and simulator accounting invariants
- LRU recency tracking and LFU frequency bucketing
- Belady future-use oracle guarantees
- Classic LeCaR, ARC, and SIEVE compliance
- SmartCache AI trained config loading, scan-resistant probation filtering, effective decayed utility protection, single-miss damping, shift detection, measurable policy switching, consensus neutrality, and ghost TTL expiration.

---

## 🏋️ Training SMARTCACHE AI (Offline Meta-Optimization)
To train SMARTCACHE AI's decay memory horizons ($\tau = C \cdot \text{halflife\_mult}$) and frequency saturation caps ($F_{\max}$) across diverse workload distributions (Zipfian skews, scan bursts, phase shifts, and loops):
```powershell
.venv\Scripts\python.exe train.py --epochs 10 --trials 5
```
Trained parameters are serialized to `models/trained_config.json` and automatically loaded by `LeCaRAdaptivePolicy` during simulation and production runs.

---

## 📊 Running the Multi-Policy Benchmarks
To execute the automated benchmark suite comparing all 7 policies (LRU, LFU, Belady, Classic LeCaR, ARC, SIEVE, and SmartCache AI) across all workloads and capacities:
```powershell
.venv\Scripts\python.exe -m benchmarks.run_benchmarks
```
Results will be output to console and saved to:
- `results/benchmark_summary.csv`
- `results/benchmark_summary.json`

---

## 🔴 Benchmarking Against Redis (Live Production Datasets)
In real-world production systems, Redis is the industry standard in-memory cache. However, Redis does not maintain exact linked lists due to pointer overhead (16–24 bytes per key). Instead, Redis implements:
- **Redis Sampled LRU (`allkeys-lru`)**: Samples $N=5$ keys into an eviction pool of size 16 sorted by idle time.
- **Redis Sampled LFU (`allkeys-lfu`)**: Tracks an 8-bit Morris logarithmic counter with LDT time decay.

SMARTCACHE AI provides a dedicated, high-fidelity Redis benchmark suite running on realistic production traces:
- **🛒 E-Commerce Flash Sale & Catalog**: Heavy Zipfian product catalog reads (`product:xxxx`) + active session churn (`session:user_xxx`) + sudden flash sale surge.
- **🛡️ Batch Scan Attack (Cache Pollution)**: Core working set interrupted by background ETL/reporting crawler scans (`batch:scan:export_xxxx`).
- **🌐 API Gateway & Session Cache**: Microservice route caching + user profiles + auth tokens.
- **🌗 Day/Night Traffic Phase Shift**: Global diurnal traffic transitions.

### Running the Head-to-Head Redis Benchmark:
```powershell
.venv\Scripts\python.exe -m benchmarks.redis_live_benchmark --length 1000 --capacities 10 25 50
```
Results are saved to:
- `results/redis_benchmark_summary.csv`
- `results/redis_benchmark_summary.json`

### Optional: Live TCP Execution against a Running Redis Instance:
```powershell
.venv\Scripts\python.exe -m benchmarks.redis_live_benchmark --live-redis "redis://localhost:6379/0"
```

---

## 🖥️ Launching the Interactive Dashboard
Launch the visual Streamlit web application:
```powershell
.venv\Scripts\streamlit.exe run app.py
```
Open your browser at `http://localhost:8501`.

### Dashboard Features:
1. **Live Simulator & Telemetry**: Configure capacity, workloads (including Redis production traces), and watch real-time rolling hit rates, cache chips, and adaptive weights.
2. **🔴 Redis vs SmartCache AI (Live Benchmark)**: Direct side-by-side head-to-head comparison against Redis Sampled LRU and LFU on live production traces, featuring rolling hit-rate timelines, scan pollution analysis, and real Redis key chip breakdown.
3. **Workload-Shift Demo**: Observe the exact moment access patterns change (Hot Set $\to$ Scan $\to$ Hot Set B) and see SmartCache dynamically re-weight recency vs frequency.
4. **Multi-Policy Benchmark Matrix**: Grouped bar charts, penalty comparisons, and pivot tables across all 9 policies (including Redis LRU and Redis LFU).
5. **Theory & Architecture**: Mathematical walkthrough of online learning, regret attribution, and comparisons with Redis, ARC, and SIEVE.

---

## 📖 In-Depth Technical References
For deep technical questions, see the dedicated documentation files:
- [DOCS.md](file:///c:/Users/Aaditya%20Paul/Documents/Projects/Personal%20Projects/neuro-cache/DOCS.md): Detailed documentation of every class, method, argument, and return value across the entire engine.
- [TECHNICAL_DEEP_DIVE.md](file:///c:/Users/Aaditya%20Paul/Documents/Projects/Personal%20Projects/neuro-cache/TECHNICAL_DEEP_DIVE.md): Full mathematical derivations (Hedge/Exp3 bandit updates, optimal learning rates, ghost regret bounds), audited ghost mechanics, transition regret telemetry, and empirical benchmark analysis.

---

## ⚖️ Scientific Integrity & Competitive Benchmarking Summary
- **Zipf Skew ($C=5, 10$)**: SmartCache AI takes **1st Place** among all online replacement policies (**38.80%** at $C=5$ vs LFU 35.60%, SIEVE 34.60%, ARC 33.60%; and **52.60%** at $C=10$ vs LFU 49.60%, SIEVE 48.20%, ARC 44.70%).
- **Mixed Workload ($C=20$)**: SmartCache AI reaches **41.30%** hit rate (up from 31.10%), effectively matching ARC (**42.10%**), SIEVE (**41.70%**), and LFU (**41.70%**), while outperforming LRU (**23.20%**) by +18.1%.
- **Phase Shift Recovery ($C=5$)**: SmartCache AI incurs only **41 extra misses** during recovery, achieving the lowest adaptation penalty of all evaluated policies (ARC incurs 50, LFU incurs 82).
- **Looping & Steady States ($C=10, 20$)**: Matches Belady's optimal oracle at **64.44%** and **94.00%**.
