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
- **Reference Baselines & Bound**:
  - $O(1)$ LRU baseline (using `OrderedDict`).
  - $O(1)$ LFU baseline with frequency buckets and deterministic LRU tie-breaking.
  - Belady's Optimal (MIN) offline oracle providing the theoretical upper bound.
- **Comprehensive Workload Generator**: Looping / Hot Sets, Sequential Scans, Zipf Skew ($\alpha=1.0$), Phase Transitions, and Mixed access patterns.
- **Rich Streamlit Dashboard**: Live simulation, real-time weight dynamics timeline, workload-shift adaptation demo, and multi-policy comparison matrices.

---

## 📁 Repository Structure

```
neuro-cache/
├── app.py                      # Interactive Streamlit dashboard
├── cache_simulator.py          # Decoupled simulator engine and accounting
├── policies/
│   ├── __init__.py             # Exported policies
│   ├── base.py                 # Base replacement policy interface
│   ├── lru.py                  # Baseline LRU (O(1))
│   ├── lfu.py                  # Baseline LFU (O(1) with LRU tie-breaking)
│   ├── belady.py               # Belady's Optimal offline MIN oracle
│   └── adaptive.py             # LeCaR-style online adaptive policy
├── workloads/
│   ├── __init__.py
│   ├── generators.py           # Synthetic trace generators
│   └── loaders.py              # File trace loader (CSV / txt)
├── benchmarks/
│   ├── __init__.py
│   ├── metrics.py              # Evaluation metrics & rolling hit rate
│   └── run_benchmarks.py       # Fair multi-policy benchmark runner
├── tests/
│   ├── test_simulator.py       # Simulator accounting & lifecycle tests
│   ├── test_lru.py             # LRU unit tests
│   ├── test_lfu.py             # LFU unit tests
│   ├── test_belady.py          # Belady oracle tests
│   ├── test_adaptive.py        # Adaptive policy weight & ghost tests
│   └── test_workloads.py       # Generator reproducibility tests
├── results/                    # Saved benchmark results (CSV/JSON)
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
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

## 🧪 Running Automated Tests
Run the complete unit and integration test suite:
```powershell
.venv\Scripts\pytest.exe -v tests/
```
All 18 tests cover capacity-1 edge cases, LRU/LFU invariant tests, Belady upper bound guarantees, and weight normalization checks.

---

## 📊 Running the Multi-Policy Benchmarks
To execute the automated benchmark suite comparing Belady, LFU, LRU, and SmartCache across all 5 workloads:
```powershell
.venv\Scripts\python.exe -m benchmarks.run_benchmarks
```
Results will be output to console and saved to:
- `results/benchmark_summary.csv`
- `results/benchmark_summary.json`

---

## 🖥️ Launching the Interactive Dashboard
Launch the visual Streamlit web application:
```powershell
.venv\Scripts\streamlit.exe run app.py
```
Open your browser at `http://localhost:8501`.

### Dashboard Features:
1. **Live Simulator & Telemetry**: Configure capacity, workloads, and watch real-time rolling hit rates, cache chips, and adaptive weights.
2. **Workload-Shift Demo**: Observe the exact moment access patterns change (Hot Set $\to$ Scan $\to$ Hot Set B) and see SmartCache dynamically re-weight recency vs frequency.
3. **Multi-Policy Benchmark Matrix**: Grouped bar charts and pivot tables across capacities.
4. **Theory & Architecture**: Mathematical walkthrough of the LeCaR formulation.

---

## ⚖️ Scientific Integrity & Honest Claims
- **Prototype Status**: SMARTCACHE AI is an algorithmic prototype evaluated on synthetic and trace-driven simulations; it is not yet a hardware cache controller or kernel module.
- **Workload Nuances**: No single online replacement policy beats all policies on every workload. SmartCache's strength lies in its **robustness to phase shifts and mixed workloads**, preventing severe degradation where static policies collapse.
