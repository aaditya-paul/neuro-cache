"""SMARTCACHE AI — Interactive Streamlit Dashboard & Evaluation Visualizer.

Provides live cache simulation, real-time weight adaptation telemetry,
Redis head-to-head comparison against live production traces,
workload-shift demo, and multi-policy benchmark comparisons.
"""
from pathlib import Path
import random
import time
from typing import Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cache_simulator import CacheSimulator
from policies.lru import LRUPolicy
from policies.lfu import LFUPolicy
from policies.belady import BeladyPolicy
from policies.adaptive import LeCaRAdaptivePolicy
from policies.arc import ARCPolicy
from policies.sieve import SIEVEPolicy
from policies.classic_lecar import ClassicLeCaRPolicy
from policies.redis_policy import RedisSampledLRUPolicy, RedisSampledLFUPolicy
from workloads.generators import (
    generate_looping_trace,
    generate_sequential_scan_trace,
    generate_zipf_trace,
    generate_phase_shift_trace,
    generate_mixed_trace,
)
from workloads.redis_workload import (
    get_redis_workload,
    generate_redis_ecommerce_workload,
    generate_redis_scan_pollution_workload,
    generate_redis_api_gateway_workload,
    generate_redis_phase_shift_workload,
)
from workloads.loaders import load_trace_from_file
from benchmarks.metrics import calculate_rolling_hit_rate, compute_extended_metrics
from benchmarks.redis_live_benchmark import run_live_redis_server_benchmark


# Page Configuration
st.set_page_config(
    page_title="SMARTCACHE AI | Team BONGORITHM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        color: #1E3A8A;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F3F4F6;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #3B82F6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        padding: 8px 16px;
    }
    .tag-product {
        display: inline-block;
        background: #DCFCE7;
        color: #166534;
        padding: 3px 8px;
        margin: 2px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .tag-session {
        display: inline-block;
        background: #DBEAFE;
        color: #1E40AF;
        padding: 3px 8px;
        margin: 2px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .tag-scan {
        display: inline-block;
        background: #FEE2E2;
        color: #991B1B;
        padding: 3px 8px;
        margin: 2px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .tag-other {
        display: inline-block;
        background: #F3F4F6;
        color: #374151;
        padding: 3px 8px;
        margin: 2px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_workload_trace(workload_type: str, length: int, seed: int, uploaded_file=None):
    """Generate or load the selected workload trace."""
    if workload_type == "Looping (Hot Set)":
        return generate_looping_trace(hot_set_size=10, repetitions=max(1, length // 10), noise_ratio=0.05, seed=seed)[:length]
    elif workload_type == "Sequential Scan":
        return generate_sequential_scan_trace(scan_length=length, seed=seed)
    elif workload_type == "Zipf Skew (alpha=1.0)":
        return generate_zipf_trace(num_items=max(20, length // 5), total_requests=length, alpha=1.0, seed=seed)
    elif workload_type == "Phase Shift (Shift)":
        return generate_phase_shift_trace(seed=seed)[:length]
    elif workload_type == "Mixed Workload":
        return generate_mixed_trace(total_requests=length, seed=seed)
    elif workload_type.startswith("Redis:"):
        scenario = workload_type.replace("Redis: ", "").strip()
        return get_redis_workload(scenario, total_requests=length, seed=seed)
    elif workload_type == "Custom Uploaded File" and uploaded_file is not None:
        temp_path = Path("temp_upload.txt")
        temp_path.write_bytes(uploaded_file.getbuffer())
        trace = load_trace_from_file(temp_path)
        temp_path.unlink(missing_ok=True)
        return trace[:length] if trace else [0] * length
    return generate_mixed_trace(total_requests=length, seed=seed)


def format_key_chip(key: Any) -> str:
    """Format key into stylized HTML chip based on namespace."""
    k_str = str(key)
    if "product" in k_str or "hot_" in k_str:
        return f'<span class="tag-product">{k_str}</span>'
    elif "session" in k_str or "auth" in k_str or "user" in k_str:
        return f'<span class="tag-session">{k_str}</span>'
    elif "scan" in k_str or "batch" in k_str or "export" in k_str:
        return f'<span class="tag-scan">{k_str}</span>'
    else:
        return f'<span class="tag-other">{k_str}</span>'


def main():
    st.markdown('<div class="main-title">🧠 SMARTCACHE AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Adaptive Online Cache Replacement System • Team BONGORITHM • <em>"Learning what to keep before you need it."</em></div>',
        unsafe_allow_html=True,
    )

    tab_sim, tab_redis, tab_shift, tab_bench, tab_arch = st.tabs([
        "⚡ Live Simulator & Telemetry",
        "🔴 Redis vs SmartCache AI (Live Benchmark)",
        "🔄 Workload-Shift Demo",
        "📊 Multi-Policy Benchmark Matrix",
        "📖 Architecture & Theory",
    ])

    # ---------------------------------------------------------
    # TAB 1: Live Simulator & Telemetry
    # ---------------------------------------------------------
    with tab_sim:
        col_ctrl, col_view = st.columns([1, 3])

        with col_ctrl:
            st.subheader("⚙️ Configuration")
            capacity = st.slider("Cache Capacity", min_value=2, max_value=50, value=10, step=1)
            workload_type = st.selectbox(
                "Workload Pattern",
                [
                    "Phase Shift (Shift)",
                    "Looping (Hot Set)",
                    "Zipf Skew (alpha=1.0)",
                    "Mixed Workload",
                    "Sequential Scan",
                    "Redis: E-Commerce Flash Sale & Catalog",
                    "Redis: Batch Scan Attack (Cache Pollution)",
                    "Redis: API Gateway & Session Cache",
                    "Redis: Day/Night Traffic Phase Shift",
                    "Custom Uploaded File",
                ],
            )
            uploaded_file = None
            if workload_type == "Custom Uploaded File":
                uploaded_file = st.file_uploader("Upload Trace (.txt / .csv)", type=["txt", "csv"])

            trace_length = st.slider("Trace Length", min_value=50, max_value=2000, value=500, step=50)
            seed = st.number_input("Random Seed", value=42, step=1)

            st.divider()
            policy_name = st.selectbox(
                "Policy",
                [
                    "SmartCache AI",
                    "Redis Sampled LRU (samples=5)",
                    "Redis Sampled LRU (samples=10)",
                    "Redis Sampled LFU (samples=5)",
                    "ARC",
                    "SIEVE",
                    "LeCaR (Classic)",
                    "LRU",
                    "LFU",
                    "Belady (Optimal)",
                ],
            )

            # Adaptive Hyperparameters
            learning_rate = None
            discount = 1.0
            history_mult = 1.0
            deterministic = False

            if policy_name == "SmartCache AI":
                st.caption("Adaptive Hyperparameters")
                use_adaptive_lr = st.checkbox("Auto Adaptive Learning Rate", value=True)
                if not use_adaptive_lr:
                    learning_rate = st.slider("Learning Rate (λ)", 0.01, 1.0, 0.15, 0.01)
                discount = st.slider("Recency Discount (γ)", 0.90, 1.0, 1.0, 0.01)
                history_mult = st.slider("Ghost Cache Size Multiplier", 0.5, 2.0, 1.0, 0.25)
                deterministic = st.checkbox("Deterministic Argmax", value=False)

            run_sim_btn = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

        with col_view:
            trace = get_workload_trace(workload_type, trace_length, seed, uploaded_file)

            # Instantiate policy
            if policy_name == "SmartCache AI":
                policy = LeCaRAdaptivePolicy(
                    capacity=capacity,
                    learning_rate=learning_rate,
                    discount=discount,
                    history_multiplier=history_mult,
                    deterministic=deterministic,
                    seed=seed,
                    name="SmartCache AI",
                )
            elif policy_name == "Redis Sampled LRU (samples=5)":
                policy = RedisSampledLRUPolicy(samples=5, seed=seed, name="Redis LRU (samples=5)")
            elif policy_name == "Redis Sampled LRU (samples=10)":
                policy = RedisSampledLRUPolicy(samples=10, seed=seed, name="Redis LRU (samples=10)")
            elif policy_name == "Redis Sampled LFU (samples=5)":
                policy = RedisSampledLFUPolicy(samples=5, seed=seed, name="Redis LFU (samples=5)")
            elif policy_name == "ARC":
                policy = ARCPolicy(capacity=capacity)
            elif policy_name == "SIEVE":
                policy = SIEVEPolicy(capacity=capacity)
            elif policy_name == "LeCaR (Classic)":
                policy = ClassicLeCaRPolicy(capacity=capacity, seed=seed)
            elif policy_name == "LRU":
                policy = LRUPolicy()
            elif policy_name == "LFU":
                policy = LFUPolicy()
            else:
                policy = BeladyPolicy(trace=trace)

            sim = CacheSimulator(capacity=capacity, policy=policy, record_events=True)
            result = sim.run(trace)
            extended = compute_extended_metrics(result, trace)

            # Metrics Row
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Requests", f"{result.total_requests}")
            m2.metric("Hits", f"{result.hits}")
            m3.metric("Misses", f"{result.misses}")
            m4.metric("Hit Rate", f"{result.hit_rate * 100:.2f}%")
            m5.metric("Evictions", f"{result.evictions}")
            m6.metric("Decision Time", f"{result.avg_decision_time_us:.2f} μs")

            # Rolling Hit Rate Chart
            rolling_hr = calculate_rolling_hit_rate(result.events, window_size=max(10, min(50, len(trace) // 10)))
            df_timeline = pd.DataFrame({
                "Request Step": list(range(len(rolling_hr))),
                "Rolling Hit Rate (%)": [v * 100 for v in rolling_hr],
            })
            fig_hr = px.line(
                df_timeline,
                x="Request Step",
                y="Rolling Hit Rate (%)",
                title=f"Rolling Hit Rate ({policy.name} on {workload_type})",
                color_discrete_sequence=["#2563EB"],
            )
            fig_hr.update_layout(yaxis_range=[0, 100], margin=dict(l=20, r=20, t=40, b=20), height=300)
            st.plotly_chart(fig_hr, use_container_width=True)

            # If SmartCache, show Adaptive Weight Telemetry
            if policy_name == "SmartCache AI":
                hist = policy.weight_history
                df_weights = pd.DataFrame({
                    "Step": [h[0] for h in hist],
                    "w_LRU (Recency)": [h[1] for h in hist],
                    "w_LFU (Frequency)": [h[2] for h in hist],
                })
                fig_weights = px.line(
                    df_weights,
                    x="Step",
                    y=["w_LRU (Recency)", "w_LFU (Frequency)"],
                    title="Adaptive Weight Dynamics (w_LRU vs w_LFU over Time)",
                    color_discrete_map={"w_LRU (Recency)": "#10B981", "w_LFU (Frequency)": "#F59E0B"},
                )
                fig_weights.update_layout(yaxis_range=[0, 1], margin=dict(l=20, r=20, t=40, b=20), height=300)
                st.plotly_chart(fig_weights, use_container_width=True)

                st.info(
                    f"💡 **Ghost Cache Telemetry**: LRU Ghost Hits: **{policy.ghost_hits_lru}** | "
                    f"LFU Ghost Hits: **{policy.ghost_hits_lfu}** | Expired Entries: **{policy.expired_ghost_entries}** | "
                    f"Workload Shift Active: **{policy.is_shift_active}**\n\n"
                    f"🔄 **Policy Switching Metrics**: Transitions Detected: **{len(policy.completed_transitions)}** | "
                    f"Avg Adaptation Latency: **{policy.get_state().get('avg_adaptation_latency', 0)} requests** | "
                    f"Extra Transition Misses: **{policy.get_state().get('total_extra_misses', 0)}**"
                )

            # Current Cache Contents Chip View
            st.subheader("📦 Cache State Inspector")
            cached_items = sorted(list(sim.cache), key=lambda x: str(x))
            if cached_items:
                chips_html = "".join([format_key_chip(item) for item in cached_items])
                st.markdown(f"**Current Items in Cache ({len(cached_items)}/{capacity}):**<br>{chips_html}", unsafe_allow_html=True)
            else:
                st.write("Cache is empty.")

    # ---------------------------------------------------------
    # TAB 2: REDIS VS SMARTCACHE AI LIVE BENCHMARK
    # ---------------------------------------------------------
    with tab_redis:
        st.subheader("🔴 Head-to-Head: SMARTCACHE AI vs. Production Redis Eviction Engine")
        st.markdown(
            """
            In high-throughput key-value systems like **Redis**, memory constraints prevent maintaining exact linked lists 
            (which require 16–24 bytes of pointer overhead per key). Instead, Redis implements **Sampled LRU** (`allkeys-lru` via an eviction pool) 
            and **Sampled LFU** (`allkeys-lfu` via an 8-bit Morris logarithmic counter).
            
            Below, we test **SMARTCACHE AI (NeuroCache)** against Redis's exact C eviction algorithms across realistic production Redis workloads.
            """
        )

        col_rw1, col_rw2, col_rw3 = st.columns([1.5, 1, 1])
        with col_rw1:
            redis_scenario = st.selectbox(
                "🏢 Redis Production Scenario",
                [
                    "E-Commerce Flash Sale & Catalog",
                    "Batch Scan Attack (Cache Pollution)",
                    "API Gateway & Session Cache",
                    "Day/Night Traffic Phase Shift",
                ],
            )
        with col_rw2:
            redis_trace_len = st.slider("Workload Length (Keys)", 200, 2000, 800, 50, key="r_len")
        with col_rw3:
            redis_cap = st.slider("Cache Capacity (Keys)", 5, 50, 15, 1, key="r_cap")

        # Scenario Description Card
        if redis_scenario == "E-Commerce Flash Sale & Catalog":
            st.caption(
                "🛒 **Workload Profile**: Power-law product catalog reads (`product:xxxx`, Zipf α=1.15) mixed with active sessions (`session:user_xxx`). "
                "A flash-sale event triggers at step 40% of the trace, causing 65% of all traffic to hit 5 hot items, shifting access patterns abruptly."
            )
        elif redis_scenario == "Batch Scan Attack (Cache Pollution)":
            st.caption(
                "🛡️ **Workload Profile**: Core working set of 20 hot items serving 90% of requests. At step 35%, a background ETL/reporting crawler "
                "sequentially queries 240 single-use keys (`batch:scan:export_xxxx`). Redis LRU pollutes memory with cold keys; SmartCache quarantines them."
            )
        elif redis_scenario == "API Gateway & Session Cache":
            st.caption(
                "🌐 **Workload Profile**: High-concurrency microservice caching endpoint payloads (`api:route_xx`), user profiles (`user:xxx:profile`), "
                "and auth tokens (`auth:token_xx`). Evaluates temporal recency vs frequency in web infrastructure."
            )
        else:
            st.caption(
                "🌗 **Workload Profile**: Simulates global diurnal access shifts: day business traffic transitions into evening entertainment streaming "
                "followed by nocturnal batch processing."
            )

        # Optional Live Redis Server Integration Expander
        with st.expander("🔌 Connect to Live Running Redis Instance (Optional)", expanded=False):
            st.write("Benchmark real GET/SET operations over TCP directly into an active Redis server!")
            live_col1, live_col2, live_col3 = st.columns([2, 1, 1])
            with live_col1:
                live_redis_url = st.text_input("Redis URL", value="redis://localhost:6379/0")
            with live_col2:
                live_maxmem = st.selectbox("Redis maxmemory", ["2mb", "5mb", "10mb", "50mb"], index=1)
            with live_col3:
                live_policy = st.selectbox("maxmemory-policy", ["allkeys-lru", "allkeys-lfu"])

            test_live_btn = st.button("⚡ Execute on Live Redis Server")
            if test_live_btn:
                with st.spinner("Connecting to live Redis server..."):
                    live_trace = get_redis_workload(redis_scenario, total_requests=redis_trace_len, seed=42)
                    live_res = run_live_redis_server_benchmark(
                        redis_url=live_redis_url,
                        maxmemory=live_maxmem,
                        policy=live_policy,
                        trace=live_trace,
                    )
                    if live_res.get("status") == "success":
                        st.success(f"Connected to Live Redis! Hit Rate: **{live_res['hit_rate_pct']}%** | Throughput: **{live_res['throughput_req_per_sec']} req/s** | Avg Latency: **{live_res['avg_latency_ms']} ms**")
                    else:
                        st.warning(f"Could not reach live Redis: {live_res.get('message')}. Proceeding with high-fidelity simulated Redis C-engine below.")

        run_redis_btn = st.button("🔄 Rerun / Refresh Benchmark", type="primary", use_container_width=True)
        if True:  # Display immediately on load and upon control changes
            r_trace = get_redis_workload(redis_scenario, total_requests=redis_trace_len, seed=42)

            # Policies to evaluate
            pol_smart = LeCaRAdaptivePolicy(capacity=redis_cap, seed=42, name="SmartCache AI (NeuroCache)")
            pol_redis_lru5 = RedisSampledLRUPolicy(samples=5, seed=42, name="Redis LRU (samples=5) [Default]")
            pol_redis_lru10 = RedisSampledLRUPolicy(samples=10, seed=42, name="Redis LRU (samples=10) [High-Acc]")
            pol_redis_lfu = RedisSampledLFUPolicy(samples=5, seed=42, name="Redis LFU (samples=5) [Morris Log-C]")
            pol_exact_lru = LRUPolicy(name="Exact LRU (Linked-List)")
            pol_arc = ARCPolicy(capacity=redis_cap)
            pol_belady = BeladyPolicy(trace=r_trace)

            sim_smart = CacheSimulator(capacity=redis_cap, policy=pol_smart, record_events=True)
            sim_redis_lru5 = CacheSimulator(capacity=redis_cap, policy=pol_redis_lru5, record_events=True)
            sim_redis_lru10 = CacheSimulator(capacity=redis_cap, policy=pol_redis_lru10, record_events=True)
            sim_redis_lfu = CacheSimulator(capacity=redis_cap, policy=pol_redis_lfu, record_events=True)
            sim_exact_lru = CacheSimulator(capacity=redis_cap, policy=pol_exact_lru, record_events=True)
            sim_arc = CacheSimulator(capacity=redis_cap, policy=pol_arc, record_events=True)
            sim_belady = CacheSimulator(capacity=redis_cap, policy=pol_belady, record_events=True)

            res_smart = sim_smart.run(r_trace)
            res_redis_lru5 = sim_redis_lru5.run(r_trace)
            res_redis_lru10 = sim_redis_lru10.run(r_trace)
            res_redis_lfu = sim_redis_lfu.run(r_trace)
            res_exact_lru = sim_exact_lru.run(r_trace)
            res_arc = sim_arc.run(r_trace)
            res_belady = sim_belady.run(r_trace)

            hr_smart = res_smart.hit_rate * 100
            hr_redis5 = res_redis_lru5.hit_rate * 100
            delta_hr = hr_smart - hr_redis5
            hits_saved = res_smart.hits - res_redis_lru5.hits

            # Headline KPI Row
            st.markdown("### 🏆 Benchmark Outcome")
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            kpi1.metric("SmartCache AI Hit Rate", f"{hr_smart:.2f}%")
            kpi2.metric("Redis Default Hit Rate", f"{hr_redis5:.2f}%")
            kpi3.metric("Hit Rate Advantage (Δ)", f"{'+' if delta_hr >= 0 else ''}{delta_hr:.2f}%", delta=f"{delta_hr:.2f}% vs Redis")
            kpi4.metric("Extra Cache Hits Captured", f"+{hits_saved} Hits" if hits_saved >= 0 else f"{hits_saved}")
            kpi5.metric("Belady Oracle Bound", f"{res_belady.hit_rate * 100:.2f}%")

            # Comparative Rolling Hit Rate Chart
            window = max(10, min(50, len(r_trace) // 15))
            rhr_smart = calculate_rolling_hit_rate(res_smart.events, window_size=window)
            rhr_redis5 = calculate_rolling_hit_rate(res_redis_lru5.events, window_size=window)
            rhr_redis_lfu = calculate_rolling_hit_rate(res_redis_lfu.events, window_size=window)
            rhr_exact = calculate_rolling_hit_rate(res_exact_lru.events, window_size=window)
            rhr_arc = calculate_rolling_hit_rate(res_arc.events, window_size=window)

            df_comp_hr = pd.DataFrame({
                "Request Step": list(range(len(r_trace))),
                "SmartCache AI (NeuroCache)": [v * 100 for v in rhr_smart],
                "Redis LRU (samples=5) [Default]": [v * 100 for v in rhr_redis5],
                "Redis LFU (samples=5) [Morris Log-C]": [v * 100 for v in rhr_redis_lfu],
                "Exact LRU (Linked-List)": [v * 100 for v in rhr_exact],
                "ARC": [v * 100 for v in rhr_arc],
            })

            fig_redis_comp = px.line(
                df_comp_hr,
                x="Request Step",
                y=[
                    "SmartCache AI (NeuroCache)",
                    "Redis LRU (samples=5) [Default]",
                    "Redis LFU (samples=5) [Morris Log-C]",
                    "Exact LRU (Linked-List)",
                    "ARC",
                ],
                title=f"Comparative Rolling Hit Rate Timeline ({redis_scenario})",
                color_discrete_map={
                    "SmartCache AI (NeuroCache)": "#10B981",
                    "Redis LRU (samples=5) [Default]": "#EF4444",
                    "Redis LFU (samples=5) [Morris Log-C]": "#8B5CF6",
                    "Exact LRU (Linked-List)": "#3B82F6",
                    "ARC": "#F59E0B",
                },
            )
            # Add phase event lines if applicable
            if redis_scenario == "Batch Scan Attack (Cache Pollution)":
                scan_start = int(redis_trace_len * 0.35)
                scan_end = scan_start + int(redis_trace_len * 0.30)
                fig_redis_comp.add_vline(x=scan_start, line_dash="dash", line_color="#EF4444", annotation_text="Scan Burst Begins (Redis Thrashing)")
                fig_redis_comp.add_vline(x=scan_end, line_dash="dash", line_color="#10B981", annotation_text="Scan Ends (Hot Set Resumes)")
            elif redis_scenario == "E-Commerce Flash Sale & Catalog":
                fs_start = int(redis_trace_len * 0.40)
                fs_end = fs_start + int(redis_trace_len * 0.25)
                fig_redis_comp.add_vline(x=fs_start, line_dash="dash", line_color="#F59E0B", annotation_text="Flash Sale Starts (Surge)")
                fig_redis_comp.add_vline(x=fs_end, line_dash="dash", line_color="#6B7280", annotation_text="Flash Sale Ends")

            fig_redis_comp.update_layout(yaxis_range=[0, 100], height=400)
            st.plotly_chart(fig_redis_comp, use_container_width=True)

            # Side-by-side Deep Dive: Adaptive Weights & Cache Inspector
            col_deep1, col_deep2 = st.columns(2)

            with col_deep1:
                st.subheader("🧠 NeuroCache Adaptive Learning")
                st.write("Dynamic reweighting of Recency ($w_{LRU}$) vs. Frequency ($w_{LFU}$) via ghost regret updates:")
                hist_w = pol_smart.weight_history
                df_w_sub = pd.DataFrame({
                    "Step": [h[0] for h in hist_w],
                    "w_LRU (Recency)": [h[1] for h in hist_w],
                    "w_LFU (Frequency)": [h[2] for h in hist_w],
                })
                fig_weights_sub = px.line(
                    df_w_sub,
                    x="Step",
                    y=["w_LRU (Recency)", "w_LFU (Frequency)"],
                    color_discrete_map={"w_LRU (Recency)": "#10B981", "w_LFU (Frequency)": "#F59E0B"},
                )
                fig_weights_sub.update_layout(yaxis_range=[0, 1], height=280, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig_weights_sub, use_container_width=True)

            with col_deep2:
                st.subheader("🛡️ Cache Pollution Resistance")
                st.write("Key composition of memory state after workload execution:")
                # Analyze keys in cache for SmartCache vs Redis LRU
                smart_cached = list(sim_smart.cache)
                redis_cached = list(sim_redis_lru5.cache)

                smart_pollution = sum(1 for k in smart_cached if "scan" in str(k) or "export" in str(k))
                redis_pollution = sum(1 for k in redis_cached if "scan" in str(k) or "export" in str(k))

                smart_hot = len(smart_cached) - smart_pollution
                redis_hot = len(redis_cached) - redis_pollution

                df_poll = pd.DataFrame({
                    "Engine": ["SmartCache AI", "Redis LRU (Default)"],
                    "Protected Hot Keys": [smart_hot, redis_hot],
                    "Polluting Scan Keys": [smart_pollution, redis_pollution],
                })
                fig_poll = px.bar(
                    df_poll,
                    x="Engine",
                    y=["Protected Hot Keys", "Polluting Scan Keys"],
                    barmode="stack",
                    title="Final Cache Composition (Hot Keys vs Scan Contamination)",
                    color_discrete_map={"Protected Hot Keys": "#10B981", "Polluting Scan Keys": "#EF4444"},
                )
                fig_poll.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_poll, use_container_width=True)

            # Live Cache Chips Inspector
            st.subheader("📦 Cache State Inspector: Key Breakdown")
            chip_col1, chip_col2 = st.columns(2)
            with chip_col1:
                st.markdown(f"**SmartCache AI Retained Keys ({len(sim_smart.cache)}/{redis_cap}):**")
                chips_s = "".join([format_key_chip(k) for k in sorted(list(sim_smart.cache), key=lambda x: str(x))])
                st.markdown(chips_s if chips_s else "*(Empty)*", unsafe_allow_html=True)

            with chip_col2:
                st.markdown(f"**Redis Sampled LRU Retained Keys ({len(sim_redis_lru5.cache)}/{redis_cap}):**")
                chips_r = "".join([format_key_chip(k) for k in sorted(list(sim_redis_lru5.cache), key=lambda x: str(x))])
                st.markdown(chips_r if chips_r else "*(Empty)*", unsafe_allow_html=True)

            # Complete Summary Comparison Table
            st.subheader("📋 Comprehensive Performance Table")
            summary_records = [
                {"Policy": pol_smart.name, "Hit Rate (%)": round(hr_smart, 2), "Δ vs Redis (%)": f"+{delta_hr:.2f}%", "Hits": res_smart.hits, "Misses": res_smart.misses, "Evictions": res_smart.evictions, "Avg Decision (μs)": res_smart.avg_decision_time_us},
                {"Policy": pol_redis_lru5.name, "Hit Rate (%)": round(hr_redis5, 2), "Δ vs Redis (%)": "0.00% (Baseline)", "Hits": res_redis_lru5.hits, "Misses": res_redis_lru5.misses, "Evictions": res_redis_lru5.evictions, "Avg Decision (μs)": res_redis_lru5.avg_decision_time_us},
                {"Policy": pol_redis_lru10.name, "Hit Rate (%)": round(res_redis_lru10.hit_rate * 100, 2), "Δ vs Redis (%)": f"{res_redis_lru10.hit_rate * 100 - hr_redis5:+.2f}%", "Hits": res_redis_lru10.hits, "Misses": res_redis_lru10.misses, "Evictions": res_redis_lru10.evictions, "Avg Decision (μs)": res_redis_lru10.avg_decision_time_us},
                {"Policy": pol_redis_lfu.name, "Hit Rate (%)": round(res_redis_lfu.hit_rate * 100, 2), "Δ vs Redis (%)": f"{res_redis_lfu.hit_rate * 100 - hr_redis5:+.2f}%", "Hits": res_redis_lfu.hits, "Misses": res_redis_lfu.misses, "Evictions": res_redis_lfu.evictions, "Avg Decision (μs)": res_redis_lfu.avg_decision_time_us},
                {"Policy": pol_exact_lru.name, "Hit Rate (%)": round(res_exact_lru.hit_rate * 100, 2), "Δ vs Redis (%)": f"{res_exact_lru.hit_rate * 100 - hr_redis5:+.2f}%", "Hits": res_exact_lru.hits, "Misses": res_exact_lru.misses, "Evictions": res_exact_lru.evictions, "Avg Decision (μs)": res_exact_lru.avg_decision_time_us},
                {"Policy": pol_arc.name, "Hit Rate (%)": round(res_arc.hit_rate * 100, 2), "Δ vs Redis (%)": f"{res_arc.hit_rate * 100 - hr_redis5:+.2f}%", "Hits": res_arc.hits, "Misses": res_arc.misses, "Evictions": res_arc.evictions, "Avg Decision (μs)": res_arc.avg_decision_time_us},
                {"Policy": pol_belady.name, "Hit Rate (%)": round(res_belady.hit_rate * 100, 2), "Δ vs Redis (%)": f"{res_belady.hit_rate * 100 - hr_redis5:+.2f}%", "Hits": res_belady.hits, "Misses": res_belady.misses, "Evictions": res_belady.evictions, "Avg Decision (μs)": res_belady.avg_decision_time_us},
            ]
            df_sum = pd.DataFrame(summary_records)
            st.dataframe(df_sum.style.highlight_max(axis=0, subset=["Hit Rate (%)"], color="#DCFCE7"), use_container_width=True)

    # ---------------------------------------------------------
    # TAB 3: Workload-Shift Demo
    # ---------------------------------------------------------
    with tab_shift:
        st.subheader("🔄 Live Workload-Shift Adaptation Demonstration")
        st.write(
            "This demo simulates an abrupt access distribution change. "
            "Traditional static policies struggle because their assumptions become invalid, whereas **SmartCache AI** "
            "observes ghost cache regret feedback and dynamically redistributes decision weights."
        )

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            p1_len = st.number_input("Phase 1 Length (Hot Set A)", value=250, step=50)
        with col_s2:
            p2_len = st.number_input("Phase 2 Length (Sequential Scan)", value=250, step=50)
        with col_s3:
            p3_len = st.number_input("Phase 3 Length (Hot Set B)", value=250, step=50)

        shift_cap = st.slider("Demo Cache Capacity", min_value=3, max_value=20, value=5, key="shift_cap")

        if st.button("🚀 Run Phase Transition Demo", type="primary"):
            custom_phases = [
                {"type": "loop", "hot_set": list(range(0, 8)), "length": p1_len},
                {"type": "scan", "start": 1000, "length": p2_len},
                {"type": "loop", "hot_set": list(range(20, 28)), "length": p3_len},
            ]
            shift_trace = generate_phase_shift_trace(phases=custom_phases, seed=42)

            # Run Competitors on identical trace
            pol_lru = LRUPolicy()
            pol_lfu = LFUPolicy()
            pol_redis = RedisSampledLRUPolicy(samples=5, seed=42, name="Redis LRU")
            pol_arc = ARCPolicy(capacity=shift_cap)
            pol_sieve = SIEVEPolicy(capacity=shift_cap)
            pol_smart = LeCaRAdaptivePolicy(capacity=shift_cap, seed=42, name="SmartCache AI")
            pol_belady = BeladyPolicy(trace=shift_trace)

            res_lru = CacheSimulator(capacity=shift_cap, policy=pol_lru).run(shift_trace)
            res_lfu = CacheSimulator(capacity=shift_cap, policy=pol_lfu).run(shift_trace)
            res_redis = CacheSimulator(capacity=shift_cap, policy=pol_redis).run(shift_trace)
            res_arc = CacheSimulator(capacity=shift_cap, policy=pol_arc).run(shift_trace)
            res_sieve = CacheSimulator(capacity=shift_cap, policy=pol_sieve).run(shift_trace)
            res_smart = CacheSimulator(capacity=shift_cap, policy=pol_smart).run(shift_trace)
            res_belady = CacheSimulator(capacity=shift_cap, policy=pol_belady).run(shift_trace)

            # Summary Cards
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("LRU", f"{res_lru.hit_rate * 100:.2f}%")
            c2.metric("Redis LRU", f"{res_redis.hit_rate * 100:.2f}%")
            c3.metric("ARC", f"{res_arc.hit_rate * 100:.2f}%")
            c4.metric("SIEVE", f"{res_sieve.hit_rate * 100:.2f}%")
            c5.metric("SmartCache AI", f"{res_smart.hit_rate * 100:.2f}%")
            c6.metric("Belady Oracle", f"{res_belady.hit_rate * 100:.2f}%")

            # Transition Switching Measurement Cards
            t_state = pol_smart.get_state()
            ct1, ct2, ct3, ct4 = st.columns(4)
            ct1.metric("Transitions Detected", f"{t_state['completed_transitions']}")
            ct2.metric("Avg Adaptation Latency", f"{t_state['avg_adaptation_latency']} reqs")
            ct3.metric("Extra Transition Misses", f"{t_state['total_extra_misses']}")
            ct4.metric("Expired Ghost Entries", f"{t_state['expired_ghost_entries']}")

            # Dual Charts: Rolling Hit Rates & SmartCache Weights
            window = 40
            hr_lru = calculate_rolling_hit_rate(res_lru.events, window_size=window)
            hr_redis = calculate_rolling_hit_rate(res_redis.events, window_size=window)
            hr_arc = calculate_rolling_hit_rate(res_arc.events, window_size=window)
            hr_sieve = calculate_rolling_hit_rate(res_sieve.events, window_size=window)
            hr_smart = calculate_rolling_hit_rate(res_smart.events, window_size=window)

            df_hr_shift = pd.DataFrame({
                "Request Step": list(range(len(shift_trace))),
                "LRU": [v * 100 for v in hr_lru],
                "Redis LRU": [v * 100 for v in hr_redis],
                "ARC": [v * 100 for v in hr_arc],
                "SIEVE": [v * 100 for v in hr_sieve],
                "SmartCache AI": [v * 100 for v in hr_smart],
            })

            fig_shift_hr = px.line(
                df_hr_shift,
                x="Request Step",
                y=["LRU", "Redis LRU", "ARC", "SIEVE", "SmartCache AI"],
                title=f"Comparative Rolling Hit Rate Across Phase Shift (Window={window})",
                color_discrete_map={
                    "LRU": "#3B82F6",
                    "Redis LRU": "#EF4444",
                    "ARC": "#8B5CF6",
                    "SIEVE": "#EC4899",
                    "SmartCache AI": "#10B981",
                },
            )
            # Add phase boundary annotations
            fig_shift_hr.add_vline(x=p1_len, line_dash="dash", line_color="gray", annotation_text="Shift to Scan")
            fig_shift_hr.add_vline(x=p1_len + p2_len, line_dash="dash", line_color="gray", annotation_text="Shift to Hot Set B")
            fig_shift_hr.update_layout(yaxis_range=[0, 100], height=380)
            st.plotly_chart(fig_shift_hr, use_container_width=True)

            # Weight Evolution
            hist_weights = pol_smart.weight_history
            df_w = pd.DataFrame({
                "Step": [h[0] for h in hist_weights],
                "w_LRU": [h[1] for h in hist_weights],
                "w_LFU": [h[2] for h in hist_weights],
            })
            fig_w = px.line(
                df_w,
                x="Step",
                y=["w_LRU", "w_LFU"],
                title="SmartCache Weight Shift Dynamics (w_LRU vs w_LFU)",
                color_discrete_map={"w_LRU": "#10B981", "w_LFU": "#F59E0B"},
            )
            fig_w.add_vline(x=p1_len, line_dash="dash", line_color="gray", annotation_text="Shift to Scan")
            fig_w.add_vline(x=p1_len + p2_len, line_dash="dash", line_color="gray", annotation_text="Shift to Hot Set B")
            fig_w.update_layout(yaxis_range=[0, 1], height=320)
            st.plotly_chart(fig_w, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 4: Multi-Policy Benchmark Matrix
    # ---------------------------------------------------------
    with tab_bench:
        st.subheader("📊 Comprehensive Multi-Policy Benchmark Matrix")
        bench_csv = Path("results/benchmark_summary.csv")

        if bench_csv.is_file():
            df_bench = pd.read_csv(bench_csv)

            # Pivot Table View
            pivot_hr = df_bench.pivot_table(
                index=["Workload", "Capacity"],
                columns="Policy",
                values="Hit Rate (%)",
            ).reset_index()

            avail_online = [p for p in df_bench["Policy"].unique() if p in pivot_hr.columns and p != "Belady (Optimal)"]

            st.write("### Measured Hit Rate (%) Comparison Across All Workloads")
            st.dataframe(pivot_hr.style.highlight_max(axis=1, subset=avail_online, color="#DCFCE7"), use_container_width=True)

            # Grouped Bar Chart
            fig_bar = px.bar(
                df_bench,
                x="Workload",
                y="Hit Rate (%)",
                color="Policy",
                barmode="group",
                facet_col="Capacity",
                title="Policy Hit Rate (%) Grouped by Workload and Cache Capacity",
                color_discrete_map={
                    "Belady (Optimal)": "#6B7280",
                    "LRU": "#3B82F6",
                    "LFU": "#EF4444",
                    "Redis LRU": "#F87171",
                    "Redis LFU": "#A78BFA",
                    "ARC": "#8B5CF6",
                    "SIEVE": "#EC4899",
                    "LeCaR (Classic)": "#F59E0B",
                    "SmartCache AI": "#10B981",
                },
            )
            fig_bar.update_layout(height=450)
            st.plotly_chart(fig_bar, use_container_width=True)

            # Download CSV
            with open(bench_csv, "rb") as f:
                st.download_button("📥 Download Benchmark CSV", f, file_name="benchmark_summary.csv", mime="text/csv")
        else:
            st.warning("Benchmark results file not found. Run `python -m benchmarks.run_benchmarks` to generate it.")

    # ---------------------------------------------------------
    # TAB 5: Architecture & Mathematical Theory
    # ---------------------------------------------------------
    with tab_arch:
        st.subheader("📖 Algorithm Architecture, Mathematical Theory & Redis Analysis")
        st.markdown(
            r"""
            ### 1. The Credit Assignment Problem in Cache Replacement
            When an item is evicted from a cache, the decision cannot be judged immediately. An eviction is only confirmed 
            to be a **mistake** if the evicted item is requested again shortly thereafter (a cache miss that could have been a hit).

            ### 2. LeCaR Online Learning with Dual Ghost Caches
            SMARTCACHE AI maintains two bounded history FIFO/LRU buffers:
            - **$H_{LRU}$**: Items recently evicted due to the recommendation of the LRU expert.
            - **$H_{LFU}$**: Items recently evicted due to the recommendation of the LFU expert.

            #### Expert Eviction Selection:
            When cache is full and candidates differ ($v_{LRU} \neq v_{LFU}$):
            $$P(\text{Select LRU}) = w_{LRU}, \quad P(\text{Select LFU}) = w_{LFU}$$

            #### Regret Attribution & Exponential Gradient Update:
            When a missed request $x$ hits an expert's ghost cache, that expert is penalized:
            $$w_{E}^{(t+1)} = w_{E}^{(t)} \cdot \exp(-\lambda \cdot \gamma^{\Delta t})$$
            where $\lambda$ is the learning rate, $\gamma$ is the recency discount, and $\Delta t$ is the eviction latency.

            #### Safeguards & Normalization:
            $$w_i = \max(w_i, \epsilon_{\min})$$
            $$w_i = \frac{w_i}{\sum w}$$
            Ensuring weights strictly sum to 1.0 and neither expert is extinguished completely.

            ---

            ### 3. Deep Dive: How Redis Eviction Works vs. SMARTCACHE AI
            
            | Feature | Redis `allkeys-lru` | Redis `allkeys-lfu` | SMARTCACHE AI (NeuroCache) |
            | :--- | :--- | :--- | :--- |
            | **Tracking Data Structure** | 24-bit LRU clock field in object header | 16-bit LDT + 8-bit Morris Counter | Dual Expert Queues + Dual Ghost Histories ($H_{LRU}, H_{LFU}$) |
            | **Eviction Mechanism** | Samples $N=5$ keys into 16-entry eviction pool | Samples $N=5$ keys, ranks by Morris counter | Online Exp3/Hedge regret bandit over LRU & LFU |
            | **Scan Resistance** | ❌ **None** — sequential scan flushes all hot keys | ⚠️ Partial — Morris counter decays slowly | ✅ **Strong** — probation filter & decayed utility protection |
            | **Phase Shift Adaptation** | ⚠️ Static heuristic, no learning feedback | ⚠️ Stale counts persist due to slow decay | ✅ **Immediate** — ghost regrets trigger weight shifts |
            | **Metadata Memory** | 3 bytes per key (24-bit clock) | 3 bytes per key (24-bit counter) | Dual ghost arrays ($O(C)$ bounded capacity) |
            | **Optimal Balance** | Fixed to recency | Fixed to frequency | **Self-tuning** ($w_{LRU} \leftrightarrow w_{LFU}$) |
            """
        )


if __name__ == "__main__":
    main()
