"""SMARTCACHE AI — Interactive Streamlit Dashboard & Evaluation Visualizer.

Provides live cache simulation, real-time weight adaptation telemetry,
workload-shift demo, and multi-policy benchmark comparisons.
"""
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cache_simulator import CacheSimulator
from policies import LRUPolicy, LFUPolicy, BeladyPolicy, LeCaRAdaptivePolicy
from workloads.generators import (
    generate_looping_trace,
    generate_sequential_scan_trace,
    generate_zipf_trace,
    generate_phase_shift_trace,
    generate_mixed_trace,
)
from workloads.loaders import load_trace_from_file
from benchmarks.metrics import calculate_rolling_hit_rate, compute_extended_metrics


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
    elif workload_type == "Custom Uploaded File" and uploaded_file is not None:
        temp_path = Path("temp_upload.txt")
        temp_path.write_bytes(uploaded_file.getbuffer())
        trace = load_trace_from_file(temp_path)
        temp_path.unlink(missing_ok=True)
        return trace[:length] if trace else [0] * length
    return generate_mixed_trace(total_requests=length, seed=seed)


def main():
    st.markdown('<div class="main-title">🧠 SMARTCACHE AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Adaptive Cache Replacement System • Team BONGORITHM • <em>"Learning what to keep before you need it."</em></div>',
        unsafe_allow_html=True,
    )

    tab_sim, tab_shift, tab_bench, tab_arch = st.tabs([
        "⚡ Live Simulator & Telemetry",
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
                ["SmartCache (LeCaR)", "LRU", "LFU", "Belady (Optimal)"],
            )

            # Adaptive Hyperparameters
            learning_rate = None
            discount = 1.0
            history_mult = 1.0
            deterministic = False

            if policy_name == "SmartCache (LeCaR)":
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
            if policy_name == "SmartCache (LeCaR)":
                policy = LeCaRAdaptivePolicy(
                    capacity=capacity,
                    learning_rate=learning_rate,
                    discount=discount,
                    history_multiplier=history_mult,
                    deterministic=deterministic,
                    seed=seed,
                )
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
                title=f"Rolling Hit Rate ({policy_name} on {workload_type})",
                color_discrete_sequence=["#2563EB"],
            )
            fig_hr.update_layout(yaxis_range=[0, 100], margin=dict(l=20, r=20, t=40, b=20), height=300)
            st.plotly_chart(fig_hr, use_container_width=True)

            # If SmartCache, show Adaptive Weight Telemetry
            if policy_name == "SmartCache (LeCaR)":
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
                    f"LFU Ghost Hits: **{policy.ghost_hits_lfu}** | Active Ghost LRU Size: {len(policy.h_lru)} | Ghost LFU Size: {len(policy.h_lfu)}"
                )

            # Current Cache Contents Chip View
            st.subheader("📦 Cache State Inspector")
            cached_items = sorted(list(sim.cache))
            if cached_items:
                chips_html = "".join([f'<span style="display:inline-block;background:#E0E7FF;color:#1E40AF;padding:4px 10px;margin:3px;border-radius:14px;font-weight:600;font-size:0.85rem">Item {item}</span>' for item in cached_items])
                st.markdown(f"**Current Items in Cache ({len(cached_items)}/{capacity}):**<br>{chips_html}", unsafe_allow_html=True)
            else:
                st.write("Cache is empty.")

    # ---------------------------------------------------------
    # TAB 2: Workload-Shift Demo
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

            # Run LRU, LFU, SmartCache, Belady on identical trace
            pol_lru = LRUPolicy()
            pol_lfu = LFUPolicy()
            pol_smart = LeCaRAdaptivePolicy(capacity=shift_cap, seed=42)
            pol_belady = BeladyPolicy(trace=shift_trace)

            res_lru = CacheSimulator(capacity=shift_cap, policy=pol_lru).run(shift_trace)
            res_lfu = CacheSimulator(capacity=shift_cap, policy=pol_lfu).run(shift_trace)
            res_smart = CacheSimulator(capacity=shift_cap, policy=pol_smart).run(shift_trace)
            res_belady = CacheSimulator(capacity=shift_cap, policy=pol_belady).run(shift_trace)

            # Summary Cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("LRU Hit Rate", f"{res_lru.hit_rate * 100:.2f}%")
            c2.metric("LFU Hit Rate", f"{res_lfu.hit_rate * 100:.2f}%")
            c3.metric("SmartCache Hit Rate", f"{res_smart.hit_rate * 100:.2f}%", delta=f"{(res_smart.hit_rate - res_lfu.hit_rate)*100:+.2f}% vs LFU")
            c4.metric("Belady Oracle (Upper Bound)", f"{res_belady.hit_rate * 100:.2f}%")

            # Dual Charts: Rolling Hit Rates & SmartCache Weights
            window = 40
            hr_lru = calculate_rolling_hit_rate(res_lru.events, window_size=window)
            hr_lfu = calculate_rolling_hit_rate(res_lfu.events, window_size=window)
            hr_smart = calculate_rolling_hit_rate(res_smart.events, window_size=window)

            df_hr_shift = pd.DataFrame({
                "Request Step": list(range(len(shift_trace))),
                "LRU": [v * 100 for v in hr_lru],
                "LFU": [v * 100 for v in hr_lfu],
                "SmartCache (LeCaR)": [v * 100 for v in hr_smart],
            })

            fig_shift_hr = px.line(
                df_hr_shift,
                x="Request Step",
                y=["LRU", "LFU", "SmartCache (LeCaR)"],
                title=f"Comparative Rolling Hit Rate Across Phase Shift (Window={window})",
                color_discrete_map={"LRU": "#3B82F6", "LFU": "#EF4444", "SmartCache (LeCaR)": "#10B981"},
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
    # TAB 3: Multi-Policy Benchmark Matrix
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

            st.write("### Measured Hit Rate (%) Comparison")
            st.dataframe(pivot_hr.style.highlight_max(axis=1, subset=["LFU", "LRU", "SmartCache (LeCaR)"], color="#DCFCE7"), use_container_width=True)

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
                    "SmartCache (LeCaR)": "#10B981",
                },
            )
            fig_bar.update_layout(height=450)
            st.plotly_chart(fig_bar, use_container_width=True)

            # Overhead Chart
            fig_overhead = px.bar(
                df_bench[df_bench["Capacity"] == 10],
                x="Policy",
                y="Avg Decision (us)",
                color="Policy",
                title="Average Policy Decision Overhead (Microseconds per Request at Capacity 10)",
            )
            fig_overhead.update_layout(height=350)
            st.plotly_chart(fig_overhead, use_container_width=True)

            # Download CSV
            with open(bench_csv, "rb") as f:
                st.download_button("📥 Download Benchmark CSV", f, file_name="benchmark_summary.csv", mime="text/csv")
        else:
            st.warning("Benchmark results file not found. Run `python -m benchmarks.run_benchmarks` to generate it.")

    # ---------------------------------------------------------
    # TAB 4: Architecture & Mathematical Theory
    # ---------------------------------------------------------
    with tab_arch:
        st.subheader("📖 SMARTCACHE AI: Algorithm Architecture & Theory")
        st.markdown(
            """
            ### 1. The Credit Assignment Problem in Cache Replacement
            When an item is evicted from a cache, the decision cannot be judged immediately. An eviction is only confirmed 
            to be a **mistake** if the evicted item is requested again shortly thereafter (a cache miss that could have been a hit).

            ### 2. LeCaR Online Learning with Dual Ghost Caches
            SMARTCACHE AI maintains two bounded history FIFO/LRU buffers:
            - **$H_{LRU}$**: Items recently evicted due to the recommendation of the LRU expert.
            - **$H_{LFU}$**: Items recently evicted due to the recommendation of the LFU expert.

            #### Expert Eviction Selection:
            When cache is full and candidates differ ($v_{LRU} \\neq v_{LFU}$):
            $$P(\\text{Select LRU}) = w_{LRU}, \\quad P(\\text{Select LFU}) = w_{LFU}$$

            #### Regret Attribution & Exponential Gradient Update:
            When a missed request $x$ hits an expert's ghost cache, that expert is penalized:
            $$w_{E}^{(t+1)} = w_{E}^{(t)} \\cdot \\exp(-\\lambda \\cdot \\gamma^{\\Delta t})$$
            where $\\lambda$ is the learning rate, $\\gamma$ is the recency discount, and $\\Delta t$ is the eviction latency.

            #### Safeguards & Normalization:
            $$w_i = \\max(w_i, \\epsilon_{\\min})$$
            $$w_i = \\frac{w_i}{\\sum w}$$
            Ensuring weights strictly sum to 1.0 and neither expert is extinguished completely.
            """
        )


if __name__ == "__main__":
    main()
