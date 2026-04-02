"""
DeepScaler Dashboard — Streamlit interface for cloud workload right-sizing.

Run:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from engine import (
    CLASS_ASYMRMSE,
    CLASS_DATASET_SHARE,
    CLASS_NAMES,
    FEATURE_GROUPS,
    SAFETY_BUFFER,
    DeepScalerAgent,
    Decision,
)

# ═══════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="DeepScaler",
    page_icon="\u2601\uFE0F",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_agent() -> DeepScalerAgent:
    return DeepScalerAgent()


agent = load_agent()

# ═══════════════════════════════════════════════════════════════════════════
# Sidebar — Job Selector
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.title("DeepScaler")

if agent.mode == "production":
    st.sidebar.success("PRODUCTION MODE — trained LightGBM + CatBoost loaded.")
else:
    st.sidebar.warning("DEMO MODE — no trained models found.")

# Load sample jobs for selection
sample_jobs = agent.sample_jobs
if sample_jobs is not None:
    st.sidebar.header("Select a Real Job")

    # Class filter
    class_filter = st.sidebar.multiselect(
        "Filter by scheduling class",
        options=[0, 1, 2, 3],
        default=[2, 3],
        format_func=lambda x: f"Class {x}: {CLASS_NAMES[x]}",
    )
    filtered = sample_jobs[sample_jobs["scheduling_class"].isin(class_filter)].copy()

    if len(filtered) == 0:
        st.sidebar.warning("No jobs match the filter.")
        selected_idx = None
    else:
        # Build display labels
        labels = []
        for i, row in filtered.iterrows():
            cls = int(row["scheduling_class"])
            label = (
                f"[C{cls}] Job {row['collection_id']} | "
                f"CPU req: {row['requested_cpus']:.3f} | "
                f"Actual peak: {row['actual_peak']:.3f}"
            )
            labels.append((i, label))

        selection = st.sidebar.selectbox(
            "Choose a job to analyze",
            options=labels,
            format_func=lambda x: x[1],
        )
        selected_idx = selection[0] if selection else None
else:
    st.sidebar.warning("No sample jobs loaded. Run train.py first.")
    selected_idx = None

# ═══════════════════════════════════════════════════════════════════════════
# Main content
# ═══════════════════════════════════════════════════════════════════════════

st.title("DeepScaler")
st.caption("Predicting Peak CPU Utilization in Cloud Workloads  &mdash;  Wang, Lee, Shankar & Maheshwari (2026)")

tab_predict, tab_performance, tab_savings, tab_about = st.tabs([
    "Prediction", "Model Performance", "Financial Impact", "About",
])

# ─── Tab 1: Prediction ───────────────────────────────────────────────────

with tab_predict:
    if selected_idx is not None:
        result = agent.predict_job(selected_idx)
        job = sample_jobs.iloc[selected_idx]

        # Decision banner
        if result.decision == Decision.REFUSE:
            st.error(f"REFUSED  &mdash;  {result.refusal_reason}")
            col_a, col_b = st.columns(2)
            col_a.metric("Scheduling Class", f"{result.scheduling_class}: {result.class_name}")
            col_b.metric("Actual Peak (ground truth)", f"{result.actual_peak:.4f}")
        else:
            freed = result.cpu_freed_pct or 0
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Predicted Peak", f"{result.predicted_peak_utilization:.4f}")
            col_b.metric("Recommended Ceiling", f"{result.recommended_cpu_ceiling:.4f}")
            col_c.metric("CPU Freed", f"{freed:.1f}%")
            col_d.metric("Actual Peak (truth)", f"{result.actual_peak:.4f}")

            # Right-sizing visualization
            fig = go.Figure()
            # Horizontal lines
            fig.add_hline(
                y=result.original_request, line_dash="solid", line_color="red",
                annotation_text=f"Requested CPUs = {result.original_request:.4f}",
            )
            fig.add_hline(
                y=result.recommended_cpu_ceiling, line_dash="dash", line_color="green",
                annotation_text=f"DeepScaler ceiling = {result.recommended_cpu_ceiling:.4f}",
            )
            fig.add_hline(
                y=result.actual_peak, line_dash="dot", line_color="blue",
                annotation_text=f"Actual peak = {result.actual_peak:.4f}",
            )
            # Reclaimed region
            if result.recommended_cpu_ceiling < result.original_request:
                fig.add_hrect(
                    y0=result.recommended_cpu_ceiling, y1=result.original_request,
                    fillcolor="green", opacity=0.12,
                    annotation_text=f"{freed:.1f}% reclaimed",
                )
            fig.update_layout(
                title="Right-Sizing Recommendation",
                yaxis_title="CPU (fraction of machine)",
                height=350, template="plotly_white",
                xaxis=dict(visible=False),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Reasoning trace
        st.subheader("Agent Reasoning Trace")
        for i, step in enumerate(result.reasoning, 1):
            with st.expander(f"Step {i}: {step.step}", expanded=(result.decision == Decision.REFUSE)):
                st.write(step.detail)

        # Job details table
        st.subheader("Job Details")
        detail_cols = [c for c in job.index if c not in ("ensemble_prediction",)]
        st.dataframe(job[detail_cols].to_frame().T, use_container_width=True, hide_index=True)

        st.caption(f"Mode: **{result.mode}**")
    else:
        st.info("Select a job from the sidebar to see its prediction.")

# ─── Tab 2: Model Performance ────────────────────────────────────────────

with tab_performance:
    st.subheader("GPU Ensemble Performance (Test Set)")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Best AsymRMSE", "23.64", "-67.2% vs baseline")
    col_m2.metric("Under-prediction Rate", "6.4%", "Target: 9.1%")
    col_m3.metric("95% CI", "[23.1, 24.2]")
    col_m4.metric("Baseline AsymRMSE", "72.11")

    # Show local model metrics if available
    if agent._metadata:
        st.markdown("#### Local Trained Model (CPU Ensemble)")
        lm = agent._metadata
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("LGB Test AsymRMSE", f"{lm.get('lgb_test_asymrmse', 0):.2f}")
        lc2.metric("CatBoost Test AsymRMSE", f"{lm.get('cat_test_asymrmse', 0):.2f}")
        lc3.metric("Ensemble Test AsymRMSE", f"{lm.get('ensemble_test_asymrmse', 0):.2f}")
        w = lm.get("ensemble_weights", {})
        st.caption(f"Ensemble weights: LGB={w.get('lgb', 0):.1%}, CatBoost={w.get('cat', 0):.1%}")

    # Model comparison bar chart
    models = ["Cond. Baseline", "Tabular ResNet", "XGBoost", "FT-Transformer",
              "LightGBM", "CatBoost", "CPU Ensemble", "GPU Ensemble"]
    scores = [72.11, 30.51, 27.48, 26.93, 26.10, 25.41, 25.51, 23.64]
    colors = ["#FFC107", "#90A4AE", "#90A4AE", "#42A5F5",
              "#90A4AE", "#90A4AE", "#66BB6A", "#2E7D32"]

    fig_models = go.Figure(go.Bar(
        x=scores, y=models, orientation="h",
        marker_color=colors,
        text=[f"{s:.2f}" for s in scores],
        textposition="outside",
    ))
    fig_models.update_layout(
        title="AsymRMSE by Model (lower is better)",
        xaxis_title="AsymRMSE",
        height=380, template="plotly_white",
        xaxis=dict(range=[0, 80]),
    )
    st.plotly_chart(fig_models, use_container_width=True)

    # AsymRMSE by class
    st.subheader("AsymRMSE by Scheduling Class")
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        fig_class = go.Figure(go.Bar(
            x=list(CLASS_ASYMRMSE.keys()),
            y=list(CLASS_ASYMRMSE.values()),
            marker_color=["#42A5F5", "#EF5350", "#66BB6A", "#66BB6A"],
            text=[f"{v:.2f}" for v in CLASS_ASYMRMSE.values()],
            textposition="outside",
        ))
        fig_class.update_layout(
            title="AsymRMSE by Class (GPU Ensemble)",
            xaxis=dict(
                tickvals=[0, 1, 2, 3],
                ticktext=["0: Best-effort", "1: Burst", "2: Std prod", "3: Hi-pri prod"],
            ),
            yaxis_title="AsymRMSE",
            height=350, template="plotly_white",
        )
        st.plotly_chart(fig_class, use_container_width=True)

    with col_c2:
        fig_share = go.Figure(go.Pie(
            labels=[f"Class {k}: {CLASS_NAMES[k]}" for k in CLASS_DATASET_SHARE],
            values=[v * 100 for v in CLASS_DATASET_SHARE.values()],
            marker_colors=["#42A5F5", "#EF5350", "#66BB6A", "#2E7D32"],
            hole=0.4,
        ))
        fig_share.update_layout(
            title="Dataset Distribution by Class",
            height=350,
        )
        st.plotly_chart(fig_share, use_container_width=True)

    # Sample-level accuracy if we have data
    if sample_jobs is not None and agent.mode == "production":
        st.subheader("Sample Job Predictions vs Actuals")
        prod_jobs = sample_jobs[sample_jobs["scheduling_class"].isin([2, 3])].copy()
        if len(prod_jobs) > 0:
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=prod_jobs["actual_peak"],
                y=prod_jobs["ensemble_prediction"],
                mode="markers",
                marker=dict(size=8, color="#2E7D32", opacity=0.7),
                name="Production jobs (Class 2-3)",
                text=[f"Job {r['collection_id']}" for _, r in prod_jobs.iterrows()],
            ))
            max_val = max(prod_jobs["actual_peak"].max(), prod_jobs["ensemble_prediction"].max()) * 1.1
            fig_scatter.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines", line=dict(dash="dash", color="gray"),
                name="Perfect prediction",
            ))
            fig_scatter.update_layout(
                title="Predicted vs Actual Peak CPU (Production Jobs)",
                xaxis_title="Actual Peak",
                yaxis_title="Ensemble Prediction",
                height=400, template="plotly_white",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    # Modeling journey
    st.subheader("Modeling Journey")
    runs = ["Cond.\nBaseline", "Run 6", "Run 8", "Run 9", "Run 10", "Run 12",
            "CPU\nFinal", "GPU-1", "GPU-2", "GPU-3\n(Best)"]
    asym_values = [72.11, 27.23, 26.85, 26.63, 26.16, 26.10, 25.51, 25.79, 23.66, 23.64]

    fig_journey = go.Figure()
    fig_journey.add_trace(go.Scatter(
        x=list(range(len(runs))), y=asym_values,
        mode="lines+markers", name="AsymRMSE",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=10),
    ))
    fig_journey.add_vrect(x0=6.5, x1=9.5, fillcolor="green", opacity=0.08,
                          annotation_text="GPU runs")
    fig_journey.add_trace(go.Scatter(
        x=[9], y=[23.64], mode="markers",
        marker=dict(size=18, color="green", symbol="star"),
        name="Best: 23.64", showlegend=True,
    ))
    fig_journey.update_layout(
        title="AsymRMSE Across Modeling Iterations",
        xaxis=dict(tickvals=list(range(len(runs))), ticktext=runs),
        yaxis_title="AsymRMSE (lower = better)",
        height=400, template="plotly_white",
    )
    st.plotly_chart(fig_journey, use_container_width=True)


# ─── Tab 3: Financial Impact ─────────────────────────────────────────────

with tab_savings:
    st.subheader("Estimated Annual Savings at Scale")

    st.markdown(
        "DeepScaler predicts a **28% reduction** in resource allocation for production "
        "workloads (classes 2-3, ~64% of tasks). The savings model below replicates "
        "the methodology from the report (Section 5). Adjust parameters to model your own fleet."
    )

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("#### Fleet Parameters")
        fleet_vcpu = st.number_input(
            "Fleet vCPU-hours/year (billions)", 1.0, 1000.0, 134.0, 1.0,
        ) * 1e9
        addressable = st.slider("Addressable fraction", 0.0, 1.0, 0.266, 0.01,
                                help="26.6% = tasks within 7 days + early reclamation premium")
        vcpu_price = st.number_input("vCPU-hour price ($)", 0.001, 1.0, 0.0316, 0.001, format="%.4f",
                                     help="GCP n2-standard: $0.0316/vCPU-hr (March 2026)")

    with col_s2:
        st.markdown("#### Deployment Assumptions")
        coverage = st.slider("Deployment coverage", 0.0, 1.0, 0.30, 0.05,
                             help="Fraction of eligible fleet where DeepScaler is deployed")
        effectiveness = st.slider("Effectiveness", 0.0, 1.0, 0.80, 0.05,
                                  help="Fraction of predicted savings actually realized")

    savings = DeepScalerAgent.estimate_savings(
        fleet_vcpu_hours=fleet_vcpu,
        addressable_fraction=addressable,
        vcpu_price=vcpu_price,
        coverage=coverage,
        effectiveness=effectiveness,
    )

    st.markdown("---")
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric(
        "Upper Bound",
        f"${savings['upper_bound_dollars'] / 1e6:,.0f}M/year",
        help="100% coverage, 100% effectiveness",
    )
    col_r2.metric(
        "Realistic Savings",
        f"${savings['realistic_savings_dollars'] / 1e6:,.0f}M/year",
        delta=f"{coverage:.0%} coverage x {effectiveness:.0%} effectiveness",
    )
    col_r3.metric(
        "Addressable Capacity",
        f"{savings['addressable_vcpu_hours'] / 1e9:,.1f}B vCPU-hrs",
    )

    # Waterfall chart
    upper_m = savings["upper_bound_dollars"] / 1e6
    realistic_m = savings["realistic_savings_dollars"] / 1e6

    fig_waterfall = go.Figure(go.Waterfall(
        x=["Fleet Total", "Addressable\n(26.6%)", f"Priced at\n${vcpu_price:.4f}",
           f"Coverage\n({coverage:.0%})", f"Effectiveness\n({effectiveness:.0%})", "Realistic\nSavings"],
        y=[
            upper_m / (addressable * coverage * effectiveness),
            0,
            upper_m,
            -(upper_m * (1 - coverage)),
            -(upper_m * coverage * (1 - effectiveness)),
            0,
        ],
        measure=["absolute", "relative", "absolute", "relative", "relative", "total"],
        text=[
            f"{fleet_vcpu / 1e9:.0f}B vCPU-hrs",
            f"{savings['addressable_vcpu_hours'] / 1e9:.1f}B vCPU-hrs",
            f"${upper_m:,.0f}M",
            f"-${upper_m * (1 - coverage):,.0f}M",
            f"-${upper_m * coverage * (1 - effectiveness):,.0f}M",
            f"${realistic_m:,.0f}M/year",
        ],
        connector=dict(line=dict(color="gray", dash="dot")),
        increasing=dict(marker=dict(color="#2E7D32")),
        decreasing=dict(marker=dict(color="#EF5350")),
        totals=dict(marker=dict(color="#1565C0")),
    ))
    fig_waterfall.update_layout(
        title="Savings Derivation (Report Section 5)",
        yaxis_title="$ Millions / year",
        height=450, template="plotly_white",
        showlegend=False,
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)

    st.markdown("#### Derivation (Report Section 5)")
    st.markdown(f"""
| Step | Calculation | Value |
|------|------------|-------|
| 1. Fleet capacity | Total production vCPU-hours/year | **{fleet_vcpu/1e9:.0f}B** |
| 2. Addressable capacity | Fleet x {addressable:.1%} | **{savings['addressable_vcpu_hours']/1e9:.1f}B** vCPU-hrs |
| 3. Upper bound | Addressable x ${vcpu_price:.4f}/vCPU-hr | **${savings['upper_bound_dollars']/1e6:,.0f}M**/year |
| 4. Realistic deployment | Upper bound x {coverage:.0%} x {effectiveness:.0%} | **${savings['realistic_savings_dollars']/1e6:,.0f}M**/year |
""")


# ─── Tab 4: About ────────────────────────────────────────────────────────

with tab_about:
    st.subheader("About DeepScaler")
    st.markdown("""
**Problem**: Cloud data centers are massively inefficient. Engineers request far more CPU
than their jobs consume, leaving expensive compute idle. Google's Autopilot requires
7 days of observation history and makes near-zero adjustments for short-lived batch jobs.

**Solution**: DeepScaler predicts peak CPU utilization from the **first 15 minutes** of a
job's execution trace, enabling quick right-sizing. The system uses a 72-feature pipeline
and a 3-model GPU ensemble (LightGBM + CatBoost + FT-Transformer).

**Key Results**:
- **67.2% improvement** over baseline (AsymRMSE: 23.64 vs 72.11)
- **6.4% under-prediction rate** (near the 9.1% theoretical optimum)
- **$270M/year** estimated savings at Google's scale

**Safety Mechanisms**:
1. Class gate: **refuses** to right-size Class 0-1 (burst) jobs due to the structural error floor
2. 10% safety buffer on all predictions, capped at original request
3. Burst-user override: substitutes user's historical 91st-percentile for high-variance users

---
*Wang, Lee, Shankar & Maheshwari. DeepScaler: Predicting Peak CPU Utilization
in Cloud Workloads. OIT 367, Stanford GSB, March 2026.*
""")

    st.subheader("Scheduling Classes")
    class_data = []
    for cls in sorted(CLASS_NAMES):
        class_data.append({
            "Class": cls,
            "Name": CLASS_NAMES[cls],
            "Share": f"{CLASS_DATASET_SHARE[cls]:.0%}",
            "AsymRMSE": CLASS_ASYMRMSE[cls],
            "Right-sizeable": "Yes" if cls >= 2 else "No",
        })
    st.table(class_data)

    st.subheader("Architecture")
    st.markdown("""
```
           +──────────────────────────────────+
           |   Job Telemetry (0-15 min)       |
           +──────────────┬───────────────────+
                          v
           +──────────────────────────────────+
           |   72-Feature Engineering Pipeline |
           |   (6 groups: metadata, user       |
           |    history, time-series, hist,    |
           |    trend, interaction)            |
           +──────────────┬───────────────────+
                          v
           +──────────────────────────────────+
           |   Class Gate                      |
           |   Class 0-1 -> REFUSE             |
           |   Class 2-3 -> predict            |
           +──────────────┬───────────────────+
                          v
     +──────────+  +──────────+  +──────────────+
     | LightGBM |  | CatBoost |  | FT-Transformer|
     | q=0.9091 |  | q=0.90   |  | (attention)   |
     +────┬─────+  +────┬─────+  +──────┬────────+
          |              |               |
          v              v               v
     +──────────────────────────────────────────+
     |   SLSQP Ensemble (weights from training) |
     |   -> log-space blend -> expm1             |
     +──────────────┬───────────────────────────+
                    v
     +──────────────────────────────────────────+
     |   Safety: +10% buffer, cap at request    |
     +──────────────┬───────────────────────────+
                    v
     +──────────────────────────────────────────+
     |   Recommended CPU Ceiling                |
     +──────────────────────────────────────────+
```
""")
