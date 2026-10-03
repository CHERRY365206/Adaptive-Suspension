"""Scientific visualization helpers using Plotly for interactive dashboards."""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from models.quarter_car import SimulationResult
from experiments.baselines import SYSTEM_DISPLAY_NAMES


def plot_simulation_time_series(res: SimulationResult, title: str = "Quarter-Car Simulation Response") -> go.Figure:
    """Create interactive multi-trace time domain plot for a quarter-car simulation run."""
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[
            "1. Vertical Displacements [m]",
            "2. Body Vertical Acceleration [m/s²]",
            "3. Suspension Travel (zs - zu) [m]",
            "4. Dynamic Tyre Load (kt * (zu - zr)) [N]",
            "5. Actuator Force (Requested vs Applied) [N]"
        ]
    )

    t = res.t

    # 1. Displacements
    fig.add_trace(go.Scatter(x=t, y=res.zr, name="Road Profile (zr)", line=dict(color="#444444", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=res.zs, name="Body Disp (zs)", line=dict(color="#1f77b4", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=res.zu, name="Wheel Disp (zu)", line=dict(color="#2ca02c", width=1.5)), row=1, col=1)

    # 2. Acceleration
    fig.add_trace(go.Scatter(x=t, y=res.zs_ddot, name="Body Accel (zs_ddot)", line=dict(color="#d62728", width=1.5)), row=2, col=1)

    # 3. Suspension Travel with limit lines
    fig.add_trace(go.Scatter(x=t, y=res.suspension_travel, name="Susp Travel", line=dict(color="#9467bd", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=[t[0], t[-1]], y=[0.08, 0.08], name="+Max Travel (80mm)", line=dict(color="red", dash="dot")), row=3, col=1)
    fig.add_trace(go.Scatter(x=[t[0], t[-1]], y=[-0.08, -0.08], name="-Max Travel (80mm)", line=dict(color="red", dash="dot"), showlegend=False), row=3, col=1)

    # 4. Dynamic Tyre Load
    fig.add_trace(go.Scatter(x=t, y=res.dynamic_tyre_load, name="Dynamic Tyre Load", line=dict(color="#ff7f0e", width=1.5)), row=4, col=1)

    # 5. Actuator Force
    fig.add_trace(go.Scatter(x=t, y=res.u_requested, name="Requested Force", line=dict(color="#8c564b", dash="dot")), row=5, col=1)
    fig.add_trace(go.Scatter(x=t, y=res.u_applied, name="Applied Force", line=dict(color="#17becf", width=2)), row=5, col=1)

    fig.update_layout(
        height=850,
        title=dict(text=title, font=dict(size=15, weight="bold")),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
        template="plotly_white"
    )
    fig.update_xaxes(title_text="Time [s]", row=5, col=1)
    return fig


def plot_comparison_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Create interactive bar chart comparing key metrics across all systems."""
    summary = df.groupby("system_type")[["rms_body_accel_mps2", "rms_susp_travel_m", "rms_dynamic_tyre_load_n"]].mean().reset_index()

    display_names = [SYSTEM_DISPLAY_NAMES.get(s, s) for s in summary["system_type"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=display_names,
        y=summary["rms_body_accel_mps2"],
        name="RMS Body Accel [m/s²]",
        marker_color="#1f77b4"
    ))
    fig.add_trace(go.Bar(
        x=display_names,
        y=summary["rms_susp_travel_m"] * 1000.0,
        name="RMS Travel [mm]",
        marker_color="#2ca02c"
    ))
    fig.add_trace(go.Bar(
        x=display_names,
        y=summary["rms_dynamic_tyre_load_n"] / 100.0,
        name="RMS Dynamic Tyre Load [x100 N]",
        marker_color="#d62728"
    ))

    fig.update_layout(
        barmode="group",
        title="Comparative Performance Across All Evaluated Suspension Architectures",
        xaxis_title="Suspension System",
        yaxis_title="Metric Value",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_pso_convergence_interactive(pso_results: Dict[str, Any]) -> go.Figure:
    """Plotly convergence curves for all tuned controller modes."""
    fig = go.Figure()

    colors = {"comfort": "#1f77b4", "balanced": "#2ca02c", "road_holding": "#d62728"}

    for mode, data in pso_results.items():
        history = data.get("history", [])
        if not history:
            continue
        iters = [h["iteration"] for h in history]
        bests = [h["best_cost"] for h in history]

        fig.add_trace(go.Scatter(
            x=iters,
            y=bests,
            mode="lines+markers",
            name=f"PSO {mode.capitalize()} (Best: {data['best_cost']:.3f})",
            line=dict(color=colors.get(mode, "#333333"), width=2)
        ))

    fig.update_layout(
        title="Particle Swarm Optimization Convergence Curves",
        xaxis_title="Iteration",
        yaxis_title="Normalized Objective Cost",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig


def plot_feature_importance_interactive(feat_dict: Dict[str, float]) -> go.Figure:
    """Plot horizontal bar chart of Random Forest feature importances."""
    sorted_items = sorted(feat_dict.items(), key=lambda x: x[1])
    names = [k.replace("_", " ").title() for k, v in sorted_items]
    vals = [v for k, v in sorted_items]

    fig = go.Figure(go.Bar(
        x=vals,
        y=names,
        orientation="h",
        marker_color="#1f77b4"
    ))
    fig.update_layout(
        title="Random Forest Gini Feature Importances",
        xaxis_title="Importance Ratio",
        template="plotly_white",
        height=600
    )
    return fig
