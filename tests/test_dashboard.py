"""Unit tests for dashboard visualization helpers and imports."""

import pytest
import numpy as np
import pandas as pd

from dashboard.visualizations import (
    plot_simulation_time_series,
    plot_comparison_bar_chart,
    plot_pso_convergence_interactive,
    plot_feature_importance_interactive
)
from models.quarter_car import SimulationResult


def test_dashboard_visualizations_create_figures():
    """Verify that dashboard Plotly figure builders execute without errors and produce valid Figure objects."""
    t = np.linspace(0, 1.0, 100)
    dummy_res = SimulationResult(
        t=t,
        zs=np.zeros_like(t),
        zs_dot=np.zeros_like(t),
        zs_ddot=np.zeros_like(t),
        zu=np.zeros_like(t),
        zu_dot=np.zeros_like(t),
        zu_ddot=np.zeros_like(t),
        suspension_travel=np.zeros_like(t),
        tyre_deflection=np.zeros_like(t),
        zr=np.zeros_like(t),
        u_requested=np.zeros_like(t),
        u_applied=np.zeros_like(t),
        dynamic_tyre_load=np.zeros_like(t),
        total_tyre_load=np.full_like(t, 4000.0),
        loss_of_contact=np.zeros(len(t), dtype=bool),
        travel_exceeded=np.zeros(len(t), dtype=bool),
        power_mechanical=np.zeros_like(t),
        power_electrical=np.zeros_like(t)
    )

    fig_sim = plot_simulation_time_series(dummy_res, "Test Sim")
    assert fig_sim is not None
    assert len(fig_sim.data) >= 5

    df_dummy = pd.DataFrame([
        {"system_type": "passive", "rms_body_accel_mps2": 1.5, "rms_susp_travel_m": 0.02, "rms_dynamic_tyre_load_n": 800.0},
        {"system_type": "proposed_ai_pso", "rms_body_accel_mps2": 0.8, "rms_susp_travel_m": 0.015, "rms_dynamic_tyre_load_n": 700.0}
    ])
    fig_bar = plot_comparison_bar_chart(df_dummy)
    assert fig_bar is not None
    assert len(fig_bar.data) == 3

    pso_data = {
        "comfort": {"history": [{"iteration": 0, "best_cost": 2.0}, {"iteration": 1, "best_cost": 1.2}], "best_cost": 1.2}
    }
    fig_pso = plot_pso_convergence_interactive(pso_data)
    assert fig_pso is not None

    feat_dict = {"body_accel_rms": 0.35, "wheel_accel_rms": 0.25}
    fig_imp = plot_feature_importance_interactive(feat_dict)
    assert fig_imp is not None
