"""Integration tests running end-to-end multi-system simulation comparisons."""

import pytest
from pathlib import Path
from roads.road_dataset import create_scenario
from experiments.runner import run_comparative_benchmark


def test_end_to_end_benchmarks_quick(tmp_path):
    """Run an end-to-end benchmark comparison across all 6 systems on a single test scenario."""
    csv_path = tmp_path / "test_summary.csv"
    json_path = tmp_path / "test_agg.json"
    fig_path = tmp_path / "test_bar.png"
    ts_fig_path = tmp_path / "test_ts.png"

    # Single held-out test scenario
    scenarios = [
        create_scenario(scenario_id=999, road_type="bumpy", speed_kmh=45.0, split="test", duration_s=1.5, seed=999)
    ]

    results = run_comparative_benchmark(
        scenarios=scenarios,
        output_csv_path=csv_path,
        output_json_path=json_path,
        output_fig_path=fig_path,
        output_ts_fig_path=ts_fig_path,
        quick_mode=True,
        dt=0.01
    )

    assert csv_path.exists()
    assert json_path.exists()
    assert fig_path.exists()
    assert ts_fig_path.exists()

    df = results["dataframe"]
    # 6 systems for 1 scenario = 6 rows
    assert len(df) == 6
    assert set(df["system_type"].unique()) == {
        "passive",
        "fixed_nominal_lqr",
        "fixed_pso_lqr",
        "adaptive_no_pso",
        "proposed_ai_pso",
        "oracle_adaptive"
    }

    # All metrics must be finite numbers
    for col in ["rms_body_accel_mps2", "rms_susp_travel_m", "rms_dynamic_tyre_load_n"]:
        assert df[col].notna().all()
        assert (df[col] >= 0.0).all()
