"""Unit tests for parameter variation and sensitivity analysis."""

import pytest
import pandas as pd
from pathlib import Path
from experiments.sensitivity_analysis import run_parameter_sensitivity_study


def test_sensitivity_study_execution(tmp_path):
    """Verify execution of parameter sensitivity study under parameter mismatch."""
    csv_path = tmp_path / "test_sens.csv"
    fig_path = tmp_path / "test_sens.png"

    df = run_parameter_sensitivity_study(
        output_csv_path=csv_path,
        output_fig_path=fig_path
    )

    assert csv_path.exists()
    assert fig_path.exists()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 7  # 7 variation cases
    assert "variation_case" in df.columns
    assert "active_rms_accel_mps2" in df.columns

    # All accelerations must be finite and bounded
    for _, row in df.iterrows():
        assert row["active_rms_accel_mps2"] < 2.5
        assert row["passive_rms_accel_mps2"] < 2.5

    # Nominal baseline must show significant active reduction
    nom_row = df[df["variation_case"] == "Nominal Baseline"].iloc[0]
    assert nom_row["active_rms_accel_mps2"] < nom_row["passive_rms_accel_mps2"]

    # Damper degradation must correctly detect unsaturated instability
    deg_row = df[df["variation_case"] == "Damper Degradation -30%"].iloc[0]
    assert not bool(deg_row["unsaturated_linear_stable"])
    assert deg_row["max_real_pole"] > 0.0
