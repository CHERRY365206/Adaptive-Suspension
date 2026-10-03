"""CLI script to run parameter sensitivity and robustness analysis."""

from pathlib import Path
from experiments.sensitivity_analysis import run_parameter_sensitivity_study


def main():
    print("Running parameter sensitivity and robustness analysis under vehicle parameter mismatch...")
    df = run_parameter_sensitivity_study()
    print("\n--- Sensitivity Results ---")
    print(df[["variation_case", "passive_rms_accel_mps2", "active_rms_accel_mps2", "accel_reduction_pct"]].to_string(index=False))
    print("\nArtifacts saved:")
    print("  - results/tables/sensitivity_analysis.csv")
    print("  - results/figures/parameter_sensitivity.png")


if __name__ == "__main__":
    main()
