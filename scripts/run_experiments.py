"""CLI script to run comparative benchmark experiments across all 6 suspension systems."""

from __future__ import annotations
import argparse
from pathlib import Path
from experiments.runner import run_comparative_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run complete comparative benchmark on held-out test scenarios.")
    parser.add_argument("--quick", action="store_true", help="Quick mode for rapid testing")
    args = parser.parse_args()

    print(f"Executing comparative suspension benchmarks (quick={args.quick})...")
    res = run_comparative_benchmark(quick_mode=args.quick)

    print("\n--- Benchmark Execution Summary ---")
    df = res["dataframe"]
    summary = df.groupby("system_display_name")[[
        "rms_body_accel_mps2",
        "peak_body_accel_mps2",
        "rms_susp_travel_m",
        "rms_dynamic_tyre_load_n",
        "saturation_pct"
    ]].mean()
    print(summary.to_string())

    print("\nArtifacts saved:")
    print("  - results/tables/experiments_summary.csv")
    print("  - results/tables/aggregated_metrics.json")
    print("  - results/figures/comparative_metrics.png")
    print("  - results/figures/time_series_comparison.png")


if __name__ == "__main__":
    main()
