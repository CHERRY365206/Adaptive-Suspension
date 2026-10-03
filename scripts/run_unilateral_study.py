"""CLI script to run unilateral tyre contact study."""

from pathlib import Path
from experiments.unilateral_contact_study import run_unilateral_contact_comparison


def main():
    print("Running unilateral tyre contact study (loss-of-contact & impact shock analysis)...")
    res = run_unilateral_contact_comparison()
    print("\n--- Unilateral Contact Comparison Results ---")
    lin = res["linear_model"]
    uni = res["unilateral_model"]
    print(f"Linear Tyre Model:     Peak Accel={lin['peak_body_accel_mps2']:.3f} m/s², Min Normal Force={lin['min_total_tyre_load_n']:.1f} N, Contact Loss Duration={lin['contact_loss_duration_s']:.3f} s")
    print(f"Unilateral Tyre Model: Peak Accel={uni['peak_body_accel_mps2']:.3f} m/s², Min Normal Force={uni['min_total_tyre_load_n']:.1f} N, Contact Loss Duration={uni['contact_loss_duration_s']:.3f} s")
    print("\nArtifacts saved:")
    print("  - results/tables/unilateral_contact_metrics.json")
    print("  - results/figures/unilateral_contact_comparison.png")


if __name__ == "__main__":
    main()
