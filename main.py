"""Master entry point for the AI-Assisted Adaptive Active Suspension System for Electric Vehicles.

Supports single-command execution of dataset generation, model training, PSO optimization,
and comparative experimental benchmarks, as well as standalone dynamic simulation runs.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.vehicle_parameters import VehicleParameters
from models.actuator import ActuatorModel
from roads.road_dataset import create_scenario
from ai.train_classifier import train_road_classifier
from optimization.optimize_lqr import run_pso_tuning
from experiments.runner import run_comparative_benchmark
from experiments.baselines import SuspensionBenchmarkSuite
from experiments.sensitivity_analysis import run_parameter_sensitivity_study
from experiments.unilateral_contact_study import run_unilateral_contact_comparison


def main():
    parser = argparse.ArgumentParser(
        description="AI-Assisted Adaptive Active Suspension System for Electric Vehicles."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "train", "optimize", "experiment", "sensitivity", "unilateral", "simulate", "test"],
        default="all",
        help="Pipeline stage to execute (default: all)"
    )
    parser.add_argument("--quick", action="store_true", help="Run with reduced iterations and scenarios for rapid verification")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed (default: 42)")
    parser.add_argument("--road", choices=["smooth", "bumpy", "rough"], default="bumpy", help="Road profile for simulate mode")
    parser.add_argument("--speed", type=float, default=50.0, help="Vehicle speed [km/h] for simulate mode")
    parser.add_argument("--duration", type=float, default=6.0, help="Duration [s] for simulate mode")
    args = parser.parse_args()

    print("================================================================================")
    print("AI-Assisted Adaptive Active Suspension System for Electric Vehicles")
    print(f"Mode: {args.mode.upper()} | Quick: {args.quick} | Master Seed: {args.seed}")
    print("================================================================================\n")

    if args.mode in ("train", "all"):
        print("[STAGE 1/5] Training Random Forest Road Classifier...")
        metrics = train_road_classifier(seed=args.seed, quick_mode=args.quick)
        print(f"  Classification Accuracy: {metrics['accuracy'] * 100.0:.2f}%")
        print(f"  Macro F1-Score:          {metrics['macro_f1'] * 100.0:.2f}%")
        print("  Artifact saved to artifacts/models/road_classifier.joblib\n")

    if args.mode in ("optimize", "all"):
        print("[STAGE 2/5] Running Offline Particle Swarm Optimization (PSO)...")
        pso_res = run_pso_tuning(seed=args.seed, quick_mode=args.quick)
        for mode, data in pso_res["results"].items():
            print(f"  Tuned {mode.capitalize()} LQR: Cost={data['best_cost']:.4f}")
        print("  Artifact saved to artifacts/controllers/pso_lqr_configs.json\n")

    if args.mode in ("experiment", "all"):
        print("[STAGE 3/5] Running Comparative Benchmark Across 6 Suspension Systems...")
        res = run_comparative_benchmark(quick_mode=args.quick)
        df = res["dataframe"]
        summary = df.groupby("system_display_name")[[
            "rms_body_accel_mps2",
            "peak_body_accel_mps2",
            "rms_susp_travel_m",
            "rms_dynamic_tyre_load_n"
        ]].mean()
        print("\nAggregated Results:")
        print(summary.round(4).to_string())
        print("\n  CSV exported to results/tables/experiments_summary.csv")
        print("  Figures exported to results/figures/\n")

    if args.mode in ("sensitivity", "all"):
        print("[STAGE 4/5] Running Parameter Sensitivity & Robustness Study...")
        df_sens = run_parameter_sensitivity_study()
        print("  Sensitivity analysis completed across payload, spring, damper, and tyre variations.")
        print(f"  CSV exported to results/tables/sensitivity_analysis.csv\n")

    if args.mode in ("unilateral", "all"):
        print("[STAGE 5/5] Running Unilateral Tyre Contact Dynamics Investigation...")
        u_metrics = run_unilateral_contact_comparison()
        lin_pk = u_metrics["linear_model"]["peak_body_accel_mps2"]
        uni_pk = u_metrics["unilateral_model"]["peak_body_accel_mps2"]
        air_ms = u_metrics["unilateral_model"]["contact_loss_duration_s"] * 1000.0
        print(f"  Linear Peak Accel:     {lin_pk:.2f} m/s^2")
        print(f"  Unilateral Peak Accel: {uni_pk:.2f} m/s^2 (Impact landing shock)")
        print(f"  Tyre Airborne Duration: {air_ms:.1f} ms")
        print("  JSON exported to results/tables/unilateral_contact_metrics.json\n")

    if args.mode == "simulate":
        print(f"Simulating single quarter-car run on {args.road.upper()} road at {args.speed:.1f} km/h...")
        suite = SuspensionBenchmarkSuite()
        sc = create_scenario(
            scenario_id=1,
            road_type=args.road,
            speed_kmh=args.speed,
            split="test",
            duration_s=args.duration,
            seed=args.seed
        )
        road_fn = sc.road_profile.get_time_callable(sc.speed_mps)

        for sys_name in ["passive", "proposed_ai_pso"]:
            res = suite.simulate_system(sys_name, road_fn, sc.road_type, duration_s=args.duration)
            print(f"\n[{sys_name.upper()}]")
            print(f"  RMS Body Accel:        {res.rms_body_accel:.4f} m/s^2")
            print(f"  RMS Suspension Travel: {res.rms_suspension_travel * 1000.0:.2f} mm")
            print(f"  RMS Dynamic Tyre Load: {res.rms_dynamic_tyre_load:.1f} N")

    if args.mode == "test":
        import subprocess
        print("Running full pytest test suite...")
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], check=True)

    print("\nWorkflow completed successfully!")
    print("To launch interactive dashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
