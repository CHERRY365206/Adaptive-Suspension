"""CLI script to run offline Particle Swarm Optimization for LQR controllers."""

from __future__ import annotations
import argparse
from pathlib import Path
from optimization.optimize_lqr import run_pso_tuning


def main():
    parser = argparse.ArgumentParser(description="Run offline PSO tuning for comfort, balanced, and road-holding LQR controllers.")
    parser.add_argument("--quick", action="store_true", help="Quick mode for rapid testing")
    parser.add_argument("--swarm-size", type=int, default=15, help="Number of particles (default: 15)")
    parser.add_argument("--iterations", type=int, default=20, help="Max iterations (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Optimization seed (default: 42)")
    args = parser.parse_args()

    print(f"Starting PSO tuning (quick={args.quick}, seed={args.seed})...")
    res = run_pso_tuning(
        swarm_size=args.swarm_size,
        max_iterations=args.iterations,
        seed=args.seed,
        quick_mode=args.quick
    )

    print("\n--- PSO Controller Tuning Results ---")
    for mode, data in res["results"].items():
        print(f"\n[{mode.upper()} LQR]")
        print(f"  Best Normalized Cost: {data['best_cost']:.4f}")
        print(f"  Runtime:              {data['runtime_seconds']:.2f} s")
        print(f"  Feedback Gain K:      {data['gain_K']}")

    print("\nArtifacts saved:")
    print("  - artifacts/controllers/pso_lqr_configs.json")
    print("  - artifacts/optimization/pso_results.json")
    print("  - results/figures/pso_convergence.png")


if __name__ == "__main__":
    main()
