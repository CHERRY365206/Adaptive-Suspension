"""CLI script to generate feature dataset from simulated road scenarios."""

from __future__ import annotations
import argparse
from pathlib import Path
from roads.road_dataset import generate_benchmark_dataset
from ai.dataset_generation import generate_scenario_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate rolling-window sensor dataset for road classification.")
    parser.add_argument("--num-per-class", type=int, default=15, help="Scenarios per road class (default: 15)")
    parser.add_argument("--quick", action="store_true", help="Run fast generation with fewer scenarios")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    num = 5 if args.quick else args.num_per_class
    print(f"Generating benchmark dataset with {num} scenarios per class (seed={args.seed})...")

    dataset = generate_benchmark_dataset(num_per_class=num, base_seed=args.seed)
    print(f"Scenarios: Train={len(dataset['train'])}, Val={len(dataset['val'])}, Test={len(dataset['test'])}")

    print("Extracting feature windows across train/val scenarios...")
    X_train, y_train, groups_train, _ = generate_scenario_dataset(dataset["train"] + dataset["val"])
    print(f"Train/Val feature matrix shape: {X_train.shape}, Unique scenario groups: {len(set(groups_train))}")

    print("Extracting feature windows across test scenarios...")
    X_test, y_test, groups_test, _ = generate_scenario_dataset(dataset["test"])
    print(f"Held-out test feature matrix shape: {X_test.shape}, Unique scenario groups: {len(set(groups_test))}")

    leakage = set(groups_train).intersection(set(groups_test))
    if len(leakage) == 0:
        print("SUCCESS: Zero data leakage confirmed between train and test scenarios.")
    else:
        print(f"ERROR: Data leakage detected! Overlapping scenarios: {leakage}")


if __name__ == "__main__":
    main()
