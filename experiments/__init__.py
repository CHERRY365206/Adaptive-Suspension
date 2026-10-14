"""Comparative experiments, baselines suite, and performance metrics."""

from experiments.metrics import compute_simulation_metrics
from experiments.scenarios import get_held_out_benchmark_scenarios
from experiments.baselines import (
    SuspensionBenchmarkSuite,
    SYSTEM_NAMES,
    ALL_SYSTEM_NAMES,
    SYSTEM_DISPLAY_NAMES
)
from experiments.statistical_analysis import (
    compute_percentage_improvement,
    aggregate_experiment_results
)
from experiments.runner import run_comparative_benchmark
from experiments.comparisons import generate_radar_comparison_plot

__all__ = [
    "compute_simulation_metrics",
    "get_held_out_benchmark_scenarios",
    "SuspensionBenchmarkSuite",
    "SYSTEM_NAMES",
    "ALL_SYSTEM_NAMES",
    "SYSTEM_DISPLAY_NAMES",
    "compute_percentage_improvement",
    "aggregate_experiment_results",
    "run_comparative_benchmark",
    "generate_radar_comparison_plot",
]
