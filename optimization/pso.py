"""From-scratch Particle Swarm Optimization (PSO) implementation.

Supports bounded search spaces, velocity clamping, dynamic inertia damping,
reproducible random seeds, iteration history tracking, and robust exception handling.
"""

from __future__ import annotations
from typing import Callable, Tuple, List, Dict, Any, Optional
import time
import numpy as np


class ParticleSwarmOptimizer:
    """Canonical Particle Swarm Optimizer implemented from first principles.

    Velocity update equation:
        v_i(k+1) = w(k) * v_i(k) + c1 * r1 * (p_best_i - x_i(k)) + c2 * r2 * (g_best - x_i(k))

    Position update equation:
        x_i(k+1) = x_i(k) + v_i(k+1)
    """

    def __init__(
        self,
        cost_function: Callable[[np.ndarray], float],
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        swarm_size: int = 20,
        max_iterations: int = 25,
        w_start: float = 0.9,
        w_end: float = 0.4,
        c1: float = 1.6,
        c2: float = 1.6,
        velocity_clamp_ratio: float = 0.20,
        seed: Optional[int] = 42
    ) -> None:
        self.cost_func = cost_function
        self.lb = np.asarray(lower_bounds, dtype=np.float64)
        self.ub = np.asarray(upper_bounds, dtype=np.float64)
        self.dim = len(self.lb)

        if len(self.ub) != self.dim:
            raise ValueError("Lower bounds and upper bounds must have the same dimension.")
        if np.any(self.lb >= self.ub):
            raise ValueError("All lower bounds must be strictly less than upper bounds.")

        self.swarm_size = swarm_size
        self.max_iter = max_iterations
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.seed = seed

        # Maximum velocity per dimension
        self.v_max = velocity_clamp_ratio * (self.ub - self.lb)

        # Swarm internal state
        self.positions: np.ndarray = np.zeros((self.swarm_size, self.dim), dtype=np.float64)
        self.velocities: np.ndarray = np.zeros((self.swarm_size, self.dim), dtype=np.float64)
        self.pbest_positions: np.ndarray = np.zeros((self.swarm_size, self.dim), dtype=np.float64)
        self.pbest_costs: np.ndarray = np.full(self.swarm_size, np.inf, dtype=np.float64)

        self.gbest_position: np.ndarray = np.zeros(self.dim, dtype=np.float64)
        self.gbest_cost: float = float("inf")

        # History log
        self.iteration_history: List[Dict[str, Any]] = []
        self.runtime_seconds: float = 0.0

    def optimize(self, verbose: bool = False) -> Tuple[np.ndarray, float, List[Dict[str, Any]]]:
        """Execute PSO optimization loop.

        Returns:
            Tuple of (best_position, best_cost, iteration_history).
        """
        rng = np.random.default_rng(self.seed)
        start_time = time.time()
        self.iteration_history.clear()

        # 1. Initialize particle positions uniformly within bounds
        self.positions = rng.uniform(self.lb, self.ub, size=(self.swarm_size, self.dim))
        # Initialize small random velocities
        self.velocities = rng.uniform(-self.v_max, self.v_max, size=(self.swarm_size, self.dim))

        # 2. Initial evaluation
        for i in range(self.swarm_size):
            pos = self.positions[i]
            try:
                cost = float(self.cost_func(pos))
                if not np.isfinite(cost):
                    cost = 1.0e6
            except Exception:
                cost = 1.0e6

            self.pbest_positions[i] = pos.copy()
            self.pbest_costs[i] = cost

            if cost < self.gbest_cost:
                self.gbest_cost = cost
                self.gbest_position = pos.copy()

        self.iteration_history.append({
            "iteration": 0,
            "best_cost": float(self.gbest_cost),
            "mean_cost": float(np.mean(self.pbest_costs[np.isfinite(self.pbest_costs)])),
            "best_position": self.gbest_position.tolist()
        })

        # 3. Main PSO iterations
        for k in range(1, self.max_iter + 1):
            # Dynamic linear inertia weight damping
            w = self.w_start - ((self.w_start - self.w_end) * (k / self.max_iter))

            for i in range(self.swarm_size):
                r1 = rng.uniform(0.0, 1.0, size=self.dim)
                r2 = rng.uniform(0.0, 1.0, size=self.dim)

                # Cognitive and social velocity components
                v_cog = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                v_soc = self.c2 * r2 * (self.gbest_position - self.positions[i])

                new_velocity = w * self.velocities[i] + v_cog + v_soc
                # Clamp velocity
                self.velocities[i] = np.clip(new_velocity, -self.v_max, self.v_max)

                # Update position and enforce box bounds
                new_pos = self.positions[i] + self.velocities[i]
                # Reflective / clamp boundary condition
                clamped_pos = np.clip(new_pos, self.lb, self.ub)
                self.positions[i] = clamped_pos

                # Evaluate candidate position
                try:
                    cost = float(self.cost_func(clamped_pos))
                    if not np.isfinite(cost):
                        cost = 1.0e6
                except Exception:
                    cost = 1.0e6

                # Update personal best
                if cost < self.pbest_costs[i]:
                    self.pbest_costs[i] = cost
                    self.pbest_positions[i] = clamped_pos.copy()

                    # Update global best
                    if cost < self.gbest_cost:
                        self.gbest_cost = cost
                        self.gbest_position = clamped_pos.copy()

            mean_c = float(np.mean(self.pbest_costs[np.isfinite(self.pbest_costs)]))
            self.iteration_history.append({
                "iteration": k,
                "best_cost": float(self.gbest_cost),
                "mean_cost": mean_c,
                "best_position": self.gbest_position.tolist()
            })

            if verbose:
                print(f"PSO Iteration {k:2d}/{self.max_iter}: Best Cost = {self.gbest_cost:.4f}, Mean = {mean_c:.4f}")

        self.runtime_seconds = float(time.time() - start_time)
        return self.gbest_position, self.gbest_cost, self.iteration_history
