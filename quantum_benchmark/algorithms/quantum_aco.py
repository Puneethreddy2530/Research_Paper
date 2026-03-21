"""
Quantum Ant Colony Optimization (QACO) — Continuous Version
=============================================================
Based on: Wang et al. (2007) "Novel QACO" in Springer LNCS
          + Panchi Li et al. (2010) "Continuous QACO" in IEEE
          + Li & Wang (2012) "BQACO with Bloch sphere"

Key idea:
  - Each ant's position is a qubit angle θ ∈ [0, π/2]
  - Real position = cos²(θ) × (ub - lb) + lb
  - Pheromone is replaced by qubit probability amplitudes:
      |ψ⟩ = cos(θ)|0⟩ + sin(θ)|1⟩
  - Pheromone update (evaporation + deposit) becomes qubit rotation:
      θ_new = θ_old + Δθ (rotation toward best path)
  - "Hadamard mutation" (NOT gate flip) for diversity:
      θ → π/2 - θ  (swaps cos² and sin² probabilities)

Classical ACO continuous (ACO-R):
  - Samples from a Gaussian mixture around archive solutions
  - Pheromone: weight w_l = rank-based

QACO continuous:
  - Pheromone intensity → cos²(θ) probability
  - Evaporation → rotation AWAY from poor paths
  - Deposition → rotation TOWARD best path
"""

import numpy as np
from mealpy import ACOR
from mealpy.utils.space import FloatVar


class QACO(ACOR.OriginalACOR):
    """
    Quantum Ant Colony Optimization for continuous spaces.

    Parameters (from Li et al. 2010):
        sample_count (int):  Archive size / number of solution samples (default 50)
        intent_factor (float): Pheromone concentration (default 0.5)
        zeta (float):         Deviation-distance ratio (default 1.0)
        delta_theta (float):  Base rotation angle in units of π (default 0.01)
        tunnel_prob (float):  Hadamard mutation probability (default 0.02)
    """

    def __init__(self, epoch=500, pop_size=30,
                 sample_count=50, intent_factor=0.5, zeta=1.0,
                 delta_theta=0.01, tunnel_prob=0.02, **kwargs):
        super().__init__(epoch=epoch, pop_size=pop_size,
                         sample_count=sample_count,
                         intent_factor=intent_factor,
                         zeta=zeta, **kwargs)
        self.delta_theta_base = delta_theta * np.pi
        self.tunnel_prob = tunnel_prob
        self.theta = None
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

    def _measure(self, theta, lb, ub):
        """Collapse qubit to real position using cos² mapping."""
        return lb + (ub - lb) * np.cos(theta) ** 2

    def _pheromone_to_quality(self, theta):
        """
        Pheromone quality ∝ cos²(θ).
        Higher cos²(θ) → more likely to be selected path.
        """
        return np.cos(theta) ** 2

    def _rotation_gate(self, theta_i, theta_best, fitness_i, fitness_best, minmax):
        """
        Quantum rotation gate update.
        Rotates θ_i toward θ_best if best has better fitness.
        Rotation angle is proportional to fitness difference.
        """
        direction = np.sign(theta_best - theta_i)
        direction[direction == 0] = np.random.choice([-1, 1],
                                    size=np.sum(direction == 0))

        # Adaptive rotation: larger when far from best
        distance = np.abs(theta_best - theta_i)
        adaptive_scale = np.tanh(distance)   # ∈ (0,1), larger when far

        delta = self.delta_theta_base * direction * adaptive_scale
        return np.clip(theta_i + delta, 0, np.pi / 2)

    def _hadamard_mutation(self, theta):
        """
        Hadamard gate = NOT gate in θ-space: θ → π/2 - θ
        Swaps probability amplitudes: cos²↔sin²
        Used for diversity maintenance (replaces classical evaporation).
        """
        mask = np.random.rand(len(theta)) < self.tunnel_prob
        theta[mask] = np.pi / 2 - theta[mask]
        return theta

    def initialization(self):
        """Initialize ant qubits uniformly."""
        super().initialization()
        n_dims = self.problem.n_dims
        lb = np.array(self.problem.lb)
        ub = np.array(self.problem.ub)

        self.theta = np.random.uniform(0, np.pi / 2,
                                       (self.pop_size, n_dims))
        for i, agent in enumerate(self.pop):
            agent.solution = self._measure(self.theta[i], lb, ub)

    def evolve(self, epoch):
        """
        QACO evolution:
        1. Find best ant (highest quality path)
        2. For each ant: rotate qubit toward best
        3. Apply Hadamard mutation for diversity
        4. Measure: θ → real position
        """
        lb = np.array(self.problem.lb)
        ub = np.array(self.problem.ub)

        # ── Find best agent in current population
        best_idx = 0
        for i in range(1, self.pop_size):
            if self.compare_target(self.pop[i].target,
                                   self.pop[best_idx].target,
                                   self.problem.minmax):
                best_idx = i
        theta_best = self.theta[best_idx].copy()

        pop_new = []
        for i in range(self.pop_size):
            theta_i = self.theta[i].copy()

            # ── Quantum rotation toward best path (pheromone deposit)
            theta_i = self._rotation_gate(
                theta_i, theta_best,
                self.pop[i].target.fitness,
                self.pop[best_idx].target.fitness,
                self.problem.minmax
            )

            # ── Hadamard mutation (diversity / evaporation analog)
            theta_i = self._hadamard_mutation(theta_i)

            # ── Measure to real position
            x_new = self._measure(theta_i, lb, ub)
            x_new = np.clip(x_new, lb, ub)
            self.theta[i] = theta_i

            agent = self.generate_empty_agent(x_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(x_new)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)

        # Greedy selection
        for i in range(self.pop_size):
            if self.compare_target(pop_new[i].target,
                                   self.pop[i].target,
                                   self.problem.minmax):
                self.pop[i] = pop_new[i]

        self.pop = self.get_sorted_and_trimmed_population(
            self.pop + pop_new, self.pop_size, self.problem.minmax)
