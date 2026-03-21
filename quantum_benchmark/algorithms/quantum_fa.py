"""
Quantum-Inspired Firefly Algorithm (QFA)
==========================================
Based on: Zouache et al. (2016) "QIFAPSO" in Soft Computing
          + Zitouni et al. (2021) "Novel QFA" in Arabian Journal

Key idea:
  - Each firefly's position encoded as qubit angle θ ∈ [0, π/2]
  - Real position = cos²(θ) × (ub - lb) + lb
  - Attractiveness between fireflies uses Euclidean distance in θ-space
    (not x-space) → "quantum attractiveness"
  - Movement uses quantum rotation gate toward brighter firefly
  - Quantum tunneling allows local optima escape

Classical FA movement:
  x_i = x_i + β₀ × exp(-γ × r²) × (x_j - x_i) + α × (rand - 0.5)

QGFA movement:
  θ_i = θ_i + β₀ × exp(-γ × r²_θ) × Δθ_sign + α × random_perturbation
  where r²_θ = ||θ_i - θ_j||² (distance in qubit angle space)
"""

import numpy as np
from mealpy import FFA
from mealpy.utils.space import FloatVar


class QFA(FFA.OriginalFFA):
    """
    Quantum-Inspired Firefly Algorithm.

    Parameters (from Zouache et al. 2016):
        max_sparks (float):  Alpha — random walk step size (default 0.5)
        p_sparks (float):    Beta0 — max attractiveness (default 1.0)
        exp_const (float):   Gamma — light absorption coefficient (default 1.0)
        delta_theta_max (float): Max qubit rotation per step (default 0.05π)
        tunnel_prob (float): Quantum tunneling probability (default 0.01)
    """

    def __init__(self, epoch=500, pop_size=30,
                 max_sparks=0.5, p_sparks=1.0, exp_const=1.0,
                 delta_theta_max=0.05, tunnel_prob=0.01, **kwargs):
        super().__init__(epoch=epoch, pop_size=pop_size,
                         max_sparks=max_sparks, p_sparks=p_sparks,
                         exp_const=exp_const, **kwargs)
        self.delta_theta_max = delta_theta_max * np.pi
        self.tunnel_prob = tunnel_prob
        self.theta = None
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

    def _measure(self, theta, lb, ub):
        """Collapse qubit angles to real-valued positions."""
        return lb + (ub - lb) * np.cos(theta) ** 2

    def _quantum_attractiveness(self, theta_i, theta_j):
        """
        Attractiveness based on distance in qubit angle space.
        r² = ||θ_i - θ_j||²
        β = β₀ × exp(-γ × r²)
        """
        r_sq = np.sum((theta_i - theta_j) ** 2)
        return self.p_sparks * np.exp(-self.exp_const * r_sq)

    def initialization(self):
        """Initialize qubits uniformly in [0, 2π]."""
        super().initialization()
        n_dims = self.problem.n_dims
        lb = np.array(self.problem.lb)
        ub = np.array(self.problem.ub)

        self.theta = np.random.uniform(0, 2 * np.pi,
                                       (self.pop_size, n_dims))

    def evolve(self, epoch):
        """
        QFA evolution step.
        For each pair (i, j): if firefly j is brighter than i,
        move i toward j using quantum rotation gate.
        """
        lb = np.array(self.problem.lb)
        ub = np.array(self.problem.ub)

        pop_new = []
        for i in range(self.pop_size):
            theta_i = self.theta[i].copy()
            moved = False

            for j in range(self.pop_size):
                if i == j:
                    continue

                # Check brightness: lower fitness = brighter (minimization)
                fi = self.pop[i].target.fitness
                fj = self.pop[j].target.fitness

                if self.compare_target(self.pop[j].target,
                                       self.pop[i].target,
                                       self.problem.minmax):
                    # j is brighter → move i toward j
                    theta_j = self.theta[j]

                    # Quantum attractiveness (distance in θ-space)
                    beta = self._quantum_attractiveness(theta_i, theta_j)

                    # Rotation direction: toward j
                    direction = np.sign(theta_j - theta_i)
                    direction[direction == 0] = np.random.choice(
                        [-1, 1], size=np.sum(direction == 0))

                    # Quantum rotation gate update
                    delta_theta = beta * direction * self.delta_theta_max

                    # Random walk (alpha term) — keeps diversity
                    rand_walk = self.max_sparks * (np.random.rand(len(theta_i)) - 0.5)
                    rand_walk_theta = rand_walk * (np.pi / 4)  # scale to θ space

                    theta_i = theta_i + delta_theta + rand_walk_theta
                    theta_i = np.clip(theta_i, 0, np.pi / 2)
                    moved = True

            # Quantum tunneling with small probability
            tunnel_mask = np.random.rand(len(theta_i)) < self.tunnel_prob
            if np.any(tunnel_mask):
                theta_i[tunnel_mask] = np.random.uniform(
                    0, np.pi / 2, np.sum(tunnel_mask))

            # If no movement, do random walk (firefly is brightest)
            if not moved:
                rand_step = self.max_sparks * (np.random.rand(len(theta_i)) - 0.5)
                theta_i += rand_step * (np.pi / 4)
                theta_i = np.clip(theta_i, 0, np.pi / 2)

            # Measure: collapse to real position
            x_new = self._measure(theta_i, lb, ub)
            x_new = np.clip(x_new, lb, ub)
            self.theta[i] = theta_i

            agent = self.generate_empty_agent(x_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(x_new)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)

        # Greedy: accept new if better
        for i in range(self.pop_size):
            if self.compare_target(pop_new[i].target,
                                   self.pop[i].target,
                                   self.problem.minmax):
                self.pop[i] = pop_new[i]

        self.pop = self.get_sorted_and_trimmed_population(
            self.pop + pop_new, self.pop_size, self.problem.minmax)
