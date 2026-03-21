"""
Quantum-Inspired Grey Wolf Optimizer (QGWO)
============================================
Based on: Srikanth et al. (2018) "Quantum-inspired binary GWO"
          + Deshmukh et al. (2023) "QEGWO with quantum entanglement"

Key idea:
  - Each wolf's position is encoded as a qubit: θ ∈ [0, π/2]
  - Real position = cos²(θ) × (ub - lb) + lb
  - Position update uses a quantum rotation gate instead of GWO's linear move
  - The rotation angle Δθ is guided by alpha, beta, delta wolves
  - Quantum tunneling (random flip of θ) helps escape local optima

How it differs from classical GWO:
  Classical: X_new = (X1 + X2 + X3) / 3
  Quantum:   θ_new = θ_old + Δθ × lookup_table_sign(best, current)
             X_new = cos²(θ_new) × (ub - lb) + lb
"""

import numpy as np
from mealpy import GWO
from mealpy.utils.space import FloatVar


class QGWO(GWO.OriginalGWO):
    """
    Quantum-Inspired GWO — extends MEALPY's OriginalGWO.

    Extra parameters:
        delta_theta_max (float): Max rotation angle per step (default 0.05π)
                                 From Srikanth et al. 2018
        tunnel_prob (float):     Quantum tunneling probability (default 0.01)
    """

    def __init__(self, epoch=500, pop_size=30,
                 delta_theta_max=0.05, tunnel_prob=0.01, **kwargs):
        super().__init__(epoch=epoch, pop_size=pop_size, **kwargs)
        self.delta_theta_max = delta_theta_max * np.pi   # convert to radians
        self.tunnel_prob = tunnel_prob
        self.theta = None   # qubit angle array [pop_size × n_dims]
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

    # ──────────────────────────────────────────
    # Lookup table for rotation direction
    # Adapted from Han & Kim (2002) Table 1
    # ──────────────────────────────────────────
    def _rotation_sign(self, x_bit, best_bit):
        """
        Determine sign of rotation angle based on current and best bit values.
        x_bit, best_bit: 0 or 1 (measured qubit values)
        Returns: +1, -1, or 0
        """
        if x_bit == 0 and best_bit == 1:
            return +1    # rotate toward best
        elif x_bit == 1 and best_bit == 0:
            return -1    # rotate away from bad
        else:
            return np.random.choice([-1, 1])   # random when equal

    def _measure_qubit(self, theta):
        """
        Quantum measurement: collapse qubit angle to real-valued position.
        x = cos²(θ) × (ub - lb) + lb
        """
        lb = np.array(self.problem.lb)
        ub = np.array(self.problem.ub)
        return lb + (ub - lb) * np.cos(theta) ** 2

    def _get_bit(self, x, lb, ub):
        """Convert real position to binary bit for lookup table."""
        mid = (lb + ub) / 2
        return (x > mid).astype(int)

    def initialization(self):
        """Initialize qubits uniformly in [0, 2π] for directional encoding."""
        super().initialization()
        n_dims = self.problem.n_dims
        # Use full [0, 2π] range — θ encodes DIRECTION of movement
        # This is more effective for continuous optimization than [0, π/2]
        # Reference: Li et al. 2010 "Continuous QACO"
        self.theta = np.random.uniform(0, 2 * np.pi,
                                       (self.pop_size, n_dims))

    def evolve(self, epoch):
        """
        QGWO evolution using directional qubit encoding.

        The qubit angle θ encodes the DIRECTION of movement toward leaders.
        Movement: x_new = x_old + step_size × cos(θ)
        This ensures the algorithm can actually converge while maintaining
        quantum diversity through the rotational update.

        Based on: Li et al. (2010) continuous quantum encoding principle
        """
        sorted_pop = sorted(self.pop,
                            key=lambda ag: ag.target.fitness)
        alpha = sorted_pop[0]
        beta  = sorted_pop[1]
        delta = sorted_pop[2]

        lb = np.array(self.problem.lb)
        ub = np.array(self.problem.ub)

        # Linearly decreasing step scale (same as classical GWO's a)
        a = 2 - epoch * (2.0 / self.epoch)
        step_size = a * (ub - lb) / 4.0

        pop_new = []
        for idx in range(self.pop_size):
            theta_i = self.theta[idx].copy()
            x_i = self.pop[idx].solution

            r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()

            # Quantum rotation gate: rotate theta toward each leader
            # The rotation angle is proportional to the direction to the leader
            # δθ = arctan(Δx / step_size) — bounded rotation

            diff_alpha = alpha.solution - x_i
            delta_theta1 = r1 * np.arctan2(diff_alpha,
                                            step_size + 1e-10) * self.delta_theta_max / np.pi

            diff_beta = beta.solution - x_i
            delta_theta2 = r2 * np.arctan2(diff_beta,
                                            step_size + 1e-10) * self.delta_theta_max / np.pi

            diff_delta = delta.solution - x_i
            delta_theta3 = r3 * np.arctan2(diff_delta,
                                            step_size + 1e-10) * self.delta_theta_max / np.pi

            # Update theta (full 2π range, wrap around)
            theta_i = (theta_i + delta_theta1 + delta_theta2 + delta_theta3) % (2 * np.pi)

            # Quantum measurement: move in direction cos(θ) × step_size
            x_new = x_i + step_size * np.cos(theta_i)

            # ── Quantum tunneling: random re-initialization of some dimensions
            tunnel_mask = np.random.rand(len(theta_i)) < self.tunnel_prob
            if np.any(tunnel_mask):
                theta_i[tunnel_mask] = np.random.uniform(0, 2 * np.pi,
                                                          np.sum(tunnel_mask))
                # Tunnel to random position in that dimension
                x_new[tunnel_mask] = np.random.uniform(
                    lb[tunnel_mask], ub[tunnel_mask])

            x_new = np.clip(x_new, lb, ub)
            self.theta[idx] = theta_i

            agent = self.generate_empty_agent(x_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(x_new)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)

        # Greedy selection: keep new only if better
        for idx in range(self.pop_size):
            if self.compare_target(pop_new[idx].target,
                                   self.pop[idx].target,
                                   self.problem.minmax):
                self.pop[idx] = pop_new[idx]

        self.pop = self.get_sorted_and_trimmed_population(
            self.pop + pop_new, self.pop_size, self.problem.minmax)
