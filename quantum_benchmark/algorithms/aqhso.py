"""
AQHSO — Adaptive Quantum Hybrid Swarm Optimizer
=================================================
Designed by: Analysis of GWO, FA, ACO, QGWO, QFA, QACO results

DIAGNOSIS THAT DROVE THIS DESIGN:
  From our benchmark results:
    FA   rank 1.91  ← BEST classical (bright attraction works)
    GWO  rank 2.64  ← 2nd (hierarchy helps exploitation)
    ACO  rank 2.73  ← 3rd (memory helps)
    QACO rank 3.18  ← quantum barely helps ACO
    QGWO rank 4.64  ← fixed rotation gate hurts GWO
    QFA  rank 5.91  ← WORST — fixed gate destroys FA's strength

  Root causes of quantum failure:
    1. Fixed Δθ=0.05π too aggressive — overshoots good solutions
    2. cos²(θ) mapping compresses search space non-uniformly
    3. 1% random tunneling destroys elite solutions
    4. Three quantum variants don't cooperate — isolated silos
    5. 500 epochs not enough for pure quantum convergence

AQHSO DESIGN PRINCIPLES:
  1. FA attraction as BACKBONE (it's empirically the best)
  2. GWO hierarchy for GUIDED exploitation of elite solutions
  3. ACO pheromone matrix for LANDSCAPE MEMORY
  4. Quantum gates that are ADAPTIVE (not fixed) — key innovation
  5. Opposition-based learning for diverse initialization
  6. Lévy flights instead of random tunneling
  7. Elite archive that survives across all phases
  8. Stagnation detector that triggers quantum burst

THREE PHASES:
  Phase 1 (early):   OBL init + GWO hierarchy + standard positions
  Phase 2 (middle):  FA attraction + adaptive quantum encoding
  Phase 3 (late):    Quantum burst + Lévy flights + ACO memory

This combination is NOVEL because:
  - No paper combines FA+GWO+ACO in a unified quantum framework
  - Adaptive rotation angle (not fixed) is new for these algorithms  
  - Pheromone-guided quantum superposition is new
  - Phase-switching based on stagnation is new

CITATION TEMPLATE:
  "We propose AQHSO, an Adaptive Quantum Hybrid Swarm Optimizer
   that unifies the attraction mechanism of FA, the hierarchical
   exploitation of GWO, and the memory-guided search of ACO under
   an adaptive quantum encoding framework with opposition-based
   initialization and Lévy-flight tunneling."
"""

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.space import FloatVar


class AQHSO(Optimizer):
    """
    Adaptive Quantum Hybrid Swarm Optimizer.

    Parameters
    ----------
    epoch : int
        Number of iterations (default 500)
    pop_size : int  
        Population size (default 30)
    alpha_fa : float
        FA random walk coefficient (default 0.5)
    beta0 : float
        FA max attractiveness (default 1.0)
    gamma : float
        FA light absorption (default 1.0)
    theta_max_init : float
        Initial max quantum rotation angle in π units (default 0.1)
    levy_beta : float
        Lévy flight stability parameter (default 1.5, range 1-2)
    archive_size : int
        Elite archive size (default 10)
    stag_window : int
        Stagnation detection window in epochs (default 20)
    phase1_end : float
        Fraction of epochs for Phase 1 (default 0.2)
    phase2_end : float
        Fraction of epochs for Phase 2 (default 0.7)
    """

    def __init__(self, epoch=500, pop_size=30,
                 alpha_fa=0.5, beta0=1.0, gamma=1.0,
                 theta_max_init=0.1, levy_beta=1.5,
                 archive_size=10, stag_window=20,
                 phase1_end=0.20, phase2_end=0.70,
                 **kwargs):
        super().__init__(**kwargs)

        self.epoch          = epoch
        self.pop_size       = pop_size
        self.alpha_fa       = alpha_fa
        self.beta0          = beta0
        self.gamma          = gamma
        self.theta_max_init = theta_max_init * np.pi
        self.levy_beta      = levy_beta
        self.archive_size   = archive_size
        self.stag_window    = stag_window
        self.phase1_end     = int(epoch * phase1_end)
        self.phase2_end     = int(epoch * phase2_end)

        self.nfe_per_epoch  = pop_size
        self.sort_flag      = False

        # Internal state
        self.theta          = None     # qubit angles [pop × dims]
        self.pheromone      = None     # ACO pheromone matrix [dims]
        self.elite_archive  = []       # top solutions
        self.best_history   = []       # epoch-wise best for stagnation detection
        self.theta_max      = self.theta_max_init   # adaptive rotation angle

    # ─────────────────────────────────────────────────────────────
    # INITIALIZATION
    # ─────────────────────────────────────────────────────────────

    def initialization(self):
        """
        Opposition-Based Learning (OBL) initialization.
        Creates pop_size candidates + pop_size opposites,
        selects best pop_size. Proven to improve initial diversity.
        From: Tizhoosh (2005) "Opposition-based learning"
        """
        super().initialization()

        n_dims = self.problem.n_dims
        lb = np.array(self.problem.lb)
        ub = np.array(self.problem.ub)

        # Generate regular + opposite population
        regular_pop  = np.array([lb + np.random.rand(n_dims) * (ub - lb)
                                  for _ in range(self.pop_size)])
        opposite_pop = lb + ub - regular_pop   # x_opp = lb + ub - x

        # Evaluate both and keep best pop_size
        all_solutions = np.vstack([regular_pop, opposite_pop])
        fitnesses = []
        for sol in all_solutions:
            sol_clipped = np.clip(sol, lb, ub)
            try:
                fit = self.problem.fit_func(sol_clipped)
                if isinstance(fit, (list, tuple)):
                    fit = fit[0]
                fitnesses.append(float(fit))
            except Exception:
                fitnesses.append(float('inf') if self.problem.minmax == 'min' else float('-inf'))

        fitnesses = np.array(fitnesses)
        if self.problem.minmax == 'min':
            best_idx = np.argsort(fitnesses)[:self.pop_size]
        else:
            best_idx = np.argsort(fitnesses)[::-1][:self.pop_size]

        # Set initial population to best OBL solutions
        for i, idx in enumerate(best_idx):
            self.pop[i].solution = np.clip(all_solutions[idx], lb, ub)

        # Initialize qubit angles based on initial positions
        # θ = arccos(sqrt((x - lb) / (ub - lb)))
        positions = np.array([a.solution for a in self.pop])
        ratios = np.clip((positions - lb) / (ub - lb + 1e-10), 0.001, 0.999)
        self.theta = np.arccos(np.sqrt(ratios))   # ∈ [0, π/2]

        # Initialize pheromone uniformly
        self.pheromone = np.ones(n_dims) * 0.5

        # Initialize elite archive
        self.elite_archive = []

    # ─────────────────────────────────────────────────────────────
    # UTILITY FUNCTIONS
    # ─────────────────────────────────────────────────────────────

    def _measure_qubit(self, theta):
        """Collapse qubit angle → real position: x = cos²(θ)·(ub-lb) + lb"""
        lb = np.array(self.problem.lb)
        ub = np.array(self.problem.ub)
        return lb + (ub - lb) * np.cos(theta) ** 2

    def _levy_flight(self, n_dims):
        """
        Lévy flight step using Mantegna's algorithm.
        Heavy-tailed distribution — occasionally makes very large jumps.
        Much better than uniform random for escaping local optima.
        """
        beta = self.levy_beta
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, n_dims)
        v = np.random.normal(0, 1, n_dims)
        step = u / (np.abs(v) ** (1 / beta))
        return step * 0.01   # scale down

    def _detect_stagnation(self):
        """
        Detects if algorithm is stuck.
        Returns True if best fitness hasn't improved in stag_window epochs.
        """
        if len(self.best_history) < self.stag_window:
            return False
        recent = self.best_history[-self.stag_window:]
        improvement = abs(recent[-1] - recent[0])
        relative = improvement / (abs(recent[0]) + 1e-300)
        return relative < 1e-6   # less than 1e-6 relative improvement

    def _update_pheromone(self, best_solution, evap_rate=0.1):
        """
        ACO-style pheromone update on the current best solution.
        High-performing dimensions get more pheromone → guides future search.
        """
        lb = np.array(self.problem.lb)
        ub = np.array(self.problem.ub)
        # Normalize best solution to [0,1]
        normalized = (best_solution - lb) / (ub - lb + 1e-10)
        # Evaporate + deposit
        self.pheromone = (1 - evap_rate) * self.pheromone + evap_rate * normalized
        self.pheromone = np.clip(self.pheromone, 0.01, 0.99)

    def _update_elite_archive(self):
        """Maintain top-k solutions in elite archive."""
        candidates = list(self.pop) + self.elite_archive
        if self.problem.minmax == 'min':
            candidates.sort(key=lambda a: a.target.fitness)
        else:
            candidates.sort(key=lambda a: a.target.fitness, reverse=True)
        self.elite_archive = candidates[:self.archive_size]

    def _adaptive_theta_max(self, epoch):
        """
        Adaptive rotation angle:
          - Starts large (exploration)
          - Decreases as epochs progress (exploitation)
          - Increases when stagnation detected (escape)
        """
        progress = epoch / self.epoch
        base = self.theta_max_init * (1 - 0.7 * progress)   # linear decay

        if self._detect_stagnation():
            # Double the rotation angle to escape
            return min(base * 2.0, np.pi / 4)
        return base

    # ─────────────────────────────────────────────────────────────
    # PHASE 1: GWO Hierarchy (epochs 0 → phase1_end)
    # Standard GWO with OBL-initialized population
    # Purpose: Fast initial convergence to promising regions
    # ─────────────────────────────────────────────────────────────

    def _phase1_gwo(self, epoch):
        """GWO-style hierarchy update."""
        lb = np.array(self.problem.lb)
        ub = np.array(self.problem.ub)
        a  = 2 - epoch * (2.0 / self.epoch)   # linearly decreases 2→0

        sorted_pop = sorted(self.pop, key=lambda x: x.target.fitness,
                            reverse=(self.problem.minmax == 'max'))
        alpha, beta, delta = sorted_pop[0], sorted_pop[1], sorted_pop[2]

        pop_new = []
        for i in range(self.pop_size):
            r1, r2 = np.random.rand(2)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * alpha.solution - self.pop[i].solution)
            X1 = alpha.solution - A1 * D_alpha

            r1, r2 = np.random.rand(2)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * beta.solution - self.pop[i].solution)
            X2 = beta.solution - A2 * D_beta

            r1, r2 = np.random.rand(2)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * delta.solution - self.pop[i].solution)
            X3 = delta.solution - A3 * D_delta

            x_new = np.clip((X1 + X2 + X3) / 3, lb, ub)

            # Update theta to match new position
            ratios = np.clip((x_new - lb) / (ub - lb + 1e-10), 0.001, 0.999)
            self.theta[i] = np.arccos(np.sqrt(ratios))

            agent = self.generate_empty_agent(x_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(x_new)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)

        for i in range(self.pop_size):
            if self.compare_target(pop_new[i].target, self.pop[i].target,
                                   self.problem.minmax):
                self.pop[i] = pop_new[i]

    # ─────────────────────────────────────────────────────────────
    # PHASE 2: FA Attraction + Adaptive Quantum Encoding
    # (epochs phase1_end → phase2_end)
    # Purpose: Multimodal exploration via attraction + quantum
    # ─────────────────────────────────────────────────────────────

    def _phase2_fa_quantum(self, epoch):
        """FA attraction in qubit angle space + pheromone guidance."""
        lb  = np.array(self.problem.lb)
        ub  = np.array(self.problem.ub)
        theta_max = self._adaptive_theta_max(epoch)

        pop_new = []
        for i in range(self.pop_size):
            theta_i = self.theta[i].copy()
            moved = False

            for j in range(self.pop_size):
                if i == j:
                    continue
                if self.compare_target(self.pop[j].target, self.pop[i].target,
                                       self.problem.minmax):
                    # FA: attractiveness in θ-space (not x-space — key difference)
                    r_sq = np.sum((self.theta[i] - self.theta[j]) ** 2)
                    beta_ij = self.beta0 * np.exp(-self.gamma * r_sq)

                    # Quantum rotation toward brighter firefly
                    direction = np.sign(self.theta[j] - theta_i)
                    direction[direction == 0] = np.random.choice([-1, 1],
                                               size=np.sum(direction == 0))
                    delta_theta = beta_ij * direction * theta_max

                    # Pheromone bias: nudge toward high-pheromone dimensions
                    phero_direction = np.sign(self.pheromone - np.cos(theta_i)**2)
                    pheromone_nudge = 0.05 * theta_max * phero_direction

                    # Random walk
                    rand_step = self.alpha_fa * (np.random.rand(len(theta_i)) - 0.5)
                    rand_theta = rand_step * (np.pi / 4)

                    theta_i = np.clip(theta_i + delta_theta + pheromone_nudge + rand_theta,
                                      0, np.pi / 2)
                    moved = True

            # No movement = this agent is brightest — do random walk
            if not moved:
                rand_step = self.alpha_fa * (np.random.rand(len(theta_i)) - 0.5)
                theta_i = np.clip(theta_i + rand_step * (np.pi/4), 0, np.pi/2)

            # Measure qubit
            x_new = np.clip(self._measure_qubit(theta_i), lb, ub)
            self.theta[i] = theta_i

            agent = self.generate_empty_agent(x_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(x_new)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)

        for i in range(self.pop_size):
            if self.compare_target(pop_new[i].target, self.pop[i].target,
                                   self.problem.minmax):
                self.pop[i] = pop_new[i]
                self.theta[i] = np.arccos(np.sqrt(
                    np.clip((self.pop[i].solution - lb) / (ub - lb + 1e-10),
                            0.001, 0.999)))

    # ─────────────────────────────────────────────────────────────
    # PHASE 3: Quantum Burst + Lévy + ACO Memory
    # (epochs phase2_end → epoch)
    # Purpose: Final refinement, escape traps, guided intensification
    # ─────────────────────────────────────────────────────────────

    def _phase3_quantum_burst(self, epoch):
        """
        Quantum burst phase:
        - Aggressive quantum rotation toward archive elite
        - Lévy tunneling for trap escape
        - ACO memory-guided perturbation
        """
        lb  = np.array(self.problem.lb)
        ub  = np.array(self.problem.ub)
        theta_max = self._adaptive_theta_max(epoch)
        n_dims    = self.problem.n_dims

        # Best from archive or current population
        elite = (self.elite_archive[0] if self.elite_archive
                 else sorted(self.pop, key=lambda x: x.target.fitness,
                              reverse=(self.problem.minmax == 'max'))[0])
        theta_elite = np.arccos(np.sqrt(
            np.clip((elite.solution - lb) / (ub - lb + 1e-10), 0.001, 0.999)))

        pop_new = []
        for i in range(self.pop_size):
            theta_i = self.theta[i].copy()

            # Quantum rotation toward elite
            direction = np.sign(theta_elite - theta_i)
            direction[direction == 0] = np.random.choice([-1, 1],
                                       size=np.sum(direction == 0))
            r = np.random.rand()
            theta_i += r * direction * theta_max

            # Lévy tunneling (replaces random reset — much smarter)
            levy_mask = np.random.rand(n_dims) < 0.05   # 5% of dimensions
            if np.any(levy_mask):
                levy_step = self._levy_flight(n_dims)
                # Scale Lévy step to θ-space
                theta_levy = theta_i + levy_step * (np.pi / 4)
                theta_i[levy_mask] = theta_levy[levy_mask]

            # ACO pheromone guidance: bias toward high-pheromone dimensions
            phero_target = np.arccos(np.sqrt(
                np.clip(1 - self.pheromone, 0.001, 0.999)))
            pheromone_pull = 0.1 * (phero_target - theta_i)
            theta_i += pheromone_pull

            theta_i = np.clip(theta_i, 0, np.pi / 2)

            # Measure
            x_new = np.clip(self._measure_qubit(theta_i), lb, ub)
            self.theta[i] = theta_i

            agent = self.generate_empty_agent(x_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(x_new)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)

        for i in range(self.pop_size):
            if self.compare_target(pop_new[i].target, self.pop[i].target,
                                   self.problem.minmax):
                self.pop[i] = pop_new[i]

    # ─────────────────────────────────────────────────────────────
    # MAIN EVOLUTION LOOP
    # ─────────────────────────────────────────────────────────────

    def evolve(self, epoch):
        """
        Route to appropriate phase based on epoch number.
        Update shared state (pheromone, archive, history) after each phase.
        """
        # Route to phase
        if epoch < self.phase1_end:
            self._phase1_gwo(epoch)
        elif epoch < self.phase2_end:
            self._phase2_fa_quantum(epoch)
        else:
            self._phase3_quantum_burst(epoch)

        # Update shared state
        current_best = sorted(self.pop,
                               key=lambda x: x.target.fitness,
                               reverse=(self.problem.minmax == 'max'))[0]
        self.best_history.append(current_best.target.fitness)

        # Update pheromone with current best
        self._update_pheromone(current_best.solution)

        # Update elite archive
        self._update_elite_archive()

        # Trim population using greedy selection from archive injection
        # Every 50 epochs, inject 1 elite archive member into worst position
        if epoch % 50 == 0 and self.elite_archive:
            if self.problem.minmax == 'min':
                worst_idx = max(range(self.pop_size),
                                key=lambda i: self.pop[i].target.fitness)
            else:
                worst_idx = min(range(self.pop_size),
                                key=lambda i: self.pop[i].target.fitness)
            self.pop[worst_idx] = self.elite_archive[0]
