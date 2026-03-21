"""
Experiment 3: Convergence Curves
==================================
Records best fitness at every epoch for selected functions.
Used to generate the convergence curve figures in the paper.

Selects representative functions:
  - F1  (simple unimodal — should all converge)
  - F9  (Rastrigin — multimodal, shows exploration difference)
  - F10 (Ackley — deceptive)
  - F23 (Shekel 10 — fixed dim, tricky)
"""

import sys, os, logging
import numpy as np

logging.disable(logging.CRITICAL)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mealpy import GWO, FFA, ACOR
from mealpy.utils.space import FloatVar
from algorithms.quantum_gwo import QGWO
from algorithms.quantum_fa  import QFA
from algorithms.quantum_aco import QACO
from benchmarks.classical_23 import BENCHMARK_FUNCTIONS
from utils.result_manager    import RESULTS_DIR

import json

# Functions to track convergence on
CONVERGENCE_FUNCS = ["F1", "F9", "F10", "F23"]
N_RUNS   = 5     # 5 runs for convergence (averaged later)
POP_SIZE = 30
EPOCH    = 500


ALGORITHMS = {
    "GWO":  lambda: GWO.OriginalGWO(epoch=EPOCH, pop_size=POP_SIZE),
    "QGWO": lambda: QGWO(epoch=EPOCH, pop_size=POP_SIZE),
    "FA":   lambda: FFA.OriginalFFA(epoch=EPOCH, pop_size=POP_SIZE,
                                    max_sparks=0.5, p_sparks=1.0, exp_const=1.0),
    "QFA":  lambda: QFA(epoch=EPOCH, pop_size=POP_SIZE),
    "ACO":  lambda: ACOR.OriginalACOR(epoch=EPOCH, pop_size=POP_SIZE,
                                      sample_count=50, intent_factor=0.5, zeta=1.0),
    "QACO": lambda: QACO(epoch=EPOCH, pop_size=POP_SIZE),
}


class ConvergenceTracker:
    """
    Wraps a mealpy algorithm to record best fitness at every epoch.
    Patches into the algorithm's history after solve().
    """
    def __init__(self, algo):
        self.algo = algo
        self.convergence = []

    def solve(self, problem):
        self.algo.solve(problem)
        # mealpy stores history in algo.history
        if hasattr(self.algo, 'history') and hasattr(self.algo.history, 'list_global_best_fit'):
            self.convergence = [float(f) for f in self.algo.history.list_global_best_fit]
        else:
            self.convergence = []
        return self.algo.g_best.target.fitness


def run_convergence():
    print("=" * 50)
    print("EXPERIMENT 3: Convergence Curves")
    print(f"  Functions: {CONVERGENCE_FUNCS}")
    print(f"  Runs per function: {N_RUNS}")
    print("=" * 50)

    # convergence_data[func][algo] = list of lists (each run's epoch-by-epoch best)
    convergence_data = {f: {a: [] for a in ALGORITHMS} for f in CONVERGENCE_FUNCS}

    func_lookup = {f[0]: f for f in BENCHMARK_FUNCTIONS}

    for fname in CONVERGENCE_FUNCS:
        if fname not in func_lookup:
            print(f"[SKIP] {fname} not in benchmark list")
            continue

        _, func, dim, lb, ub, _ = func_lookup[fname]
        print(f"\n── {fname} (dim={dim}) ──")

        bounds = FloatVar(
            lb=tuple([lb] * dim),
            ub=tuple([ub] * dim),
            name="vars"
        )
        problem = {
            "obj_func": func,
            "bounds":   bounds,
            "minmax":   "min",
            "n_dims":   dim,
            "log_to":   None,
        }

        for algo_name, algo_ctor in ALGORITHMS.items():
            runs_convergence = []
            for run in range(N_RUNS):
                try:
                    model = algo_ctor()
                    tracker = ConvergenceTracker(model)
                    tracker.solve(problem)
                    runs_convergence.append(tracker.convergence)
                    print(f"    {algo_name} run {run+1}/{N_RUNS} ✓")
                except Exception as e:
                    print(f"    {algo_name} run {run+1} ERROR: {e}")
                    runs_convergence.append([])

            convergence_data[fname][algo_name] = runs_convergence

    # Save
    path = os.path.join(RESULTS_DIR, "convergence_data.json")
    with open(path, 'w') as f:
        json.dump(convergence_data, f)
    print(f"\n✓ Convergence data saved: {path}")
    return convergence_data


if __name__ == "__main__":
    run_convergence()
