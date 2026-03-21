"""
Experiment 1: Classical 23 Benchmark Functions
================================================
Runs all 6 algorithms on all 23 functions for 30 independent runs.
Uses paper-standard settings:
  - Population: 30
  - Epochs: 500
  - Runs: 30 independent
  - Records: Mean, Std, Best, Worst, Median

Runtime estimate: ~10-15 minutes on a normal laptop
"""

import sys, os, time, logging
import numpy as np

# ── suppress mealpy's verbose logging ──────────────────────
logging.disable(logging.CRITICAL)

# ── path setup ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mealpy import GWO, FFA, ACOR
from mealpy.utils.space import FloatVar

from algorithms.quantum_gwo import QGWO
from algorithms.quantum_fa  import QFA
from algorithms.quantum_aco import QACO
from benchmarks.classical_23 import BENCHMARK_FUNCTIONS
from utils.result_manager    import save_run_results, save_raw_runs

# ── Experiment settings ─────────────────────────────────────
N_RUNS      = 30    # 30 independent runs (paper standard)
POP_SIZE    = 30    # population size
EPOCH       = 500   # iterations

# ── Algorithm definitions ───────────────────────────────────
# Each is (display_name, constructor_function)
ALGORITHMS = {
    "GWO":  lambda: GWO.OriginalGWO(epoch=EPOCH, pop_size=POP_SIZE),
    "QGWO": lambda: QGWO(epoch=EPOCH, pop_size=POP_SIZE,
                         delta_theta_max=0.05, tunnel_prob=0.01),
    "FA":   lambda: FFA.OriginalFFA(epoch=EPOCH, pop_size=POP_SIZE,
                                    max_sparks=0.5, p_sparks=1.0, exp_const=1.0),
    "QFA":  lambda: QFA(epoch=EPOCH, pop_size=POP_SIZE,
                        max_sparks=0.5, p_sparks=1.0, exp_const=1.0,
                        delta_theta_max=0.05, tunnel_prob=0.01),
    "ACO":  lambda: ACOR.OriginalACOR(epoch=EPOCH, pop_size=POP_SIZE,
                                      sample_count=50, intent_factor=0.5, zeta=1.0),
    "QACO": lambda: QACO(epoch=EPOCH, pop_size=POP_SIZE,
                         sample_count=50, intent_factor=0.5, zeta=1.0,
                         delta_theta=0.01, tunnel_prob=0.02),
}


def run_single(algo_constructor, func, dim, lb, ub):
    """
    Run one algorithm on one function for one trial.
    Returns best fitness value found.
    """
    bounds = FloatVar(
        lb=tuple([lb] * dim),
        ub=tuple([ub] * dim),
        name="vars"
    )

    problem_dict = {
        "obj_func": func,
        "bounds":   bounds,
        "minmax":   "min",
        "n_dims":   dim,
        "log_to":   None,    # suppress all output
    }

    model = algo_constructor()
    model.solve(problem_dict, seed=None)   # seed=None for independent runs
    return model.g_best.target.fitness


def run_experiment():
    """
    Main experiment loop.
    For each function × algorithm: run N_RUNS times, record all fitness values.
    """
    print("=" * 60)
    print("EXPERIMENT 1: Classical 23 Benchmark Functions")
    print(f"  Algorithms : {list(ALGORITHMS.keys())}")
    print(f"  Functions  : F1–F23 (23 total)")
    print(f"  Runs each  : {N_RUNS}")
    print(f"  Dim        : 30 (F1–F13), varies (F14–F23)")
    print("=" * 60)

    # results[func_name][algo_name] = [run1_fitness, run2_fitness, ...]
    results  = {f[0]: {a: [] for a in ALGORITHMS} for f in BENCHMARK_FUNCTIONS}
    raw_runs = {f[0]: {a: [] for a in ALGORITHMS} for f in BENCHMARK_FUNCTIONS}

    total_tasks = len(BENCHMARK_FUNCTIONS) * len(ALGORITHMS) * N_RUNS
    done = 0
    start_time = time.time()

    for fname, func, dim, lb, ub, f_opt in BENCHMARK_FUNCTIONS:
        print(f"\n── {fname} (dim={dim}, lb={lb}, ub={ub}) ──")

        for algo_name, algo_constructor in ALGORITHMS.items():
            run_values = []

            for run in range(N_RUNS):
                try:
                    fitness = run_single(algo_constructor, func, dim, lb, ub)
                    run_values.append(float(fitness))
                except Exception as e:
                    print(f"    [ERROR] {fname}/{algo_name}/run{run}: {e}")
                    run_values.append(float('nan'))

                done += 1
                if done % 30 == 0:
                    elapsed = time.time() - start_time
                    pct = 100 * done / total_tasks
                    eta = elapsed / done * (total_tasks - done) if done > 0 else 0
                    print(f"    Progress: {pct:.1f}% | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

            mean_val = np.nanmean(run_values)
            std_val  = np.nanstd(run_values)
            best_val = np.nanmin(run_values)
            print(f"    {algo_name:6s} → Mean: {mean_val:.4e}  Std: {std_val:.4e}  Best: {best_val:.4e}")

            results[fname][algo_name]  = run_values
            raw_runs[fname][algo_name] = run_values

    # ── Save results ────────────────────────────────────────
    print("\n── Saving results ──")
    df = save_run_results(results,  "classical23_results.csv")
    save_raw_runs(raw_runs,          "classical23_raw_runs.json")

    total_time = time.time() - start_time
    print(f"\n✓ Experiment 1 complete in {total_time/60:.1f} minutes")
    return df


if __name__ == "__main__":
    run_experiment()
