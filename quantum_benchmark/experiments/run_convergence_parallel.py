"""
Experiment 3: Convergence Curves — PARALLEL VERSION
====================================================
Put this in: experiments/run_convergence_parallel.py
Run directly: python experiments/run_convergence_parallel.py

Records best fitness at every epoch for selected functions.
Parallelizes across the 5 runs per function per algorithm.
"""

import sys, os, time, logging, warnings, json
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.disable(logging.CRITICAL)
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.result_manager import RESULTS_DIR

CONVERGENCE_FUNCS = ["F1", "F9", "F10", "F23"]
N_RUNS   = 5
POP_SIZE = 30
EPOCH    = 500
N_CORES  = max(1, multiprocessing.cpu_count() - 1)
ALGO_NAMES = ["GWO", "QGWO", "FA", "QFA", "ACO", "QACO"]


def _one_convergence_trial(args):
    """One trial — records epoch-by-epoch best fitness."""
    import logging, warnings, sys, os
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings('ignore')
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    algo_name, func_name, run_id, epoch, pop_size = args

    from mealpy import GWO, FFA, ACOR
    from mealpy.utils.space import FloatVar
    from algorithms.quantum_gwo import QGWO
    from algorithms.quantum_fa  import QFA
    from algorithms.quantum_aco import QACO
    from benchmarks.classical_23 import BENCHMARK_FUNCTIONS

    func_info = next((f for f in BENCHMARK_FUNCTIONS if f[0] == func_name), None)
    if func_info is None:
        return algo_name, func_name, run_id, []

    fname, func, dim, lb, ub, _ = func_info

    algo_map = {
        "GWO":  lambda: GWO.OriginalGWO(epoch=epoch, pop_size=pop_size),
        "QGWO": lambda: QGWO(epoch=epoch, pop_size=pop_size,
                             delta_theta_max=0.05, tunnel_prob=0.01),
        "FA":   lambda: FFA.OriginalFFA(epoch=epoch, pop_size=pop_size,
                                        max_sparks=0.5, p_sparks=1.0, exp_const=1.0),
        "QFA":  lambda: QFA(epoch=epoch, pop_size=pop_size,
                            max_sparks=0.5, p_sparks=1.0, exp_const=1.0,
                            delta_theta_max=0.05, tunnel_prob=0.01),
        "ACO":  lambda: ACOR.OriginalACOR(epoch=epoch, pop_size=pop_size,
                                          sample_count=50, intent_factor=0.5, zeta=1.0),
        "QACO": lambda: QACO(epoch=epoch, pop_size=pop_size,
                             sample_count=50, intent_factor=0.5, zeta=1.0,
                             delta_theta=0.01, tunnel_prob=0.02),
    }

    try:
        bounds  = FloatVar(lb=tuple([lb]*dim), ub=tuple([ub]*dim), name="v")
        problem = {"obj_func": func, "bounds": bounds, "minmax": "min",
                   "n_dims": dim, "log_to": None}
        model = algo_map[algo_name]()
        model.solve(problem, seed=None)

        # Extract epoch-by-epoch best fitness from history
        if hasattr(model, 'history') and hasattr(model.history, 'list_global_best_fit'):
            curve = [float(v) for v in model.history.list_global_best_fit]
        else:
            curve = [float(model.g_best.target.fitness)] * epoch

        return algo_name, func_name, run_id, curve
    except Exception:
        return algo_name, func_name, run_id, []


def run_convergence():
    print("=" * 65)
    print("EXPERIMENT 3: Convergence Curves [PARALLEL]")
    print(f"  Functions  : {CONVERGENCE_FUNCS}")
    print(f"  Runs each  : {N_RUNS}")
    print(f"  CPU cores  : {N_CORES} of {multiprocessing.cpu_count()} available")
    print("=" * 65)

    # Build all tasks
    all_tasks = []
    for fname in CONVERGENCE_FUNCS:
        for algo_name in ALGO_NAMES:
            for run_id in range(N_RUNS):
                all_tasks.append((algo_name, fname, run_id, EPOCH, POP_SIZE))

    total = len(all_tasks)
    print(f"\n  Total trials: {total} "
          f"({len(CONVERGENCE_FUNCS)} funcs × {len(ALGO_NAMES)} algos × {N_RUNS} runs)\n")

    # Storage: conv_data[func][algo] = list of N_RUNS curves
    conv_data = {f: {a: [None]*N_RUNS for a in ALGO_NAMES}
                 for f in CONVERGENCE_FUNCS}
    completed = set()

    done = 0
    start = time.time()

    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        future_map = {executor.submit(_one_convergence_trial, t): t for t in all_tasks}

        for future in as_completed(future_map):
            algo_name, func_name, run_id, curve = future.result()
            conv_data[func_name][algo_name][run_id] = curve
            done += 1

            key = (func_name, algo_name)
            done_runs = [v for v in conv_data[func_name][algo_name] if v is not None]
            if len(done_runs) == N_RUNS and key not in completed:
                completed.add(key)
                print(f"  ✓ {func_name}/{algo_name:6s} — {N_RUNS} convergence curves done")

    # Clean up Nones → empty lists
    for fname in CONVERGENCE_FUNCS:
        for algo in ALGO_NAMES:
            conv_data[fname][algo] = [v if v is not None else []
                                      for v in conv_data[fname][algo]]

    # Save
    path = os.path.join(RESULTS_DIR, "convergence_data.json")
    with open(path, 'w') as f:
        json.dump(conv_data, f)

    elapsed = time.time() - start
    print(f"\n  ✓ Convergence data saved: {path}")
    print(f"  ✓ Complete in {elapsed/60:.1f} min")
    return conv_data


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_convergence()
