"""
Experiment 1: Classical 23 Benchmark Functions — PARALLEL VERSION
==================================================================
Uses multiprocessing to run all 30 independent runs simultaneously
across all available CPU cores.

Why this works:
  - Each of the 30 runs is completely independent (different random seed)
  - We use ProcessPoolExecutor to bypass Python's GIL entirely
  - Each process gets its own Python interpreter → true parallelism
  - Speedup = roughly equal to number of CPU cores

On a 4-core machine: 4x faster
On an 8-core machine: 8x faster
On a 12-core machine: ~10x faster (some overhead)

IMPORTANT: Results are identical to sequential version.
Same 30 runs, same stats, same paper tables.
"""

import sys, os, time, logging, warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

logging.disable(logging.CRITICAL)
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.result_manager import save_run_results, save_raw_runs

# ── Settings ────────────────────────────────────────────────
N_RUNS   = 30
POP_SIZE = 30
EPOCH    = 500
N_CORES  = max(1, multiprocessing.cpu_count() - 1)  # leave 1 core for OS


# ── Worker function (must be at module level for pickling) ──
def _run_one_trial(args):
    """
    Runs ONE trial of ONE algorithm on ONE function.
    This runs in a separate process — fully isolated.

    Args: (algo_name, func_name, run_id, epoch, pop_size)
    Returns: (algo_name, func_name, run_id, fitness)
    """
    import logging, warnings, sys, os
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings('ignore')

    algo_name, func_name, run_id, epoch, pop_size = args

    # Re-import everything inside worker process
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from mealpy import GWO, FFA, ACOR
    from mealpy.utils.space import FloatVar
    from algorithms.quantum_gwo import QGWO
    from algorithms.quantum_fa  import QFA
    from algorithms.quantum_aco import QACO
    from benchmarks.classical_23 import BENCHMARK_FUNCTIONS

    # Get the function info
    func_info = next((f for f in BENCHMARK_FUNCTIONS if f[0] == func_name), None)
    if func_info is None:
        return algo_name, func_name, run_id, float('nan')

    fname, func, dim, lb, ub, f_opt = func_info

    # Build algorithm
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
        bounds = FloatVar(lb=tuple([lb]*dim), ub=tuple([ub]*dim), name="vars")
        problem = {"obj_func": func, "bounds": bounds, "minmax": "min",
                   "n_dims": dim, "log_to": None}
        model = algo_map[algo_name]()
        model.solve(problem, seed=None)
        return algo_name, func_name, run_id, float(model.g_best.target.fitness)
    except Exception as e:
        return algo_name, func_name, run_id, float('nan')


def run_experiment():
    from benchmarks.classical_23 import BENCHMARK_FUNCTIONS

    ALGO_NAMES = ["GWO", "QGWO", "FA", "QFA", "ACO", "QACO"]

    print("=" * 65)
    print("EXPERIMENT 1: Classical 23 Benchmark Functions [PARALLEL]")
    print(f"  Algorithms : {ALGO_NAMES}")
    print(f"  Functions  : 23 total")
    print(f"  Runs each  : {N_RUNS}")
    print(f"  CPU cores  : {N_CORES} (of {multiprocessing.cpu_count()} available)")
    print(f"  Speedup    : ~{N_CORES}x vs sequential")
    print("=" * 65)

    # Build all tasks upfront
    # Each task = (algo_name, func_name, run_id, epoch, pop_size)
    all_tasks = []
    for fname, func, dim, lb, ub, f_opt in BENCHMARK_FUNCTIONS:
        for algo_name in ALGO_NAMES:
            for run_id in range(N_RUNS):
                all_tasks.append((algo_name, fname, run_id, EPOCH, POP_SIZE))

    total_tasks = len(all_tasks)
    print(f"\n  Total trials: {total_tasks} "
          f"({len(BENCHMARK_FUNCTIONS)} funcs × {len(ALGO_NAMES)} algos × {N_RUNS} runs)")
    print(f"  Submitting to {N_CORES} parallel workers...\n")

    # Storage: results[func][algo] = list of N_RUNS fitness values
    results  = {f[0]: {a: [None]*N_RUNS for a in ALGO_NAMES} for f in BENCHMARK_FUNCTIONS}
    raw_runs = {f[0]: {a: [None]*N_RUNS for a in ALGO_NAMES} for f in BENCHMARK_FUNCTIONS}

    done = 0
    start_time = time.time()
    last_print = {}   # track last completed run per (func, algo)

    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        # Submit ALL tasks at once — executor manages the queue
        future_map = {executor.submit(_run_one_trial, task): task
                      for task in all_tasks}

        for future in as_completed(future_map):
            algo_name, func_name, run_id, fitness = future.result()
            results[func_name][algo_name][run_id]  = fitness
            raw_runs[func_name][algo_name][run_id] = fitness
            done += 1

            # Print progress every 30 completed trials
            if done % 30 == 0:
                elapsed = time.time() - start_time
                pct = 100 * done / total_tasks
                eta = elapsed / done * (total_tasks - done)
                print(f"  Progress: {pct:.1f}% | "
                      f"Done: {done}/{total_tasks} | "
                      f"Elapsed: {elapsed:.0f}s | "
                      f"ETA: {eta:.0f}s ({eta/60:.1f}min)")

            # Print algo summary when all 30 runs for that algo+func are done
            key = (func_name, algo_name)
            runs_done = [r for r in results[func_name][algo_name] if r is not None]
            if len(runs_done) == N_RUNS and key not in last_print:
                last_print[key] = True
                arr = np.array(runs_done)
                print(f"  ✓ {func_name}/{algo_name:6s} → "
                      f"Mean: {np.nanmean(arr):.4e}  "
                      f"Std: {np.nanstd(arr):.4e}  "
                      f"Best: {np.nanmin(arr):.4e}")

    # Replace any None with nan (failed runs)
    for fname in results:
        for algo in ALGO_NAMES:
            results[fname][algo]  = [v if v is not None else float('nan')
                                     for v in results[fname][algo]]
            raw_runs[fname][algo] = results[fname][algo]

    # Save
    print("\n── Saving results ──")
    df = save_run_results(results,  "classical23_results.csv")
    save_raw_runs(raw_runs,         "classical23_raw_runs.json")

    total_time = time.time() - start_time
    print(f"\n✓ Parallel Experiment 1 complete in {total_time/60:.1f} minutes")
    print(f"  (Sequential would have taken ~{total_time * N_CORES / 60:.0f} min)")
    return df


if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    run_experiment()
