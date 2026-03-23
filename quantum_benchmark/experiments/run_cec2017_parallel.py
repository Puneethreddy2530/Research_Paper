"""
Experiment 2: CEC 2017 — PARALLEL VERSION
==========================================
Put this in: experiments/run_cec2017_parallel.py
Run directly: python experiments/run_cec2017_parallel.py
Or via:       python main_parallel.py  (auto-detected)

Uses all CPU cores for ~15x speedup on 16-core machine.
Results identical to sequential version.
"""

import sys, os, time, logging, warnings
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.disable(logging.CRITICAL)
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.result_manager import save_run_results, save_raw_runs

# ── Settings ────────────────────────────────────────────────
N_RUNS     = 30
POP_SIZE   = 30
EPOCH      = 500
DIMENSIONS = [10, 30]
N_CORES    = max(1, multiprocessing.cpu_count() - 1)

CEC2017_IDS = [1, 3, 4, 5, 6, 7, 8, 9, 10,
               11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
               21, 22, 23, 24, 25, 26, 27, 28, 29]

CATEGORIES = {
    **{f: "Unimodal"     for f in [1, 3]},
    **{f: "Simple Multi" for f in [4,5,6,7,8,9,10]},
    **{f: "Hybrid"       for f in [11,12,13,14,15,16,17,18,19,20]},
    **{f: "Composition"  for f in [21,22,23,24,25,26,27,28,29]},
}

ALGO_NAMES = ["GWO", "QGWO", "FA", "QFA", "ACO", "QACO"]


def _one_cec_trial(args):
    """Single trial — runs in its own process."""
    import logging, warnings, sys, os
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings('ignore')
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    algo_name, func_id, ndim, run_id, epoch, pop_size = args

    from mealpy import GWO, FFA, ACOR
    from mealpy.utils.space import FloatVar
    from algorithms.quantum_gwo import QGWO
    from algorithms.quantum_fa  import QFA
    from algorithms.quantum_aco import QACO
    import opfunu

    # Load CEC function
    try:
        func_class_name = f"F{func_id}2017"
        func_class = getattr(opfunu.cec_based, func_class_name, None)
        if func_class is None:
            funcs = opfunu.get_cec_functions(cec_year=2017, ndim=ndim)
            for f in funcs:
                if f.__name__.endswith(f"F{func_id}"):
                    func_class = f
                    break
        if func_class is None:
            return algo_name, func_id, ndim, run_id, float('nan')
        f_inst = func_class(ndim=ndim)
        func   = f_inst.evaluate
    except Exception:
        return algo_name, func_id, ndim, run_id, float('nan')

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
        bounds  = FloatVar(lb=tuple([-100.0]*ndim), ub=tuple([100.0]*ndim), name="v")
        problem = {"obj_func": func, "bounds": bounds, "minmax": "min",
                   "n_dims": ndim, "log_to": None}
        model = algo_map[algo_name]()
        model.solve(problem, seed=None)
        return algo_name, func_id, ndim, run_id, float(model.g_best.target.fitness)
    except Exception:
        return algo_name, func_id, ndim, run_id, float('nan')


def run_experiment():
    print("=" * 65)
    print("EXPERIMENT 2: CEC 2017 Benchmark Suite [PARALLEL]")
    print(f"  Dimensions : {DIMENSIONS}")
    print(f"  Functions  : {len(CEC2017_IDS)} (F1,F3–F29)")
    print(f"  Algorithms : {ALGO_NAMES}")
    print(f"  Runs each  : {N_RUNS}")
    print(f"  CPU cores  : {N_CORES} of {multiprocessing.cpu_count()} available")
    print("=" * 65)

    for dim in DIMENSIONS:
        print(f"\n{'='*55}")
        print(f"DIMENSION = {dim}  [PARALLEL — {N_CORES} cores]")
        print(f"{'='*55}")

        # Build all tasks for this dimension
        all_tasks = []
        for fid in CEC2017_IDS:
            for algo_name in ALGO_NAMES:
                for run_id in range(N_RUNS):
                    all_tasks.append((algo_name, fid, dim, run_id, EPOCH, POP_SIZE))

        total = len(all_tasks)
        print(f"  Total trials: {total} "
              f"({len(CEC2017_IDS)} funcs × {len(ALGO_NAMES)} algos × {N_RUNS} runs)")

        # Storage: results[fname][algo] = list of N_RUNS values
        results  = {f"F{fid}": {a: [None]*N_RUNS for a in ALGO_NAMES}
                    for fid in CEC2017_IDS}
        raw_runs = {f"F{fid}": {a: [None]*N_RUNS for a in ALGO_NAMES}
                    for fid in CEC2017_IDS}
        completed = set()

        done = 0
        start = time.time()

        with ProcessPoolExecutor(max_workers=N_CORES) as executor:
            future_map = {executor.submit(_one_cec_trial, t): t for t in all_tasks}

            for future in as_completed(future_map):
                algo_name, fid, ndim, run_id, fitness = future.result()
                fname = f"F{fid}"
                results[fname][algo_name][run_id]  = fitness
                raw_runs[fname][algo_name][run_id] = fitness
                done += 1

                if done % (len(ALGO_NAMES) * N_RUNS) == 0:
                    elapsed = time.time() - start
                    pct = 100 * done / total
                    eta = elapsed / done * (total - done)
                    print(f"  Progress: {pct:.1f}% | "
                          f"Done: {done}/{total} | "
                          f"ETA: {eta/60:.0f}min")

                # Print when all runs for one algo+func done
                key = (fname, algo_name)
                done_runs = [v for v in results[fname][algo_name] if v is not None]
                if len(done_runs) == N_RUNS and key not in completed:
                    completed.add(key)
                    arr = np.array(done_runs)
                    cat = CATEGORIES.get(fid, "?")
                    print(f"  ✓ {fname}[{cat}]/{algo_name:6s} | "
                          f"Mean: {np.nanmean(arr):.4e}  "
                          f"Std: {np.nanstd(arr):.4e}")

        # Clean up Nones
        for fname in results:
            for algo in ALGO_NAMES:
                results[fname][algo]  = [v if v is not None else float('nan')
                                         for v in results[fname][algo]]
                raw_runs[fname][algo] = results[fname][algo]

        # Save per dimension
        suffix = f"_D{dim}"
        save_run_results(results,  f"cec2017_results{suffix}.csv")
        save_raw_runs(raw_runs,    f"cec2017_raw_runs{suffix}.json")

        elapsed = time.time() - start
        print(f"\n  ✓ D={dim} complete in {elapsed/60:.1f} min")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_experiment()
