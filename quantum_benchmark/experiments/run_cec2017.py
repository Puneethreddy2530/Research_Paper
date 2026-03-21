"""
Experiment 2: CEC 2017 Benchmark Suite
=========================================
Runs all 6 algorithms on CEC 2017 functions (F1–F29, F2 excluded).
KEY CONTRIBUTION: No quantum GWO/FA/ACO variant has been tested on CEC 2017 before.

Uses opfunu library for CEC 2017 function implementations.
Tested dimensions: D = 10, 30 (D=50 optional, very slow)

Runtime:
  D=10: ~20-30 min
  D=30: ~1-2 hours
  D=50: ~3-5 hours (skip for initial runs)

CEC 2017 function categories:
  F1:        Shifted and Rotated Bent Cigar     (unimodal)
  F3:        Shifted and Rotated Zakharov       (unimodal)
  F4-F10:    Simple multimodal functions
  F11-F20:   Hybrid functions
  F21-F29:   Composition functions
  (F2 excluded — known instability)
"""

import sys, os, time, logging, warnings
import numpy as np

logging.disable(logging.CRITICAL)
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mealpy import GWO, FFA, ACOR
from mealpy.utils.space import FloatVar
import opfunu

from algorithms.quantum_gwo import QGWO
from algorithms.quantum_fa  import QFA
from algorithms.quantum_aco import QACO
from utils.result_manager   import save_run_results, save_raw_runs

# ── Experiment settings ─────────────────────────────────────
N_RUNS      = 30
POP_SIZE    = 30
EPOCH       = 500
DIMENSIONS  = [10, 30]   # Add 50 if you have time

# CEC 2017 function IDs (skip F2)
CEC2017_IDS = [1, 3, 4, 5, 6, 7, 8, 9, 10,
               11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
               21, 22, 23, 24, 25, 26, 27, 28, 29]

# Category labels (useful for paper table)
CATEGORIES = {
    **{f: "Unimodal"      for f in [1, 3]},
    **{f: "Simple Multi"  for f in [4,5,6,7,8,9,10]},
    **{f: "Hybrid"        for f in [11,12,13,14,15,16,17,18,19,20]},
    **{f: "Composition"   for f in [21,22,23,24,25,26,27,28,29]},
}

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


def get_cec2017_function(func_id, ndim):
    """
    Load CEC 2017 function using opfunu.
    Returns (func_callable, lb, ub, optimal_value)
    """
    func_class_name = f"F{func_id}2017"
    func_class = getattr(opfunu.cec_based, func_class_name, None)
    if func_class is None:
        # Try alternative naming
        funcs = opfunu.get_cec_functions(cec_year=2017, ndim=ndim)
        for f in funcs:
            if f.__name__.endswith(f"F{func_id}"):
                func_class = f
                break

    if func_class is None:
        raise ImportError(f"CEC2017 F{func_id} not found in opfunu")

    f_instance = func_class(ndim=ndim)
    return f_instance.evaluate, -100.0, 100.0, f_instance.f_global


def run_experiment():
    print("=" * 60)
    print("EXPERIMENT 2: CEC 2017 Benchmark Suite")
    print(f"  Dimensions : {DIMENSIONS}")
    print(f"  Functions  : {len(CEC2017_IDS)} (F1,F3-F29)")
    print(f"  Algorithms : {list(ALGORITHMS.keys())}")
    print(f"  Runs each  : {N_RUNS}")
    print("=" * 60)

    for dim in DIMENSIONS:
        print(f"\n{'='*50}")
        print(f"DIMENSION = {dim}")
        print(f"{'='*50}")

        results  = {}
        raw_runs = {}
        start_time = time.time()

        for fid in CEC2017_IDS:
            fname = f"F{fid}"
            cat   = CATEGORIES.get(fid, "Unknown")
            print(f"\n── CEC2017 {fname} [{cat}] (D={dim}) ──")

            try:
                func, lb, ub, f_opt = get_cec2017_function(fid, dim)
            except Exception as e:
                print(f"  [SKIP] Could not load {fname}: {e}")
                continue

            results[fname]  = {a: [] for a in ALGORITHMS}
            raw_runs[fname] = {a: [] for a in ALGORITHMS}

            for algo_name, algo_ctor in ALGORITHMS.items():
                run_values = []
                for run in range(N_RUNS):
                    try:
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
                        model = algo_ctor()
                        model.solve(problem, seed=None)
                        fitness = model.g_best.target.fitness
                        run_values.append(float(fitness))
                    except Exception as e:
                        run_values.append(float('nan'))

                mean_val = np.nanmean(run_values)
                std_val  = np.nanstd(run_values)
                print(f"    {algo_name:6s} → Mean: {mean_val:.4e}  Std: {std_val:.4e}")
                results[fname][algo_name]  = run_values
                raw_runs[fname][algo_name] = run_values

        # Save per dimension
        suffix = f"_D{dim}"
        save_run_results(results,  f"cec2017_results{suffix}.csv")
        save_raw_runs(raw_runs,    f"cec2017_raw_runs{suffix}.json")

        elapsed = time.time() - start_time
        print(f"\n✓ D={dim} complete in {elapsed/60:.1f} min")

    print("\n✓ CEC 2017 experiments complete!")


if __name__ == "__main__":
    run_experiment()
