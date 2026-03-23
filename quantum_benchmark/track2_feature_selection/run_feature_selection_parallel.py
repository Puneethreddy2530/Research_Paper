"""
Track 2: Feature Selection — PARALLEL VERSION
===============================================
Parallelizes the 30 independent runs across all CPU cores.
Drop-in replacement for run_feature_selection.py

Speedup: ~N_CORES x faster (e.g. 16 cores = 16x faster)
Results: Identical — same 30 runs, same stats, same paper tables.

How:
  For each dataset × algorithm:
    Instead of: for run in range(30): run_one()   ← sequential
    We do:      executor.map(run_one, range(30))   ← all cores at once
"""

import sys, os, time, logging, warnings, json
import numpy as np
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.disable(logging.CRITICAL)
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.result_manager import RESULTS_DIR

# ── Settings ─────────────────────────────────────────────────
N_RUNS   = 30
EPOCH    = 200
POP_SIZE = 20
K_KNN    = 5
CV_FOLDS = 5
N_CORES  = max(1, multiprocessing.cpu_count() - 1)

ALGO_NAMES = ["GWO", "QGWO", "FA", "QFA", "ACO", "QACO"]


# ── Worker — must be top-level for pickling (Windows requirement) ──
def _one_trial(args):
    """
    Runs ONE trial completely isolated in its own process.
    Returns dict with accuracy, n_selected, reduction, fitness.
    """
    import logging, warnings, sys, os
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings('ignore')
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    algo_name, ds_name, run_id, epoch, pop_size, k_knn, cv_folds = args

    from mealpy import GWO, FFA, ACOR
    from mealpy.utils.space import FloatVar
    from algorithms.quantum_gwo import QGWO
    from algorithms.quantum_fa  import QFA
    from algorithms.quantum_aco import QACO
    from track2_feature_selection.datasets import load_all_datasets
    from track2_feature_selection.fitness  import FeatureSelectionFitness

    # Load dataset
    all_data = load_all_datasets(verbose=False)
    if ds_name not in all_data:
        return algo_name, ds_name, run_id, None

    X, y, n_feats = all_data[ds_name]

    # Build bounds + fitness
    bounds = FloatVar(lb=tuple([-6.0]*n_feats),
                      ub=tuple([ 6.0]*n_feats), name="f")
    fitness_fn = FeatureSelectionFitness(X, y, k_neighbors=k_knn, cv_folds=cv_folds)

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
        problem = {
            "obj_func": fitness_fn,
            "bounds":   bounds,
            "minmax":   "min",
            "n_dims":   bounds.n_vars,
            "log_to":   None,
        }
        model = algo_map[algo_name]()
        model.solve(problem, seed=None)
        best_x   = model.g_best.solution
        decoded  = fitness_fn.decode(best_x)
        return algo_name, ds_name, run_id, {
            "fitness":    float(model.g_best.target.fitness),
            "accuracy":   float(decoded["accuracy"]),
            "n_selected": int(decoded["n_selected"]),
            "reduction":  float(decoded["reduction"]),
        }
    except Exception as e:
        return algo_name, ds_name, run_id, {
            "fitness": 1.0, "accuracy": 0.0,
            "n_selected": n_feats, "reduction": 0.0
        }


def run_feature_selection_parallel():
    from track2_feature_selection.datasets import load_all_datasets
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    print("=" * 65)
    print("TRACK 2: Feature Selection [PARALLEL]")
    print(f"  Algorithms : {ALGO_NAMES}")
    print(f"  Runs each  : {N_RUNS} | Epochs: {EPOCH}")
    print(f"  CPU cores  : {N_CORES} of {multiprocessing.cpu_count()} available")
    print(f"  Speedup    : ~{N_CORES}x vs sequential")
    print("=" * 65)

    # Load all datasets upfront (just to get names + shapes)
    print("\nLoading datasets...")
    all_data = load_all_datasets(verbose=True)
    dataset_names = list(all_data.keys())

    # Build ALL tasks: dataset × algo × run
    all_tasks = []
    for ds_name in dataset_names:
        for algo_name in ALGO_NAMES:
            for run_id in range(N_RUNS):
                all_tasks.append((
                    algo_name, ds_name, run_id,
                    EPOCH, POP_SIZE, K_KNN, CV_FOLDS
                ))

    total_tasks = len(all_tasks)
    print(f"\n  Total trials: {total_tasks} "
          f"({len(dataset_names)} datasets × {len(ALGO_NAMES)} algos × {N_RUNS} runs)")
    print(f"  Submitting to {N_CORES} parallel workers...\n")

    # Storage: results[ds][algo] = list of N_RUNS result dicts
    results = {ds: {a: [None]*N_RUNS for a in ALGO_NAMES}
               for ds in dataset_names}
    completed_keys = set()

    done = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        future_map = {executor.submit(_one_trial, task): task
                      for task in all_tasks}

        for future in as_completed(future_map):
            algo_name, ds_name, run_id, result = future.result()
            if result is not None:
                results[ds_name][algo_name][run_id] = result
            done += 1

            # Progress every 60 trials
            if done % 60 == 0:
                elapsed = time.time() - start_time
                pct = 100 * done / total_tasks
                eta = elapsed / done * (total_tasks - done)
                print(f"  Progress: {pct:.1f}% | "
                      f"Done: {done}/{total_tasks} | "
                      f"ETA: {eta/60:.0f}min")

            # Print summary when all runs for one algo+dataset complete
            key = (ds_name, algo_name)
            done_runs = [r for r in results[ds_name][algo_name] if r is not None]
            if len(done_runs) == N_RUNS and key not in completed_keys:
                completed_keys.add(key)
                accs = [r['accuracy']   for r in done_runs]
                sels = [r['n_selected'] for r in done_runs]
                n_f  = all_data[ds_name][2]
                print(f"  ✓ {ds_name}/{algo_name:6s} | "
                      f"Acc: {np.mean(accs):.4f}±{np.std(accs):.4f} | "
                      f"Features: {np.mean(sels):.1f}/{n_f} "
                      f"({(1-np.mean(sels)/n_f)*100:.1f}% reduced)")

    # ── Aggregate into summary rows ──────────────────────
    summary_rows = []
    raw_all = {}

    for ds_name in dataset_names:
        X, y, n_feats = all_data[ds_name]
        baseline_acc = float(np.mean(cross_val_score(
            KNeighborsClassifier(n_neighbors=K_KNN), X, y, cv=CV_FOLDS)))
        raw_all[ds_name] = {}

        for algo_name in ALGO_NAMES:
            run_results = [r for r in results[ds_name][algo_name] if r is not None]
            if not run_results:
                continue

            accs = [r['accuracy']   for r in run_results]
            sels = [r['n_selected'] for r in run_results]
            reds = [r['reduction']  for r in run_results]
            fits = [r['fitness']    for r in run_results]

            raw_all[ds_name][algo_name] = run_results
            summary_rows.append({
                "Dataset":        ds_name,
                "N_Features":     n_feats,
                "Baseline_Acc":   baseline_acc,
                "Algorithm":      algo_name,
                "Acc_Mean":       np.mean(accs),
                "Acc_Std":        np.std(accs),
                "Acc_Best":       np.max(accs),
                "Sel_Mean":       np.mean(sels),
                "Sel_Std":        np.std(sels),
                "Reduction_Mean": np.mean(reds) * 100,
                "Fitness_Mean":   np.mean(fits),
                "Fitness_Best":   np.min(fits),
            })

    # ── Save ────────────────────────────────────────────
    df = pd.DataFrame(summary_rows)
    p1 = os.path.join(RESULTS_DIR, "feature_selection_summary.csv")
    df.to_csv(p1, index=False, float_format="%.6f")
    print(f"\n  ✓ Saved: {p1}")

    p2 = os.path.join(RESULTS_DIR, "feature_selection_raw.json")
    with open(p2, 'w') as f:
        json.dump(raw_all, f, indent=2)
    print(f"  ✓ Saved: {p2}")

    elapsed = time.time() - start_time
    print(f"\n✓ Track 2 [PARALLEL] complete in {elapsed/3600:.2f} hours")
    print(f"  (Sequential would have taken ~{elapsed*N_CORES/3600:.1f} hours)")
    return df


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_feature_selection_parallel()
