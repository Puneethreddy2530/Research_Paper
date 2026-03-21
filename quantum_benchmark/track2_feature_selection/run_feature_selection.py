"""
Track 2: Feature Selection Experiment
=======================================
Runs all 6 algorithms on 12 UCI datasets as binary feature selectors.

Output per dataset per algorithm (30 runs):
  - Classification accuracy (mean +/- std)
  - Number of features selected (mean +/- std)
  - Feature reduction % 
  - Best fitness achieved

Paper tables this generates:
  Table A: Accuracy comparison
  Table B: Feature count comparison  
  Table C: Feature reduction %
  Table D: Wilcoxon significance
"""

import sys, os, time, logging, warnings
import numpy as np
import pandas as pd
import json

logging.disable(logging.CRITICAL)
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mealpy import GWO, FFA, ACOR
from mealpy.utils.space import FloatVar
from algorithms.quantum_gwo import QGWO
from algorithms.quantum_fa  import QFA
from algorithms.quantum_aco import QACO
from track2_feature_selection.datasets import load_all_datasets
from track2_feature_selection.fitness  import FeatureSelectionFitness
from utils.result_manager import RESULTS_DIR

# ── Settings ────────────────────────────────────────────
N_RUNS   = 30
POP_SIZE = 20
EPOCH    = 100
K_KNN    = 5
CV_FOLDS = 5

ALGO_NAMES = ["GWO", "QGWO", "FA", "QFA", "ACO", "QACO"]


def make_algo(name, n_features):
    """Instantiate algorithm with correct bounds for n_features."""
    bounds = FloatVar(lb=tuple([-6.0]*n_features), ub=tuple([6.0]*n_features), name="f")
    makers = {
        "GWO":  lambda: GWO.OriginalGWO(epoch=EPOCH, pop_size=POP_SIZE),
        "QGWO": lambda: QGWO(epoch=EPOCH, pop_size=POP_SIZE, delta_theta_max=0.05, tunnel_prob=0.01),
        "FA":   lambda: FFA.OriginalFFA(epoch=EPOCH, pop_size=POP_SIZE, max_sparks=0.5, p_sparks=1.0, exp_const=1.0),
        "QFA":  lambda: QFA(epoch=EPOCH, pop_size=POP_SIZE, max_sparks=0.5, p_sparks=1.0, exp_const=1.0),
        "ACO":  lambda: ACOR.OriginalACOR(epoch=EPOCH, pop_size=POP_SIZE, sample_count=50, intent_factor=0.5, zeta=1.0),
        "QACO": lambda: QACO(epoch=EPOCH, pop_size=POP_SIZE, sample_count=50, intent_factor=0.5, zeta=1.0),
    }
    return bounds, makers[name]


def run_one_trial(algo_ctor, bounds, fitness_fn):
    """One run: solve + decode result."""
    problem = {
        "obj_func": fitness_fn,
        "bounds":   bounds,
        "minmax":   "min",
        "n_dims":   bounds.n_vars,
        "log_to":   None,
    }
    model = algo_ctor()
    model.solve(problem, seed=None)
    best_x = model.g_best.solution
    decoded = fitness_fn.decode(best_x)
    return {
        "fitness":    float(model.g_best.target.fitness),
        "accuracy":   float(decoded["accuracy"]),
        "n_selected": int(decoded["n_selected"]),
        "reduction":  float(decoded["reduction"]),
    }


def run_feature_selection():
    print("=" * 65)
    print("TRACK 2: Feature Selection on UCI Datasets")
    print(f"  Algorithms: {ALGO_NAMES}")
    print(f"  Runs: {N_RUNS} | Epochs: {EPOCH} | KNN k={K_KNN} | CV={CV_FOLDS}-fold")
    print("=" * 65)

    print("\nLoading datasets...")
    all_data = load_all_datasets(verbose=True)

    all_results  = {}
    summary_rows = []
    total_start  = time.time()

    for ds_name, (X, y, n_feats) in all_data.items():
        print(f"\n{'='*55}")
        print(f"Dataset: {ds_name} ({X.shape[0]}x{n_feats})")
        print(f"{'='*55}")

        # Baseline: all features
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        baseline_acc = float(np.mean(cross_val_score(
            KNeighborsClassifier(n_neighbors=K_KNN), X, y, cv=CV_FOLDS)))
        print(f"  Baseline KNN (all {n_feats} features): {baseline_acc:.4f}")

        all_results[ds_name] = {}

        for algo_name in ALGO_NAMES:
            bounds, algo_ctor = make_algo(algo_name, n_feats)
            run_results = []
            t0 = time.time()

            for run in range(N_RUNS):
                fitness_fn = FeatureSelectionFitness(X, y, k_neighbors=K_KNN, cv_folds=CV_FOLDS)
                try:
                    res = run_one_trial(algo_ctor, bounds, fitness_fn)
                    run_results.append(res)
                except Exception as e:
                    run_results.append({"fitness": 1.0, "accuracy": 0.0, "n_selected": n_feats, "reduction": 0.0})

                if (run + 1) % 10 == 0:
                    valid = [r for r in run_results if r['accuracy'] > 0]
                    if valid:
                        print(f"    {algo_name} [{run+1}/{N_RUNS}] "
                              f"acc={np.mean([r['accuracy'] for r in valid]):.4f} "
                              f"sel={np.mean([r['n_selected'] for r in valid]):.1f}/{n_feats} "
                              f"({time.time()-t0:.0f}s)")

            accs = [r['accuracy']   for r in run_results]
            sels = [r['n_selected'] for r in run_results]
            reds = [r['reduction']  for r in run_results]
            fits = [r['fitness']    for r in run_results]

            all_results[ds_name][algo_name] = run_results
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
                "Reduction_Std":  np.std(reds) * 100,
                "Fitness_Mean":   np.mean(fits),
                "Fitness_Best":   np.min(fits),
            })

            print(f"  {algo_name:6s}: acc={np.mean(accs):.4f}+/-{np.std(accs):.4f}  "
                  f"features={np.mean(sels):.1f}/{n_feats} (reduction={np.mean(reds)*100:.1f}%)")

    # Save
    df = pd.DataFrame(summary_rows)
    p1 = os.path.join(RESULTS_DIR, "feature_selection_summary.csv")
    df.to_csv(p1, index=False, float_format="%.6f")
    print(f"\n  Saved: {p1}")

    p2 = os.path.join(RESULTS_DIR, "feature_selection_raw.json")
    with open(p2, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: {p2}")

    elapsed = time.time() - total_start
    print(f"\nTrack 2 complete in {elapsed/3600:.2f} hours")
    return df


if __name__ == "__main__":
    run_feature_selection()
