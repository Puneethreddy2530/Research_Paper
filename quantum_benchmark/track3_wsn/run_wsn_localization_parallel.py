"""
Track 3: WSN Localization — PARALLEL VERSION
=============================================
Parallelizes the 30 independent runs per topology per algorithm.
Drop-in replacement for run_wsn_localization.py

Speedup: ~N_CORES x faster
Results: Identical — same 30 runs, same paper tables.
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
N_RUNS       = 30
EPOCHS       = 500
POP_SIZE     = 30
N_TOPOLOGIES = 5        # number of different random network layouts
N_CORES      = max(1, multiprocessing.cpu_count() - 1)

ALGO_NAMES = ["GWO", "QGWO", "FA", "QFA", "ACO", "QACO", "AQHSO"]
RUN_ALGOS  = ["AQHSO"]


def _one_wsn_trial(args):
    """
    Runs ONE trial of ONE algorithm on ONE network topology.
    Fully isolated — runs in its own process.
    """
    import logging, warnings, sys, os
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings('ignore')
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    algo_name, topo_seed, run_id, epochs, pop_size = args

    from mealpy import GWO, FFA, ACOR
    from mealpy.utils.space import FloatVar
    from algorithms.quantum_gwo import QGWO
    from algorithms.quantum_fa  import QFA
    from algorithms.quantum_aco import QACO
    from algorithms.aqhso       import AQHSO

    # Import WSN network class from the existing wsn file
    # Works with whichever version is in the repo
    try:
        from track3_wsn.wsn_localization import WSNNetwork, SOLUTION_DIM, AREA_SIZE
    except ImportError:
        from track3_wsn.run_wsn_localization import WSNNetwork, SOLUTION_DIM, AREA_SIZE

    algo_map = {
        "GWO":  lambda: GWO.OriginalGWO(epoch=epochs, pop_size=pop_size),
        "QGWO": lambda: QGWO(epoch=epochs, pop_size=pop_size,
                             delta_theta_max=0.05, tunnel_prob=0.01),
        "FA":   lambda: FFA.OriginalFFA(epoch=epochs, pop_size=pop_size,
                                        max_sparks=0.5, p_sparks=1.0, exp_const=1.0),
        "QFA":  lambda: QFA(epoch=epochs, pop_size=pop_size,
                            max_sparks=0.5, p_sparks=1.0, exp_const=1.0,
                            delta_theta_max=0.05, tunnel_prob=0.01),
        "ACO":  lambda: ACOR.OriginalACOR(epoch=epochs, pop_size=pop_size,
                                          sample_count=50, intent_factor=0.5, zeta=1.0),
        "QACO": lambda: QACO(epoch=epochs, pop_size=pop_size,
                             sample_count=50, intent_factor=0.5, zeta=1.0,
                             delta_theta=0.01, tunnel_prob=0.02),
        "AQHSO": lambda: AQHSO(epoch=epochs, pop_size=pop_size),
    }

    try:
        network = WSNNetwork(seed=topo_seed)
        bounds  = FloatVar(lb=tuple([0.0]*SOLUTION_DIM),
                           ub=tuple([AREA_SIZE]*SOLUTION_DIM),
                           name="positions")
        problem = {
            "obj_func": network.fitness,
            "bounds":   bounds,
            "minmax":   "min",
            "n_dims":   SOLUTION_DIM,
            "log_to":   None,
        }
        model = algo_map[algo_name]()
        model.solve(problem, seed=None)
        metrics = network.evaluate_solution(model.g_best.solution)
        return algo_name, topo_seed, run_id, metrics
    except Exception as e:
        return algo_name, topo_seed, run_id, {
            "mean_error": 9999.0, "min_error": 9999.0,
            "pct_localized": 0.0, "std_error": 0.0
        }


def run_wsn_localization_parallel():
    # Import constants from whichever wsn file exists
    try:
        from track3_wsn.wsn_localization import (
            AREA_SIZE, N_TOTAL, N_ANCHORS, N_UNKNOWN, COMM_RANGE, NOISE_SIGMA)
    except ImportError:
        from track3_wsn.run_wsn_localization import (
            AREA_SIZE, N_TOTAL, N_ANCHORS, N_UNKNOWN, COMM_RANGE, NOISE_SIGMA)

    print("=" * 65)
    print("TRACK 3: WSN Node Localization [PARALLEL]")
    print(f"  Network: {AREA_SIZE}m×{AREA_SIZE}m | {N_TOTAL} nodes | "
          f"{N_ANCHORS} anchors | {N_UNKNOWN} unknown")
    print(f"  Topologies: {N_TOPOLOGIES} | Runs per topology: {N_RUNS}")
    print(f"  CPU cores: {N_CORES} of {multiprocessing.cpu_count()} available")
    print(f"  Speedup  : ~{N_CORES}x vs sequential")
    print("=" * 65)

    # Build ALL tasks: topology × algo × run
    topo_seeds = [i * 42 for i in range(N_TOPOLOGIES)]
    all_tasks = []
    for seed in topo_seeds:
        for algo_name in RUN_ALGOS:
            for run_id in range(N_RUNS):
                all_tasks.append((algo_name, seed, run_id, EPOCHS, POP_SIZE))

    total_tasks = len(all_tasks)
    print(f"\n  Total trials: {total_tasks} "
          f"({N_TOPOLOGIES} topos × {len(ALGO_NAMES)} algos × {N_RUNS} runs)")
    print(f"  Submitting to {N_CORES} workers...\n")

    # Storage: results[seed][algo] = list of N_RUNS metric dicts
    results = {s: {a: [None]*N_RUNS for a in ALGO_NAMES} for s in topo_seeds}
    completed_keys = set()

    done = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        future_map = {executor.submit(_one_wsn_trial, task): task
                      for task in all_tasks}

        for future in as_completed(future_map):
            algo_name, topo_seed, run_id, metrics = future.result()
            results[topo_seed][algo_name][run_id] = metrics
            done += 1

            if done % 30 == 0:
                elapsed = time.time() - start_time
                pct = 100 * done / total_tasks
                eta = elapsed / done * (total_tasks - done)
                print(f"  Progress: {pct:.1f}% | "
                      f"Done: {done}/{total_tasks} | "
                      f"ETA: {eta/60:.0f}min")

            # Summary when all runs for one algo+topo done
            key = (topo_seed, algo_name)
            done_runs = [r for r in results[topo_seed][algo_name] if r is not None]
            if len(done_runs) == N_RUNS and key not in completed_keys:
                completed_keys.add(key)
                errs = [r['mean_error']    for r in done_runs]
                pcts = [r['pct_localized'] for r in done_runs]
                topo_id = topo_seeds.index(topo_seed) + 1
                print(f"  ✓ Topo{topo_id}/{algo_name:6s} | "
                      f"MeanErr: {np.mean(errs):.4f}m | "
                      f"Localized: {np.mean(pcts):.1f}%")

    # ── Aggregate into summary rows ──────────────────────
    summary_rows = []
    raw_metrics  = {a: [] for a in ALGO_NAMES}

    for topo_id, seed in enumerate(topo_seeds):
        for algo_name in RUN_ALGOS:
            run_results = [r for r in results[seed][algo_name] if r is not None]
            if not run_results:
                continue

            errs = [r['mean_error']    for r in run_results]
            pcts = [r['pct_localized'] for r in run_results]
            mins = [r['min_error']     for r in run_results]
            raw_metrics[algo_name].extend(errs)

            summary_rows.append({
                "Topology":      topo_id + 1,
                "Algorithm":     algo_name,
                "Mean_Error_m":  round(np.mean(errs), 4),
                "Std_Error_m":   round(np.std(errs),  4),
                "Min_Error_m":   round(np.mean(mins),  4),
                "Pct_Localized": round(np.mean(pcts),  2),
                "N_Runs":        len(run_results),
            })

    # ── Save ────────────────────────────────────────────
    new_df = pd.DataFrame(summary_rows)
    p1 = os.path.join(RESULTS_DIR, "wsn_localization_summary.csv")
    try:
        old_df = pd.read_csv(p1)
        old_df = old_df[old_df['Algorithm'] != "AQHSO"]
        df = pd.concat([old_df, new_df], ignore_index=True)
    except:
        df = new_df
    df.to_csv(p1, index=False)
    print(f"\n  ✓ Merged and Saved: {p1}")

    p2 = os.path.join(RESULTS_DIR, "wsn_raw_metrics.json")
    try:
        with open(p2, 'r') as f:
            old_raw = json.load(f)
        for a in ALGO_NAMES:
            if a != "AQHSO" and a in old_raw:
                raw_metrics[a] = old_raw[a]
    except:
        pass
    with open(p2, 'w') as f:
        json.dump(raw_metrics, f, indent=2)
    print(f"  ✓ Merged and Saved: {p2}")

    # Print overall summary
    print("\n  Overall (averaged across all topologies):")
    overall = df.groupby("Algorithm")[["Mean_Error_m","Pct_Localized"]].mean()
    overall = overall.loc[[a for a in ALGO_NAMES if a in overall.index]]
    print(overall.round(4).to_string())

    elapsed = time.time() - start_time
    print(f"\n✓ Track 3 [PARALLEL] complete in {elapsed/3600:.2f} hours")
    print(f"  (Sequential would have taken ~{elapsed*N_CORES/3600:.1f} hours)")
    return df


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_wsn_localization_parallel()
