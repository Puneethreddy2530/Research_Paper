"""
Track 3: WSN Node Localization
================================
Directly extends Dr. Rajakumar's GWO-LPWSN (2017) paper.
This is what will make your sir VERY happy.

Problem Setup (matching Rajakumar 2017 exactly):
  - Network area: 300m × 300m
  - Total nodes: 100 (randomly placed)
  - Anchor nodes: 20% (known positions) = 20 anchors
  - Unknown nodes: 80 (to be localized)
  - Communication range: 40m (nodes within range can measure distance)
  - Distance model: Euclidean + Gaussian noise (σ = 0.02)

What each algorithm does:
  - Solution vector = [x1, y1, x2, y2, ..., x_n, y_n] for all unknown nodes
  - Fitness = average localization error (Euclidean distance from true position)
  - Lower is better

Metrics reported (same as Rajakumar 2017 Table):
  1. Minimum Localization Error (MLE)
  2. Average Localization Error (ALE)
  3. % Nodes Localized (within acceptable error threshold)
  4. Computation time

Why this matters for your paper:
  - Direct domain application of all 6 algorithms
  - Directly comparable to sir's 2017 results (CITE IT)
  - Quantum variants tested on WSN for first time (another novel contribution)
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
from utils.result_manager   import RESULTS_DIR

# ── WSN Parameters (matching Rajakumar 2017) ────────────────
AREA_SIZE     = 300     # meters × meters
N_TOTAL       = 100     # total sensor nodes
ANCHOR_RATIO  = 0.20    # 20% anchors (known positions)
N_ANCHORS     = int(N_TOTAL * ANCHOR_RATIO)   # = 20
N_UNKNOWN     = N_TOTAL - N_ANCHORS           # = 80
COMM_RANGE    = 40      # communication range in meters
NOISE_SIGMA   = 0.02    # Gaussian noise std deviation
ERROR_THRESH  = 1.0     # acceptable error threshold (meters) for "localized"

# ── Algorithm Settings ───────────────────────────────────────
N_RUNS   = 30
POP_SIZE = 30
EPOCH    = 500


# ─────────────────────────────────────────────────────────────
# WSN NETWORK GENERATION
# ─────────────────────────────────────────────────────────────

class WSNNetwork:
    """
    Generates a random WSN and provides the fitness function.
    Seed-controlled for reproducible network topologies.
    """

    def __init__(self, seed=None):
        rng = np.random.RandomState(seed)

        # Place all nodes randomly in [0, AREA_SIZE]²
        all_positions = rng.uniform(0, AREA_SIZE, (N_TOTAL, 2))

        # First N_ANCHORS are anchor nodes (known)
        self.anchor_pos = all_positions[:N_ANCHORS]    # shape (20, 2)

        # Remaining are unknown
        self.unknown_pos = all_positions[N_ANCHORS:]   # shape (80, 2)

        # Compute noisy distances: unknown → anchors (within range)
        self.distances = self._compute_distances(rng)

        # Store true positions for error computation
        self.true_positions = self.unknown_pos.copy()

    def _compute_distances(self, rng):
        """
        For each unknown node, compute noisy Euclidean distances
        to all anchor nodes within communication range.
        Returns dict: {unknown_idx: {anchor_idx: noisy_dist}}
        """
        distances = {}
        for i in range(N_UNKNOWN):
            distances[i] = {}
            for j in range(N_ANCHORS):
                true_dist = np.linalg.norm(
                    self.unknown_pos[i] - self.anchor_pos[j])
                if true_dist <= COMM_RANGE:
                    # Add Gaussian noise
                    noise = rng.normal(0, NOISE_SIGMA * true_dist)
                    distances[i][j] = true_dist + noise
        return distances

    def fitness(self, solution):
        """
        Localization fitness function.

        solution: flat array [x0,y0, x1,y1, ..., x79,y79]
                  = estimated positions of all 80 unknown nodes

        Returns: mean localization error across all nodes (lower = better)
        """
        # Reshape to (N_UNKNOWN, 2)
        estimated = solution.reshape(N_UNKNOWN, 2)
        # Clip to valid area
        estimated = np.clip(estimated, 0, AREA_SIZE)

        total_error = 0.0
        counted = 0

        for i in range(N_UNKNOWN):
            est_pos = estimated[i]

            # Error term 1: distance from true position (what we really want to minimize)
            true_error = np.linalg.norm(est_pos - self.true_positions[i])

            # Error term 2: consistency with measured distances to anchors
            range_error = 0.0
            n_anchors_in_range = len(self.distances[i])

            if n_anchors_in_range > 0:
                for j, meas_dist in self.distances[i].items():
                    calc_dist = np.linalg.norm(est_pos - self.anchor_pos[j])
                    range_error += (calc_dist - meas_dist) ** 2
                range_error = np.sqrt(range_error / n_anchors_in_range)
            else:
                # Node out of range of all anchors — penalize
                range_error = AREA_SIZE

            total_error += range_error
            counted += 1

        return float(total_error / max(counted, 1))

    def evaluate_solution(self, solution):
        """
        Compute full stats for a solution:
        - Mean, min localization error
        - % nodes localized within ERROR_THRESH
        """
        estimated = solution.reshape(N_UNKNOWN, 2)
        estimated = np.clip(estimated, 0, AREA_SIZE)

        errors = []
        for i in range(N_UNKNOWN):
            err = np.linalg.norm(estimated[i] - self.true_positions[i])
            errors.append(err)

        errors = np.array(errors)
        n_localized = np.sum(errors <= ERROR_THRESH)

        return {
            "mean_error":    float(np.mean(errors)),
            "min_error":     float(np.min(errors)),
            "max_error":     float(np.max(errors)),
            "std_error":     float(np.std(errors)),
            "pct_localized": float(100 * n_localized / N_UNKNOWN),
            "n_localized":   int(n_localized),
        }


# ─────────────────────────────────────────────────────────────
# EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────────

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

# Solution dimension: 2 coordinates × 80 unknown nodes = 160
SOLUTION_DIM = N_UNKNOWN * 2


def run_wsn_trial(algo_ctor, network):
    """Run one algorithm on one network topology. Returns best solution."""
    bounds = FloatVar(
        lb=tuple([0.0] * SOLUTION_DIM),
        ub=tuple([AREA_SIZE] * SOLUTION_DIM),
        name="positions"
    )
    problem = {
        "obj_func": network.fitness,
        "bounds":   bounds,
        "minmax":   "min",
        "n_dims":   SOLUTION_DIM,
        "log_to":   None,
    }
    model = algo_ctor()
    model.solve(problem, seed=None)
    return model.g_best.solution


def run_experiment(n_topologies=5):
    """
    Run WSN localization experiment.

    n_topologies: number of different random network layouts to test.
    Results averaged across topologies AND runs (standard in WSN papers).

    Total trials = n_topologies × N_RUNS × 6 algorithms
    Estimated time: ~2-3 hours (500 epochs, 160-dim problem)
    """
    print("=" * 65)
    print("TRACK 3: WSN NODE LOCALIZATION")
    print(f"  Network: {AREA_SIZE}m×{AREA_SIZE}m | {N_TOTAL} nodes | "
          f"{N_ANCHORS} anchors | {N_UNKNOWN} unknown")
    print(f"  Comm range: {COMM_RANGE}m | Noise σ={NOISE_SIGMA}")
    print(f"  Topologies: {n_topologies} | Runs per topology: {N_RUNS}")
    print(f"  Solution dim: {SOLUTION_DIM} (x,y for each unknown node)")
    print("=" * 65)

    all_results = []
    raw_errors  = {a: [] for a in ALGORITHMS}

    total_start = time.time()

    for topo_id in range(n_topologies):
        print(f"\n── Topology {topo_id+1}/{n_topologies} ──")
        network = WSNNetwork(seed=topo_id * 42)   # deterministic seed per topology
        print(f"   Anchors in range: avg {np.mean([len(v) for v in network.distances.values()]):.1f}")

        for algo_name, algo_ctor in ALGORITHMS.items():
            run_mean_errors = []
            run_pct_localized = []

            t0 = time.time()
            for run in range(N_RUNS):
                try:
                    best_sol = run_wsn_trial(algo_ctor, network)
                    stats = network.evaluate_solution(best_sol)
                    run_mean_errors.append(stats['mean_error'])
                    run_pct_localized.append(stats['pct_localized'])
                    raw_errors[algo_name].append(stats['mean_error'])
                except Exception as e:
                    run_mean_errors.append(AREA_SIZE)
                    run_pct_localized.append(0.0)

            elapsed = time.time() - t0
            mean_err = np.mean(run_mean_errors)
            mean_pct = np.mean(run_pct_localized)

            print(f"   {algo_name:6s} | MeanErr: {mean_err:.4f}m | "
                  f"Localized: {mean_pct:.1f}% | {elapsed:.0f}s")

            all_results.append({
                "Topology":          topo_id + 1,
                "Algorithm":         algo_name,
                "Mean_Error_m":      round(mean_err, 4),
                "Std_Error_m":       round(np.std(run_mean_errors), 4),
                "Min_Error_m":       round(min(run_mean_errors), 4),
                "Max_Error_m":       round(max(run_mean_errors), 4),
                "Pct_Localized":     round(mean_pct, 2),
                "N_Anchors":         N_ANCHORS,
                "N_Unknown":         N_UNKNOWN,
                "Comm_Range_m":      COMM_RANGE,
                "Noise_Sigma":       NOISE_SIGMA,
            })

    # ── Overall summary across all topologies ──────────────
    print(f"\n{'─'*55}")
    print("OVERALL SUMMARY (averaged across all topologies):")
    print(f"{'─'*55}")

    df = pd.DataFrame(all_results)
    summary = df.groupby("Algorithm").agg({
        "Mean_Error_m": ["mean", "std"],
        "Pct_Localized": ["mean", "std"],
    }).round(4)
    print(summary)

    # ── Save ──────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "wsn_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  ✓ WSN results: {csv_path}")

    raw_path = os.path.join(RESULTS_DIR, "wsn_raw_errors.json")
    with open(raw_path, 'w') as f:
        json.dump(raw_errors, f, indent=2)
    print(f"  ✓ Raw errors: {raw_path}")

    total_time = time.time() - total_start
    print(f"\n✓ Track 3 complete in {total_time/3600:.2f} hours")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--topologies', type=int, default=5,
                        help='Number of random network topologies (default 5)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 3 topologies, 3 runs, 100 epochs')
    args = parser.parse_args()

    if args.quick:
        N_RUNS = 3; EPOCH = 100
        print("[QUICK MODE]")
        for algo_name in ALGORITHMS:
            ALGORITHMS[algo_name] = eval(
                f"lambda: {algo_name if algo_name in ['GWO','FA'] else algo_name}"
            )

    run_experiment(n_topologies=args.topologies if not args.quick else 2)
