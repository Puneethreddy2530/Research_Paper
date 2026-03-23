"""
main_parallel.py — Parallel Version of main.py
================================================
Drop-in replacement for main.py.
Uses ALL CPU cores for ~Nx speedup (N = core count).

Usage:
  python main_parallel.py                  # Full run, all cores
  python main_parallel.py --skip-cec2017  # Skip the long CEC 2017
  python main_parallel.py --cores 4       # Limit to 4 cores

How it works:
  The 30 independent runs per algo per function are embarrassingly
  parallel — no communication needed between runs.
  ProcessPoolExecutor spawns N worker processes (bypasses GIL).
  Each process runs one trial independently.

Windows note:
  multiprocessing.freeze_support() is called automatically.
  Must run from command line, not Jupyter notebook.
"""

import sys, os, time, argparse, multiprocessing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows multiprocessing requirement — MUST be before any imports
if __name__ == "__main__":
    multiprocessing.freeze_support()


def print_banner(text):
    print("\n" + "─"*55)
    print(text)
    print("─"*55)


def main():
    parser = argparse.ArgumentParser(description='Parallel Quantum Benchmark Suite')
    parser.add_argument('--skip-cec2017', action='store_true',
                        help='Skip CEC 2017 (saves ~8 hrs)')
    parser.add_argument('--cores', type=int, default=None,
                        help='Number of CPU cores to use (default: all-1)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 5 runs, 100 epochs')
    args = parser.parse_args()

    n_cores_available = multiprocessing.cpu_count()
    n_cores_use = args.cores if args.cores else max(1, n_cores_available - 1)

    print("=" * 65)
    print("  QUANTUM vs CLASSICAL METAHEURISTIC BENCHMARK SUITE")
    print("  [PARALLEL MODE]")
    print(f"  CPU cores: {n_cores_use} of {n_cores_available} available")
    print(f"  Expected speedup: ~{n_cores_use}x")
    print("=" * 65)

    start_total = time.time()

    # Patch N_CORES into the parallel experiment module
    import experiments.run_classical23_parallel as exp23
    exp23.N_CORES = n_cores_use
    if args.quick:
        exp23.N_RUNS = 5
        exp23.EPOCH  = 100
        print("\n[QUICK MODE] 5 runs, 100 epochs")

    # ── STEP 1: Classical 23 ──────────────────────────────
    print_banner("STEP 1: Classical 23 Benchmark Functions [PARALLEL]")
    exp23.run_experiment()

    # ── STEP 2: CEC 2017 ─────────────────────────────────
    if not args.skip_cec2017:
        print_banner("STEP 2: CEC 2017 [PARALLEL]")
        try:
            import experiments.run_cec2017_parallel as cec
            cec.N_CORES = n_cores_use
            if args.quick:
                cec.N_RUNS = 5; cec.EPOCH = 100
                cec.DIMENSIONS = [10]
            cec.run_experiment()
        except ImportError:
            # Fallback to sequential if parallel CEC not available
            print("  [Falling back to sequential CEC 2017]")
            import experiments.run_cec2017 as cec_seq
            if args.quick:
                cec_seq.N_RUNS = 5; cec_seq.EPOCH = 100
            cec_seq.run_experiment()
    else:
        print("\n[SKIPPED] CEC 2017 — use --skip-cec2017 intentionally")

    # ── STEP 3: Convergence ───────────────────────────────
    print_banner("STEP 3: Convergence Curves")
    try:
        from experiments.run_convergence import run_convergence
        run_convergence()
    except Exception as e:
        print(f"  [WARN] Convergence failed: {e}")

    # ── STEP 4: Statistical Tests ─────────────────────────
    print_banner("STEP 4: Statistical Tests")
    try:
        from stats.statistical_tests import run_all_tests
        run_all_tests("classical23_raw_runs.json")
    except Exception as e:
        print(f"  [WARN] Stats failed: {e}")

    # ── STEP 5: Plots ─────────────────────────────────────
    print_banner("STEP 5: Generating Figures")
    try:
        from plots.plot_results import run_all_plots
        run_all_plots()
    except Exception as e:
        print(f"  [WARN] Plots failed: {e}")

    # ── Done ──────────────────────────────────────────────
    total = time.time() - start_total
    print("\n" + "="*65)
    print(f"  ✓ ALL DONE in {total/60:.1f} minutes")
    print(f"  Results: quantum_benchmark/results/")
    print(f"  Figures: quantum_benchmark/plots/output/")
    print("="*65)


if __name__ == "__main__":
    main()
