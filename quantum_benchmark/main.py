"""
main.py — One-Click Runner
===========================
Runs the complete benchmarking pipeline:
  1. Classical 23 benchmark experiments
  2. CEC 2017 benchmark experiments (D=10, D=30)
  3. Convergence curve generation
  4. Statistical tests (Wilcoxon, Friedman, Nemenyi)
  5. All paper figures

Usage:
  python main.py                    # Run everything
  python main.py --quick            # Quick test (5 runs, 100 epochs)
  python main.py --skip-cec2017     # Skip the long CEC 2017 run
  python main.py --stats-only       # Only run stats on existing results

Estimated runtimes:
  Quick mode (--quick):  ~2-3 minutes
  Classical 23 only:     ~15 minutes
  Full (everything):     ~3-4 hours
"""

import argparse, os, sys, time

def main():
    parser = argparse.ArgumentParser(description='Quantum Metaheuristic Benchmark Suite')
    parser.add_argument('--quick',        action='store_true', help='Fast test run (5 runs, 100 epochs)')
    parser.add_argument('--skip-cec2017', action='store_true', help='Skip CEC 2017 (saves ~2hrs)')
    parser.add_argument('--stats-only',   action='store_true', help='Only run stats on existing results')
    parser.add_argument('--plots-only',   action='store_true', help='Only generate plots')
    parser.add_argument('--no-plots',     action='store_true', help='Skip plot generation')
    args = parser.parse_args()

    print("=" * 65)
    print("  QUANTUM vs CLASSICAL METAHEURISTIC BENCHMARK SUITE")
    print("  Paper: Comparative Analysis of GWO, FA, ACO")
    print("         vs QGWO, QFA, QACO")
    print("=" * 65)

    start = time.time()

    if args.plots_only:
        from plots.plot_results import run_all_plots
        run_all_plots()
        return

    if args.stats_only:
        from stats.statistical_tests import run_all_tests
        run_all_tests()
        return

    # ── Patch settings for quick mode ────────────────────────
    if args.quick:
        print("\n[QUICK MODE] Using 5 runs, 100 epochs for fast testing")
        import experiments.run_classical23 as e23
        import experiments.run_cec2017    as ecec
        import experiments.run_convergence as econv
        e23.N_RUNS   = 5;   e23.EPOCH = 100
        ecec.N_RUNS  = 5;   ecec.EPOCH = 100; ecec.DIMENSIONS = [10]
        econv.N_RUNS = 2;   econv.EPOCH = 100

    # ── Step 1: Classical 23 ─────────────────────────────────
    print("\n" + "─"*50)
    print("STEP 1: Classical 23 Benchmark Functions")
    print("─"*50)
    from experiments.run_classical23 import run_experiment
    run_experiment()

    # ── Step 2: CEC 2017 ─────────────────────────────────────
    if not args.skip_cec2017:
        print("\n" + "─"*50)
        print("STEP 2: CEC 2017 Benchmark Suite")
        print("─"*50)
        from experiments.run_cec2017 import run_experiment as run_cec
        run_cec()
    else:
        print("\n[SKIPPED] CEC 2017 experiments")

    # ── Step 3: Convergence ───────────────────────────────────
    print("\n" + "─"*50)
    print("STEP 3: Convergence Curves")
    print("─"*50)
    from experiments.run_convergence import run_convergence
    run_convergence()

    # ── Step 4: Statistical Tests ─────────────────────────────
    print("\n" + "─"*50)
    print("STEP 4: Statistical Tests")
    print("─"*50)
    from stats.statistical_tests import run_all_tests
    run_all_tests("classical23_raw_runs.json")

    # ── Step 5: Plots ─────────────────────────────────────────
    if not args.no_plots:
        print("\n" + "─"*50)
        print("STEP 5: Generating Paper Figures")
        print("─"*50)
        from plots.plot_results import run_all_plots
        run_all_plots()

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.time() - start
    print("\n" + "="*65)
    print(f"  ✓ ALL DONE in {elapsed/60:.1f} minutes")
    print(f"  Results: quantum_benchmark/results/")
    print(f"  Figures: quantum_benchmark/plots/output/")
    print("="*65)
    print("\n  Files to use in your paper:")
    print("  📊 classical23_results.csv    → Tables 1, 2")
    print("  📊 cec2017_results_D30.csv    → Table 3")
    print("  📊 wilcoxon_results.csv       → Table 4 (sig test)")
    print("  📊 friedman_ranks.csv         → Table 5")
    print("  📊 significance_table.csv     → Table 6 (+/=/- table)")
    print("  🖼  fig1_barchart_F1_F7.png    → Figure 1")
    print("  🖼  fig2_boxplots.png          → Figure 2")
    print("  🖼  fig3_convergence.png       → Figure 3")
    print("  🖼  fig4_ranking_heatmap.png   → Figure 4")
    print("  🖼  fig5_quantum_improvement.png → Figure 5")
    print("  🖼  cd_diagram.png             → Figure 6")


if __name__ == "__main__":
    main()
