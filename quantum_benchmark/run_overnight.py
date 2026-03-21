"""
run_overnight.py
=================
Run this before you go to sleep.
It runs Track 2 (Feature Selection) + Track 3 (WSN) back to back.
All results + figures will be ready by morning.

Usage:
  python run_overnight.py              # Full run (~6-8 hours)
  python run_overnight.py --quick      # Test run (~5-10 min)
  python run_overnight.py --track2     # Feature selection only
  python run_overnight.py --track3     # WSN only

Estimated times (full run):
  Track 2 (12 datasets × 6 algos × 30 runs × 200 epochs): ~4-5 hours
  Track 3 (5 topologies × 6 algos × 30 runs × 500 epochs): ~2-3 hours
  Total: ~6-8 hours overnight
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_banner(text):
    print("\n" + "═"*65)
    print(f"  {text}")
    print("═"*65)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick',  action='store_true', help='Fast test run')
    parser.add_argument('--track2', action='store_true', help='Track 2 only')
    parser.add_argument('--track3', action='store_true', help='Track 3 only')
    args = parser.parse_args()

    # If neither specified, run both
    run_t2 = args.track2 or (not args.track2 and not args.track3)
    run_t3 = args.track3 or (not args.track2 and not args.track3)

    start_total = time.time()
    print_banner("OVERNIGHT RUN: TRACKS 2 & 3")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Quick mode: {args.quick}")

    # ── TRACK 2: Feature Selection ────────────────────────
    if run_t2:
        print_banner("TRACK 2: FEATURE SELECTION (UCI Datasets)")
        import track2_feature_selection.run_feature_selection as t2

        if args.quick:
            t2.N_RUNS = 5
            t2.EPOCH  = 50
            datasets_to_run = ["Iris", "Wine", "BreastCancer"]
        else:
            t2.N_RUNS = 30
            t2.EPOCH  = 200
            datasets_to_run = None   # all 12

        t2_start = time.time()
        try:
            df_t2 = t2.run_experiment(datasets_to_run)
            print(f"\n✓ Track 2 done in {(time.time()-t2_start)/3600:.2f}h")
        except Exception as e:
            print(f"✗ Track 2 failed: {e}")
            import traceback; traceback.print_exc()

        # Generate Track 2 plots immediately
        try:
            from plots.plot_tracks_2_3 import run_all_track2_plots
            run_all_track2_plots()
        except Exception as e:
            print(f"[WARN] Track 2 plots failed: {e}")

    # ── TRACK 3: WSN Localization ─────────────────────────
    if run_t3:
        print_banner("TRACK 3: WSN NODE LOCALIZATION")
        import track3_wsn.wsn_localization as t3

        if args.quick:
            t3.N_RUNS = 3
            t3.EPOCH  = 50
            n_topos   = 2
        else:
            t3.N_RUNS = 30
            t3.EPOCH  = 500
            n_topos   = 5

        # Rebuild ALGORITHMS dict with new EPOCH
        from mealpy import GWO, FFA, ACOR
        from algorithms.quantum_gwo import QGWO
        from algorithms.quantum_fa  import QFA
        from algorithms.quantum_aco import QACO

        t3.ALGORITHMS = {
            "GWO":  lambda: GWO.OriginalGWO(epoch=t3.EPOCH, pop_size=t3.POP_SIZE),
            "QGWO": lambda: QGWO(epoch=t3.EPOCH, pop_size=t3.POP_SIZE),
            "FA":   lambda: FFA.OriginalFFA(epoch=t3.EPOCH, pop_size=t3.POP_SIZE,
                                            max_sparks=0.5, p_sparks=1.0, exp_const=1.0),
            "QFA":  lambda: QFA(epoch=t3.EPOCH, pop_size=t3.POP_SIZE),
            "ACO":  lambda: ACOR.OriginalACOR(epoch=t3.EPOCH, pop_size=t3.POP_SIZE,
                                              sample_count=50, intent_factor=0.5, zeta=1.0),
            "QACO": lambda: QACO(epoch=t3.EPOCH, pop_size=t3.POP_SIZE),
        }

        t3_start = time.time()
        try:
            df_t3 = t3.run_experiment(n_topologies=n_topos)
            print(f"\n✓ Track 3 done in {(time.time()-t3_start)/3600:.2f}h")
        except Exception as e:
            print(f"✗ Track 3 failed: {e}")
            import traceback; traceback.print_exc()

        # Generate Track 3 plots
        try:
            from plots.plot_tracks_2_3 import run_all_track3_plots
            run_all_track3_plots()
        except Exception as e:
            print(f"[WARN] Track 3 plots failed: {e}")

    # ── Statistical tests on feature selection ────────────
    print_banner("STATISTICAL TESTS (Feature Selection)")
    try:
        from stats.statistical_tests import run_all_tests
        run_all_tests("feature_selection_raw_accuracy.json")
    except Exception as e:
        print(f"[WARN] Stats failed: {e}")

    # ── Final summary ─────────────────────────────────────
    total = time.time() - start_total
    print_banner(f"ALL DONE in {total/3600:.2f} hours")
    print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("  Results files:")
    print("  📊 results/feature_selection_results.csv  → Tables 4,5")
    print("  📊 results/wsn_results.csv                → Table 6")
    print("  📊 results/wilcoxon_results.csv           → Table 7")
    print()
    print("  Figures:")
    print("  🖼  plots/output/track2_accuracy_comparison.png  → Figure 7")
    print("  🖼  plots/output/track2_feature_reduction.png    → Figure 8")
    print("  🖼  plots/output/track2_accuracy_vs_features.png → Figure 9")
    print("  🖼  plots/output/track2_fs_heatmap.png           → Figure 10")
    print("  🖼  plots/output/track3_wsn_network.png          → Figure 11")
    print("  🖼  plots/output/track3_wsn_error.png            → Figure 12")
    print("  🖼  plots/output/track3_wsn_localized_pct.png    → Figure 13")


if __name__ == "__main__":
    main()
