"""
run_overnight_parallel.py
==========================
Parallel version of run_overnight.py — uses ALL CPU cores.
Run this before you go to sleep.

Usage:
  python run_overnight_parallel.py              # Full run, all cores
  python run_overnight_parallel.py --quick      # 5 min test
  python run_overnight_parallel.py --track2     # Track 2 only
  python run_overnight_parallel.py --track3     # Track 3 only
  python run_overnight_parallel.py --cores 8    # Limit cores

Estimated time with 16 cores:
  Track 2: ~20-30 min  (was 4-5 hours sequential)
  Track 3: ~15-20 min  (was 2-3 hours sequential)
  Total  : ~1 hour     (was 6-8 hours sequential)
"""

import sys, os, time, argparse, multiprocessing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows MUST have this before any other multiprocessing code
if __name__ == "__main__":
    multiprocessing.freeze_support()

import logging, warnings
warnings.filterwarnings('ignore')

log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'overnight_parallel.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(log_path, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.getLogger('mealpy').setLevel(logging.CRITICAL)
log = logging.getLogger(__name__)


def banner(text):
    log.info("═" * 65)
    log.info(f"  {text}")
    log.info("═" * 65)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick',  action='store_true', help='Fast test (5 runs, 50 epochs)')
    parser.add_argument('--track2', action='store_true', help='Track 2 only')
    parser.add_argument('--track3', action='store_true', help='Track 3 only')
    parser.add_argument('--cores',  type=int, default=None, help='Cores to use (default: all-1)')
    args = parser.parse_args()

    run_t2 = args.track2 or (not args.track2 and not args.track3)
    run_t3 = args.track3 or (not args.track2 and not args.track3)

    n_cores = args.cores if args.cores else max(1, multiprocessing.cpu_count() - 1)
    total_start = time.time()

    banner("PARALLEL OVERNIGHT RUN: TRACKS 2 & 3")
    log.info(f"  Started  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Cores    : {n_cores} of {multiprocessing.cpu_count()} available")
    log.info(f"  Quick    : {args.quick}")
    log.info(f"  Log file : {log_path}")

    # ── TRACK 2: Feature Selection ────────────────────────
    if run_t2:
        banner("TRACK 2: FEATURE SELECTION (UCI Datasets) [PARALLEL]")
        t2_start = time.time()
        try:
            import track2_feature_selection.run_feature_selection_parallel as t2

            t2.N_CORES = n_cores
            if args.quick:
                t2.N_RUNS = 5
                t2.EPOCH  = 50
                log.info("  [QUICK] 5 runs, 50 epochs")
            else:
                t2.N_RUNS = 30
                t2.EPOCH  = 200

            t2.run_feature_selection_parallel()
            log.info(f"\n✓ Track 2 done in {(time.time()-t2_start)/3600:.2f}h")

        except Exception as e:
            log.error(f"✗ Track 2 FAILED: {e}")
            import traceback
            log.error(traceback.format_exc())

        # Plots
        try:
            from plots.plot_tracks import run_all_track_plots
            run_all_track_plots()
            log.info("  ✓ Track 2 figures generated")
        except Exception as e:
            log.warning(f"  [WARN] Track 2 plots failed: {e}")

    # ── TRACK 3: WSN Localization ─────────────────────────
    if run_t3:
        banner("TRACK 3: WSN NODE LOCALIZATION [PARALLEL]")
        t3_start = time.time()
        try:
            import track3_wsn.run_wsn_localization_parallel as t3

            t3.N_CORES = n_cores
            if args.quick:
                t3.N_RUNS       = 3
                t3.EPOCHS       = 100
                t3.N_TOPOLOGIES = 2
                log.info("  [QUICK] 3 runs, 100 epochs, 2 topologies")
            else:
                t3.N_RUNS       = 30
                t3.EPOCHS       = 500
                t3.N_TOPOLOGIES = 5

            t3.run_wsn_localization_parallel()
            log.info(f"\n✓ Track 3 done in {(time.time()-t3_start)/3600:.2f}h")

        except Exception as e:
            log.error(f"✗ Track 3 FAILED: {e}")
            import traceback
            log.error(traceback.format_exc())

        # Plots
        try:
            from plots.plot_tracks import run_all_track_plots
            run_all_track_plots()
            log.info("  ✓ Track 3 figures generated")
        except Exception as e:
            log.warning(f"  [WARN] Track 3 plots failed: {e}")

    # ── Statistical Tests ─────────────────────────────────
    banner("STATISTICAL TESTS")
    try:
        from stats.statistical_tests import run_all_tests
        from utils.result_manager import RESULTS_DIR

        fs_raw = os.path.join(RESULTS_DIR, "feature_selection_raw.json")
        c23_raw = os.path.join(RESULTS_DIR, "classical23_raw_runs.json")

        if os.path.exists(fs_raw):
            run_all_tests(os.path.basename(fs_raw))
        elif os.path.exists(c23_raw):
            run_all_tests(os.path.basename(c23_raw))
        else:
            log.warning("  No raw runs file found — run Track 1 first for stats")
    except Exception as e:
        log.warning(f"  [WARN] Stats failed: {e}")

    # ── Final Summary ─────────────────────────────────────
    total = time.time() - total_start
    banner(f"ALL DONE in {total/3600:.2f} hours")
    log.info(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Log: {log_path}")
    log.info("")

    from utils.result_manager import RESULTS_DIR
    results_files = [f for f in os.listdir(RESULTS_DIR)
                     if f.endswith('.csv') or f.endswith('.json')]
    log.info(f"  Results ({len(results_files)} files in results/):")
    for f in sorted(results_files):
        size = os.path.getsize(os.path.join(RESULTS_DIR, f))
        log.info(f"    ✓ {f}  ({size//1024}KB)")

    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'plots', 'output')
    if os.path.exists(plots_dir):
        pngs = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        log.info(f"\n  Figures ({len(pngs)} files in plots/output/):")
        for f in sorted(pngs):
            log.info(f"    ✓ {f}")


if __name__ == "__main__":
    main()
