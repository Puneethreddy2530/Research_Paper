"""
Result Manager
==============
Handles saving/loading experiment results as CSV and JSON.
"""

import os, json
import numpy as np
import pandas as pd


RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_run_results(results_dict, filename):
    """
    Save results dict to CSV.
    results_dict format:
      {
        'F1': {'GWO': [run1, run2, ...], 'QGWO': [...], ...},
        'F2': {...},
        ...
      }
    Output CSV columns:
      Function | Algorithm | Mean | Std | Best | Worst | Median
    """
    rows = []
    for func_name, algo_data in results_dict.items():
        for algo_name, runs in algo_data.items():
            arr = np.array(runs)
            rows.append({
                'Function': func_name,
                'Algorithm': algo_name,
                'Mean':   np.mean(arr),
                'Std':    np.std(arr),
                'Best':   np.min(arr),
                'Worst':  np.max(arr),
                'Median': np.median(arr),
                'Runs':   len(arr),
            })

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(path, index=False, float_format='%.6e')
    print(f"  ✓ Saved: {path}")
    return df


def save_raw_runs(all_runs, filename):
    """Save raw per-run data for statistical tests."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'w') as f:
        json.dump(all_runs, f, indent=2)
    print(f"  ✓ Raw runs saved: {path}")


def load_raw_runs(filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path) as f:
        return json.load(f)


def load_results(filename):
    path = os.path.join(RESULTS_DIR, filename)
    return pd.read_csv(path)


def print_summary_table(df, functions=None):
    """Pretty-print a results table to console."""
    if functions:
        df = df[df['Function'].isin(functions)]

    algos = df['Algorithm'].unique()
    funcs = df['Function'].unique()

    header = f"{'Function':<8}" + "".join(f"{'Mean_'+a:<20}" for a in algos)
    print(header)
    print("-" * len(header))

    for func in funcs:
        row = f"{func:<8}"
        for algo in algos:
            val = df[(df['Function']==func) & (df['Algorithm']==algo)]['Mean'].values
            if len(val) > 0:
                row += f"{val[0]:<20.4e}"
            else:
                row += f"{'N/A':<20}"
        print(row)
