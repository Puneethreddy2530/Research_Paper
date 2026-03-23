"""
Plot Results — All Figures for the Paper
==========================================
Generates all publication-ready figures:

Figure 1: Bar chart — Mean fitness comparison (F1–F13)
Figure 2: Box plots — Distribution across 30 runs (selected functions)
Figure 3: Convergence curves — Best fitness per epoch
Figure 4: Heatmap — Algorithm rankings across all functions
Figure 5: Radar chart — Multi-metric comparison
"""

import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.result_manager import load_results, load_raw_runs, RESULTS_DIR

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Color scheme (colorblind-friendly) ──────────────────────
COLORS = {
    "GWO":  "#2196F3",   # Blue
    "QGWO": "#0D47A1",   # Dark Blue
    "FA":   "#FF9800",   # Orange
    "QFA":  "#E65100",   # Dark Orange
    "ACO":  "#4CAF50",   # Green
    "QACO": "#1B5E20",   # Dark Green
    "AQHSO": "#9C27B0",  # Purple
}

MARKERS = {
    "GWO":  "o", "QGWO": "s",
    "FA":   "^", "QFA":  "v",
    "ACO":  "D", "QACO": "P",
    "AQHSO": "X",
}

LINESTYLES = {
    "GWO":  "--", "QGWO": "-",
    "FA":   "--", "QFA":  "-",
    "ACO":  "--", "QACO": "-",
    "AQHSO": "-.",
}

ALGORITHMS = ["GWO", "QGWO", "FA", "QFA", "ACO", "QACO", "AQHSO"]


def fig1_mean_fitness_barchart(df, func_range=("F1", "F13"),
                                title="Mean Fitness — Unimodal & Multimodal Functions"):
    """
    Grouped bar chart: X=functions, groups=algorithms, Y=log(mean fitness)
    """
    funcs = df['Function'].unique()
    # Filter to range
    f_start = int(func_range[0][1:])
    f_end   = int(func_range[1][1:])
    funcs = [f for f in funcs if f.startswith('F') and
             f_start <= int(f[1:]) <= f_end]

    if len(funcs) == 0:
        print(f"[SKIP] No functions in range {func_range}")
        return

    x = np.arange(len(funcs))
    bar_width = 0.13
    offsets = np.linspace(-(len(ALGORITHMS)-1)/2, (len(ALGORITHMS)-1)/2, len(ALGORITHMS))

    # Precompute offsets per function to make values positive
    func_offsets = {}
    for fname in funcs:
        fmin = df[df['Function'] == fname]['Mean'].min()
        func_offsets[fname] = fmin - 1e-10 if fmin <= 0 else 0

    fig, ax = plt.subplots(figsize=(16, 6))

    for i, algo in enumerate(ALGORITHMS):
        vals = []
        for fname in funcs:
            subset = df[(df['Function'] == fname) & (df['Algorithm'] == algo)]
            if len(subset) > 0:
                v = subset.iloc[0]['Mean']
                vals.append(v - func_offsets[fname])
            else:
                vals.append(1e-10)

        # Log scale values for display
        log_vals = np.log10(np.array(vals))
        log_vals = np.clip(log_vals, -300, 300)

        ax.bar(x + offsets[i] * bar_width,
               log_vals, bar_width * 0.9,
               label=algo, color=COLORS[algo], alpha=0.85,
               edgecolor='white', linewidth=0.5)

    ax.set_xlabel("Benchmark Function", fontsize=13)
    ax.set_ylabel("log₁₀(Mean Fitness)", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(funcs, rotation=0, fontsize=10)
    ax.legend(ncol=3, fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    fname_out = f"fig1_barchart_{func_range[0]}_{func_range[1]}.png"
    path = os.path.join(OUTPUT_DIR, fname_out)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 1 saved: {path}")


def fig2_boxplots(raw_data, selected_funcs=["F1", "F9", "F10", "F23"]):
    """
    Box plots for selected functions showing distribution of 30 runs.
    """
    n_funcs = len(selected_funcs)
    fig, axes = plt.subplots(1, n_funcs, figsize=(5 * n_funcs, 5))
    if n_funcs == 1:
        axes = [axes]

    for ax, fname in zip(axes, selected_funcs):
        if fname not in raw_data:
            continue

        data_per_algo = []
        labels = []
        for algo in ALGORITHMS:
            if algo in raw_data[fname]:
                vals = np.array(raw_data[fname][algo], dtype=float)
                vals = vals[~np.isnan(vals)]
                data_per_algo.append(vals)
                labels.append(algo)

        if not data_per_algo:
            continue

        bp = ax.boxplot(data_per_algo, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))

        for patch, algo in zip(bp['boxes'], labels):
            patch.set_facecolor(COLORS.get(algo, 'gray'))
            patch.set_alpha(0.7)

        ax.set_xticklabels(labels, rotation=45, fontsize=9)
        ax.set_title(fname, fontsize=12, fontweight='bold')
        ax.set_ylabel("Fitness Value", fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # Log scale if values span many orders of magnitude
        try:
            all_vals = np.concatenate(data_per_algo)
            all_vals = all_vals[all_vals > 0]
            if len(all_vals) > 0 and np.max(all_vals) / (np.min(all_vals) + 1e-300) > 1000:
                ax.set_yscale('log')
        except:
            pass

    plt.suptitle("Distribution of Fitness Values (30 Runs)", fontsize=14,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig2_boxplots.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 2 saved: {path}")


def fig3_convergence(convergence_file="convergence_data.json",
                     selected_funcs=["F1", "F9", "F10", "F23"]):
    """
    Convergence curves: epoch vs best fitness (averaged over runs).
    """
    conv_path = os.path.join(RESULTS_DIR, convergence_file)
    if not os.path.exists(conv_path):
        print(f"  [SKIP] {convergence_file} not found. Run run_convergence.py first.")
        return

    with open(conv_path) as f:
        conv_data = json.load(f)

    n_funcs = len(selected_funcs)
    cols = min(2, n_funcs)
    rows = (n_funcs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    axes = np.array(axes).flatten() if n_funcs > 1 else [axes]

    for ax, fname in zip(axes, selected_funcs):
        if fname not in conv_data:
            ax.set_visible(False)
            continue

        for algo in ALGORITHMS:
            if algo not in conv_data[fname]:
                continue

            all_runs = conv_data[fname][algo]
            # Pad / trim to same length
            valid_runs = [r for r in all_runs if len(r) > 0]
            if not valid_runs:
                continue

            min_len = min(len(r) for r in valid_runs)
            trimmed = np.array([r[:min_len] for r in valid_runs])
            mean_curve = np.mean(trimmed, axis=0)

            epochs = np.arange(1, len(mean_curve) + 1)
            ax.plot(epochs, mean_curve,
                       label=algo,
                       color=COLORS.get(algo, 'gray'),
                       linestyle=LINESTYLES.get(algo, '-'),
                       marker=MARKERS.get(algo, 'o'),
                       markevery=max(1, len(epochs)//10),
                       linewidth=2, markersize=5)

        ax.set_xlabel("Iteration (Epoch)", fontsize=11)
        
        # Apply log scale only if strictly positive
        ymin, ymax = ax.get_ylim()
        if ymin > 0 and ymax / (ymin + 1e-10) > 100:
            ax.set_yscale('log')
        ax.set_ylabel("Best Fitness (log scale)", fontsize=11)
        ax.set_title(f"Convergence — {fname}", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for ax in axes[n_funcs:]:
        ax.set_visible(False)

    plt.suptitle("Convergence Curves: Classical vs Quantum Algorithms",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig3_convergence.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 3 saved: {path}")


def fig4_ranking_heatmap(df):
    """
    Heatmap: functions × algorithms, values = rank (1=best, 6=worst).
    Color: green=good rank, red=bad rank.
    """
    funcs = sorted(df['Function'].unique(),
                   key=lambda f: int(f[1:]) if f[1:].isdigit() else 999)

    rank_matrix = np.zeros((len(funcs), len(ALGORITHMS)))

    for i, fname in enumerate(funcs):
        func_data = df[df['Function'] == fname]
        means = []
        for algo in ALGORITHMS:
            subset = func_data[func_data['Algorithm'] == algo]
            means.append(subset.iloc[0]['Mean'] if len(subset) > 0 else np.inf)

        means = np.array(means)
        # Rank 1 = best (lowest mean)
        sorted_idx = np.argsort(means)
        for rank, idx in enumerate(sorted_idx):
            rank_matrix[i, idx] = rank + 1

    fig, ax = plt.subplots(figsize=(10, max(8, len(funcs) * 0.4)))
    im = ax.imshow(rank_matrix, cmap='RdYlGn_r', aspect='auto',
                   vmin=1, vmax=len(ALGORITHMS))

    # Labels
    ax.set_xticks(range(len(ALGORITHMS)))
    ax.set_xticklabels(ALGORITHMS, fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(funcs)))
    ax.set_yticklabels(funcs, fontsize=9)

    # Rank numbers in cells
    for i in range(len(funcs)):
        for j in range(len(ALGORITHMS)):
            rank = int(rank_matrix[i, j])
            color = 'white' if rank <= 2 or rank >= 5 else 'black'
            ax.text(j, i, str(rank), ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label="Rank (1=best)")
    ax.set_title("Algorithm Rankings per Function\n(1=best, green=good)",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_ylabel("Benchmark Function", fontsize=12)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig4_ranking_heatmap.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 4 saved: {path}")


def fig5_quantum_improvement(df):
    """
    Side-by-side comparison: how much does quantum improve each classical?
    Bar chart: % improvement or degradation.
    """
    pairs = [("GWO", "QGWO"), ("FA", "QFA"), ("ACO", "QACO")]
    pair_colors = [("#2196F3", "#0D47A1"), ("#FF9800", "#E65100"), ("#4CAF50", "#1B5E20")]

    funcs = sorted(df['Function'].unique(),
                   key=lambda f: int(f[1:]) if f[1:].isdigit() else 999)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (classic, quantum), (c1, c2) in zip(axes, pairs, pair_colors):
        improvements = []
        func_labels = []

        for fname in funcs:
            c_val = df[(df['Function']==fname) & (df['Algorithm']==classic)]['Mean'].values
            q_val = df[(df['Function']==fname) & (df['Algorithm']==quantum)]['Mean'].values

            if len(c_val) > 0 and len(q_val) > 0:
                c, q = c_val[0], q_val[0]
                fmin = df[df['Function']==fname]['Mean'].min()
                offset = fmin - 1e-10 if fmin <= 0 else 0
                c_shifted, q_shifted = c - offset, q - offset
                
                # Positive = quantum better (lower is better, so C > Q -> C_shift > Q_shift -> log10 diff > 0)
                pct = np.log10(c_shifted) - np.log10(q_shifted)
                improvements.append(pct)
                func_labels.append(fname)

        x = np.arange(len(func_labels))
        bar_colors = [c2 if v > 0 else '#F44336' for v in improvements]
        ax.bar(x, improvements, color=bar_colors, alpha=0.8, edgecolor='white')
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(func_labels, rotation=90, fontsize=8)
        ax.set_title(f"{quantum} vs {classic}\n(+ve = quantum better)", fontsize=11)
        ax.set_ylabel("log₁₀(improvement)", fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        wins = sum(1 for v in improvements if v > 0)
        ax.text(0.02, 0.98, f"Wins: {wins}/{len(improvements)}",
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle("Quantum Enhancement Effect per Function",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig5_quantum_improvement.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 5 saved: {path}")


def run_all_plots(results_file="classical23_results.csv",
                  raw_file="classical23_raw_runs.json"):
    print("=" * 50)
    print("GENERATING ALL PAPER FIGURES")
    print("=" * 50)

    # Load data
    try:
        df = load_results(results_file)
    except FileNotFoundError:
        print(f"[ERROR] {results_file} not found. Run experiments first.")
        return

    print(f"  Loaded {len(df)} rows from {results_file}")

    # All figures
    print("\nFigure 1: Bar charts")
    fig1_mean_fitness_barchart(df, ("F1", "F7"),  "Mean Fitness — Unimodal Functions (F1–F7)")
    fig1_mean_fitness_barchart(df, ("F8", "F13"), "Mean Fitness — Multimodal Functions (F8–F13)")

    print("Figure 2: Box plots")
    try:
        raw_data = load_raw_runs(raw_file)
        fig2_boxplots(raw_data)
    except FileNotFoundError:
        print(f"  [SKIP] {raw_file} not found")

    print("Figure 3: Convergence curves")
    fig3_convergence()

    print("Figure 4: Ranking heatmap")
    fig4_ranking_heatmap(df)

    print("Figure 5: Quantum improvement analysis")
    fig5_quantum_improvement(df)

    print(f"\n✓ All figures saved to: {OUTPUT_DIR}")
    print("  Insert these PNG files directly into your paper (300 DPI).")


if __name__ == "__main__":
    run_all_plots()
