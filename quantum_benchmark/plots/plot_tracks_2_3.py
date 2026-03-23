"""
Plots for Track 2 (Feature Selection) and Track 3 (WSN)
=========================================================
Generates all figures for sections 5.2 and 5.3 of the paper.
"""

import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.result_manager import RESULTS_DIR

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    "GWO":  "#2196F3", "QGWO": "#0D47A1",
    "FA":   "#FF9800", "QFA":  "#E65100",
    "ACO":  "#4CAF50", "QACO": "#1B5E20",
}
ALGORITHMS = ["GWO", "QGWO", "FA", "QFA", "ACO", "QACO"]


# ─────────────────────────────────────────────────────────────
# TRACK 2 PLOTS
# ─────────────────────────────────────────────────────────────

def plot_accuracy_comparison(df):
    """
    Grouped bar chart: accuracy per dataset per algorithm.
    Shows classical vs quantum side by side.
    """
    datasets = df['Dataset'].unique()
    n = len(datasets)
    x = np.arange(n)
    bw = 0.13
    offsets = np.linspace(-(len(ALGORITHMS)-1)/2,
                          (len(ALGORITHMS)-1)/2, len(ALGORITHMS))

    fig, ax = plt.subplots(figsize=(max(14, n*1.2), 6))

    for i, algo in enumerate(ALGORITHMS):
        vals = []
        for ds in datasets:
            row = df[(df['Dataset']==ds) & (df['Algorithm']==algo)]
            vals.append(row['Accuracy_Mean'].values[0] if len(row) else 0)

        ax.bar(x + offsets[i]*bw, vals, bw*0.9,
               label=algo, color=COLORS[algo], alpha=0.85,
               edgecolor='white', linewidth=0.5)

    # Baseline dots
    for di, ds in enumerate(datasets):
        baseline_rows = df[df['Dataset'] == ds]
        if len(baseline_rows):
            bl = baseline_rows.iloc[0]['Baseline_Acc']
            ax.plot(di, bl, 'k^', markersize=8, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Classification Accuracy (%)", fontsize=12)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_title("Feature Selection: Accuracy Comparison\n"
                 "(▲ = KNN baseline using all features)", fontsize=13, fontweight='bold')
    ax.legend(ncol=3, fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "track2_accuracy_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {path}")


def plot_feature_reduction(df):
    """
    Bar chart: % features removed per algorithm per dataset.
    Higher reduction = better (fewer features, simpler model).
    """
    datasets = df['Dataset'].unique()
    n = len(datasets)
    x = np.arange(n)
    bw = 0.13
    offsets = np.linspace(-(len(ALGORITHMS)-1)/2,
                          (len(ALGORITHMS)-1)/2, len(ALGORITHMS))

    fig, ax = plt.subplots(figsize=(max(14, n*1.2), 6))

    for i, algo in enumerate(ALGORITHMS):
        vals = []
        for ds in datasets:
            row = df[(df['Dataset']==ds) & (df['Algorithm']==algo)]
            vals.append(row['Reduction_Pct'].values[0] if len(row) else 0)

        ax.bar(x + offsets[i]*bw, vals, bw*0.9,
               label=algo, color=COLORS[algo], alpha=0.85,
               edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Feature Reduction (%)", fontsize=12)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_title("Feature Reduction Rate per Algorithm\n"
                 "(higher = fewer features selected)", fontsize=13, fontweight='bold')
    ax.legend(ncol=3, fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='50% line')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "track2_feature_reduction.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {path}")


def plot_accuracy_vs_features(df):
    """
    Scatter: x=features selected, y=accuracy. One point per algo per dataset.
    Shows trade-off between accuracy and compactness.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for algo in ALGORITHMS:
        sub = df[df['Algorithm'] == algo]
        ax.scatter(sub['Features_Selected_Mean'],
                   sub['Accuracy_Mean'],
                   color=COLORS[algo], label=algo,
                   s=80, alpha=0.8, edgecolors='white', linewidth=0.5)

        # Connect classical-quantum pairs with light line
        if algo.startswith('Q'):
            classic = algo[1:]
            for ds in sub['Dataset'].unique():
                q_row = sub[sub['Dataset']==ds]
                c_row = df[(df['Algorithm']==classic) & (df['Dataset']==ds)]
                if len(q_row) and len(c_row):
                    ax.plot([c_row.iloc[0]['Features_Selected_Mean'],
                             q_row.iloc[0]['Features_Selected_Mean']],
                            [c_row.iloc[0]['Accuracy_Mean'],
                             q_row.iloc[0]['Accuracy_Mean']],
                            color='gray', alpha=0.3, linewidth=0.8)

    ax.set_xlabel("Mean Features Selected", fontsize=12)
    ax.set_ylabel("Classification Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs Feature Count Trade-off\n"
                 "(ideal: top-left = high accuracy, few features)",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Ideal region annotation
    ax.annotate('← Ideal region', xy=(0.1, 0.95),
                xycoords='axes fraction', fontsize=10, color='green',
                fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "track2_accuracy_vs_features.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {path}")


def plot_fs_heatmap(df):
    """
    Heatmap: datasets × algorithms, color = accuracy.
    Clean, publication-ready table visualization.
    """
    pivot = df.pivot_table(index='Dataset', columns='Algorithm',
                           values='Accuracy_Mean', aggfunc='mean')
    pivot = pivot[ALGORITHMS]   # fix column order

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot)*0.5)))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                   vmin=pivot.values.min()-5, vmax=100)

    ax.set_xticks(range(len(ALGORITHMS)))
    ax.set_xticklabels(ALGORITHMS, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    for i in range(len(pivot)):
        for j, algo in enumerate(ALGORITHMS):
            val = pivot.iloc[i, j]
            color = 'white' if val < pivot.values.mean() - 10 else 'black'
            ax.text(j, i, f"{val:.1f}%", ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label="Accuracy (%)")
    ax.set_title("Feature Selection Accuracy Heatmap\n"
                 "(green=high accuracy, red=low accuracy)",
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "track2_fs_heatmap.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────
# TRACK 3 PLOTS
# ─────────────────────────────────────────────────────────────

def plot_wsn_error_comparison(df):
    """
    Bar chart: mean localization error per algorithm.
    Lower = better.
    """
    summary = df.groupby('Algorithm')['Mean_Error_m'].agg(['mean','std']).reset_index()
    summary.columns = ['Algorithm', 'Mean', 'Std']
    summary = summary.set_index('Algorithm').loc[ALGORITHMS].reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ALGORITHMS))
    bars = ax.bar(x, summary['Mean'],
                  color=[COLORS[a] for a in ALGORITHMS],
                  alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.errorbar(x, summary['Mean'], yerr=summary['Std'],
                fmt='none', color='black', capsize=5, linewidth=1.5)

    # Value labels on bars
    for bar, val in zip(bars, summary['Mean']):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f"{val:.3f}m",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(ALGORITHMS, fontsize=12)
    ax.set_ylabel("Mean Localization Error (meters)", fontsize=12)
    ax.set_title("WSN Node Localization — Mean Error Comparison\n"
                 "(lower = better localization accuracy)",
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Highlight quantum improvements
    pairs = [("GWO","QGWO"), ("FA","QFA"), ("ACO","QACO")]
    for c, q in pairs:
        ci = ALGORITHMS.index(c); qi = ALGORITHMS.index(q)
        cv = summary.loc[summary['Algorithm']==c, 'Mean'].values[0]
        qv = summary.loc[summary['Algorithm']==q, 'Mean'].values[0]
        color = 'green' if qv < cv else 'red'
        arrow_y = max(cv, qv) + 0.3
        ax.annotate('', xy=(qi, qv+0.05), xytext=(ci, cv+0.05),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "track3_wsn_error.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {path}")


def plot_wsn_localized_pct(df):
    """
    Bar chart: % nodes successfully localized per algorithm.
    Higher = better.
    """
    summary = df.groupby('Algorithm')['Pct_Localized'].agg(['mean','std']).reset_index()
    summary.columns = ['Algorithm', 'Mean', 'Std']
    summary = summary.set_index('Algorithm').loc[ALGORITHMS].reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ALGORITHMS))
    bars = ax.bar(x, summary['Mean'],
                  color=[COLORS[a] for a in ALGORITHMS],
                  alpha=0.85, edgecolor='white')
    ax.errorbar(x, summary['Mean'], yerr=summary['Std'],
                fmt='none', color='black', capsize=5)

    for bar, val in zip(bars, summary['Mean']):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(ALGORITHMS, fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_ylabel("% Nodes Localized (error < 1m)", fontsize=12)
    ax.set_title("WSN Localization Rate per Algorithm\n"
                 "(higher = more nodes successfully located)",
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "track3_wsn_localized_pct.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {path}")


def plot_wsn_network_viz():
    """
    Visualize the WSN network layout.
    Shows anchor nodes, unknown nodes, and comm range circles.
    """
    from track3_wsn.wsn_localization import WSNNetwork, N_ANCHORS, N_UNKNOWN, COMM_RANGE
    net = WSNNetwork(seed=0)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Comm range circles for anchors (light)
    for pos in net.anchor_pos:
        circle = plt.Circle(pos, COMM_RANGE, color='blue',
                            fill=False, alpha=0.1, linewidth=0.5)
        ax.add_patch(circle)

    # Unknown nodes
    ax.scatter(net.unknown_pos[:, 0], net.unknown_pos[:, 1],
               c='orange', s=40, zorder=3, label=f'Unknown nodes ({N_UNKNOWN})',
               edgecolors='darkorange', linewidth=0.5)

    # Anchor nodes
    ax.scatter(net.anchor_pos[:, 0], net.anchor_pos[:, 1],
               c='blue', s=120, marker='^', zorder=4,
               label=f'Anchor nodes ({N_ANCHORS})',
               edgecolors='darkblue', linewidth=0.8)

    # Draw connections (lines between nodes in range)
    for i in range(N_UNKNOWN):
        for j in net.distances[i].keys():
            ax.plot([net.unknown_pos[i, 0], net.anchor_pos[j, 0]],
                    [net.unknown_pos[i, 1], net.anchor_pos[j, 1]],
                    'gray', alpha=0.15, linewidth=0.5)

    ax.set_xlim(-10, 310); ax.set_ylim(-10, 310)
    ax.set_xlabel("X coordinate (m)", fontsize=12)
    ax.set_ylabel("Y coordinate (m)", fontsize=12)
    ax.set_title(f"WSN Network Topology\n"
                 f"300×300m area | Comm range={COMM_RANGE}m",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_aspect('equal')
    ax.grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "track3_wsn_network.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run_all_track2_plots():
    print("\n── Track 2 Plots ──")
    # Try both naming conventions
    for fname in ("feature_selection_results.csv", "feature_selection_summary.csv"):
        fs_path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fs_path):
            break
    else:
        print(f"  [SKIP] No feature selection CSV found in {RESULTS_DIR}. Run Track 2 first.")
        return
    df = pd.read_csv(fs_path)
    # Normalise column names to what the plot functions expect
    rename = {
        'Acc_Mean':          'Accuracy_Mean',
        'Acc_Std':           'Accuracy_Std',
        'Acc_Best':          'Accuracy_Best',
        'Sel_Mean':          'Features_Selected_Mean',
        'Sel_Std':           'Features_Selected_Std',
        'Red_Mean':          'Reduction_Pct',
        'Reduction_Mean':    'Reduction_Pct',
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)
    # Reduction_Pct: convert 0-1 fraction to percentage if needed
    if 'Reduction_Pct' in df.columns and df['Reduction_Pct'].max() <= 1.0:
        df['Reduction_Pct'] = df['Reduction_Pct'] * 100
    plot_accuracy_comparison(df)
    plot_feature_reduction(df)
    plot_accuracy_vs_features(df)
    plot_fs_heatmap(df)
    print("  Track 2 plots done.")


def run_all_track3_plots():
    print("\n── Track 3 Plots ──")
    for fname in ("wsn_results.csv", "wsn_localization_summary.csv"):
        wsn_path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(wsn_path):
            break
    else:
        print(f"  [SKIP] No WSN CSV found in {RESULTS_DIR}. Run Track 3 first.")
        return
    df = pd.read_csv(wsn_path)
    plot_wsn_error_comparison(df)
    plot_wsn_localized_pct(df)
    try:
        plot_wsn_network_viz()
    except Exception as e:
        print(f"  [WARN] Network viz failed: {e}")
    print("  Track 3 plots done.")


if __name__ == "__main__":
    run_all_track2_plots()
    run_all_track3_plots()
