"""
Statistical Tests for Paper
==============================
Runs all required statistical tests for a publishable metaheuristic paper:

1. Wilcoxon Rank-Sum Test
   - Pairwise comparison: classical vs quantum variant
   - GWO vs QGWO, FA vs QFA, ACO vs QACO
   - p < 0.05 = statistically significant difference
   - Reports +/=/- (quantum better/similar/worse)

2. Friedman Test
   - Non-parametric ANOVA across all 6 algorithms
   - Tests if there's ANY significant difference between all algorithms
   - Reports chi-square statistic and p-value

3. Nemenyi Post-hoc Test + Critical Difference Diagram
   - Pairwise comparison after Friedman
   - Creates the CD diagram figure standard in ML papers

These are MANDATORY for top-tier metaheuristic papers.
"""

import sys, os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.result_manager import load_raw_runs, RESULTS_DIR

PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots', 'output')
os.makedirs(PLOTS_DIR, exist_ok=True)

ALGORITHMS = ["GWO", "QGWO", "FA", "QFA", "ACO", "QACO", "AQHSO"]
ALPHA = 0.05  # significance level


# ─────────────────────────────────────────────────────────────
# 1. WILCOXON RANK-SUM TEST
# ─────────────────────────────────────────────────────────────

def wilcoxon_classical_vs_quantum(raw_data):
    """
    Pairwise Wilcoxon test: classical vs quantum for each function.
    Pairs: (GWO, QGWO), (FA, QFA), (ACO, QACO)
    """
    pairs = [("GWO", "QGWO"), ("FA", "QFA"), ("ACO", "QACO")]
    results = []

    for fname, algo_data in raw_data.items():
        for classic, quantum in pairs:
            try:
                def _extract(runs):
                    """Handle both plain floats and dicts like {fitness, n_selected, ...}"""
                    if runs and isinstance(runs[0], dict):
                        return [r.get('fitness', float('nan')) for r in runs]
                    return runs

                c_runs = np.array(_extract(algo_data[classic]), dtype=float)
                q_runs = np.array(_extract(algo_data[quantum]), dtype=float)

                # Remove NaN
                mask = ~(np.isnan(c_runs) | np.isnan(q_runs))
                c_runs, q_runs = c_runs[mask], q_runs[mask]

                if len(c_runs) < 5:
                    continue

                stat, p_val = stats.wilcoxon(c_runs, q_runs,
                                              alternative='two-sided',
                                              zero_method='wilcox')

                # Determine winner
                c_mean = np.mean(c_runs)
                q_mean = np.mean(q_runs)

                if p_val < ALPHA:
                    if q_mean < c_mean:
                        outcome = "+"   # quantum significantly better
                    else:
                        outcome = "-"   # quantum significantly worse
                else:
                    outcome = "="       # no significant difference

                results.append({
                    'Function': fname,
                    'Classical': classic,
                    'Quantum': quantum,
                    'Classical_Mean': c_mean,
                    'Quantum_Mean': q_mean,
                    'W_Statistic': stat,
                    'p_value': p_val,
                    'Outcome': outcome,  # +/=/- 
                })
            except Exception as e:
                pass

    df = pd.DataFrame(results)

    # Summary table
    print("\n── Wilcoxon Test Summary (+ = quantum better, = = tie, - = classical better) ──")
    if df.empty:
        print("  [WARNING] No Wilcoxon results — check algorithm names match data keys.")
        return df
    pivot = df.pivot_table(index='Function',
                           columns='Classical',
                           values='Outcome',
                           aggfunc='first')
    print(pivot.to_string())

    # Count wins
    print("\n── Win/Tie/Loss Summary ──")
    for classic, quantum in pairs:
        subset = df[df['Classical'] == classic]
        wins   = (subset['Outcome'] == '+').sum()
        ties   = (subset['Outcome'] == '=').sum()
        losses = (subset['Outcome'] == '-').sum()
        print(f"  {quantum} vs {classic}: {wins}W / {ties}T / {losses}L")

    # Save
    path = os.path.join(RESULTS_DIR, "wilcoxon_results.csv")
    df.to_csv(path, index=False, float_format='%.6e')
    print(f"\n  ✓ Wilcoxon results saved: {path}")
    return df


# ─────────────────────────────────────────────────────────────
# 2. FRIEDMAN TEST
# ─────────────────────────────────────────────────────────────

def friedman_test(raw_data):
    """
    Friedman test across all 6 algorithms.
    Input: raw_data[func][algo] = [run1, run2, ...]
    Returns: chi2 statistic, p-value, per-algorithm average ranks
    """
    funcs = list(raw_data.keys())
    algos = ALGORITHMS

    # Build matrix: rows=functions, cols=algorithms, values=mean fitness
    matrix = []
    valid_funcs = []
    for fname in funcs:
        row = []
        valid = True
        for algo in algos:
            if algo not in raw_data[fname]:
                valid = False
                break
            raw = raw_data[fname][algo]
            if raw and isinstance(raw[0], dict):
                raw = [r.get('fitness', float('nan')) for r in raw]
            vals = np.array(raw, dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                valid = False
                break
            row.append(np.mean(vals))
        if valid:
            matrix.append(row)
            valid_funcs.append(fname)

    if len(matrix) < 3:
        print("[WARNING] Not enough functions for Friedman test")
        return None, None, None

    matrix = np.array(matrix)

    # Rank within each row (function)
    ranks = np.zeros_like(matrix)
    for i in range(len(matrix)):
        row = matrix[i]
        # Rank 1 = best (lowest for minimization)
        sorted_idx = np.argsort(row)
        for rank, idx in enumerate(sorted_idx):
            ranks[i, idx] = rank + 1

    avg_ranks = np.mean(ranks, axis=0)

    # Friedman statistic
    n = len(matrix)    # number of functions
    k = len(algos)     # number of algorithms

    chi2 = (12 * n) / (k * (k + 1)) * (
        np.sum(avg_ranks ** 2) - (k * (k + 1) ** 2) / 4
    )
    p_val = 1 - stats.chi2.cdf(chi2, df=k-1)

    print("\n── Friedman Test ──")
    print(f"  Functions used: {n}")
    print(f"  Chi² statistic: {chi2:.4f}")
    print(f"  p-value: {p_val:.6f} ({'SIGNIFICANT' if p_val < ALPHA else 'not significant'})")
    print("\n  Average Ranks (lower = better):")
    for algo, rank in zip(algos, avg_ranks):
        print(f"    {algo:6s}: {rank:.4f}")

    # Save ranks
    ranks_df = pd.DataFrame({
        'Algorithm': algos,
        'Average_Rank': avg_ranks,
        'Chi2': [chi2] * len(algos),
        'p_value': [p_val] * len(algos),
    })
    path = os.path.join(RESULTS_DIR, "friedman_ranks.csv")
    ranks_df.to_csv(path, index=False)
    print(f"  ✓ Friedman ranks saved: {path}")

    return chi2, p_val, dict(zip(algos, avg_ranks))


# ─────────────────────────────────────────────────────────────
# 3. NEMENYI POST-HOC TEST
# ─────────────────────────────────────────────────────────────

def nemenyi_critical_difference(avg_ranks_dict, n_functions):
    """
    Compute Nemenyi critical difference (CD) for pairwise comparison.
    CD = q_alpha × sqrt(k(k+1) / 6n)
    where q_alpha comes from the Studentized range distribution.

    If |rank_i - rank_j| > CD → algorithms i and j are significantly different.
    """
    k = len(avg_ranks_dict)
    n = n_functions

    # Critical q values for alpha=0.05, k=6 (from Demsar 2006 Table 5)
    q_alpha_table = {
        2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
    }
    q_alpha = q_alpha_table.get(k, 2.850)

    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

    print(f"\n── Nemenyi Critical Difference ──")
    print(f"  k={k}, n={n}, q_α={q_alpha}, CD = {cd:.4f}")
    print(f"\n  Significantly different pairs (|rank diff| > {cd:.4f}):")

    algos = list(avg_ranks_dict.keys())
    ranks = list(avg_ranks_dict.values())
    sig_pairs = []

    for i in range(len(algos)):
        for j in range(i+1, len(algos)):
            diff = abs(ranks[i] - ranks[j])
            if diff > cd:
                sig_pairs.append((algos[i], algos[j], diff))
                print(f"    {algos[i]:6s} vs {algos[j]:6s}: diff={diff:.4f} ✓")

    if not sig_pairs:
        print("    None (no significant pairwise differences)")

    return cd, sig_pairs


# ─────────────────────────────────────────────────────────────
# 4. CRITICAL DIFFERENCE DIAGRAM (Figure for paper)
# ─────────────────────────────────────────────────────────────

def plot_cd_diagram(avg_ranks_dict, cd, n_functions, title="CD Diagram"):
    """
    Creates the Critical Difference diagram.
    Standard figure in ML comparison papers (Demsar 2006 style).
    """
    algos = sorted(avg_ranks_dict.keys(), key=lambda a: avg_ranks_dict[a])
    ranks = [avg_ranks_dict[a] for a in algos]

    k = len(algos)
    fig, ax = plt.subplots(figsize=(10, 3))

    # Main axis line
    min_rank = max(1, min(ranks) - 0.5)
    max_rank = min(k, max(ranks) + 0.5)
    ax.plot([min_rank, max_rank], [0, 0], 'k-', linewidth=2)

    # Algorithm ticks
    for i, (algo, rank) in enumerate(zip(algos, ranks)):
        # Alternate label above/below for readability
        if i % 2 == 0:
            ax.plot([rank, rank], [0, 0.1], 'k-', linewidth=1.5)
            ax.text(rank, 0.18, algo, ha='center', va='bottom',
                    fontsize=11, fontweight='bold', color='darkblue')
            ax.text(rank, 0.12, f"({rank:.2f})", ha='center', va='bottom',
                    fontsize=8, color='gray')
        else:
            ax.plot([rank, rank], [0, -0.1], 'k-', linewidth=1.5)
            ax.text(rank, -0.18, algo, ha='center', va='top',
                    fontsize=11, fontweight='bold', color='darkred')
            ax.text(rank, -0.12, f"({rank:.2f})", ha='center', va='top',
                    fontsize=8, color='gray')

    # CD bar
    best_rank = ranks[0]
    ax.annotate('', xy=(best_rank + cd, 0.35), xytext=(best_rank, 0.35),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(best_rank + cd/2, 0.40, f'CD={cd:.3f}',
            ha='center', va='bottom', color='red', fontsize=10)

    # Draw cliques (non-significantly different groups) as horizontal bars
    for i in range(len(algos)):
        clique_end = ranks[i]
        for j in range(i+1, len(algos)):
            if abs(ranks[j] - ranks[i]) <= cd:
                clique_end = ranks[j]
        if clique_end > ranks[i]:
            ax.plot([ranks[i], clique_end], [0.05, 0.05],
                    color='green', linewidth=4, alpha=0.4)

    ax.set_xlim(min_rank - 0.3, max_rank + 0.3)
    ax.set_ylim(-0.5, 0.7)
    ax.set_xlabel("Average Rank (lower = better)", fontsize=12)
    ax.set_title(f"{title}\n(n={n_functions} functions, α=0.05)", fontsize=12)
    ax.axis('off')

    # Add x-axis manually
    ax.annotate('', xy=(max_rank + 0.2, 0), xytext=(min_rank - 0.2, 0),
                arrowprops=dict(arrowstyle='->', color='black'))
    for r in range(int(min_rank), int(max_rank)+1):
        ax.text(r, -0.07, str(r), ha='center', va='top', fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "cd_diagram.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ CD diagram saved: {path}")


# ─────────────────────────────────────────────────────────────
# 5. SIGNIFICANCE TABLE (for paper body)
# ─────────────────────────────────────────────────────────────

def build_significance_table(wilcoxon_df):
    """
    Creates the standard +/=/- significance table.
    Row: function, Col: pair comparison
    Used directly in the paper.
    """
    pairs = [("GWO", "QGWO"), ("FA", "QFA"), ("ACO", "QACO")]
    funcs = wilcoxon_df['Function'].unique()

    rows = []
    for fname in funcs:
        row = {'Function': fname}
        for classic, quantum in pairs:
            subset = wilcoxon_df[
                (wilcoxon_df['Function'] == fname) &
                (wilcoxon_df['Classical'] == classic)
            ]
            if len(subset) > 0:
                row[f"{quantum}/{classic}"] = subset.iloc[0]['Outcome']
            else:
                row[f"{quantum}/{classic}"] = "N/A"
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "significance_table.csv")
    df.to_csv(path, index=False)
    print(f"\n  ✓ Significance table saved: {path}")
    print("\nSignificance Table (+ = quantum better, = = tie, - = worse):")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run_all_tests(raw_data_file="classical23_raw_runs.json"):
    print("=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)

    try:
        raw_data = load_raw_runs(raw_data_file)
    except FileNotFoundError:
        print(f"[ERROR] Run experiments first. File not found: {raw_data_file}")
        return

    n_functions = len(raw_data)

    # 1. Wilcoxon
    wilcoxon_df = wilcoxon_classical_vs_quantum(raw_data)

    # 2. Friedman
    chi2, p_val, avg_ranks = friedman_test(raw_data)

    if avg_ranks:
        # 3. Nemenyi
        cd, sig_pairs = nemenyi_critical_difference(avg_ranks, n_functions)

        # 4. CD Diagram
        plot_cd_diagram(avg_ranks, cd, n_functions,
                        title="Critical Difference Diagram — All 6 Algorithms")

    # 5. Significance table
    if wilcoxon_df is not None and len(wilcoxon_df) > 0:
        build_significance_table(wilcoxon_df)

    print("\n✓ All statistical tests complete!")


if __name__ == "__main__":
    # Optionally pass CEC2017 results too
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='classical23_raw_runs.json',
                        help='Raw runs JSON file to analyze')
    args = parser.parse_args()
    run_all_tests(args.file)
