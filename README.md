# Quantum-Inspired Metaheuristic Optimization — Research Code

> **Paper:** *Quantum-Enhanced Grey Wolf Optimizer, Firefly Algorithm, and Ant Colony Optimization: A Comparative Benchmarking Study with Feature Selection and WSN Localization Applications*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-green.svg)]()

---

## 📋 Overview

This repository contains the **complete implementation** for our comparative study of three classical metaheuristic algorithms (GWO, FA, ACO) versus their quantum-enhanced counterparts (QGWO, QFA, QACO) across three experimental tracks:

| Track | Scope | Benchmark |
|-------|-------|-----------|
| **Track 1** | Optimization benchmarking | 23 classical functions + CEC 2017 (29 functions, D=10/30) |
| **Track 2** | Feature selection (ML) | 12 UCI datasets with KNN wrapper |
| **Track 3** | WSN localization | 300×300 m network, 100 nodes, 20 anchors |

The quantum enhancement uses **qubit rotation gates** and **quantum tunneling** — no quantum hardware required. All algorithms run on standard CPU.

---

## 📁 Project Structure

```
quantum_benchmark/
├── algorithms/                         # 6 algorithms (3 classical + 3 quantum)
│   ├── quantum_gwo.py                  # QGWO — qubit rotation gate, tunneling
│   ├── quantum_fa.py                   # QFA  — quantum attractiveness in θ-space
│   └── quantum_aco.py                  # QACO — qubit pheromone + Hadamard mutation
│
├── benchmarks/                         # Benchmark function definitions
│   ├── classical_23.py                 # F1–F23 (unimodal, multimodal, fixed-dim)
│   └── cec2017_loader.py               # CEC 2017 via opfunu
│
├── experiments/                        # Track 1 experiment runners
│   ├── run_classical23.py              # 6 algos × 23 funcs × 30 runs
│   ├── run_cec2017.py                  # 6 algos × 29 CEC funcs × 30 runs, D=10/30
│   └── run_convergence.py              # Epoch-by-epoch best fitness (convergence curves)
│
├── track2_feature_selection/           # Track 2 — ML feature selection
│   ├── dataset_loader.py               # Loads 12 UCI datasets (with fallbacks)
│   ├── fitness_function.py             # KNN wrapper fitness (10-fold CV)
│   └── run_feature_selection.py        # 6 algos × 12 datasets × 30 runs
│
├── track3_wsn/                         # Track 3 — WSN localization
│   └── wsn_localization.py             # 300×300m, 100 nodes, 160-dim optimization
│
├── stats/
│   └── statistical_tests.py            # Wilcoxon + Friedman + Nemenyi + CD diagram
│
├── plots/
│   ├── plot_results.py                 # Figures 1–6 (Track 1)
│   └── plot_tracks_2_3.py              # Figures 7–13 (Tracks 2 & 3)
│
├── utils/
│   └── result_manager.py               # Save/load CSV and JSON results
│
├── main.py                             # Run all of Track 1 in one command
└── run_overnight.py                    # Run Tracks 2 & 3 overnight
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install mealpy opfunu scipy scikit-learn matplotlib seaborn numpy pandas
```

### 2. Run everything (Track 1 — ~3–6 hours)
```bash
cd quantum_benchmark
python main.py
```

### 3. Run Tracks 2 & 3 (overnight — ~8–10 hours)
```bash
python run_overnight.py
```

### 4. Quick test (5 runs, 50 epochs, 3 datasets only)
```bash
python run_overnight.py --quick
```

---

## 🔬 Algorithm Design

### Classical counterparts (via mealpy)
`GWO` (OriginalGWO), `FA` (OriginalFA), `ACO-R` (OriginalACOR) — used as-is from the mealpy library.

### Quantum-Inspired variants

All three quantum algorithms share the same core design principle:

| Component | Classical | Quantum Analog |
|-----------|-----------|----------------|
| Position | `x ∈ [lb, ub]` | Qubit angle `θ ∈ [0, π/2]` |
| Position recovery | Direct | `x = cos²(θ) × (ub - lb) + lb` |
| Position update | Arithmetic (e.g., `x += step`) | Rotation gate `θ += Δθ` |
| Diversity | Random walk / evaporation | Quantum tunneling (QGWO/QFA) or Hadamard gate (QACO) |

**QGWO** — Each wolf stores θ angles. Movement guided by alpha/beta/delta wolves via lookup-table rotation signs.

**QFA** — Attractiveness computed in θ-space. `β = β₀ × exp(−γ × |θᵢ − θⱼ|²)`.

**QACO** — Pheromone strength = `cos²(θ)`. Update via rotation gate with tanh-scaled magnitude. Evaporation replaced by Hadamard gate mutation (2% probability).

---

## ⚙️ Parameter Settings (from original papers — do not change)

| Algorithm | Population | Epochs | Key Parameters |
|-----------|-----------|--------|----------------|
| GWO       | 30 | 500 | `a` decreases linearly 2 → 0 |
| QGWO      | 30 | 500 | `Δθ_max = 0.05π`, `p_tunnel = 0.01` |
| FA        | 30 | 500 | `α=0.5`, `β₀=1.0`, `γ=1.0` |
| QFA       | 30 | 500 | `α=0.5`, `β₀=1.0`, `Δθ_max = 0.05π` |
| ACO-R     | 30 | 500 | `sample_count=50`, `intent_factor=0.5` |
| QACO      | 30 | 500 | `Δθ=0.01π`, `p_hadamard=0.02` |

---

## 📊 Track 1 — Benchmark Functions

### Unimodal (F1–F7): Single global minimum — tests exploitation precision
| Fn | Name | Dimension | Known Optimum |
|----|------|-----------|---------------|
| F1 | Sphere | 30 | 0 |
| F2 | Schwefel 2.22 | 30 | 0 |
| F5 | Rosenbrock | 30 | 0 |
| F6 | Step | 30 | 0 |

### Multimodal High-Dim (F8–F13): Many local minima — tests exploration
| Fn | Name | Why Famous |
|----|------|------------|
| F8 | Schwefel 2.26 | Global min far from origin — deceptive |
| F9 | Rastrigin | Regular grid of traps |
| F10 | Ackley | Most widely used benchmark |

### Fixed-Dim Multimodal (F14–F23): 2D–6D complex surfaces
Shekel's Foxholes, Kowalik, Six-Hump Camel, Branin, Goldstein-Price, Hartmann-3/6, Shekel variants.

---

## 🤖 Track 2 — Feature Selection

**Datasets (12 UCI):** Iris, Wine, Breast Cancer (WDBC), Pima Diabetes, Glass, Heart (Statlog), Zoo, Vehicle, Vowel, Ionosphere, Sonar, Lymphography

**Fitness function** (El-ashry et al. 2020):
```
fitness = 0.99 × (1 - KNN_accuracy) + 0.01 × (selected_features / total_features)
```
- Solution space: `[0, 1]^n_features` (thresholded at 0.5)
- KNN: k=5, 10-fold cross-validation

---

## 📡 Track 3 — WSN Localization

**Setup** (matching Rajakumar 2017 GWO-LPWSN exactly):

| Parameter | Value |
|-----------|-------|
| Area | 300 × 300 m |
| Total nodes | 100 |
| Anchor nodes | 20 (known position) |
| Unknown nodes | 80 (to be localized) |
| Communication range | 40 m |
| Distance noise | Gaussian σ = 0.02 |

**Optimization dimension:** 160 (x,y for each of 80 unknown nodes)

**Fitness:** Mean RMSE between estimated and anchored distances across all unknown nodes.

---

## 📈 Output Files

All results are auto-saved to `results/` and `plots/output/` as experiments complete:

| File | Paper Section |
|------|--------------|
| `classical23_results.csv` | Tables 1 & 2 |
| `cec2017_results_D10.csv` | Table 3 (D=10) |
| `cec2017_results_D30.csv` | Table 3 (D=30) |
| `wilcoxon_results.csv` | Table 4 |
| `significance_table.csv` | Table 5 (+/=/−) |
| `friedman_ranks.csv` | Table 6 |
| `feature_selection_results.csv` | Tables 7 & 8 |
| `wsn_results.csv` | Table 9 |
| `fig1_barchart_*.png` | Figure 1 |
| `fig2_boxplots.png` | Figure 2 |
| `fig3_convergence.png` | Figure 3 |
| `fig4_ranking_heatmap.png` | Figure 4 |
| `fig5_quantum_improvement.png` | Figure 5 |
| `cd_diagram.png` | Figure 6 |
| `track2_*.png` (4 figures) | Figures 7–10 |
| `track3_*.png` (3 figures) | Figures 11–13 |

---

## 📦 Running Individual Components

```bash
# Track 1 only (classical 23 benchmarks)
python experiments/run_classical23.py          # ~10 min

# Track 1 — CEC 2017
python experiments/run_cec2017.py             # ~2–3 hrs

# Track 1 — Convergence curves
python experiments/run_convergence.py         # ~20 min

# Statistical tests (after Track 1 done)
python stats/statistical_tests.py

# Generate all Track 1 figures
python plots/plot_results.py

# Track 2 only
python track2_feature_selection/run_feature_selection.py

# Track 3 only
python track3_wsn/wsn_localization.py

# Track 2 + 3 overnight (with plots + stats)
python run_overnight.py

# Track 2 + 3 quick test
python run_overnight.py --quick
```

---

## 📚 Dependencies

```
mealpy>=3.0.1        # Classical algorithm baselines (GWO, FA, ACO-R)
opfunu>=1.0.0        # CEC 2017 benchmark functions
scipy>=1.11.0        # Wilcoxon, Friedman statistical tests
scikit-learn>=1.3.0  # KNN classifier for feature selection
matplotlib>=3.7.0    # All figures
seaborn>=0.12.0      # Heatmaps
numpy>=1.24.0
pandas>=2.0.0
```

---

## 📄 Citation

If you use this code, please cite:

```bibtex
@article{rajakumar2026quantum,
  title   = {Quantum-Enhanced Grey Wolf Optimizer, Firefly Algorithm, and Ant Colony 
             Optimization: A Comparative Benchmarking Study with Feature Selection 
             and WSN Localization Applications},
  author  = {Rajakumar, B.R. and ...},
  journal = {[Journal Name]},
  year    = {2026}
}
```

---

## 📧 Contact

For questions or reproducibility issues, open a GitHub Issue.

---

*Code developed to accompany the research paper. All results are reproducible with fixed random seeds (30 independent runs per configuration).*
