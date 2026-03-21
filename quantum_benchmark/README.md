# Quantum-Inspired Metaheuristic Benchmarking Suite
## Paper: Comparative Study of Classical vs Quantum-Enhanced GWO, FA, and ACO

---

## Project Structure
```
quantum_benchmark/
├── algorithms/
│   ├── classical_gwo.py      # Standard Grey Wolf Optimizer
│   ├── classical_fa.py       # Standard Firefly Algorithm  
│   ├── classical_aco.py      # Standard ACO (Continuous)
│   ├── quantum_gwo.py        # Quantum-Inspired GWO (QGWO)
│   ├── quantum_fa.py         # Quantum-Inspired Firefly (QFA)
│   └── quantum_aco.py        # Quantum-Inspired ACO (QACO)
├── benchmarks/
│   ├── classical_23.py       # 23 classical benchmark functions
│   └── cec2017_loader.py     # CEC 2017 functions via opfunu
├── experiments/
│   ├── run_classical23.py    # Run all 6 algos on 23 functions
│   ├── run_cec2017.py        # Run all 6 algos on CEC 2017
│   └── run_convergence.py    # Generate convergence data
├── stats/
│   └── statistical_tests.py  # Wilcoxon, Friedman, Nemenyi
├── plots/
│   └── plot_results.py       # All figures for the paper
├── utils/
│   └── result_manager.py     # Save/load results as CSV/JSON
├── results/                  # Auto-generated CSVs go here
├── main.py                   # ONE CLICK: runs everything
└── requirements.txt
```

## How to Run

### Step 1 — Install dependencies
```bash
pip install mealpy opfunu scipy scikit-learn matplotlib seaborn numpy pandas
```

### Step 2 — Run everything (paper-ready)
```bash
python main.py
```

### Step 3 — Run individual experiments
```bash
python experiments/run_classical23.py    # ~10 mins
python experiments/run_cec2017.py        # ~2-3 hours
python experiments/run_convergence.py    # ~20 mins
python stats/statistical_tests.py        # seconds
python plots/plot_results.py             # seconds
```

## Parameter Settings (from original papers — DO NOT change for paper)
| Algorithm | Population | Epochs | Key Params |
|-----------|-----------|--------|------------|
| GWO | 30 | 500 | a linearly decreases 2→0 |
| QGWO | 30 | 500 | Δθ_max=0.05π |
| FA | 30 | 500 | α=0.5, β₀=1.0, γ=1.0 |
| QFA | 30 | 500 | α=0.5, β₀=1.0, quantum rotation |
| ACO-R | 30 | 500 | sample_count=50, intent_factor=0.5 |
| QACO | 30 | 500 | Q-bit pheromone, Δθ=0.01π |

## Results Output
- `results/classical23_results.csv` — Mean/Std/Best/Worst for all 23 functions
- `results/cec2017_results.csv` — Mean/Std for CEC 2017
- `results/statistical_tests.csv` — Wilcoxon p-values  
- `results/friedman_ranks.csv` — Friedman rankings
- `plots/` — All figures ready for paper insertion
