# Quantum-Inspired Metaheuristic Benchmarking Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This repository contains the official implementation and reproducibility codes for the paper: **"Comparative Analysis of Classical vs. Quantum-Enhanced Metaheuristic Algorithms."**

It provides a comprehensive benchmarking framework comparing Standard Grey Wolf Optimizer (GWO), Firefly Algorithm (FA), and Ant Colony Optimization (ACO) against their Quantum-Inspired counterparts (QGWO, QFA, QACO).

## 🚀 Features

*   **Implementations:** Clean and standardized implementations of GWO, FA, ACO, QGWO, QFA, and QACO.
*   **Benchmarks:** Supports the classical 23 benchmark functions and the IEEE CEC-2017 suite.
*   **Analysis:** Automated statistical analysis (Wilcoxon rank-sum, Friedman, Nemenyi tests).
*   **Visualization:** Automated chart generation (convergence curves, boxplots, ranking heatmaps, CD diagrams).
*   **Reproducibility:** A `main.py` entry point designed for one-click reproduction of all results presented in the paper.

## 📂 Repository Structure

```
Research_Paper/
├── quantum_benchmark/
│   ├── algorithms/           # Core algorithm implementations (Classical & Quantum)
│   ├── benchmarks/           # Objective function definitions (23 Classical & CEC2017)
│   ├── experiments/          # Scripts to run specific experiment batches
│   ├── stats/                # Statistical testing suite
│   ├── plots/                # Visualization generation scripts
│   ├── utils/                # Helper functions (I/O, result management)
│   ├── results/              # Directory where generated CSV/JSON results are saved
│   ├── main.py               # Main execution script
│   └── requirements.txt      # Python dependencies
├── .gitignore
└── README.md                 # This file
```

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Puneethreddy2530/Research_Paper.git
    cd Research_Paper/quantum_benchmark
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *Dependencies include: `mealpy`, `opfunu`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `numpy`, `pandas`.*

## 🏃 Usage & Reproducibility

To reproduce the study's findings, run the `main.py` script located in the `quantum_benchmark` directory.

```bash
cd quantum_benchmark

# 1. Run the complete suite (Paper Mode - approx. 3-4 hours)
python main.py

# 2. Quick Test Mode (5 runs, 100 epochs - approx. 2-3 mins)
python main.py --quick

# 3. Skip CEC-2017 (runs Classical 23 only - approx. 15 mins)
python main.py --skip-cec2017
```

**Output:**
*   **Data:** Performance metrics (mean, std, best, worst) and statistical test results are saved to `quantum_benchmark/results/`.
*   **Figures:** All generated plots (boxplots, convergence curves, etc.) are saved to `quantum_benchmark/plots/output/`.

*Note: For the exact parameter settings used in the paper, please refer to the `Parameter Settings` table within the paper or the default configurations in `main.py`.*

## 📜 Citation

If you use this code or benchmark in your research, please cite our paper:
*(Citation details will be added upon publication)*

## 📄 License
This project is licensed under the MIT License.