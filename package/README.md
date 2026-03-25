# AQHSO: Adaptive Quantum Hybrid Swarm Optimizer

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*AQHSO is a state-of-the-art three-phase metaheuristic optimization algorithm that unifies Grey Wolf Optimization (GWO), Firefly Algorithm (FA), and Ant Colony Optimization (ACO) under an adaptive quantum rotation framework.*

This standalone library executes locally and relies natively on `mealpy` and `numpy`.

## 📦 Installation

Install AQHSO directly from the repository using pip:

```bash
pip install git+https://github.com/Puneethreddy2530/Research_Paper.git#subdirectory=package
```
Alternatively, if you cloned the source directory:
```bash
pip install .
```

## 🚀 Quick Start Example

It integrates flawlessly as a `<mealpy.Optimizer>` module into any project.

```python
import numpy as np
from aqhso import AQHSO
from mealpy.utils.space import FloatVar

# 1. Define your objective function (e.g. Sphere function)
def sphere_function(solution):
    return np.sum(solution ** 2)

# 2. Define problem boundaries
problem_dict = {
    "fit_func": sphere_function,
    "obj_func": sphere_function,
    "bounds": FloatVar(lb=[-10.0]*30, ub=[10.0]*30),
    "minmax": "min",      # Goal: minimize
}

# 3. Initialize the Adaptive Quantum optimizer
optimizer = AQHSO(
    epoch=500,           # Number of iterations
    pop_size=30,         # Agents in the swarm
    theta_max_init=0.1,  # Initial quantum rotation max angle
    levy_beta=1.5        # Heavy-tail tunneling parameter
)

# 4. Execute optimization
best_agent = optimizer.solve(problem_dict)

print(f"Optimal Minimum Discovered: {best_agent.target.fitness}")
print(f"Coordinates: {best_agent.solution}")
```

## 🧠 Architectural Phases

1. **Phase 1 (Epochs 0–20%) | GWO Exploitation:** Fast initial basin targeting. Uses OBL (Opposition-based learning) for initialization.
2. **Phase 2 (Epochs 20–70%) | Quantum FA Attraction:** Superpositioned spatial brightness evaluated strictly inside angular bounds ($\theta$-space). Dynamic stagnation handling.
3. **Phase 3 (Epochs 70–100%) | Lévy Tunneling & ACO Pheromone:** Final granular traps optimization utilizing mathematical heavy-tailed distributions and an historical 1-D coordinate probability matrix.

## 📄 Citation 
If you find our work useful in your research, cite our original paper:
```bibtex
@article{reddy2026aqhso,
  title   = {AQHSO: Adaptive Quantum Hybrid Swarm Optimizer — A Novel Three-Phase Metaheuristic},
  author  = {Puneeth Reddy T and Katyayni Aarya},
  journal = {Pending Publication},
  year    = {2026}
}
```
