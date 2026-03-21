"""
Binary Feature Selection Wrapper
==================================
Converts any continuous metaheuristic into a binary feature selector.

How it works:
  1. Each agent/solution = a vector of n_features continuous values
  2. Transfer function maps continuous → binary (0/1): selected or not
  3. Fitness = 0.99 × (1 - KNN_accuracy) + 0.01 × (selected/total)
     ↑ This exact formula is used in dozens of published FS papers

Transfer functions (V-shaped, standard in literature):
  V1: |tanh(x)|          ← most common, smooth
  V2: |x / sqrt(1+x²)|  ← normalized sigmoid-like
  S1: 1/(1+e^-x)         ← standard sigmoid

We use V1 (tanh-based) as default — matches most published QGWO-FS papers.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


# ─────────────────────────────────────────────────────────
# TRANSFER FUNCTIONS
# Map continuous position → probability of selecting feature
# ─────────────────────────────────────────────────────────

def transfer_v1(x):
    """V-shaped: |tanh(x)| — most widely used in FS papers"""
    return np.abs(np.tanh(x))

def transfer_v2(x):
    """V-shaped: |x/sqrt(1+x²)|"""
    return np.abs(x / np.sqrt(1 + x**2))

def transfer_s1(x):
    """S-shaped sigmoid"""
    return 1.0 / (1.0 + np.exp(-x))

def binarize(x, transfer_fn=transfer_v1):
    """
    Convert continuous solution to binary feature mask.
    Probability = transfer_fn(x)
    bit_i = 1 if rand() < probability_i else 0
    """
    prob = transfer_fn(x)
    bits = (np.random.rand(len(x)) < prob).astype(int)
    # Ensure at least 1 feature is selected
    if np.sum(bits) == 0:
        bits[np.random.randint(len(bits))] = 1
    return bits


# ─────────────────────────────────────────────────────────
# FITNESS FUNCTION
# ─────────────────────────────────────────────────────────

class FeatureSelectionFitness:
    """
    Computes fitness for a binary feature selection solution.

    Fitness (minimize) = α × (1 - accuracy) + β × (|selected| / |total|)
    α = 0.99, β = 0.01  — heavily weighted toward accuracy, slightly toward fewer features
    This exact formulation from: Mirjalili et al. (2016) "SCA for FS"

    Uses 5-fold stratified cross-validation with KNN (k=5).
    Results are cached per binary mask string for efficiency.
    """

    def __init__(self, X, y, k_neighbors=5, cv_folds=5, alpha=0.99, beta=0.01):
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.alpha = alpha
        self.beta  = beta
        self.knn   = KNeighborsClassifier(n_neighbors=k_neighbors)
        self.cv    = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        self._cache = {}
        self.eval_count = 0

    def __call__(self, x):
        """
        x: continuous solution vector (length = n_features)
        Returns: scalar fitness value (lower = better)
        """
        bits = binarize(x)
        key  = bits.tobytes()

        if key in self._cache:
            return self._cache[key]

        self.eval_count += 1

        selected = np.where(bits == 1)[0]
        n_selected = len(selected)

        if n_selected == 0:
            fitness = 1.0   # worst possible
            self._cache[key] = fitness
            return fitness

        X_sub = self.X[:, selected]

        # Cross-validation accuracy
        try:
            accuracies = []
            for train_idx, val_idx in self.cv.split(X_sub, self.y):
                self.knn.fit(X_sub[train_idx], self.y[train_idx])
                pred = self.knn.predict(X_sub[val_idx])
                accuracies.append(accuracy_score(self.y[val_idx], pred))
            acc = np.mean(accuracies)
        except Exception:
            acc = 0.0

        # Fitness formula (standard in all FS papers)
        fitness = self.alpha * (1 - acc) + self.beta * (n_selected / self.n_features)
        self._cache[key] = fitness
        return float(fitness)

    def decode(self, x):
        """Convert continuous solution to binary and return stats."""
        bits = binarize(x)
        selected = np.where(bits == 1)[0]
        n_selected = len(selected)

        if n_selected == 0:
            return {'accuracy': 0, 'n_selected': 0, 'reduction': 1.0, 'bits': bits}

        X_sub = self.X[:, selected]
        accuracies = []
        for train_idx, val_idx in self.cv.split(X_sub, self.y):
            self.knn.fit(X_sub[train_idx], self.y[train_idx])
            pred = self.knn.predict(X_sub[val_idx])
            accuracies.append(accuracy_score(self.y[val_idx], pred))

        acc = np.mean(accuracies)
        return {
            'accuracy':    acc,
            'n_selected':  n_selected,
            'reduction':   1 - (n_selected / self.n_features),
            'bits':        bits,
            'selected_idx': selected,
        }
