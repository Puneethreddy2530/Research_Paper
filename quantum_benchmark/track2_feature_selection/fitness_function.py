"""
Feature Selection via Metaheuristic Wrapper
============================================
The "ML" part of the paper.

How it works:
  - Each algorithm's "solution" = binary vector of length n_features
    e.g. [1, 0, 1, 1, 0] means "use features 0, 2, 3"
  - Fitness function = KNN accuracy on selected features
  - Algorithm searches for the subset that gives highest accuracy
    while using fewest features

Fitness function (from El-ashry et al. 2020, standard in literature):
  fitness = 0.99 × (1 - accuracy) + 0.01 × (selected / total)

  This means:
    - Accuracy matters 99x more than feature count
    - But ties are broken by using fewer features
    - Perfect accuracy + all features selected = 0.01
    - Perfect accuracy + 1 feature selected  = very close to 0.0

Why KNN (k=5)?
  - Standard in feature selection papers (no training needed)
  - Sensitive to irrelevant features → rewards good selection
  - Fast to evaluate during optimization loop

Why 10-fold cross-validation?
  - Stable accuracy estimate without needing separate test set
  - Standard in all published feature selection comparisons
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


# ─────────────────────────────────────────────────────────────
# CORE FITNESS FUNCTION
# ─────────────────────────────────────────────────────────────

def feature_selection_fitness(solution, X, y, k=5, n_folds=10, alpha=0.99):
    """
    Evaluate a binary feature mask using KNN cross-validation.

    Args:
        solution  : numpy array of floats in [0,1] — threshold at 0.5 to get binary mask
        X         : feature matrix (n_samples × n_features)
        y         : class labels
        k         : KNN neighbors (default 5, from literature)
        n_folds   : cross-validation folds (default 10)
        alpha     : weight of accuracy vs feature count (default 0.99)

    Returns:
        fitness (float) — lower = better
        (minimization problem: 0 = perfect, 1 = worst)
    """
    n_features = X.shape[1]

    # Convert continuous solution → binary feature mask
    # threshold at 0.5: above → feature selected
    mask = solution > 0.5

    # Edge case: if no features selected, select all (penalize heavily)
    if not np.any(mask):
        return 1.0   # worst possible fitness

    X_selected = X[:, mask]
    n_selected = np.sum(mask)

    # ── 10-fold cross-validation ──────────────────────────
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=1)

    acc_scores = []
    for train_idx, val_idx in kf.split(X_selected, y):
        X_train = X_selected[train_idx]
        X_val   = X_selected[val_idx]
        y_train = y[train_idx]
        y_val   = y[val_idx]

        # Handle edge: if a class has only 1 sample in train, skip
        if len(np.unique(y_train)) < 2:
            continue

        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        acc_scores.append(accuracy_score(y_val, y_pred))

    if not acc_scores:
        return 1.0

    accuracy = np.mean(acc_scores)

    # ── Standard fitness formula (El-ashry 2020, dozens of papers) ──
    fitness = alpha * (1.0 - accuracy) + (1.0 - alpha) * (n_selected / n_features)

    return float(fitness)


# ─────────────────────────────────────────────────────────────
# EVALUATION HELPER — called after optimization to get final stats
# ─────────────────────────────────────────────────────────────

def evaluate_solution(solution, X, y, k=5, n_folds=10):
    """
    Get full stats for a solution:
    Returns dict with accuracy, n_selected, feature_indices, reduction_pct
    """
    n_features = X.shape[1]
    mask = solution > 0.5

    if not np.any(mask):
        mask = np.ones(n_features, dtype=bool)   # fallback

    X_sel = X[:, mask]
    n_sel = int(np.sum(mask))

    kf  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

    accs = []
    for tr, va in kf.split(X_sel, y):
        if len(np.unique(y[tr])) < 2:
            continue
        knn.fit(X_sel[tr], y[tr])
        accs.append(accuracy_score(y[va], knn.predict(X_sel[va])))

    # Also get baseline: all features
    baseline_accs = []
    knn2 = KNeighborsClassifier(n_neighbors=k)
    for tr, va in kf.split(X, y):
        if len(np.unique(y[tr])) < 2:
            continue
        knn2.fit(X[tr], y[tr])
        baseline_accs.append(accuracy_score(y[va], knn2.predict(X[va])))

    return {
        "accuracy":         round(np.mean(accs) * 100, 2) if accs else 0.0,
        "accuracy_std":     round(np.std(accs) * 100, 2)  if accs else 0.0,
        "baseline_acc":     round(np.mean(baseline_accs) * 100, 2) if baseline_accs else 0.0,
        "n_features_total": n_features,
        "n_features_selected": n_sel,
        "reduction_pct":    round((1 - n_sel / n_features) * 100, 1),
        "selected_indices": list(np.where(mask)[0]),
    }
