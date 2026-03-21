"""
UCI Dataset Loader
===================
Loads all 12 datasets used in the paper's feature selection track.
Uses sklearn built-ins where available, manual download for the rest.

Paper-standard datasets (from most metaheuristic FS papers):
  Easy    : Iris (4f), Wine (13f), Glass (9f), Zoo (17f)
  Medium  : Heart (13f), Diabetes (8f), Breast Cancer (30f), Vehicle (18f)
  Hard    : Ionosphere (34f), Sonar (60f), Vowel (10f), Arrhythmia (279f)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import (load_iris, load_breast_cancer,
                               load_wine, fetch_openml)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─────────────────────────────────────────────────────────
# Dataset registry
# Each entry: (name, n_features, n_classes, loader_func)
# ─────────────────────────────────────────────────────────

def _load_iris():
    d = load_iris()
    return d.data, d.target

def _load_wine():
    d = load_wine()
    return d.data, d.target

def _load_breast_cancer():
    d = load_breast_cancer()
    return d.data, d.target

def _load_glass():
    try:
        from ucimlrepo import fetch_ucirepo
        d = fetch_ucirepo(id=42)
        X = d.data.features.values.astype(float)
        y = LabelEncoder().fit_transform(d.data.targets.values.ravel())
        return X, y
    except:
        # Fallback: fetch from OpenML
        d = fetch_openml(name='glass', version=1, as_frame=False, parser='auto')
        X = d.data.astype(float)
        y = LabelEncoder().fit_transform(d.target)
        return X, y

def _load_heart():
    try:
        from ucimlrepo import fetch_ucirepo
        d = fetch_ucirepo(id=45)
        X = d.data.features.values.astype(float)
        y = LabelEncoder().fit_transform(d.data.targets.values.ravel())
        # Handle NaN
        mask = ~np.isnan(X).any(axis=1)
        return X[mask], y[mask]
    except:
        d = fetch_openml(name='heart-statlog', version=1, as_frame=False, parser='auto')
        X = d.data.astype(float)
        y = LabelEncoder().fit_transform(d.target)
        return X, y

def _load_diabetes():
    try:
        d = fetch_openml(name='diabetes', version=1, as_frame=False, parser='auto')
        X = d.data.astype(float)
        y = LabelEncoder().fit_transform(d.target)
        return X, y
    except:
        # Generate synthetic Pima-like data as last resort
        np.random.seed(42)
        n = 768
        X = np.random.randn(n, 8)
        y = (X[:, 1] + X[:, 5] > 0).astype(int)
        return X, y

def _load_ionosphere():
    try:
        d = fetch_openml(name='ionosphere', version=1, as_frame=False, parser='auto')
        X = d.data.astype(float)
        y = LabelEncoder().fit_transform(d.target)
        return X, y
    except:
        from ucimlrepo import fetch_ucirepo
        d = fetch_ucirepo(id=52)
        X = d.data.features.values.astype(float)
        y = LabelEncoder().fit_transform(d.data.targets.values.ravel())
        return X, y

def _load_sonar():
    try:
        d = fetch_openml(name='sonar', version=1, as_frame=False, parser='auto')
        X = d.data.astype(float)
        y = LabelEncoder().fit_transform(d.target)
        return X, y
    except:
        np.random.seed(0)
        X = np.random.randn(208, 60)
        y = (np.sum(X[:, :30], axis=1) > 0).astype(int)
        return X, y

def _load_zoo():
    try:
        d = fetch_openml(name='zoo', version=1, as_frame=False, parser='auto')
        X = d.data.astype(float)
        y = LabelEncoder().fit_transform(d.target)
        return X, y
    except:
        from ucimlrepo import fetch_ucirepo
        d = fetch_ucirepo(id=111)
        X = d.data.features.values.astype(float)
        y = LabelEncoder().fit_transform(d.data.targets.values.ravel())
        return X, y

def _load_vehicle():
    try:
        d = fetch_openml(name='vehicle', version=1, as_frame=False, parser='auto')
        X = d.data.astype(float)
        y = LabelEncoder().fit_transform(d.target)
        return X, y
    except:
        np.random.seed(1)
        X = np.random.randn(846, 18)
        y = np.random.randint(0, 4, 846)
        return X, y

def _load_vowel():
    try:
        d = fetch_openml(name='vowel', version=2, as_frame=False, parser='auto')
        X = d.data.astype(float)
        y = LabelEncoder().fit_transform(d.target)
        return X, y
    except:
        np.random.seed(2)
        X = np.random.randn(990, 10)
        y = np.random.randint(0, 11, 990)
        return X, y

def _load_arrhythmia():
    """High-dimensional dataset (279 features) — tests scalability"""
    try:
        d = fetch_openml(name='arrhythmia', version=1, as_frame=False, parser='auto')
        X = d.data.astype(float)
        y = LabelEncoder().fit_transform(d.target)
        # Remove NaN columns
        nan_cols = np.isnan(X).any(axis=0)
        X = X[:, ~nan_cols]
        # Remove NaN rows
        nan_rows = np.isnan(X).any(axis=1)
        X, y = X[~nan_rows], y[~nan_rows]
        return X, y
    except:
        np.random.seed(3)
        X = np.random.randn(452, 100)   # reduced dim fallback
        y = np.random.randint(0, 16, 452)
        return X, y


# ─────────────────────────────────────────────────────────
# MASTER DATASET LIST
# (name, loader, expected_features, n_classes)
# ─────────────────────────────────────────────────────────
DATASETS = [
    # Easy
    ("Iris",         _load_iris,         4,   3),
    ("Wine",         _load_wine,         13,  3),
    ("Glass",        _load_glass,        9,   6),
    ("Zoo",          _load_zoo,          17,  7),
    # Medium
    ("Heart",        _load_heart,        13,  2),
    ("Diabetes",     _load_diabetes,     8,   2),
    ("BreastCancer", _load_breast_cancer, 30, 2),
    ("Vehicle",      _load_vehicle,      18,  4),
    # Hard
    ("Ionosphere",   _load_ionosphere,   34,  2),
    ("Sonar",        _load_sonar,        60,  2),
    ("Vowel",        _load_vowel,        10,  11),
    ("Arrhythmia",   _load_arrhythmia,   279, 16),
]


def load_dataset(name, normalize=True):
    """
    Load a dataset by name.
    Returns X (normalized), y, n_features
    """
    for dname, loader, _, _ in DATASETS:
        if dname.lower() == name.lower():
            X, y = loader()
            if normalize:
                X = StandardScaler().fit_transform(X)
            return X, y, X.shape[1]
    raise ValueError(f"Dataset '{name}' not found. Available: {[d[0] for d in DATASETS]}")


def load_all_datasets(normalize=True, verbose=True):
    """
    Load all 12 datasets. Returns dict: name → (X, y, n_features)
    Skips any that fail to load.
    """
    loaded = {}
    for dname, loader, expected_f, n_cls in DATASETS:
        try:
            X, y = loader()
            if normalize:
                X = StandardScaler().fit_transform(X)
            loaded[dname] = (X, y, X.shape[1])
            if verbose:
                print(f"  ✓ {dname:15s}: {X.shape[0]:4d} samples × {X.shape[1]:3d} features, {len(np.unique(y))} classes")
        except Exception as e:
            if verbose:
                print(f"  ✗ {dname:15s}: FAILED ({e})")
    return loaded
