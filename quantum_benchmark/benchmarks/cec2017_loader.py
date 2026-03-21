"""
CEC 2017 Function Loader
=========================
Helper to load CEC 2017 functions from opfunu library.

CEC 2017 has 30 functions (F1-F30), but F2 is excluded
due to numerical instability (as per the original paper).

Usage:
    from benchmarks.cec2017_loader import get_cec2017_function, CEC2017_LIST
    func, lb, ub, f_opt = get_cec2017_function(func_id=1, ndim=30)
"""

import opfunu
import numpy as np

# All valid CEC 2017 function IDs (F2 excluded)
CEC2017_LIST = [1, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29]

# Categories for paper labeling
CATEGORY_MAP = {
    1:  "Unimodal",
    3:  "Unimodal",
    **{i: "Simple Multimodal" for i in range(4, 11)},
    **{i: "Hybrid"            for i in range(11, 21)},
    **{i: "Composition"       for i in range(21, 30)},
}


def get_cec2017_function(func_id, ndim=30):
    """
    Load a CEC 2017 benchmark function.

    Args:
        func_id (int): Function ID (1–29, not 2)
        ndim (int): Dimensionality (10, 30, 50, or 100)

    Returns:
        (callable, lb, ub, f_global)
        - callable: function that takes numpy array, returns float
        - lb: lower bound (float)
        - ub: upper bound (float)
        - f_global: known global optimum value
    """
    if func_id == 2:
        raise ValueError("F2 is excluded from CEC 2017 (numerical instability)")

    # Try standard opfunu naming
    class_name = f"F{func_id}2017"

    try:
        func_class = getattr(opfunu.cec_based, class_name)
        f_instance = func_class(ndim=ndim)
        return f_instance.evaluate, -100.0, 100.0, f_instance.f_global
    except AttributeError:
        pass

    # Fallback: search opfunu
    try:
        all_funcs = opfunu.get_cec_functions(cec_year=2017, ndim=ndim)
        for fc in all_funcs:
            if fc.__name__.endswith(str(func_id)):
                inst = fc(ndim=ndim)
                return inst.evaluate, -100.0, 100.0, inst.f_global
    except Exception:
        pass

    raise ImportError(
        f"CEC2017 F{func_id} not found in opfunu. "
        f"Install with: pip install opfunu --upgrade"
    )


def list_available_cec2017(ndim=30):
    """Print all available CEC 2017 functions."""
    print(f"CEC 2017 Functions (ndim={ndim}):")
    print(f"{'ID':>4}  {'Name':>12}  {'Category':>20}  {'f_global':>12}")
    print("-" * 55)
    for fid in CEC2017_LIST:
        try:
            _, _, _, f_opt = get_cec2017_function(fid, ndim)
            cat = CATEGORY_MAP.get(fid, "Unknown")
            print(f"  F{fid:>2}  {'CEC2017':>12}  {cat:>20}  {f_opt:>12.2f}")
        except Exception as e:
            print(f"  F{fid:>2}  [NOT AVAILABLE: {e}]")


if __name__ == "__main__":
    list_available_cec2017(ndim=30)
