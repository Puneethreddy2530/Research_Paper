"""
Classical 23 Benchmark Functions
=================================
Standard test suite used in every metaheuristic paper since Mirjalili's GWO (2014).
Source: Yao et al. (1999) + Mirjalili et al. (2014)

Categories:
  F1–F7  : Unimodal (tests exploitation / local search precision)
  F8–F13 : High-dimensional Multimodal (tests exploration)
  F14–F23: Fixed-dimension Multimodal (low-dim, tricky landscapes)
"""

import numpy as np

# ─────────────────────────────────────────────────────────
# UNIMODAL FUNCTIONS F1–F7
# These have ONE global minimum. Good algos should nail these.
# ─────────────────────────────────────────────────────────

def F1(x):
    """Sphere — simplest possible. f* = 0"""
    return float(np.sum(x ** 2))

def F2(x):
    """Schwefel 2.22 — combines sum and product of abs"""
    return float(np.sum(np.abs(x)) + np.prod(np.abs(x)))

def F3(x):
    """Schwefel 1.2 — non-separable unimodal"""
    total = 0
    for i in range(len(x)):
        total += np.sum(x[:i+1]) ** 2
    return float(total)

def F4(x):
    """Schwefel 2.21 — max absolute value"""
    return float(np.max(np.abs(x)))

def F5(x):
    """Rosenbrock — banana-shaped valley. Tricky unimodal."""
    s = 0
    for i in range(len(x) - 1):
        s += 100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
    return float(s)

def F6(x):
    """Step function — discontinuous"""
    return float(np.sum((np.floor(x + 0.5)) ** 2))

def F7(x):
    """Quartic with noise — stochastic component"""
    n = len(x)
    return float(np.sum(np.arange(1, n+1) * x**4) + np.random.uniform(0, 1))


# ─────────────────────────────────────────────────────────
# MULTIMODAL FUNCTIONS F8–F13
# Many local minima. Tests exploration heavily.
# ─────────────────────────────────────────────────────────

def F8(x):
    """Schwefel 2.26 — deceptive, global optimum far from origin"""
    n = len(x)
    return float(-np.sum(x * np.sin(np.sqrt(np.abs(x)))))

def F9(x):
    """Rastrigin — many regular local minima"""
    n = len(x)
    return float(np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10))

def F10(x):
    """Ackley — nearly flat outer region, deep hole at center"""
    n = len(x)
    a, b, c = 20, 0.2, 2 * np.pi
    s1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / n))
    s2 = -np.exp(np.sum(np.cos(c * x)) / n)
    return float(s1 + s2 + a + np.e)

def F11(x):
    """Griewank — many local minima with product term"""
    n = len(x)
    s = np.sum(x**2) / 4000
    p = np.prod(np.cos(x / np.sqrt(np.arange(1, n+1))))
    return float(s - p + 1)

def F12(x):
    """Penalized 1 (Levy) — penalized with boundary conditions"""
    n = len(x)
    def u(xi, a, k, m):
        if xi > a:
            return k * (xi - a) ** m
        elif xi < -a:
            return k * (-xi - a) ** m
        return 0

    w = 1 + (x + 1) / 4
    s = (np.pi / n) * (
        10 * np.sin(np.pi * w[0])**2
        + np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[1:])**2))
        + (w[-1] - 1)**2
    )
    penalty = np.sum([u(xi, 10, 100, 4) for xi in x])
    return float(s + penalty)

def F13(x):
    """Penalized 2 (Zakharov style)"""
    n = len(x)
    def u(xi, a, k, m):
        if xi > a:
            return k * (xi - a) ** m
        elif xi < -a:
            return k * (-xi - a) ** m
        return 0

    s = (0.1) * (
        np.sin(3 * np.pi * x[0])**2
        + np.sum((x[:-1] - 1)**2 * (1 + np.sin(3 * np.pi * x[1:])**2))
        + (x[-1] - 1)**2 * (1 + np.sin(2 * np.pi * x[-1])**2)
    )
    penalty = np.sum([u(xi, 5, 100, 4) for xi in x])
    return float(s + penalty)


# ─────────────────────────────────────────────────────────
# FIXED-DIMENSION MULTIMODAL F14–F23
# Low dimensional (2D or small). Multiple peaks/valleys.
# ─────────────────────────────────────────────────────────

def F14(x):
    """Shekel's Foxholes (2D)"""
    aS = np.array([[-32,-16,0,16,32]*5,
                   [-32,-32,-32,-32,-32,
                    -16,-16,-16,-16,-16,
                    0,0,0,0,0,
                    16,16,16,16,16,
                    32,32,32,32,32]])
    o = 0
    for j in range(25):
        col = aS[:, j]
        t = np.sum((x[:2] - col)**6)
        o += 1.0 / (j + 1 + t)
    return float((1/500 + o) ** -1)

def F15(x):
    """Kowalik (4D)"""
    aK = np.array([0.1957,0.1947,0.1735,0.1600,0.0844,
                   0.0627,0.0456,0.0342,0.0323,0.0235,0.0246])
    bK = 1 / np.array([0.25,0.5,1,2,4,6,8,10,12,14,16])
    return float(np.sum((aK - (x[0]*(bK**2 + x[1]*bK)) /
                         (bK**2 + x[2]*bK + x[3]))**2))

def F16(x):
    """Six-Hump Camel (2D)"""
    x1, x2 = x[0], x[1]
    return float(4*x1**2 - 2.1*x1**4 + x1**6/3 + x1*x2 - 4*x2**2 + 4*x2**4)

def F17(x):
    """Branin (2D)"""
    x1, x2 = x[0], x[1]
    return float((x2 - (5.1/(4*np.pi**2))*x1**2 + (5/np.pi)*x1 - 6)**2
                 + 10*(1 - 1/(8*np.pi))*np.cos(x1) + 10)

def F18(x):
    """Goldstein-Price (2D)"""
    x1, x2 = x[0], x[1]
    a = 1 + (x1+x2+1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
    b = 30 + (2*x1-3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
    return float(a * b)

def F19(x):
    """Hartmann 3 (3D)"""
    aH = [[3,10,30],[0.1,10,35],[3,10,30],[0.1,10,35]]
    cH = [1,1.2,3,3.2]
    pH = [[0.3689,0.117,0.2673],[0.4699,0.4387,0.747],
          [0.1091,0.8732,0.5547],[0.0381,0.5743,0.8828]]
    o = 0
    for i in range(4):
        s = sum(aH[i][j] * (x[j] - pH[i][j])**2 for j in range(3))
        o += cH[i] * np.exp(-s)
    return float(-o)

def F20(x):
    """Hartmann 6 (6D)"""
    aH = [[10,3,17,3.5,1.7,8],[0.05,10,17,0.1,8,14],
          [3,3.5,1.7,10,17,8],[17,8,0.05,10,0.1,14]]
    cH = [1,1.2,3,3.2]
    pH = [[0.1312,0.1696,0.5569,0.0124,0.8283,0.5886],
          [0.2329,0.4135,0.8307,0.3736,0.1004,0.9991],
          [0.2348,0.1415,0.3522,0.2883,0.3047,0.6650],
          [0.4047,0.8828,0.8732,0.5743,0.1091,0.0381]]
    o = 0
    for i in range(4):
        s = sum(aH[i][j] * (x[j] - pH[i][j])**2 for j in range(6))
        o += cH[i] * np.exp(-s)
    return float(-o)

def F21(x):
    """Shekel 5 (4D)"""
    aSH = [[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],
           [2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]]
    cSH = [0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5]
    o = 0
    for i in range(5):
        s = sum((x[j] - aSH[i][j])**2 for j in range(4))
        o += 1.0 / (s + cSH[i])
    return float(-o)

def F22(x):
    """Shekel 7 (4D)"""
    aSH = [[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],
           [2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]]
    cSH = [0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5]
    o = 0
    for i in range(7):
        s = sum((x[j] - aSH[i][j])**2 for j in range(4))
        o += 1.0 / (s + cSH[i])
    return float(-o)

def F23(x):
    """Shekel 10 (4D)"""
    aSH = [[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],
           [2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]]
    cSH = [0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5]
    o = 0
    for i in range(10):
        s = sum((x[j] - aSH[i][j])**2 for j in range(4))
        o += 1.0 / (s + cSH[i])
    return float(-o)


# ─────────────────────────────────────────────────────────
# FUNCTION REGISTRY
# All 23 functions in one list — used by experiment scripts
# ─────────────────────────────────────────────────────────

BENCHMARK_FUNCTIONS = [
    # name,  func,  dim,   lb,    ub,    f_opt
    ("F1",  F1,    30,   -100,   100,   0),
    ("F2",  F2,    30,   -10,    10,    0),
    ("F3",  F3,    30,   -100,   100,   0),
    ("F4",  F4,    30,   -100,   100,   0),
    ("F5",  F5,    30,   -30,    30,    0),
    ("F6",  F6,    30,   -100,   100,   0),
    ("F7",  F7,    30,   -1.28,  1.28,  0),
    ("F8",  F8,    30,   -500,   500,   -12569.5),
    ("F9",  F9,    30,   -5.12,  5.12,  0),
    ("F10", F10,   30,   -32,    32,    0),
    ("F11", F11,   30,   -600,   600,   0),
    ("F12", F12,   30,   -50,    50,    0),
    ("F13", F13,   30,   -50,    50,    0),
    ("F14", F14,   2,    -65.536, 65.536, 1),
    ("F15", F15,   4,    -5,     5,     0.00030),
    ("F16", F16,   2,    -5,     5,     -1.0316),
    ("F17", F17,   2,    -5,     15,    0.398),
    ("F18", F18,   2,    -2,     2,     3),
    ("F19", F19,   3,    0,      1,     -3.86),
    ("F20", F20,   6,    0,      1,     -3.32),
    ("F21", F21,   4,    0,      10,    -10.1532),
    ("F22", F22,   4,    0,      10,    -10.4028),
    ("F23", F23,   4,    0,      10,    -10.5363),
]

def get_function(name):
    """Get a benchmark function by name (e.g. 'F1')"""
    for entry in BENCHMARK_FUNCTIONS:
        if entry[0] == name:
            return entry
    raise ValueError(f"Function {name} not found")
