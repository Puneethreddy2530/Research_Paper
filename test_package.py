import sys
import numpy as np

try:
    from aqhso import AQHSO
    print("Successfully imported AQHSO from the new pip-installed package!")
except ImportError as e:
    print(f"Failed to import AQHSO: {e}")
    sys.exit(1)

def sphere_function(solution):
    return np.sum(solution ** 2)

from mealpy.utils.space import FloatVar

problem_dict = {
    "fit_func": sphere_function,  # For internal AQHSO OBL calls
    "obj_func": sphere_function,  # For mealpy>3.0 compatibility
    "bounds": FloatVar(lb=[-10.0]*10, ub=[10.0]*10),
    "minmax": "min",
}

print("Initializing AQHSO Optimizer...")
opt = AQHSO(epoch=10, pop_size=20)
best_agent = opt.solve(problem_dict)

print(f"Test Successful!")
print(f"Optimal Minimum Found: {best_agent.target.fitness}")
print(f"Position: {best_agent.solution}")
