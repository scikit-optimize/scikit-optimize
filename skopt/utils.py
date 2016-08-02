import numpy as np
from scipy.optimize import OptimizeResult

def pack_optimize_result(Xi, yi, space, rng, specs, models=None):
    """
    Instantiate OptimizeResult() with the required information.
    """
    if models is None:
        models = []
    res = OptimizeResult()
    best = np.argmin(yi)
    res.x = Xi[best]
    res.fun = yi[best]
    res.func_vals = np.array(yi)
    res.x_iters = Xi
    res.models = models
    res.space = space
    res.random_state = rng
    res.specs = specs
    return res
