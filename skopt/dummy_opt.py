import numpy as np

from scipy.optimize import OptimizeResult
from sklearn.utils import check_random_state

from .utils import extract_bounds


def dummy_minimize(func, bounds, maxiter=1000, random_state=None):
    """
    Sample each parameter uniformly within the given bounds.

    Parameters
    ----------
    func: callable
        Function to minimize. Should take a array of parameters and
        return the function value.

    bounds: array-like, shape (n_params, 2)
        ``bounds[i][0]`` should give the lower bound of each parameter and
        ``bounds[i][1]`` should give the upper bound of each parameter.

    maxiter: int, default 1000
        Number of iterations to find the minimum. In other words, the
        number of function evaluations.

    random_state: int, RandomState instance, or None (default)
        Set random state to something other than None for reproducible
        results.

    Returns
    -------
    res: OptimizeResult, scipy object
        The optimization result returned as a OptimizeResult object.
        Important attributes are
        ``x`` - float, the optimization solution,
        ``fun`` - float, the value of the function at the optimum.
        ``func_vals`` - the function value at the ith iteration.
        ``x_iters`` - the value of ``x`` corresponding to the function value
                      at the ith iteration.
        For more details related to the OptimizeResult object, refer
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """
    rng = check_random_state(random_state)

    n_params = len(bounds)
    lb, ub = extract_bounds(bounds)

    X = np.zeros((maxiter, n_params))
    y = np.zeros(maxiter)

    for i in range(maxiter):
        X[i] = lb + (ub - lb) * rng.rand(n_params)
        y[i] = func(X[i])

    res = OptimizeResult()
    best = np.argmin(y)
    res.x = X[best]
    res.fun = y[best]
    res.func_vals = y
    res.x_iters = X

    return res
