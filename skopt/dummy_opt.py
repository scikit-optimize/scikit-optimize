import numpy as np

from scipy.optimize import OptimizeResult
from sklearn.utils import check_random_state

from .utils import extract_bounds


def dummy_minimize(func, bounds, maxiter=1000, random_state=None):
    """Random search by uniform sampling within the given bounds.

    Parameters
    ----------
    * `func` [callable]:
        Function to minimize. Should take a array of parameters and
        return the function values.

    * `bounds` [array-like, shape=(n_parameters, 2)]:
        - ``bounds[i][0]`` should give the lower bound of each parameter and
        - ``bounds[i][1]`` should give the upper bound of each parameter.

    * `maxiter` [int, default=1000]:
        Number of iterations to find the minimum. In other words, the
        number of function evaluations.

    * `random_state` [int, RandomState instance, or None (default)]:
        Set random state to something other than None for reproducible
        results.

    Returns
    -------
    * `res` [`OptimizeResult`, scipy object]:
        The optimization result returned as a OptimizeResult object.
        Important attributes are:

        - `x` [float]: location of the minimum.
        - `fun` [float]: function value at the minimum.
        - `x_iters` [array]: location of function evaluation for each
           iteration.
        - `func_vals` [array]: function value for each iteration.

        For more details related to the OptimizeResult object, refer
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """
    rng = check_random_state(random_state)

    n_params = len(bounds)
    lb, ub = extract_bounds(bounds)

    X = lb + (ub - lb) * rng.rand(maxiter, n_params)
    init_y = func(X[0])
    if not np.isscalar(init_y):
        raise ValueError(
            "The function to be optimized should return a scalar")
    y = np.asarray([init_y] + [func(X[i]) for i in range(maxiter - 1)])

    res = OptimizeResult()
    best = np.argmin(y)
    res.x = X[best]
    res.fun = y[best]
    res.func_vals = y
    res.x_iters = X

    return res
