import numpy as np

from scipy.optimize import OptimizeResult
from sklearn.utils import check_random_state

from .space import Space


def dummy_minimize(func, dimensions, n_calls=1000, random_state=None):
    """Random search by uniform sampling within the given bounds.

    Parameters
    ----------
    * `func` [callable]:
        Function to minimize. Should take a array of parameters and
        return the function values.

    * `dimensions` [list, shape=(n_dims,)]:
        List of search space dimensions.
        Each search dimension can be defined either as

        - a `(upper_bound, lower_bound)` tuple (for `Real` or `Integer`
          dimensions),
        - a `(upper_bound, lower_bound, "prior")` tuple (for `Real`
          dimensions),
        - as a list of categories (for `Categorical` dimensions), or
        - an instance of a `Dimension` object (`Real`, `Integer` or
          `Categorical`).

    * `n_calls` [int, default=100]:
        Number of calls to `func` to find the minimum.

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
        - `space` [Space]: the optimisation space.

        For more details related to the OptimizeResult object, refer
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """
    rng = check_random_state(random_state)
    space = Space(dimensions)
    X = space.rvs(n_samples=n_calls, random_state=rng)

    init_y = func(X[0])
    if not np.isscalar(init_y):
        raise ValueError(
            "The function to be optimized should return a scalar")
    y = np.asarray([init_y] + [func(X[i]) for i in range(1, n_calls)])

    res = OptimizeResult()
    best = np.argmin(y)
    res.x = X[best]
    res.fun = y[best]
    res.func_vals = y
    res.x_iters = X
    res.space = space

    return res
