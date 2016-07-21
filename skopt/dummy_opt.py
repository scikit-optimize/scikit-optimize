from collections import Iterable
import numpy as np

from scipy.optimize import OptimizeResult
from sklearn.utils import check_random_state

from .space import Space


def dummy_minimize(func, dimensions, n_calls=100,
                   x0=None, y0=None, random_state=None):
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

    * `n_calls` [int, default=1000]:
        Maximum number of calls to `func` to find the minimum.

    * `x0` [list or list of lists or None]:
        List of initial input points (if it is a list of lists)
        or an initial input point (if it is a list). If it is
        `None`, no initial input points are used.

    * `y0` [list or scalar]
        if `y0` is a list, then it corresponds to evaluations of the function
        at each element of `x0` : the i-th element of `y0` corresponds
        to the function evaluated at the i-th element of `x0`. If `y0`
        is a scalar then it corresponds to the evaluation of the function at
        `x0`.
        If only `x0` is provided but not `y0`, the function is evaluated
        at each element of `x0`, otherwise the values provided in `y0`
        are used.

    * `random_state` [int, RandomState instance, or None (default)]:
        Set random state to something other than None for reproducible
        results.

    Returns
    -------
    * `res` [`OptimizeResult`, scipy object]:
        The optimization result returned as a OptimizeResult object.
        Important attributes are:

        - `x` [list]: location of the minimum.
        - `fun` [float]: function value at the minimum.
        - `x_iters` [list of lists]: location of function evaluation for each
           iteration.
        - `func_vals` [array]: function value for each iteration.
        - `space` [Space]: the optimisation space.

        For more details related to the OptimizeResult object, refer
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """
    rng = check_random_state(random_state)
    space = Space(dimensions)
    if x0 is None:
        x0 = []
    x0 = list(x0)
    if x0 and not isinstance(x0[0], list):
        x0 = [x0]
    n_random_starts = n_calls
    if y0 is None:
        y0 = [func(x) for x in x0]
        n_random_starts -= len(y0)
    if isinstance(y0, Iterable):
        y0 = list(y0)
    else:
        y0 = [y0]
    if len(x0) != len(y0):
        raise ValueError("x0 and y0 should have the same length")
    X = x0 + space.rvs(n_samples=n_random_starts, random_state=rng)
    init_provided = len(y0) != 0
    init_y = y0[0] if init_provided else func(X[0])
    if not np.isscalar(init_y):
        raise ValueError(
            "The function to be optimized should return a scalar")
    if init_provided:
        y = y0 + [func(x) for x in X[len(y0):]]
    else:
        y = [init_y] + [func(x) for x in X[1:]]

    y = np.array(y)
    res = OptimizeResult()
    best = np.argmin(y)
    res.x = X[best]
    res.fun = y[best]
    res.func_vals = y
    res.x_iters = X
    res.space = space
    # create attribute even though it is empty
    res.models = []

    return res
