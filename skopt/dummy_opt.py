"""Random search."""

import copy
import inspect
import numbers
import numpy as np

from collections import Iterable
from sklearn.utils import check_random_state

from .callbacks import check_callback
from .callbacks import VerboseCallback
from .space import make_space
from .utils import create_result


def dummy_minimize(func, dimensions, n_calls=100,
                   x0=None, y0=None, random_state=None, verbose=False,
                   callback=None):
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

    * `x0` [list, list of lists or `None`]:
        Initial input points.

        - If it is a list of lists, use it as a list of input points.
        - If it is a list, use it as a single initial input point.
        - If it is `None`, no initial input points are used.

    * `y0` [list, scalar or `None`]
        Evaluation of initial input points.

        - If it is a list, then it corresponds to evaluations of the function
          at each element of `x0` : the i-th element of `y0` corresponds
          to the function evaluated at the i-th element of `x0`.
        - If it is a scalar, then it corresponds to the evaluation of the
          function at `x0`.
        - If it is None and `x0` is provided, then the function is evaluated
          at each element of `x0`.

    * `random_state` [int, RandomState instance, or None (default)]:
        Set random state to something other than None for reproducible
        results.

    * `verbose` [boolean, default=False]:
        Control the verbosity. It is advised to set the verbosity to True
        for long optimization runs.

    * `callback` [callable, list of callables, optional]
        If callable then `callback(res)` is called after each call to `func`.
        If list of callables, then each callable in the list is called.

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
        - `specs` [dict]: the call specifications.
        - `rng` [RandomState instance]: State of the random state
           at the end of minimization.

        For more details related to the OptimizeResult object, refer
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """
    # Save call args
    specs = {"args": copy.copy(inspect.currentframe().f_locals),
             "function": inspect.currentframe().f_code.co_name}

    # Check params
    rng = check_random_state(random_state)
    space = make_space(dimensions)

    if x0 is None:
        x0 = []
    elif type(x0) is dict or not isinstance(x0[0], list):
        x0 = [x0]

    if not isinstance(x0, list):
        raise ValueError("`x0` should be a list, got %s" % type(x0))

    n_init_func_calls = 0
    if len(x0) > 0 and y0 is not None:
        if isinstance(y0, Iterable):
            y0 = list(y0)
        elif isinstance(y0, numbers.Number):
            y0 = [y0]
        else:
            raise ValueError("`y0` should be an iterable or a scalar, got %s"
                             % type(y0))
        if len(x0) != len(y0):
            raise ValueError("`x0` and `y0` should have the same length")

        if not all(map(np.isscalar, y0)):
            raise ValueError("`y0` elements should be scalars")

    elif len(x0) > 0 and y0 is None:
        y0 = []
        n_calls -= len(x0)
        n_init_func_calls = len(x0)

    elif len(x0) == 0 and y0 is not None:
        raise ValueError("`x0`cannot be `None` when `y0` is provided")

    else:  # len(x0) == 0 and y0 is None
        y0 = []

    callbacks = check_callback(callback)
    if verbose:
        callbacks.append(VerboseCallback(
            n_init=n_init_func_calls, n_total=n_calls))

    X = x0
    y = y0

    # Random search
    X = X + space.rvs(n_samples=n_calls, random_state=rng)
    first = True

    for i in range(len(y0), len(X)):
        y_i = func(X[i])

        if first:
            first = False
            if not np.isscalar(y_i):
                raise ValueError("`func` should return a scalar")

        y.append(y_i)
        if callbacks:
            curr_res = create_result(X[: i + 1], y, space, rng, specs)
            for c in callbacks:
                c(curr_res)

    y = np.array(y)
    return create_result(X, y, space, rng, specs)
