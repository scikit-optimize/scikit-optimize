from collections import Callable
from time import time

import numpy as np
from scipy.optimize import OptimizeResult


def create_result(Xi, yi, space=None, rng=None, specs=None, models=None):
    """
    Return an instance of OptimizeResult() with the required
    information.

    Parameters
    ----------
    * `Xi` [list of lists, shape=(n_iters, n_features)]:
        Location of the minimum at every iteration.

    * `yi` [array-like, shape=(n_iters,)]:
        Minimum value obtained at every iteration.

    * `space` [Space instance, optional]:
        Search space.

    * `rng` [RandomState instance, optional]:
        State of the random state.

    * `specs` [dict, optional]:
        Call specifications.

    * `models` [list, optional]:
        List of fit surrogate models.

    Returns
    -------
    * `res` [`OptimizeResult`, scipy object]:
        OptimizeResult instance with the required information.
    """
    res = OptimizeResult()
    yi = np.asarray(yi)
    best = np.argmin(yi)
    res.x = Xi[best]
    res.fun = yi[best]
    res.func_vals = yi
    res.x_iters = Xi
    res.models = models
    res.space = space
    res.random_state = rng
    res.specs = specs
    return res


def verbose_func(func, x, verbose=False, prev_ys=None, x_info='',
                 func_call_no=1):
    """
    Call func at a given point x, but with a set verbosity

    Parameters
    ----------
    * `func` [callable] :
        Expensive function to evaluate.

    * `x` [list, shape=(n_features,)] :
        Point at which the function should be evaluated

    * `verbose` [boolean, optional] :
        Set verbosity

    * `prev_ys` [list, shape=(n_iters-1,)] :
        Function evaluations at previous (n_iters-1,) iterations.

    * `x_info` [str, 'provided' or 'random', optional]:
        Whether the point `x` is provided by the user, random.
    """
    prev_ys = list(prev_ys)
    if verbose:
        print("Function evaluation No: %d at %s point started." %
              (func_call_no, x_info))
        t = time()

    curr_y = func(x)
    if verbose:
        print("Function evaluation No: %d at %s point ended." %
              (func_call_no, x_info))
        print("Time taken: %0.4f" % (time() - t))
        print("Function value obtained: %0.4f" % curr_y)
        print("Current minimum: %0.4f" % np.min(prev_ys + [curr_y]))
    return curr_y


def check_callback(callback):
    """
    Check if callback is a callable or a list of callables.
    """
    if callback is not None:
        if isinstance(callback, Callable):
            callback = [callback]
        elif not (isinstance(callback, list) and
                  all([isinstance(c, Callable) for c in callback])):
            raise ValueError("callback should be either a callable or "
                             "a list of callables.")
    return callback
