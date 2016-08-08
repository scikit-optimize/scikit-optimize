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


def check_callback(callback, verbose=False):
    """
    Check if callback is a callable or a list of callables.
    """
    if verbose:
        callbacks = []
    else:
        callbacks = [VerboseCallback()]
    if callback is not None:
        if isinstance(callback, Callable):
            callbacks.append(callback)
        elif (isinstance(callback, list) and
              all([isinstance(c, Callable) for c in callback])):
            callbacks.extend(callback)
        else:
            raise ValueError("callback should be either a callable or "
                             "a list of callables.")
    return callbacks


class VerboseCallback(object):
    def __init__(self):
        self._func_call_no = 1
        self.start_msg = "Function evaluation No: %d started"
        self.end_msg = "Function evaluation No: %d ended"
        self.start_time = time()
        self.prev_results = None
        self.curr_results = None
        print(self.start_msg % self._func_call_no)

    def __call__(self, res):
        self.curr_results = res
        time_taken = time() - self.start_time
        print(self.end_msg % self._func_call_no)

        curr_y = res.func_vals[-1]
        curr_min = res.fun
        print("Time taken: %0.4f" % time_taken)
        print("Function value obtained: %0.4f" % curr_y)
        print("Current minimum: %0.4f" % curr_min)
        self._func_call_no += 1
        print(self.start_msg % self._func_call_no)
        self.prev_results = res
