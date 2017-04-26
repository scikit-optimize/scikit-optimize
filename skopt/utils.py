from copy import deepcopy
from collections import Iterable

import numpy as np
from scipy.optimize import OptimizeResult
from sklearn.externals.joblib import dump as dump_
from sklearn.externals.joblib import load as load_

__all__ = (
    "load",
    "dump",
)


def create_result(Xi, yi, space=None, rng=None, specs=None, models=None):
    """
    Initialize an `OptimizeResult` object.

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
    if np.ndim(yi) == 2:
        yi = np.ravel(yi[:, 0])
        res.log_time = np.ravel(yi[:, 1])
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


def eval_callbacks(callbacks, result):
    """Evaluate list of callbacks on result.

    The return values of the `callbacks` are ORed together to give the
    overall decision on whether or not the optimization procedure should
    continue.

    Parameters
    ----------
    * `callbacks` [list of callables]:
        Callbacks to evaluate.

    * `result` [`OptimizeResult`, scipy object]:
        Optimization result object to be stored.

    Returns
    -------
    * `decision` [bool]:
        Decision of the callbacks whether or not to keep optimizing
    """
    stop = False
    if callbacks:
        for c in callbacks:
            decision = c(result)
            if decision is not None:
                stop = stop or decision

    return stop


def dump(res, filename, store_objective=True, **kwargs):
    """
    Store an skopt optimization result into a file.

    Parameters
    ----------
    * `res` [`OptimizeResult`, scipy object]:
        Optimization result object to be stored.

    * `filename` [string or `pathlib.Path`]:
        The path of the file in which it is to be stored. The compression
        method corresponding to one of the supported filename extensions ('.z',
        '.gz', '.bz2', '.xz' or '.lzma') will be used automatically.

    * `store_objective` [boolean, default=True]:
        Whether the objective function should be stored. Set `store_objective`
        to `False` if your objective function (`.specs['args']['func']`) is
        unserializable (i.e. if an exception is raised when trying to serialize
        the optimization result).

        Notice that if `store_objective` is set to `False`, a deep copy of the
        optimization result is created, potentially leading to performance
        problems if `res` is very large. If the objective function is not
        critical, one can delete it before calling `skopt.dump()` and thus
        avoid deep copying of `res`.

    * `**kwargs` [other keyword arguments]:
        All other keyword arguments will be passed to `joblib.dump`.
    """
    if store_objective:
        dump_(res, filename, **kwargs)

    elif 'func' in res.specs['args']:
        # If the user does not want to store the objective and it is indeed
        # present in the provided object, then create a deep copy of it and
        # remove the objective function before dumping it with joblib.dump.
        res_without_func = deepcopy(res)
        del res_without_func.specs['args']['func']
        dump_(res_without_func, filename, **kwargs)

    else:
        # If the user does not want to store the objective and it is already
        # missing in the provided object, dump it without copying.
        dump_(res, filename, **kwargs)


def load(filename, **kwargs):
    """
    Reconstruct a skopt optimization result from a file
    persisted with skopt.dump.

    Notice that the loaded optimization result can be missing
    the objective function (`.specs['args']['func']`) if `skopt.dump`
    was called with `store_objective=False`.

    Parameters
    ----------
    * `filename` [string or `pathlib.Path`]:
        The path of the file from which to load the optimization result.

    * `**kwargs` [other keyword arguments]:
        All other keyword arguments will be passed to `joblib.load`.

    Returns
    -------
    * `res` [`OptimizeResult`, scipy object]:
        Reconstructed OptimizeResult instance.
    """
    return load_(filename, **kwargs)

def is_listlike(x):
    return isinstance(x, (list, tuple))

def is_2Dlistlike(x):
    return np.all([is_listlike(xi) for xi in x])

def check_x_in_space(x, space):
    if is_2Dlistlike(x):
        if not np.all([p in space for p in x]):
            raise ValueError("Not all points are within the bounds of"
                             " the space.")
    elif is_listlike(x):
        if x not in space:
            raise ValueError("Point (%s) is not within the bounds of"
                             " the space (%s)."
                             % (x, space.bounds))
