from copy import deepcopy
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize as sp_minimize
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


def expected_minimum(res, n_random_starts=20):
    """
    Compute the minimum over the predictions of the last surrogate model.

    Note that the returned minimum may not necessarily be an accurate
    prediction of the minimum of the true objective function.

    Parameters
    ----------
    * `res`  [`OptimizeResult`, scipy object]:
        The optimization result returned by a `skopt` minimizer.

    * `n_random_starts` [int, default=20]:
        The number of random starts for the minimization of the surrogate
        model.

    Returns
    -------
    * `x` [list]: location of the minimum.

    * `fun` [float]: the surrogate function value at the minimum.
    """
    def func(x):
        reg = res.models[-1]
        return reg.predict(x.reshape(1, -1))[0]

    xs = [res.x]
    if n_random_starts > 0:
        xs.extend(res.space.rvs(n_random_starts))

    best_x = None
    best_fun = np.inf

    for x0 in xs:
        r = sp_minimize(func, x0=x0, bounds=res.space.bounds)

        if r.fun < best_fun:
            best_x = r.x
            best_fun = r.fun

    return [v for v in best_x], best_fun
