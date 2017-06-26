from copy import deepcopy
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize as sp_minimize
from sklearn.base import is_regressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals.joblib import dump as dump_
from sklearn.externals.joblib import load as load_

from .learning import ExtraTreesRegressor
from .learning import GaussianProcessRegressor
from .learning import GradientBoostingQuantileRegressor
from .learning import RandomForestRegressor
from .learning.gaussian_process.kernels import ConstantKernel
from .learning.gaussian_process.kernels import HammingKernel
from .learning.gaussian_process.kernels import Matern

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


def expected_minimum(res, n_random_starts=20, random_state=None):
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

    * `random_state` [int, RandomState instance, or None (default)]:
        Set random state to something other than None for reproducible
        results.

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
        xs.extend(res.space.rvs(n_random_starts, random_state=random_state))

    best_x = None
    best_fun = np.inf

    for x0 in xs:
        r = sp_minimize(func, x0=x0, bounds=res.space.bounds)

        if r.fun < best_fun:
            best_x = r.x
            best_fun = r.fun

    return [v for v in best_x], best_fun


def has_gradients(estimator):
    """
    Check if an estimators predict method can provide gradients.

    Parameters
    ----------
    estimator: sklearn BaseEstimator instance.
    """
    tree_estimators = (
            ExtraTreesRegressor, RandomForestRegressor,
            GradientBoostingQuantileRegressor
    )
    cat_gp = False
    if hasattr(estimator, "kernel"):
        params = estimator.get_params()
        cat_gp = (
            isinstance(estimator.kernel, HammingKernel) or
            any([isinstance(params[p], HammingKernel) for p in params])
        )

    return isinstance(estimator, tree_estimators) or cat_gp


def cook_estimator(base_estimator, space=None, **kwargs):
    """
    Cook a default estimator.

    Parameters
    ----------
    * `base_estimator` ["GP", "RF", "ET", "GBRT" or sklearn regressor, default="GP"]:
        Should inherit from `sklearn.base.RegressorMixin`.
        In addition the `predict` method should have an optional `return_std`
        argument, which returns `std(Y | x)`` along with `E[Y | x]`.
        If base_estimator is one of ["GP", "RF", "ET", "GBRT"], a default
        surrogate model of the corresponding type is used corresponding to what
        is used in the minimize functions.

    * `space` [Space instance]:
        Has to be provided if the base_estimator is a gaussian process.
        Ignored otherwise.

    * `kwargs` [dict]:
        Extra parameters provided to the base_estimator at init time.
    """
    if space is not None:
        n_dims = space.transformed_n_dims
        is_cat = space.is_categorical
    if isinstance(base_estimator, str):
        base_estimator = base_estimator.upper()
        if base_estimator not in ["GP", "ET", "RF", "GBRT"]:
            raise ValueError("Valid strings for the base_estimator parameter "
                             " are: 'RF', 'ET' or 'GP' not %s" % base_estimator)
    elif not is_regressor(base_estimator):
        raise ValueError("base_estimator has to be a regressor.")

    if base_estimator == "GP":
        cov_amplitude = ConstantKernel(1.0, (0.01, 1000.0))
        if is_cat:
            other_kernel = HammingKernel(length_scale=np.ones(n_dims))
        else:
            other_kernel = Matern(
                length_scale=np.ones(n_dims),
                length_scale_bounds=[(0.01, 100)] * n_dims, nu=2.5)

        base_estimator = GaussianProcessRegressor(
            kernel=cov_amplitude * other_kernel,
            normalize_y=True, random_state=None, alpha=0.0, noise="gaussian",
            n_restarts_optimizer=2)
    elif base_estimator == "RF":
        base_estimator = RandomForestRegressor(n_estimators=100,
                                               min_samples_leaf=3)
    elif base_estimator == "ET":
        base_estimator = ExtraTreesRegressor(n_estimators=100,
                                             min_samples_leaf=3)
    elif base_estimator == "GBRT":
        gbrt = GradientBoostingRegressor(n_estimators=30, loss="quantile")
        base_estimator = GradientBoostingQuantileRegressor(base_estimator=gbrt)

    base_estimator.set_params(**kwargs)
    return base_estimator
