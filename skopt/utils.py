from copy import deepcopy
from collections import Iterable

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

from .space import check_dimension
from .space import Categorical

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
        res.log_time = np.ravel(yi[:, 1])
        yi = np.ravel(yi[:, 0])
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

    For the special base_estimator called "DUMMY" the return value is None.
    This corresponds to sampling points at random, hence there is no need
    for an estimator.

    Parameters
    ----------
    * `base_estimator` ["GP", "RF", "ET", "GBRT", "DUMMY"
                        or sklearn regressor, default="GP"]:
        Should inherit from `sklearn.base.RegressorMixin`.
        In addition the `predict` method should have an optional `return_std`
        argument, which returns `std(Y | x)`` along with `E[Y | x]`.
        If base_estimator is one of ["GP", "RF", "ET", "GBRT", "DUMMY"], a
        surrogate model corresponding to the relevant `X_minimize` function
        is created.

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
        if base_estimator not in ["GP", "ET", "RF", "GBRT", "DUMMY"]:
            raise ValueError("Valid strings for the base_estimator parameter "
                             " are: 'RF', 'ET', 'GP', 'GBRT' or 'DUMMY' not "
                             "%s." % base_estimator)
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

    elif base_estimator == "DUMMY":
        return None

    base_estimator.set_params(**kwargs)
    return base_estimator


def dimensions_aslist(search_space):
    """Convert a dict representation of a search space into a list of
    dimensions, ordered by sorted(search_space.keys()).

    Parameters
    ----------
    search_space : dict
        Represents search space. The keys are dimension names (strings)
        and values are instances of classes that inherit from the class
        skopt.space.Dimension (Real, Integer or Categorical)
        Example:
            {'name1': Real(0,1), 'name2': Integer(2,4), 'name3': Real(-1,1)}

    Returns
    -------
    params_space_list: list of skopt.space.Dimension instances.
        Example output with example inputs:
            [Real(0,1), Integer(2,4), Real(-1,1)]
    """
    params_space_list = [
        search_space[k] for k in sorted(search_space.keys())
    ]
    return params_space_list


def point_asdict(search_space, point_as_list):
    """Convert the list representation of a point from a search space
    to the dictionary representation, where keys are dimension names
    and values are corresponding to the values of dimensions in the list.

    Counterpart to parameters_aslist.

    Parameters
    ----------
    search_space : dict
        Represents search space. The keys are dimension names (strings)
        and values are instances of classes that inherit from the class
        skopt.space.Dimension (Real, Integer or Categorical)
        Example:
            {'name1': Real(0,1), 'name2': Integer(2,4), 'name3': Real(-1,1)}

    point_as_list : list
        list with parameter values.The order of parameters in the list
        is given by sorted(params_space.keys()).
        Example:
            [0.66, 3, -0.15]

    Returns
    -------
    params_dict: dictionary with parameter names as keys to which
        corresponding parameter values are assigned.
        Example output with inputs:
            {'name1': 0.66, 'name2': 3, 'name3': -0.15}
    """
    params_dict = {
        k: v for k, v in zip(sorted(search_space.keys()), point_as_list)
    }
    return params_dict


def point_aslist(search_space, point_as_dict):
    """Convert a dictionary representation of a point from a search space to
    the list representation. The list of values is created from the values of
    the dictionary, sorted by the names of dimensions used as keys.

    Counterpart to parameters_asdict.

    Parameters
    ----------
    search_space : dict
        Represents search space. The keys are dimension names (strings)
        and values are instances of classes that inherit from the class
        skopt.space.Dimension (Real, Integer or Categorical)
        Example:
            {'name1': Real(0,1), 'name2': Integer(2,4), 'name3': Real(-1,1)}

    point_as_dict : dict
        dict with parameter names as keys to which corresponding
        parameter values are assigned.
        Example:
            {'name1': 0.66, 'name2': 3, 'name3': -0.15}

    Returns
    -------
    point_as_list: list with point values.The order of
        parameters in the list is given by sorted(params_space.keys()).
        Example output with example inputs:
            [0.66, 3, -0.15]
    """
    point_as_list = [
        point_as_dict[k] for k in sorted(search_space.keys())
    ]
    return point_as_list


def normalize_dimensions(dimensions):
    """
    Normalize all dimensions.

    This is particularly useful for Gaussian process based regressors and is used
    internally by `gp_minimize.`

    Parameters
    ----------
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

         NOTE: The upper and lower bounds are inclusive for `Integer`
         dimensions.

    * `estimator` ["GP", "RF", "ET", "GBRT", or sklearn regressor, default="GP"]:
        optimization estimator to be used.
        each estimator may use different transformations for dimensions.
    """
    if isinstance(estimator, str):
        estimator = estimator.upper()
        if estimator not in ["GP", "ET", "RF", "GBRT"]:
            raise ValueError("Valid strings for estimator parameter "
                             " are: 'RF, 'ET' or 'GP', not %s" % estimator)
    elif not is_regressor(estimator):
        raise ValueError("estimator has to be a regressor.")

    if estimator == "GP" or estimator == GaussianProcessRegressor:
        dim_types = [check_dimension(d) for d in dimensions]
        is_cat = all([isinstance(check_dimension(d), Categorical)
                      for d in dim_types])
        if is_cat:
            transformed_dims = [check_dimension(d, transform="identity")
                                for d in dimensions]
        else:
            transformed_dims = []
            for dim_type, dim in zip(dim_types, dimensions):
                if isinstance(dim_type, Categorical):
                    transformed_dims.append(
                        check_dimension(dim, transform="onehot")
                        )
                # To make sure that GP operates in the [0, 1] space
                else:
                    transformed_dims.append(
                        check_dimension(dim, transform="normalize")
                        )
    else:
        transformed_dims = dimensions
    return transformed_dims
