"""Tree based minimization algorithms."""

import numpy as np

from scipy.optimize import OptimizeResult

from sklearn.base import clone
from sklearn.base import is_regressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import check_random_state

from .acquisition import gaussian_ei
from .learning import DecisionTreeRegressor
from .learning import ExtraTreesRegressor
from .learning import GradientBoostingQuantileRegressor
from .learning import RandomForestRegressor
from .space import Space


def _tree_minimize(func, dimensions, base_estimator, n_calls,
                   n_points, n_random_starts, random_state=None):
    rng = check_random_state(random_state)
    space = Space(dimensions)

    # Initialize with random points
    if n_random_starts <= 0:
        raise ValueError(
            "Expected n_random_starts > 0, got %d" % n_random_starts)

    if n_calls <= 0:
        raise ValueError(
            "Expected n_calls > 0, got %d" % n_random_starts)

    if n_calls < n_random_starts:
        raise ValueError(
            "Expected n_calls >= %d, got %d" % (n_random_starts, n_calls))

    Xi = space.rvs(n_samples=n_random_starts, random_state=rng)
    yi = [func(x) for x in Xi]
    if np.ndim(yi) != 1:
        raise ValueError(
            "The function to be optimized should return a scalar")

    # Tree-based optimization loop
    models = []

    n_model_iter = n_calls - n_random_starts
    for i in range(n_model_iter):
        rgr = clone(base_estimator)
        rgr.fit(space.transform(Xi), yi)
        models.append(rgr)

        # `rgr` predicts constants for each leaf which means that the EI
        # has zero gradient over large distances. As a result we can not
        # use gradient based optimizers like BFGS, so using random sampling
        # for the moment.
        X = space.transform(space.rvs(n_samples=n_points,
                                      random_state=rng))
        values = -gaussian_ei(X, rgr, np.min(yi))
        next_x = X[np.argmin(values)]

        next_x = space.inverse_transform(next_x.reshape((1, -1)))[0]
        next_y = func(next_x)
        Xi.append(next_x)
        yi.append(next_y)

    res = OptimizeResult()
    best = np.argmin(yi)
    res.x = Xi[best]
    res.fun = yi[best]
    res.func_vals = np.array(yi)
    res.x_iters = Xi
    res.models = models
    res.space = space

    return res


def gbrt_minimize(func, dimensions, base_estimator=None, n_calls=100,
                  n_points=20, n_random_starts=10, random_state=None):
    """Sequential optimization using gradient boosted trees.

    Gradient boosted regression trees are used to model the (very)
    expensive to evaluate function `func`. The model is improved
    by sequentially evaluating the expensive function at the next
    best point. Thereby finding the minimum of `func` with as
    few evaluations as possible.

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

    * `base_estimator` [`GradientBoostingQuantileRegressor`]:
        The regressor to use as surrogate model

    * `n_calls` [int, default=100]:
        Number of calls to `func`.
        If `n_random_starts` > 0, `n_calls - n_random_starts`
        additional evaluations of `func` are made that are guided
        by the `base_estimator`.

    * `n_random_starts` [int, default=10]:
        Number of evaluations of `func` with random initialization points
        before approximating the `func` with `base_estimator`.

    * `n_points` [int, default=20]:
        Number of points to sample when minimizing the acquisition function.

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
        - `models`: surrogate models used for each iteration.
        - `x_iters` [list of lists]: location of function evaluation for each
           iteration.
        - `func_vals` [array]: function value for each iteration.
        - `space` [Space]: the optimization space.

        For more details related to the OptimizeResult object, refer
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """
    rng = check_random_state(random_state)

    # Default estimator
    if base_estimator is None:
        gbrt = GradientBoostingRegressor(n_estimators=20, loss='quantile')
        base_estimator = GradientBoostingQuantileRegressor(base_estimator=gbrt,
                                                           random_state=rng)

    return _tree_minimize(func, dimensions, base_estimator,
                          n_calls=n_calls,
                          n_points=n_points, n_random_starts=n_random_starts,
                          random_state=random_state)


def forest_minimize(func, dimensions, base_estimator='et', n_calls=100,
                    n_points=10000, n_random_starts=10, random_state=None):
    """Sequential optimization using decision trees.

    A tree based regression model is used to model the expensive to evaluate
    function `func`. The model is improved by sequentially evaluating
    the expensive function at the next best point. Thereby finding the
    minimum of `func` with as few evaluations as possible.

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

    * `base_estimator` [string or `Regressor`, default=`"et"`]:
        The regressor to use as surrogate model. Can be either

        - `"rf"` for random forest regressor
        - `"et"` for extra trees regressor
        - `"dt"` for single decision tree regressor
        - instance of regressor with support for `return_std` in its predict
          method

        The predefined models are initilized with good defaults. If you
        want to adjust the model parameters pass your own instance of
        a regressor which returns the mean and standard deviation when
        making predictions.

    * `n_calls` [int, default=100]:
        Number of calls to `func`.
        If `n_random_starts` > 0, `n_calls - n_random_starts`
        additional evaluations of `func` are made that are guided
        by the `base_estimator`.

    * `n_random_starts` [int, default=10]:
        Number of evaluations of `func` with random initialization points
        before approximating the `func` with `base_estimator`.

    * `n_points` [int, default=10000]:
        Number of points to sample when minimizing the acquisition function.

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
        - `models`: surrogate models used for each iteration.
        - `x_iters` [list of lists]: location of function evaluation for each
           iteration.
        - `func_vals` [array]: function value for each iteration.
        - `space` [Space]: the optimization space.

        For more details related to the OptimizeResult object, refer
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """
    rng = check_random_state(random_state)

    if isinstance(base_estimator, str):
        if base_estimator not in ("rf", "et", "dt"):
            raise ValueError("Valid values for the base_estimator parameter"
                             " are: 'rf', 'et' or 'dt', not '%s'" % base_estimator)

        if base_estimator == "rf":
            base_estimator = RandomForestRegressor(n_estimators=100,
                                                   min_samples_leaf=3,
                                                   random_state=rng)

        elif base_estimator == "et":
            base_estimator = ExtraTreesRegressor(n_estimators=100,
                                                 min_samples_leaf=3,
                                                 random_state=rng)

        elif base_estimator == "dt":
            base_estimator = DecisionTreeRegressor(min_samples_leaf=3,
                                                   random_state=rng)

    else:
        if not is_regressor(base_estimator):
            raise ValueError("The base_estimator parameter has to either"
                             " be a string or a regressor instance."
                             " '%s' is neither." % base_estimator)

    return _tree_minimize(func, dimensions, base_estimator,
                          n_calls=n_calls,
                          n_points=n_points, n_random_starts=n_random_starts,
                          random_state=random_state)
