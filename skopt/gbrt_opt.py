"""Gradient boosted trees based minimization algorithms."""

import numpy as np

from scipy import stats
from scipy.optimize import OptimizeResult

from sklearn.base import clone
from sklearn.utils import check_random_state

from .learning import GradientBoostingQuantileRegressor
from .utils import extract_bounds
from .acquisition import gaussian_ei


def _random_points(lower, upper, n_points=1, random_state=None):
    """Sample a random point"""
    rng = check_random_state(random_state)

    num_params = len(lower)
    delta = upper - lower
    return lower + rng.rand(n_points, num_params) * delta


def gbrt_minimize(func, bounds, base_estimator=None, maxiter=100,
                  n_points=20, n_start=10, random_state=None):
    """Sequential optimisation using gradient boosted trees.

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

    * `bounds` [array-like, shape=(n_parameters, 2)]:
        - ``bounds[i][0]`` should give the lower bound of each parameter and
        - ``bounds[i][1]`` should give the upper bound of each parameter.

    * `base_estimator` [`GradientBoostingQuantileRegressor`]:
        The regressor to use as surrogate model

    * `maxiter` [int, default=100]:
        Number of iterations used to find the minimum. This corresponds
        to the total number of evaluations of `func`. If `n_start` > 0
        only `maxiter - n_start` iterations are used.

    * `n_start` [int, default=10]:
        Number of random points to draw before fitting `base_estimator`
        for the first time. If `n_start > maxiter` this degrades to
        a random search for the minimum.

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

        - `x` [float]: location of the minimum.
        - `fun` [float]: function value at the minimum.
        - `models`: surrogate models used for each iteration.
        - `x_iters` [array]: location of function evaluation for each
           iteration.
        - `func_vals` [array]: function value for each iteration.

        For more details related to the OptimizeResult object, refer
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """
    rng = check_random_state(random_state)

    # Bounds
    num_params = len(bounds)
    lower_bounds, upper_bounds = extract_bounds(bounds)

    # Default estimator
    if base_estimator is None:
        base_estimator = GradientBoostingQuantileRegressor(random_state=rng)

    # Record the points and function values evaluated as part of
    # the minimization
    Xi = np.zeros((maxiter, num_params))
    yi = np.zeros(maxiter)

    # Initialize with random points
    if n_start == 0:
        raise ValueError("Need at least one starting point.")

    if maxiter == 0:
        raise ValueError("Need to perform at least one iteration.")

    n_start = min(n_start, maxiter)

    Xi[:n_start] = _random_points(
        lower_bounds, upper_bounds, n_points=n_start, random_state=rng)
    best_x = Xi[:n_start].ravel()
    yi[:n_start] = [func(xi) for xi in Xi[:n_start]]
    best_y = np.min(yi[:n_start])

    models = []

    for i in range(n_start, maxiter):
        rgr = clone(base_estimator)
        # only the first i points are meaningful
        rgr.fit(Xi[:i, :], yi[:i])
        models.append(rgr)

        # `rgr` predicts constants for each leaf which means that the EI
        # has zero gradient over large distances. As a result we can not
        # use gradient based optimisers like BFGS, use random sampling
        # for the moment.
        x0 = _random_points(lower_bounds, upper_bounds,
                            n_points=n_points,
                            random_state=rng)
        best = np.argmax(gaussian_ei(x0, rgr, best_y))

        Xi[i] = x0[best].ravel()
        yi[i] = func(x0[best])

        if yi[i] < best_y:
            best_y = yi[i]
            best_x = Xi[i]

    res = OptimizeResult()
    res.x = best_x
    res.fun = best_y
    res.func_vals = yi
    res.x_iters = Xi
    res.models = models

    return res
