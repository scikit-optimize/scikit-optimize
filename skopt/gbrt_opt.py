"""Gradient boosted trees based minimization algorithms."""

import numpy as np

from scipy import stats
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import OptimizeResult

from sklearn.base import clone
from sklearn.utils import check_random_state

from .gbt import GradientBoostingQuantileRegressor
from .utils import extract_bounds


def _expected_improvement(X, surrogate, y_opt, xi=0.01):
    """Evaluate expected improvement for `surrogate` model at `x`

    Parameters
    ----------
    X : array-like
        Values at which to evaluate the acquisition function.

    y_opt: float, optional
        The previous best value on which we want to improve.

    xi: float, default 0.01
        By how much must a new point improve over the previous best
        value.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)

    # low and high are assumed to be the 16% and 84% quantiles
    low, mu, high = surrogate.predict(X).T
    # approximate the std dev, if the pdf is gaussian this is exact
    std = (high - low) / 2.

    ei = np.zeros(len(mu))

    mask = std > 0
    improvement = y_opt - xi - mu[mask]
    exploit = improvement * stats.norm.cdf(improvement / std[mask])
    explore = std[mask] * stats.norm.pdf(improvement / std[mask])
    ei[mask] = exploit + explore

    ei = -ei # we are being used in a minimizer
    return ei


def _random_point(lower, upper, n_points=1, random_state=None):
    """Sample a random point"""
    rng = check_random_state(random_state)

    num_params = len(lower)
    delta = upper - lower
    return lower + rng.rand(n_points, num_params) * delta


def gbrt_minimize(func, bounds, base_estimator=None, maxiter=100,
                 random_state=None):
    """Sequential optimisation using gradient boosted trees.

    Gradient boosted regression trees are used to model the (very)
    expensive to evaluate function ``func``. The model is improved
    by sequentially evaluating the expensive function at the next
    best point. Thereby finding the minimum of ``func`` with as
    few evaluations as possible.

    Parameters
    ----------
    func: callable
        Function to minimize. Should take an array of parameters and
        return the function value.

    bounds: array-like, shape (n_parameters, 2)
        ``bounds[i][0]`` should give the lower bound of each parameter and
        ``bounds[i][1]`` should give the upper bound of each parameter.

    base_estimator: a GradientBoostingQuantileRegressor
        The regressor to use as surrogate model

    maxiter: int, default 100
        Number of iterations used to find the minimum.

    random_state: int, RandomState instance, or None (default)
        Set random state to something other than None for reproducible
        results.

    Returns
    -------
    res: OptimizeResult, scipy object
        The optimization result returned as a OptimizeResult object.
        Important attributes are
        ``x`` - float, location of the minimum,
        ``fun`` - float, function value at the minimum,
        ``models``- surrogate models used for each iteration,
        ``x_iters`` - location of function evaluation for each iteration,
        ``func_vals`` - function value for each iteration.
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
    Xi = np.zeros((maxiter + 1, num_params))
    yi = np.zeros(maxiter + 1)

    # Initialize with a random point
    Xi[0] = _random_point(lower_bounds, upper_bounds, random_state=rng)
    best_x = Xi[0].ravel()
    yi[0] = best_y = func(Xi[0])

    models = []

    # XXX should there be an early stopping criterion?
    for i in range(1, maxiter + 1):
        rgr = clone(base_estimator)
        # only the first i points are meaningful
        rgr.fit(Xi[:i, :], yi[:i])
        models.append(rgr)

        # `rgr` predicts constants for each leaf which means that the EI
        # has zero gradient over large distances. As a result we can not
        # use gradient based optimisers like BFGS, use random sampling
        # for the moment.
        x0 = _random_point(lower_bounds, upper_bounds,
                           n_points=20,
                           random_state=rng)
        aq = _expected_improvement(x0, rgr, best_y)
        best = np.argmin(aq)

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
