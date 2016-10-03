"""
Abstraction for optimizers.

It is sufficient that one re-implements the base estimator.
"""

import copy
import inspect
import numbers
import warnings
from collections import Iterable

import numpy as np

from scipy.optimize import fmin_l_bfgs_b

from sklearn.base import clone
from sklearn.base import is_regressor
from sklearn.utils import check_random_state

from ..acquisition import _gaussian_acquisition
from ..callbacks import check_callback
from ..callbacks import VerboseCallback
from ..space import Space
from ..utils import create_result

def _acquisition(X, model, y_opt=None, method="LCB", xi=0.01, kappa=1.96):
    """
    A wrapper around the acquisition function that is called by fmin_l_bfgs_b.

    This is because lbfgs allows only 1-D input.
    """
    X = np.expand_dims(X, axis=0)
    return _gaussian_acquisition(X, model, y_opt, method, xi, kappa)


def base_minimize(func, dimensions, base_estimator,
                  n_calls=100, n_random_starts=10,
                  acq_func="EI", acq_optimizer="auto",
                  x0=None, y0=None, random_state=None, verbose=False,
                  callback=None, n_points=10000, n_restarts_optimizer=5,
                  xi=0.01, kappa=1.96):
    """
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

    * `base_estimator` [sklearn regressor]:
        Should inherit from `sklearn.base.RegressorMixin`.
        In addition, should have an optional `return_std` argument,
        which returns `std(Y | x)`` along with `E[Y | x]`

    * `n_calls` [int, default=100]:
        Number of calls to `func`.

    * `n_random_starts` [int, default=10]:
        Number of evaluations of `func` with random initialization points
        before approximating the `func` with `base_estimator`.

    * `acq_func` [string, default=`"EI"`]:
        Function to minimize over the posterior distribution. Can be either

        - `"LCB"` for lower confidence bound,
        - `"EI"` for negative expected improvement,
        - `"PI"` for negative probability of improvement.

    * `acq_optimizer` [string, `"auto"`, `"sampling"` or `"lbfgs"`, default=`"auto"`]:
        Method to minimize the acquistion function. The fit model
        is updated with the optimal value obtained by optimizing `acq_func`
        with `acq_optimizer`.

        - If set to `"sampling"`, then `acq_func` is optimized by computing
          `acq_func` at `n_points` sampled randomly.
        - If set to `"lbfgs"`, then `acq_func` is optimized by
              - Sampling `n_restarts_optimizer` points randomly.
              - `"lbfgs"` is run for 20 iterations with these points as initial
                points to find local minima.
              - The optimal of these local minima is used to update the prior.
        - If set to `"auto"`, then it is set to `"lbfgs"`` if
          all the search dimensions are Real (continuous). It defaults to
          `"sampling"` for all other cases.

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

    * `n_points` [int, default=500]:
        Number of points to sample to determine the next "best" point.
        Useless if acq_optimizer is set to `"lbfgs"`.

    * `n_restarts_optimizer` [int, default=5]:
        The number of restarts of the optimizer when `acq_optimizer` is `"lbfgs"`.

    * `xi` [float, default=0.01]:
        Controls how much improvement one wants over the previous best
        values. Used when the acquisition is either `"EI"` or `"PI"`.

    * `kappa` [float, default=1.96]:
        Controls how much of the variance in the predicted values should be
        taken into account. If set to be very high, then we are favouring
        exploration over exploitation and vice versa.
        Used when the acquisition is `"LCB"`.

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
        - `specs` [dict]`: the call specifications.
        - `rng` [RandomState instance]: State of the random state
           at the end of minimization.

        For more details related to the OptimizeResult object, refer
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """
    specs = {"args": copy.copy(inspect.currentframe().f_locals),
             "function": inspect.currentframe().f_code.co_name}

    if not is_regressor(base_estimator):
        raise ValueError(
            "%s has to be a regressor." % base_estimator)

    # Check params
    space = Space(dimensions)
    rng = check_random_state(random_state)

    # Initialize with provided points (x0 and y0) and/or random points
    if x0 is None:
        x0 = []
    elif not isinstance(x0[0], list):
        x0 = [x0]

    if not isinstance(x0, list):
        raise ValueError("`x0` should be a list, but got %s" % type(x0))

    n_init_func_calls = len(x0) if y0 is None else 0
    n_total_init_calls = n_random_starts + n_init_func_calls

    if n_calls <= 0:
        raise ValueError("Expected `n_calls` > 0, got %d" % n_calls)

    if n_random_starts < 0:
        raise ValueError(
            "Expected `n_random_starts` >= 0, got %d" % n_random_starts)

    if n_random_starts == 0 and not x0:
        raise ValueError("Either set `n_random_starts` > 0, or provide `x0`")

    if n_calls < n_total_init_calls:
        raise ValueError(
            "Expected `n_calls` >= %d, got %d" % (n_total_init_calls, n_calls))

    callbacks = check_callback(callback)
    if verbose:
        callbacks.append(VerboseCallback(
            n_init=n_init_func_calls, n_random=n_random_starts,
            n_total=n_calls))

    if y0 is None and x0:
        y0 = []
        for i, x in enumerate(x0):
            y0.append(func(x))
            curr_res = create_result(x0[:i + 1], y0, space, rng, specs)

            if callbacks:
                for c in callbacks:
                    c(curr_res)

    elif x0:
        if isinstance(y0, Iterable):
            y0 = list(y0)
        elif isinstance(y0, numbers.Number):
            y0 = [y0]
        else:
            raise ValueError(
                "`y0` should be an iterable or a scalar, got %s" % type(y0))
        if len(x0) != len(y0):
            raise ValueError("`x0` and `y0` should have the same length")
        if not all(map(np.isscalar, y0)):
            raise ValueError(
                "`y0` elements should be scalars")
    else:
        y0 = []

    # Random function evaluations.
    X_rand = space.rvs(n_samples=n_random_starts, random_state=rng)
    Xi = x0 + X_rand
    yi = y0

    for i, x in enumerate(X_rand):
        yi.append(func(x))

        if callbacks is not None:
            curr_res = create_result(
                x0 + X_rand[:i + 1], yi, space, rng, specs)
            if callbacks:
                for c in callbacks:
                    c(curr_res)

    if np.ndim(yi) != 1:
        raise ValueError("`func` should return a scalar")

    if acq_optimizer == "auto":
        if space.is_real:
            acq_optimizer = "lbfgs"
        else:
            acq_optimizer = "sampling"
    elif acq_optimizer not in ["lbfgs", "sampling"]:
        raise ValueError(
            "Expected acq_optimizer to be 'lbfgs', 'sampling' or 'auto', "
            "got %s" % acq_optimizer)

    # Bayesian optimization loop
    models = []
    n_model_iter = n_calls - n_total_init_calls
    for i in range(n_model_iter):
        gp = clone(base_estimator)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(space.transform(Xi), yi)

        models.append(gp)

        if acq_optimizer == "sampling":
            X = space.transform(space.rvs(n_samples=n_points,
                                          random_state=rng))
            values = _gaussian_acquisition(
                X=X, model=gp,  y_opt=np.min(yi), method=acq_func,
                xi=xi, kappa=kappa)
            next_x = X[np.argmin(values)]

        elif acq_optimizer == "lbfgs":
            best = np.inf

            for j in range(n_restarts_optimizer):
                x0 = space.transform(space.rvs(n_samples=1,
                                               random_state=rng))[0]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    x, a, _ = fmin_l_bfgs_b(
                        _acquisition, x0,
                        args=(gp, np.min(yi), acq_func, xi, kappa),
                        bounds=space.transformed_bounds,
                        approx_grad=True, maxiter=20)

                if a < best:
                    next_x, best = x, a

        next_x = space.inverse_transform(next_x.reshape((1, -1)))[0]
        yi.append(func(next_x))
        Xi.append(next_x)
        curr_res = create_result(Xi, yi, space, rng, specs)
        for c in callbacks:
            if callbacks:
                c(curr_res)

    # Pack results
    return create_result(Xi, yi, space, rng, specs, models)
