"""
Abstraction for optimizers.

It is sufficient that one re-implements the base estimator.
"""

import copy
import inspect
import numbers
from collections import Iterable

import numpy as np

from ..callbacks import check_callback
from ..callbacks import VerboseCallback
from .optimizer import Optimizer
from ..utils import eval_callbacks


def base_minimize(func, dimensions, base_estimator,
                  n_calls=100, n_random_starts=10,
                  acq_func="EI", acq_optimizer="lbfgs",
                  x0=None, y0=None, random_state=None, verbose=False,
                  callback=None, n_points=10000, n_restarts_optimizer=5,
                  xi=0.01, kappa=1.96, n_jobs=1):
    """
    Parameters
    ----------
    * `func` [callable]:
        Function to minimize. Should take a single list of parameters
        and return the objective value.
    
        If you have a search-space where all dimensions have names,
        then you can use `skopt.utils.use_named_args` as a decorator
        on your objective function, in order to call it directly
        with the named arguments. See `use_named_args` for an example.

    * `dimensions` [list, shape=(n_dims,)]:
        List of search space dimensions.
        Each search dimension can be defined either as

        - a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer`
          dimensions),
        - a `(lower_bound, upper_bound, "prior")` tuple (for `Real`
          dimensions),
        - as a list of categories (for `Categorical` dimensions), or
        - an instance of a `Dimension` object (`Real`, `Integer` or
          `Categorical`).

         NOTE: The upper and lower bounds are inclusive for `Integer`
         dimensions.

    * `base_estimator` [sklearn regressor]:
        Should inherit from `sklearn.base.RegressorMixin`.
        In addition, should have an optional `return_std` argument,
        which returns `std(Y | x)`` along with `E[Y | x]`.

    * `n_calls` [int, default=100]:
        Maximum number of calls to `func`.

    * `n_random_starts` [int, default=10]:
        Number of evaluations of `func` with random points before
        approximating it with `base_estimator`.

    * `acq_func` [string, default=`"EI"`]:
        Function to minimize over the posterior distribution. Can be either

        - `"LCB"` for lower confidence bound,
        - `"EI"` for negative expected improvement,
        - `"PI"` for negative probability of improvement.
        - `"EIps" for negated expected improvement per second to take into
          account the function compute time. Then, the objective function is
          assumed to return two values, the first being the objective value and
          the second being the time taken in seconds.
        - `"PIps"` for negated probability of improvement per second. The
          return type of the objective function is assumed to be similar to
          that of `"EIps

    * `acq_optimizer` [string, `"sampling"` or `"lbfgs"`, default=`"lbfgs"`]:
        Method to minimize the acquistion function. The fit model
        is updated with the optimal value obtained by optimizing `acq_func`
        with `acq_optimizer`.

        - If set to `"sampling"`, then `acq_func` is optimized by computing
          `acq_func` at `n_points` randomly sampled points and the smallest
          value found is used.
        - If set to `"lbfgs"`, then
              - The `n_restarts_optimizer` no. of points which the acquisition
                function is least are taken as start points.
              - `"lbfgs"` is run for 20 iterations with these points as initial
                points to find local minima.
              - The optimal of these local minima is used to update the prior.

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

    * `n_points` [int, default=10000]:
        If `acq_optimizer` is set to `"sampling"`, then `acq_func` is
        optimized by computing `acq_func` at `n_points` randomly sampled
        points.

    * `n_restarts_optimizer` [int, default=5]:
        The number of restarts of the optimizer when `acq_optimizer`
        is `"lbfgs"`.

    * `xi` [float, default=0.01]:
        Controls how much improvement one wants over the previous best
        values. Used when the acquisition is either `"EI"` or `"PI"`.

    * `kappa` [float, default=1.96]:
        Controls how much of the variance in the predicted values should be
        taken into account. If set to be very high, then we are favouring
        exploration over exploitation and vice versa.
        Used when the acquisition is `"LCB"`.

    * `n_jobs` [int, default=1]:
        Number of cores to run in parallel while running the lbfgs
        optimizations over the acquisition function. Valid only when
        `acq_optimizer` is set to "lbfgs."
        Defaults to 1 core. If `n_jobs=-1`, then number of jobs is set
        to number of cores.

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

    acq_optimizer_kwargs = {
        "n_points": n_points, "n_restarts_optimizer": n_restarts_optimizer,
        "n_jobs": n_jobs}
    acq_func_kwargs = {"xi": xi, "kappa": kappa}

    # Initialize optimization
    # Suppose there are points provided (x0 and y0), record them

    # check x0: list-like, requirement of minimal points
    if x0 is None:
        x0 = []
    elif not isinstance(x0[0], (list, tuple)):
        x0 = [x0]
    if not isinstance(x0, list):
        raise ValueError("`x0` should be a list, but got %s" % type(x0))
    if n_random_starts <= 0 and not x0:
        raise ValueError("Either set `n_random_starts` > 0,"
                         " or provide `x0`")
    # check y0: list-like, requirement of maximal calls
    if isinstance(y0, Iterable):
        y0 = list(y0)
    elif isinstance(y0, numbers.Number):
        y0 = [y0]
    required_calls = n_random_starts + (len(x0) if not y0 else 0)
    if n_calls < required_calls:
        raise ValueError(
            "Expected `n_calls` >= %d, got %d" % (required_calls, n_calls))
    # calculate the total number of initial points
    n_initial_points = n_random_starts + len(x0)

    # Build optimizer

    # create optimizer class
    optimizer = Optimizer(dimensions, base_estimator,
                          n_initial_points=n_initial_points,
                          acq_func=acq_func, acq_optimizer=acq_optimizer,
                          random_state=random_state,
                          acq_optimizer_kwargs=acq_optimizer_kwargs,
                          acq_func_kwargs=acq_func_kwargs)
    # check x0: element-wise data type, dimensionality
    assert all(isinstance(p, Iterable) for p in x0)
    if not all(len(p) == optimizer.space.n_dims for p in x0):
        raise RuntimeError("Optimization space (%s) and initial points in x0 "
                           "use inconsistent dimensions." % optimizer.space)
    # check callback
    callbacks = check_callback(callback)
    if verbose:
        callbacks.append(VerboseCallback(
            n_init=len(x0) if not y0 else 0,
            n_random=n_random_starts,
            n_total=n_calls))

    # Record provided points

    # create return object
    result = None
    # evaluate y0 if only x0 is provided
    if x0 and y0 is None:
        y0 = list(map(func, x0))
        n_calls -= len(y0)
    # record through tell function
    if x0:
        if not (isinstance(y0, Iterable) or isinstance(y0, numbers.Number)):
            raise ValueError(
                "`y0` should be an iterable or a scalar, got %s" % type(y0))
        if len(x0) != len(y0):
            raise ValueError("`x0` and `y0` should have the same length")
        result = optimizer.tell(x0, y0)
        result.specs = specs
        if eval_callbacks(callbacks, result):
            return result

    # Optimize
    for n in range(n_calls):
        next_x = optimizer.ask()
        next_y = func(next_x)
        result = optimizer.tell(next_x, next_y)
        result.specs = specs
        if eval_callbacks(callbacks, result):
            break

    return result
