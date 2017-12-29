# TODO: This doc-string is way too short and quite meaningless.
"""
Abstraction for optimizers.

It is sufficient that one re-implements the base estimator.
"""

from copy import copy
import inspect
import numbers
from collections import Iterable

import numpy as np

from ..callbacks import check_callback
from ..callbacks import VerboseCallback
from .optimizer import Optimizer
from ..utils import eval_callbacks


class _WrapFuncNamedArgs:
    """
    Wrapper for the objective function. This calls the objective
    function `func()` with named arguments instead of a list of parameters.

    For example, instead of calling `func([123, 3.0, 'hello'])`,
    it calls `func(foo=123, bar=3.0, baz='hello')` for a search-space
    with dimensions named `['foo', 'bar', 'baz']`
    """

    def __init__(self, func, dimensions):
        """Create a callable object instance that wraps an objective function.

        Parameters
        ----------
        * `func` [callable]:
            Function to minimize. Takes a list of parameters and
            returns the objective or fitness value.

        * `dimensions` [list of Dimension instances]:
            List of search-space dimensions. These must all be named.
            Use a list of instances of `Real`, `Categorical` or `Integer`
            from `skopt.space`.
        """

        # Set the objective function.
        self.func = func

        # Get the names of all search-space dimensions.
        self.dim_names = [dim.name for dim in dimensions]

        # Ensure all dimension names exist.
        if None in self.dim_names:
            raise ValueError("All dimensions must have names when `use_arg_names=True`.")

    def __call__(self, x):
        """This makes the object callable so it seamlessly wraps `func(x)`.

        Parameters
        ----------
        * `x` [list]:
            Parameters for a location in the search-space.
            
        Returns
        -------
        * `fitness` [float]:
            Fitness of the objective function at the given location `x`.
        """

        # Create a dictionary with the named parameters.
        x_dict = {name: value for name, value in zip(self.dim_names, x)}

        # Call the objective function with these named arguments.
        fitness = self.func(**x_dict)

        return fitness


# TODO: There is no doc-string here! What does this function do?!
def base_minimize(func, dimensions, base_estimator,
                  use_arg_names=False,
                  n_calls=100, n_random_starts=10,
                  acq_func="EI", acq_optimizer="lbfgs",
                  x0=None, y0=None, random_state=None, verbose=False,
                  callback=None, n_points=10000, n_restarts_optimizer=5,
                  xi=0.01, kappa=1.96, n_jobs=1):
    """
    Parameters
    ----------
    * `func` [callable]:
        Function to minimize. Should take a array of parameters and
        return the function values.

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

    * `use_arg_names` [bool, default=False]:
        Whether to use use names for the search-space dimensions
        when calling `func()`, so instead of calling `func(x)` 
        with a single list `x`, it calls e.g.
        `func(foo=123, bar=3.0, baz='hello')` for a search-space
         with dimensions named `['foo', 'bar', 'baz']`

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

        For more details related to the OptimizeResult object, refer to:
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """

    # TODO: Please clean up and comment the rest of this code.
    # TODO: It is VERY hard to understand and there appears to be tautologies,
    # TODO: which indicates that several people have worked on this and they
    # TODO: also don't understand it and just keep adding code to fix bugs etc.

    # Wrap the objective function if it uses dimension-names as arguments.
    if use_arg_names:
        func = _WrapFuncNamedArgs(func=func, dimensions=dimensions)

    specs = {"args": copy(inspect.currentframe().f_locals),
             "function": inspect.currentframe().f_code.co_name}

    acq_optimizer_kwargs = {
        "n_points": n_points, "n_restarts_optimizer": n_restarts_optimizer,
        "n_jobs": n_jobs}
    acq_func_kwargs = {"xi": xi, "kappa": kappa}

    # TODO: These manipulations to x0 and y0 are VERY confusing!
    # TODO: And further below there is a check: "if x0 is not None"
    # TODO: but that is apparently always True after this.
    # Initialize with provided points (x0 and y0) and/or random points
    if x0 is None:
        x0 = []
    elif not isinstance(x0[0], (list, tuple)):
        x0 = [x0]

    if not isinstance(x0, list):
        raise ValueError("`x0` should be a list, but got %s" % type(x0))

    if n_random_starts == 0 and not x0:
        raise ValueError("Either set `n_random_starts` > 0,"
                         " or provide `x0`")

    if isinstance(y0, Iterable):
        y0 = list(y0)
    elif isinstance(y0, numbers.Number):
        y0 = [y0]

    # Is the budget for calling `func` large enough?
    required_calls = n_random_starts + (len(x0) if not y0 else 0)
    if n_calls < required_calls:
        raise ValueError(
            "Expected `n_calls` >= %d, got %d" % (required_calls, n_calls))

    # Number of points the user wants to evaluate
    # before it makes sense to fit a surrogate model.
    n_initial_points = n_random_starts + len(x0)

    optimizer = Optimizer(dimensions=dimensions,
                          base_estimator=base_estimator,
                          n_initial_points=n_initial_points,
                          acq_func=acq_func, acq_optimizer=acq_optimizer,
                          random_state=random_state,
                          acq_optimizer_kwargs=acq_optimizer_kwargs,
                          acq_func_kwargs=acq_func_kwargs)

    assert all(isinstance(p, Iterable) for p in x0)

    if not all(len(p) == optimizer.space.n_dims for p in x0):
        raise RuntimeError("Optimization space (%s) and initial points in x0 "
                           "use inconsistent dimensions." % optimizer.space)

    callbacks = check_callback(callback)
    if verbose:
        callbacks.append(VerboseCallback(
            n_init=len(x0) if not y0 else 0,
            n_random=n_random_starts,
            n_total=n_calls))

    # setting the scope for these variables
    result = None

    # TODO: I tried to clean this up a little, but it is VERY confusing.
    # Pass user suggested initialisation points to the optimizer.
    if x0 is not None:
        # If objective values are not given for these points, calculate them.
        if y0 is None:
            # Calculate the objective function for each point in x0.
            y0 = list(map(func, x0))

            # Decrease the required number of function calls.
            n_calls -= len(y0)

        if not (isinstance(y0, Iterable) or isinstance(y0, numbers.Number)):
            raise ValueError(
                "`y0` should be an iterable or a scalar, got %s" % type(y0))

        if len(x0) != len(y0):
            raise ValueError("`x0` and `y0` should have the same length")

        result = optimizer.tell(x0, y0)
        result.specs = specs

        if eval_callbacks(callbacks, result):
            return result

    # Bayesian optimization loop.
    for n in range(n_calls):
        # Get next point in the search-space.
        next_x = optimizer.ask()

        # Call the objective function with this point.
        next_y = func(next_x)

        # Report the value of the objective function at this point.
        result = optimizer.tell(next_x, next_y)

        # TODO: What is this? Please doc this and everything else.
        result.specs = specs

        if eval_callbacks(callbacks, result):
            break

    return result
