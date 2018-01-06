
from sklearn.utils import check_random_state

from .base import base_minimize
from ..utils import cook_estimator


def gbrt_minimize(func, dimensions, base_estimator=None,
                  n_calls=100, n_random_starts=10,
                  acq_func="EI", acq_optimizer="auto",
                  x0=None, y0=None, random_state=None, verbose=False,
                  callback=None, n_points=10000, xi=0.01, kappa=1.96,
                  n_jobs=1):
    """Sequential optimization using gradient boosted trees.

    Gradient boosted regression trees are used to model the (very)
    expensive to evaluate function `func`. The model is improved
    by sequentially evaluating the expensive function at the next
    best point. Thereby finding the minimum of `func` with as
    few evaluations as possible.

    The total number of evaluations, `n_calls`, are performed like the
    following. If `x0` is provided but not `y0`, then the elements of `x0`
    are first evaluated, followed by `n_random_starts` evaluations.
    Finally, `n_calls - len(x0) - n_random_starts` evaluations are
    made guided by the surrogate model. If `x0` and `y0` are both
    provided then `n_random_starts` evaluations are first made then
    `n_calls - n_random_starts` subsequent evaluations are made
    guided by the surrogate model.

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

    * `base_estimator` [`GradientBoostingQuantileRegressor`]:
        The regressor to use as surrogate model

    * `n_calls` [int, default=100]:
        Number of calls to `func`.

    * `n_random_starts` [int, default=10]:
        Number of evaluations of `func` with random points before
        approximating it with `base_estimator`.

    * `acq_func` [string, default=`"LCB"`]:
        Function to minimize over the forest posterior. Can be either

        - `"LCB"` for lower confidence bound.
        - `"EI"` for negative expected improvement.
        - `"PI"` for negative probability of improvement.
        - ``"EIps"`` for negated expected improvement per second to take into
          account the function compute time. Then, the objective function is
          assumed to return two values, the first being the objective value and
          the second being the time taken.
        - `"PIps"` for negated probability of improvement per second.

    * `x0` [list, list of lists or `None`]:
        Initial input points.

        - If it is a list of lists, use it as a list of input points.
        - If it is a list, use it as a single initial input point.
        - If it is `None`, no initial input points are used.

    * `y0` [list, scalar or `None`]:
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

    * `callback` [callable, optional]
        If provided, then `callback(res)` is called after call to func.

    * `n_points` [int, default=10000]:
        Number of points to sample when minimizing the acquisition function.

    * `xi` [float, default=0.01]:
        Controls how much improvement one wants over the previous best
        values. Used when the acquisition is either `"EI"` or `"PI"`.

    * `kappa` [float, default=1.96]:
        Controls how much of the variance in the predicted values should be
        taken into account. If set to be very high, then we are favouring
        exploration over exploitation and vice versa.
        Used when the acquisition is `"LCB"`.

    * `n_jobs` [int, default=1]:
        The number of jobs to run in parallel for `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

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
    # Check params
    rng = check_random_state(random_state)

    if base_estimator is None:
        base_estimator = cook_estimator("GBRT", random_state=rng,
                                        n_jobs=n_jobs)
    return base_minimize(func, dimensions, base_estimator,
                         n_calls=n_calls, n_points=n_points,
                         n_random_starts=n_random_starts,
                         x0=x0, y0=y0, random_state=random_state, xi=xi,
                         kappa=kappa, acq_func=acq_func, verbose=verbose,
                         callback=callback, acq_optimizer="sampling")
