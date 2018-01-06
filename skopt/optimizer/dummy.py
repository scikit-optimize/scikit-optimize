"""Random search."""

from .base import base_minimize


def dummy_minimize(func, dimensions, n_calls=100, x0=None, y0=None,
                   random_state=None, verbose=False, callback=None):
    """Random search by uniform sampling within the given bounds.

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
        - a `(lower_bound, upper_bound, prior)` tuple (for `Real`
          dimensions),
        - as a list of categories (for `Categorical` dimensions), or
        - an instance of a `Dimension` object (`Real`, `Integer` or
          `Categorical`).

    * `n_calls` [int, default=100]:
        Number of calls to `func` to find the minimum.

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

    * `callback` [callable, list of callables, optional]
        If callable then `callback(res)` is called after each call to `func`.
        If list of callables, then each callable in the list is called.

    Returns
    -------
    * `res` [`OptimizeResult`, scipy object]:
        The optimization result returned as a OptimizeResult object.
        Important attributes are:

        - `x` [list]: location of the minimum.
        - `fun` [float]: function value at the minimum.
        - `x_iters` [list of lists]: location of function evaluation for each
           iteration.
        - `func_vals` [array]: function value for each iteration.
        - `space` [Space]: the optimisation space.
        - `specs` [dict]: the call specifications.
        - `rng` [RandomState instance]: State of the random state
           at the end of minimization.

        For more details related to the OptimizeResult object, refer
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """
    # all our calls want random suggestions, except if we need to evaluate
    # some initial points
    if x0 is not None and y0 is None:
        n_random_calls = n_calls - len(x0)
    else:
        n_random_calls = n_calls

    return base_minimize(func, dimensions, base_estimator="dummy",
                         # explicitly set optimizer to sampling as "dummy"
                         # minimizer does not provide gradients.
                         acq_optimizer="sampling",
                         n_calls=n_calls, n_random_starts=n_random_calls,
                         x0=x0, y0=y0, random_state=random_state,
                         verbose=verbose,
                         callback=callback)
