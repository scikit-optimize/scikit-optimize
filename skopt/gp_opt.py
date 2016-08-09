"""Gaussian process-based minimization algorithms."""

import copy
import inspect
import numbers
import warnings
from collections import Iterable

import numpy as np

from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import OptimizeResult

from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.utils import check_random_state

from .acquisition import _gaussian_acquisition
from .callbacks import check_callback
from .callbacks import VerboseCallback
from .space import Space
from .utils import create_result


def _acquisition(X, model, y_opt=None, method="LCB", xi=0.01, kappa=1.96):
    """
    A wrapper around the acquisition function that is called by fmin_l_bfgs_b.

    This is because lbfgs allows only 1-D input.
    """
    X = np.expand_dims(X, axis=0)
    return _gaussian_acquisition(X, model, y_opt, method, xi, kappa)


def gp_minimize(func, dimensions, base_estimator=None, alpha=10e-10,
                acq="EI", xi=0.01, kappa=1.96, search="auto", n_calls=100,
                n_points=500, n_random_starts=10, n_restarts_optimizer=5,
                x0=None, y0=None, random_state=None, verbose=False,
                callback=None):
    """Bayesian optimization using Gaussian Processes.

    If every function evaluation is expensive, for instance
    when the parameters are the hyperparameters of a neural network
    and the function evaluation is the mean cross-validation score across
    ten folds, optimizing the hyperparameters by standard optimization
    routines would take for ever!

    The idea is to approximate the function using a Gaussian process.
    In other words the function values are assumed to follow a multivariate
    gaussian. The covariance of the function values are given by a
    GP kernel between the parameters. Then a smart choice to choose the
    next parameter to evaluate can be made by the acquisition function
    over the Gaussian prior which is much quicker to evaluate.

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

    * `base_estimator` [a Gaussian process estimator]:
        The Gaussian process estimator to use for optimization.

    * `alpha` [float, default=1e-10]:
        Value added to the diagonal of the kernel matrix during fitting.
        Larger values correspond to an increased noise level in the
        observations and reduce potential numerical issues during fitting.

    * `acq` [string, default=`"EI"`]:
        Function to minimize over the gaussian prior. Can be either

        - `"LCB"` for lower confidence bound,
        - `"EI"` for expected improvement,
        - `"PI"` for probability of improvement.

    * `xi` [float, default=0.01]:
        Controls how much improvement one wants over the previous best
        values. Used when the acquisition is either `"EI"` or `"PI"`.

    * `kappa` [float, default=1.96]:
        Controls how much of the variance in the predicted values should be
        taken into account. If set to be very high, then we are favouring
        exploration over exploitation and vice versa.
        Used when the acquisition is `"LCB"`.

    * `search` [string, `"auto"`, `"sampling"` or `"lbfgs"`, default=`"auto"`]:
        Searching for the next possible candidate to update the Gaussian prior
        with.

        If search is set to `"auto"`, then it is set to `"lbfgs"`` if
        all the search dimensions are Real(continuous). It defaults to
        `"sampling"` for all other cases.

        If search is set to `"sampling"`, `n_points` are sampled randomly
        and the Gaussian Process prior is updated with the point that gives
        the best acquisition value over the Gaussian prior.

        If search is set to `"lbfgs"`, then a point is sampled randomly, and
        lbfgs is run for 10 iterations optimizing the acquisition function
        over the Gaussian prior.

    * `n_calls` [int, default=100]:
        Number of calls to `func`.

    * `n_points` [int, default=500]:
        Number of points to sample to determine the next "best" point.
        Useless if search is set to `"lbfgs"`.

    * `n_random_starts` [int, default=10]:
        Number of evaluations of `func` with random initialization points
        before approximating the `func` with `base_estimator`.

    * `n_restarts_optimizer` [int, default=10]:
        The number of restarts of the optimizer when `search` is `"lbfgs"`.

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
    # Save call args
    specs = {"args": copy.copy(inspect.currentframe().f_locals),
             "function": inspect.currentframe().f_code.co_name}

    # Check params
    rng = check_random_state(random_state)
    space = Space(dimensions)

    # Default GP
    if base_estimator is None:
        base_estimator = GaussianProcessRegressor(
            kernel=(ConstantKernel(1.0, (0.01, 1000.0)) *
                    Matern(length_scale=np.ones(space.transformed_n_dims),
                           length_scale_bounds=[(0.01, 100)] * space.transformed_n_dims,
                           nu=2.5)),
            normalize_y=True, alpha=alpha, random_state=random_state)

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

    if search == "auto":
        if space.is_real:
            search = "lbfgs"
        else:
            search = "sampling"
    elif search not in ["lbfgs", "sampling"]:
        raise ValueError(
            "Expected search to be 'lbfgs', 'sampling' or 'auto', "
            "got %s" % search)

    # Bayesian optimization loop
    models = []
    n_model_iter = n_calls - n_total_init_calls
    for i in range(n_model_iter):
        gp = clone(base_estimator)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(space.transform(Xi), yi)

        models.append(gp)

        if search == "sampling":
            X = space.transform(space.rvs(n_samples=n_points,
                                          random_state=rng))
            values = _gaussian_acquisition(
                X=X, model=gp,  y_opt=np.min(yi), method=acq,
                xi=xi, kappa=kappa)
            next_x = X[np.argmin(values)]

        elif search == "lbfgs":
            best = np.inf

            for j in range(n_restarts_optimizer):
                x0 = space.transform(space.rvs(n_samples=1,
                                               random_state=rng))[0]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    x, a, _ = fmin_l_bfgs_b(
                        _acquisition, x0,
                        args=(gp, np.min(yi), acq, xi, kappa),
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
