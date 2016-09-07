"""Gaussian process-based minimization algorithms."""

import numpy as np

from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.utils import check_random_state

from .base_opt import base_minimize
from ..space import Space


def gp_minimize(func, dimensions, base_estimator=None,
                acq_func="EI", xi=0.01, kappa=1.96,
                acq_optimizer="auto",
                n_calls=100, n_points=500, n_random_starts=10,
                n_restarts_optimizer=5, x0=None, y0=None,
                random_state=None, verbose=False, callback=None):
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
        By default, a Matern kernel is used with the following
        hyperparameters tuned.
        - All the length scales of the Matern kernel.
        - The covariance amplitude that each element is multiplied with.
        - Noise that is added to the matern kernel. The noise is assumed
          to be iid gaussian.

    * `acq_func` [string, default=`"EI"`]:
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

    * `acq_optimizer` [string, `"auto"`, `"sampling"` or `"lbfgs"`, default=`"auto"`]:
        Searching for the next possible candidate to update the Gaussian prior
        with.

        If acq_optimizer is set to `"auto"`, then it is set to `"lbfgs"`` if
        all the search dimensions are Real(continuous). It defaults to
        `"sampling"` for all other cases.

        If acq_optimizer is set to `"sampling"`, `n_points` are sampled randomly
        and the Gaussian Process prior is updated with the point that gives
        the best acquisition value over the Gaussian prior.

        If acq_optimizer is set to `"lbfgs"`, then a point is sampled randomly, and
        lbfgs is run for 10 iterations optimizing the acquisition function
        over the Gaussian prior.

    * `n_calls` [int, default=100]:
        Number of calls to `func`.

    * `n_points` [int, default=500]:
        Number of points to sample to determine the next "best" point.
        Useless if acq_optimizer is set to `"lbfgs"`.

    * `n_random_starts` [int, default=10]:
        Number of evaluations of `func` with random initialization points
        before approximating the `func` with `base_estimator`.

    * `n_restarts_optimizer` [int, default=5]:
        The number of restarts of the optimizer when `acq_optimizer` is `"lbfgs"`.

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
    # Check params
    rng = check_random_state(random_state)
    space = Space(dimensions)

    # Default GP
    if base_estimator is None:
        cov_amplitude = ConstantKernel(1.0, (0.01, 1000.0))
        matern = Matern(length_scale=np.ones(space.transformed_n_dims),
                        length_scale_bounds=[(0.01, 100)] * space.transformed_n_dims,
                        nu=2.5)
        noise = WhiteKernel()
        base_estimator = GaussianProcessRegressor(
            kernel=cov_amplitude * matern + noise,
            normalize_y=True, random_state=random_state, alpha=0.0)

    return base_minimize(
        func, dimensions, base_estimator=base_estimator,
        acq_func=acq_func,
        xi=xi, kappa=kappa, acq_optimizer=acq_optimizer, n_calls=n_calls,
        n_points=n_points, n_random_starts=n_random_starts,
        n_restarts_optimizer=n_restarts_optimizer,
        x0=x0, y0=y0, random_state=random_state, verbose=verbose,
        callback=callback)
