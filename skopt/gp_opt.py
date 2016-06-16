import numpy as np
import warnings

from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import OptimizeResult

from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.utils import check_random_state

from .acquisition import _gaussian_acquisition
from .space import Space


def _acquisition(X, model, y_opt=None, method="LCB", xi=0.01, kappa=1.96):
    """
    A wrapper around the acquisition function that is called by fmin_l_bfgs_b.

    This is because lbfgs allows only 1-D input.
    """
    X = np.expand_dims(X, axis=0)
    return _gaussian_acquisition(X, model, y_opt, method, xi, kappa)


def gp_minimize(func, dimensions, base_estimator=None, acq="LCB", xi=0.01,
                kappa=1.96, search="sampling", maxiter=1000, n_points=500,
                n_start=10, n_restarts_optimizer=5, random_state=None):
    """Bayesian optimization using Gaussian Processes.

    If every function evaluation is expensive, for instance
    when the parameters are the hyperparameters of a neural network
    and the function evaluation is the mean cross-validation score across
    ten folds, optimizing the hyperparameters by standared optimization
    routines would take for ever!

    The idea is to approximate the function using a Gaussian process.
    In other words the function values are assumed to follow a multivariate
    gaussian. The covariance of the function values are given by a
    GP kernel between the parameters. Then a smart choice to choose the
    next parameter to evaluate can be made by the acquisition function
    over the Gaussian prior which is much quicker to evaluate.

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

    * `acq` [string, default=`"LCB"`]:
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

    * `search` [string, `"sampling"` or `"lbfgs"`]:
        Searching for the next possible candidate to update the Gaussian prior
        with.

        If search is set to `"sampling"`, `n_points` are sampled randomly
        and the Gaussian Process prior is updated with the point that gives
        the best acquisition value over the Gaussian prior.

        If search is set to `"lbfgs"`, then a point is sampled randomly, and
        lbfgs is run for 10 iterations optimizing the acquisition function
        over the Gaussian prior.

    * `maxiter` [int, default=1000]:
        Number of iterations to find the minimum. Note that `n_start`
        iterations are effectively discounted, such that total number of
        function evaluations is at most `maxiter`.

    * `n_points` [int, default=500]:
        Number of points to sample to determine the next "best" point.
        Useless if search is set to `"lbfgs"`.

    * `n_start` [int, default=10]:
        Number of random initialization points.

    * `n_restarts_optimizer` [int, default=10]:
        The number of restarts of the optimizer when `search` is `"lbfgs"`.

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
    space = Space(dimensions)

    # Default GP
    if base_estimator is None:
        base_estimator = GaussianProcessRegressor(
            kernel=(ConstantKernel(1.0, (0.01, 1000.0)) *
                    Matern(length_scale=np.ones(space.transformed_n_dims),
                           length_scale_bounds=[(0.01, 100)] * space.transformed_n_dims,
                           nu=2.5)),
            normalize_y=True, alpha=10e-6, random_state=random_state)

    # First points
    Xi = space.rvs(n_samples=n_start, random_state=rng)
    yi = [func(x) for x in Xi]
    if np.ndim(yi) != 1:
        raise ValueError(
            "The function to be optimized should return a scalar")

    # Bayesian optimization loop
    models = []

    for i in range(maxiter - n_start):
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
                        approx_grad=True, maxiter=10)

                if a < best:
                    next_x, best = x, a

        next_x = space.inverse_transform(next_x.reshape((1, -1)))[0]
        next_y = func(next_x)
        Xi = np.vstack((Xi, next_x))
        yi.append(next_y)

    # Pack results
    res = OptimizeResult()
    best = np.argmin(yi)
    res.x = Xi[best]
    res.fun = yi[best]
    res.func_vals = np.array(yi)
    res.x_iters = Xi
    res.models = models

    return res
