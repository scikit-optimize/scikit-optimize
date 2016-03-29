import numpy as np

from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.utils import check_random_state

from scipy import stats
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import OptimizeResult


def acquisition(X, model, x_opt=None, method="UCB", xi=0.01, kappa=1.96,
                bounds=None):
    """
    Returns the acquistion function computed at values x0, when the
    posterior of the unknown function is approximated by a Gaussian process.

    Parameters
    ----------
    X : array-like
        Values where the acquistion function should be computed.

    model: sklearn estimator that implements predict with ``return_std``
        The fit sklearn gaussian process estimator that approximates
        the function. It should have a ``return_std`` parameter
        that returns the standard deviation.

    x_opt: float, optional
        The previous best value over which we want to improve.
        Useful only when `acq` is set to "EI"

    method: string, "UCB" or "EI"
        If set to "UCB", then the lower confidence bound is taken.
        If set to "EI", then the expected improvement condition is taken.

    xi: float, default 0.01
        Controls how much improvement one wants over the previous best
        values.

    kappa: float, default 1.96
        Controls how much of the variance in the predicted values should be
        taken into account. If set to be very high, then we are favouring
        exploration over exploitation and vice versa.
        Useless if acq is set to "EI"

    Returns
    -------
    values: array-like, length X
        Acquisition function values computed at X.
    """
    # Check inputs
    X = np.asarray(X)
    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)

    # Rescale to between 0 and 1.
    if bounds is not None:
        lower_bounds, upper_bounds = zip(*bounds)
        lower_bounds = np.asarray(lower_bounds)
        upper_bounds = np.asarray(upper_bounds)
        X = (X - lower_bounds) / (upper_bounds - lower_bounds)

    # Compute posterior
    mu, std = model.predict(X, return_std=True)

    # Evaluate acquisition function
    if method == "UCB":
        values = mu - kappa * std

    elif method == "EI":
        improvement = x_opt - mu
        exploit = improvement * stats.norm.cdf(improvement / std)
        explore = std * stats.norm.pdf(improvement / std)
        values = exploit + explore

    else:
        raise ValueError("Acquisition function not implemented yet: " + method)

    # Return
    if values.shape == (1, 1):
        return values[0]

    return values


def gp_minimize(func, bounds, base_estimator=None, acq="UCB", xi=0.01,
                kappa=1.96, search="sampling", maxiter=1000, num_points=500,
                random_state=None):
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
    next parameter to evaluate can be made by the acquistion function
    over the Gaussian posterior which is much quicker to evaluate.

    Parameters
    ----------
    func: callable
        Function to minimize. Should take a array of parameters and
        return the function value.

    bounds: array-like, shape (n_parameters, 2)
        ``bounds[i][0]`` should give the lower bound of each parameter and
        ``bounds[i][1]`` should give the upper bound of each parameter.

    base_estimator: a Gaussian process estimator
        The Gaussian process estimator to use for optimization.

    acq: string, default "UCB"
        Function to minimize over the gaussian posterior. Can be either
        the "UCB" which refers to the Upper Confidence Bound or "EI" which
        is the Expected Improvement.

    xi: float, default 0.01
        Controls how much improvement one wants over the previous best
        values.

    kappa: float, default 1.96
        Controls how much of the variance in the predicted values should be
        taken into account. If set to be very high, then we are favouring
        exploration over exploitation and vice versa.
        Useless if acq is set to "EI"

    search: string, "sampling" or "lbfgs"
        Searching for the next possible candidate to update the Gaussian prior
        with.

        If search is set to "sampling", ``num_points`` are sampled randomly
        and the Gaussian Process prior is updated with that point that gives
        the best acquision value over the Gaussian posterior.

        If search is set to "lbfgs", then a point is sampled randomly, and
        lbfgs is run for 10 iterations optimizing the acquistion function
        over the Gaussian posterior.

    maxiter: int, default 1000
        Number of iterations to find the minimum. In other words, the
        number of function evaluations.

    num_points: int, default 500
        Number of points to sample to determine the next "best" point.
        Useless if search is set to "lbfgs".

    random_state: int, RandomState instance, or None (default)
        Set random state to something other than None for reproducible
        results.

    Returns
    -------
    res: OptimizeResult, scipy object
        The optimization result returned as a OptimizeResult object.
        Important attributes are
        ``x`` - float, the optimization solution,
        ``fun`` - float, the value of the function at the optimum,
        ``models``- gp_models[i], the posterior on the function fit at
                       iteration[i].
        ``func_vals`` - the function value at the ith iteration.
        ``x_iters`` - the value of ``x`` corresponding to the function value
                      at the ith iteration.
        For more details related to the OptimizeResult object, refer
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """
    rng = check_random_state(random_state)

    # Bounds
    num_params = len(bounds)
    lower_bounds, upper_bounds = zip(*bounds)
    upper_bounds = np.asarray(upper_bounds)
    lower_bounds = np.asarray(lower_bounds)
    lbfgs_bounds = np.tile((0, 1), (num_params, 1))

    # Default GP
    if base_estimator is None:
        base_estimator = GaussianProcessRegressor(
            kernel=Matern(length_scale=np.ones(num_params), nu=2.5),
            normalize_y=True, random_state=random_state)

    # First point
    x0 = rng.rand(num_params)
    Xi = np.reshape(x0, (1, -1))
    yi = [func(lower_bounds + (upper_bounds - lower_bounds) * x0)]

    # Bayesian optimization loop
    models = []

    for i in range(maxiter):
        gp = clone(base_estimator)
        gp.fit(Xi, yi)
        models.append(gp)

        if search == "sampling":
            X = rng.rand(num_points, num_params)
            values = acquisition(X=X, model=gp,
                                 x_opt=np.min(yi), method=acq,
                                 xi=xi, kappa=kappa)
            next_x = X[np.argmin(values)]

        elif search == "lbfgs":
            x0 = rng.rand(num_params)
            next_x, _, _ = fmin_l_bfgs_b(
                acquisition,
                np.asfortranarray(x0),
                args=(gp, np.min(yi), acq, xi, kappa),
                bounds=lbfgs_bounds, approx_grad=True, maxiter=10)

        next_y = func(lower_bounds + (upper_bounds - lower_bounds) * next_x)
        Xi = np.vstack((Xi, next_x))
        yi.append(next_y)

    # Rescale
    Xi = lower_bounds + (upper_bounds - lower_bounds) * Xi
    best = np.argmin(yi)
    best_x = Xi[best]
    best_y = yi[best]

    # Pack results
    res = OptimizeResult()
    res.x = best_x
    res.fun = best_y
    res.func_vals = yi
    res.x_iters = Xi
    res.models = models

    return res
