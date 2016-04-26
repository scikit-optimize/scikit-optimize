import warnings

import numpy as np

from scipy.stats import norm


def gaussian_acquisition(X, model, y_opt=None, method="LCB",
                         xi=0.01, kappa=1.96):
    """
    Returns the acquisition function computed at values x0 where the
    conditional is assumed to be a Gaussian with the mean and std
    provided by the underlying base estimator.

    Parameters
    ----------
    X : array-like
        Values where the acquisition function should be computed.

    model: sklearn estimator that implements predict with ``return_std``
        The fit sklearn gaussian process estimator that approximates
        the function. It should have a ``return_std`` parameter
        that returns the standard deviation.

    y_opt: float, optional
        The previous best value over which we want to improve.
        Useful only when `acq` is set to "EI"

    method: string, "LCB" or "EI"
        If set to "LCB", then the lower confidence bound is taken.
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
    if X.ndim != 2:
        raise ValueError("X should be 2-dimensional.")

    # Compute posterior.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, std = model.predict(X, return_std=True)

    # Evaluate acquisition function
    if method == "LCB":
        values = mu - kappa * std

    elif method == "EI":
        values = np.zeros(len(mu))
        mask = std > 0
        improvement = y_opt - xi - mu[mask]
        exploit = improvement * norm.cdf(improvement / std[mask])
        explore = std[mask] * norm.pdf(improvement / std[mask])
        values[mask] = exploit + explore
        values = -values  # acquisition is minimized

    elif method == "PI":
        values = np.zeros(len(mu))
        mask = std > 0
        improvement = y_opt - xi - mu[mask]
        values[mask] = norm.cdf(improvement / std[mask])
        values = -values  # acquisition is minimized

    else:
        raise ValueError("Acquisition function not implemented.")

    return values
