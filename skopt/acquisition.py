import numpy as np
import warnings

from scipy.stats import norm


def _gaussian_acquisition(X, model, y_opt=None, method="LCB",
                          xi=0.01, kappa=1.96, return_grad=False):
    """
    Wrapper so that the output of this function can be
    directly passed to a minimizer.
    """
    # Check inputs
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X should be 2-dimensional.")

    # Evaluate acquisition function
    if method == "LCB":
        return gaussian_lcb(X, model, kappa, return_grad)
    elif method == "EI":
        return -gaussian_ei(X, model, y_opt, xi)
    elif method == "PI":
        return -gaussian_pi(X, model, y_opt, xi)
    else:
        raise ValueError("Acquisition function not implemented.")


def gaussian_lcb(X, model, kappa=1.96, return_grad=False):
    """
    Use the lower confidence bound to estimate the acquisition
    values.

    The trade-off between exploitation and exploration is left to
    be controlled by the user through the parameter ``kappa``.

    Parameters
    ----------
    * `X` [array-like, shape=(n_samples, n_features)]:
        Values where the acquisition function should be computed.

    * `model` [sklearn estimator that implements predict with ``return_std``]:
        The fit estimator that approximates the function through the
        method ``predict``.
        It should have a ``return_std`` parameter that returns the standard
        deviation.

    * `kappa`: [float, default 1.96]:
        Controls how much of the variance in the predicted values should be
        taken into account. If set to be very high, then we are favouring
        exploration over exploitation and vice versa.
        Useless if ``method`` is set to "LCB".

    Returns
    -------
    * `values`: [array-like, shape=(X.shape[0],)]:
        Acquisition function values computed at X.
    """
    # Compute posterior.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if return_grad:
            mu, std, mu_grad, std_grad = model.predict(
                X, return_std=True, return_mean_grad=True,
                return_std_grad=True)
            return mu - kappa * std, mu_grad - kappa * std_grad
        else:
            mu, std = model.predict(X, return_std=True)
            return mu - kappa * std


def gaussian_pi(X, model, y_opt=0.0, xi=0.01):
    """
    Use the probability of improvement to calculate the acquisition values.

    The conditional probability `P(y=f(x) | x)`form a gaussian with a
    certain mean and standard deviation approximated by the model.

    The PI condition is derived by computing ``E[u(f(x))]``
    where ``u(f(x)) = 1``, if ``f(x) > y_opt`` and ``u(f(x)) = 0``,
    if``f(x) < y_opt``.

    This means that the PI condition does not care about how "better" the
    predictions are than the previous values, since it gives an equal reward
    to all of them.

    Note that the value returned by this function should be maximized to
    obtain the ``X`` with maximum improvement.

    Parameters
    ----------
    * `X` [array-like, shape=(n_samples, n_features)]:
        Values where the acquisition function should be computed.

    * `model` [sklearn estimator that implements predict with ``return_std``]:
        The fit estimator that approximates the function through the
        method ``predict``.
        It should have a ``return_std`` parameter that returns the standard
        deviation.

    * `y_opt` [float, default 0]:
        Previous minimum value which we would like to improve upon.

    * `xi`: [float, default=0.01]:
        Controls how much improvement one wants over the previous best
        values. Useful only when ``method`` is set to "EI"

    Returns
    -------
    * `values`: [array-like, shape=(X.shape[0],)]:
        Acquisition function values computed at X.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, std = model.predict(X, return_std=True)

    values = np.zeros_like(mu)
    mask = std > 0
    improvement = y_opt - xi - mu[mask]
    values[mask] = norm.cdf(improvement / std[mask])
    return values


def gaussian_ei(X, model, y_opt=0.0, xi=0.01):
    """
    Use the expected improvement to calculate the acquisition values.

    The conditional probability `P(y=f(x) | x)`form a gaussian with a certain
    mean and standard deviation approximated by the model.

    The EI condition is derived by computing ``E[u(f(x))]``
    where ``u(f(x)) = 0``, if ``f(x) > y_opt`` and ``u(f(x)) = y_opt - f(x)``,
    if``f(x) < y_opt``.

    This solves one of the issues of the PI condition by giving a reward
    proportional to the amount of improvement got.

    Note that the value returned by this function should be maximized to
    obtain the ``X`` with maximum improvement.

    Parameters
    ----------
    * `X` [array-like, shape=(n_samples, n_features)]:
        Values where the acquisition function should be computed.

    * `model` [sklearn estimator that implements predict with ``return_std``]:
        The fit estimator that approximates the function through the
        method ``predict``.
        It should have a ``return_std`` parameter that returns the standard
        deviation.

    * `y_opt` [float, default 0]:
        Previous minimum value which we would like to improve upon.

    * `xi`: [float, default=0.01]:
        Controls how much improvement one wants over the previous best
        values. Useful only when ``method`` is set to "EI"

    Returns
    -------
    * `values`: [array-like, shape=(X.shape[0],)]:
        Acquisition function values computed at X.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, std = model.predict(X, return_std=True)

    values = np.zeros_like(mu)
    mask = std > 0
    improvement = y_opt - xi - mu[mask]
    exploit = improvement * norm.cdf(improvement / std[mask])
    explore = std[mask] * norm.pdf(improvement / std[mask])
    values[mask] = exploit + explore
    return values
