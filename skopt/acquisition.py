import numpy as np
import warnings

from scipy.stats import norm


def gaussian_acquisition_1D(X, model, y_opt=None, acq_func="LCB", xi=0.01,
                            kappa=1.96):
    """
    A wrapper around the acquisition function that is called by fmin_l_bfgs_b.

    This is because lbfgs allows only 1-D input.
    """
    return _gaussian_acquisition(np.expand_dims(X, axis=0),
                                 model, y_opt, acq_func, xi, kappa,
                                 return_grad=True)


def _gaussian_acquisition(X, model, y_opt=None, acq_func="LCB",
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
    if acq_func == "LCB":
        return gaussian_lcb(X, model, kappa, return_grad)

    elif acq_func in ["EI", "PI"]:
        if acq_func == "EI":
            func_and_grad = gaussian_ei(X, model, y_opt, xi, return_grad)
        else:
            func_and_grad = gaussian_pi(X, model, y_opt, xi, return_grad)

        if return_grad:
            return -func_and_grad[0], -func_and_grad[1]
        else:
            return -func_and_grad

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

    * `return_grad`: [boolean, optional]:
        Whether or not to return the grad. Implemented only for the case where
        ``X`` is a single sample.

    Returns
    -------
    * `values`: [array-like, shape=(X.shape[0],)]:
        Acquisition function values computed at X.

    * `grad`: [array-like, shape=(n_samples, n_features)]:
        Gradient at X.
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


def gaussian_pi(X, model, y_opt=0.0, xi=0.01, return_grad=False):
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

    * `return_grad`: [boolean, optional]:
        Whether or not to return the grad. Implemented only for the case where
        ``X`` is a single sample.

    Returns
    -------
    * `values`: [array-like, shape=(X.shape[0],)]:
        Acquisition function values computed at X.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if return_grad:
            mu, std, mu_grad, std_grad = model.predict(
                X, return_std=True, return_mean_grad=True,
                return_std_grad=True)

        else:
            mu, std = model.predict(X, return_std=True)

    values = np.zeros_like(mu)
    mask = std > 0
    improve = y_opt - xi - mu[mask]
    scaled = improve / std[mask]
    values[mask] = norm.cdf(scaled)

    if return_grad:
        if not np.all(mask):
            return values, np.zeros_like(std_grad)

        # Substitute (y_opt - xi - mu) / sigma = t and apply chain rule.
        # improve_grad is the gradient of t wrt x.
        improve_grad = -mu_grad * std - std_grad * improve
        improve_grad /= std**2

        return values, improve_grad * norm.pdf(scaled)

    else:
        return values


def gaussian_ei(X, model, y_opt=0.0, xi=0.01, return_grad=False):
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

    * `return_grad`: [boolean, optional]:
        Whether or not to return the grad. Implemented only for the case where
        ``X`` is a single sample.

    Returns
    -------
    * `values`: [array-like, shape=(X.shape[0],)]:
        Acquisition function values computed at X.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if return_grad:
            mu, std, mu_grad, std_grad = model.predict(
                X, return_std=True, return_mean_grad=True,
                return_std_grad=True)

        else:
            mu, std = model.predict(X, return_std=True)

    values = np.zeros_like(mu)
    mask = std > 0
    improve = y_opt - xi - mu[mask]
    scaled = improve / std[mask]
    cdf = norm.cdf(scaled)
    pdf = norm.pdf(scaled)
    exploit = improve * cdf
    explore = std[mask] * pdf
    values[mask] = exploit + explore

    if return_grad:
        if not np.all(mask):
            return values, np.zeros_like(std_grad)

        # Substitute (y_opt - xi - mu) / sigma = t and apply chain rule.
        # improve_grad is the gradient of t wrt x.
        improve_grad = -mu_grad * std - std_grad * improve
        improve_grad /= std ** 2
        cdf_grad = improve_grad * pdf
        pdf_grad = -improve * cdf_grad
        exploit_grad = -mu_grad * cdf - pdf_grad
        explore_grad = std_grad * pdf + pdf_grad

        grad = exploit_grad + explore_grad
        return values, grad

    return values
