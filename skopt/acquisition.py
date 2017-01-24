import numpy as np
import warnings

from scipy.stats import entropy
from scipy.stats import norm


def gaussian_acquisition_1D(X, model, y_opt=None, acq_func="LCB",
                            acq_func_kwargs=None):
    """
    A wrapper around the acquisition function that is called by fmin_l_bfgs_b.

    This is because lbfgs allows only 1-D input.
    """
    return _gaussian_acquisition(np.expand_dims(X, axis=0),
                                 model, y_opt, acq_func=acq_func,
                                 acq_func_kwargs=acq_func_kwargs,
                                 return_grad=True)


def _gaussian_acquisition(X, model, y_opt=None, acq_func="LCB",
                          return_grad=False, acq_func_kwargs=None):
    """
    Wrapper so that the output of this function can be
    directly passed to a minimizer.
    """
    # Check inputs
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X should be 2-dimensional.")

    if acq_func_kwargs is None:
        acq_func_kwargs = dict()
    xi = acq_func_kwargs.get("xi", 0.01)
    kappa = acq_func_kwargs.get("kappa", 1.96)

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

    elif acq_func == "entropy_search":
        return gaussian_entropy_search(X, model, **acq_func_kwargs)
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
    where ``u(f(x)) = 1``, if ``f(x) < y_opt`` and ``u(f(x)) = 0``,
    if``f(x) > y_opt``.

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


def gaussian_entropy_search(X, model, n_rep_points=50,
                            n_trial_points=200, measure="EI", n_samples=500,
                            random_state=None, rep_cands=None, **kwargs):
    """
    Use entropy search to optimize the acquisition function.

    1. It approximates the search space by a set of representer points sampled from
       a suitable measure such as the "probability of improvement" or
       "expected improvement".

    2. p_min at a representer point X_i is calculated by the Integral
       p(f(x)) \prod_{j=1}^N_rep \Theta(f(X_j) - f(X_i)) \\
       Where P(f(x)) is the GP posterior and \Theta is the heaviside step function.
       This can be used to calculate the base entropy.

    3. Then for each candidate point, the change in the mean and the covariance
       of each of these representer points can be computed. This in turn
       can be used to compute the change in p_min for each of these representer
       points.

    Parameters
    ----------
    * `X` [array-like, shape=(n_samples, n_features)]:
        Values where the acquisition function should be computed.

    * `model` [sklearn estimator that implements predict with ``return_std``]:
        The fit estimator that approximates the function through the
        method ``predict``.
        It should have a ``return_std`` parameter that returns the standard
        deviation.

    * `space` [Space instance]:
        This is required to sample representer points.

    * `n_rep_points` [int, default 50]:
        Number of representer points.

    * `n_trial_points` [int, default 200]:
        Number of trial points sampled uniformly for each representer point
        candidate.

    * `n_samples` [int, default 500]:
        Number of functions to sample from the GP posterior at the
        representer points to approximate p_min at each of the representer
        points.

    * `random_state` [int, RandomState instance, or None (default)]:
        For reproducible results.
    """
    rng = np.random.RandomState(random_state)
    n_features = X.shape[1]

    # Step 1: Sample `n_representer_points` from the given measure.
    # XXX: Use MCMC sampling to sample from the quadrature measure.
    if measure == "LCB":
        quad_measure = gaussian_lcb

    elif measure in ["EI", "PI"]:
        if measure == "EI":
            quad_measure = lambda X : -gaussian_ei(X, model)
        else:
            quad_measure = lambda X : -gaussian_pi(X, model)

    n_total = n_rep_points * n_trial_points
    all_values = np.reshape(quad_measure(rep_cands), (n_rep_points, n_trial_points))
    rep_inds = np.arange(0, n_total, n_trial_points) + np.argmin(all_values, axis=1)
    rep_points = rep_cands[rep_inds]
    X_all = np.vstack((rep_points, X))
    X_all_mean, X_all_cov = model.predict(X_all, return_cov=True)

    # Store mu(X_i) and Sigma(X_i, X_j) of all the representer points based on
    # previous evaluations.
    init_cov = X_all_cov[:n_rep_points, :n_rep_points]
    init_mean = X_all_mean[:n_rep_points]

    # For each candidate point calculate change in p_min of all the represnter
    # points and hence change in entropy (information gain)
    inf_gain = np.zeros(X.shape[0])
    for cand_ind in range(n_rep_points, n_rep_points + X.shape[0]):

        # delta(cov(X_rep)) = \Sigma(X_rep, X_cand) \Sigma^-1(X_cand, X_cand) \Sigma(X_cand, X_rep)
        # Note that \Sigma(X_cand, X_cand) is a scalar since we only require the next
        # best evaluation.
        cov_rep_X = X_all_cov[cand_ind][: n_rep_points]
        cov_rep_X_row = np.reshape(cov_rep_X, (-1, 1))
        cov_rep_X_col = np.reshape(cov_rep_X, (1, -1))
        cov_cand_inv = 1.0 / X_all_cov[cand_ind, cand_ind]
        cov_delta = cov_cand_inv * np.dot(cov_rep_X_row, cov_rep_X_col)

        # delta(cov(X_rep)) = \Sigma(X_rep, X_cand) \Sigma^-1(X_cand, X_cand) \Sigma(X_cand, X_rep) \Omega
        # where \Omega is a distributed normally.
        mean_delta = cov_cand_inv * rng.randn() * (X_all_cov[cand_ind, cand_ind] + model.noise_) * cov_rep_X

        # New simulated mean and simulated covariance.
        new_mean = init_mean + mean_delta
        new_cov = init_cov - cov_delta

        # p_min(Xrep_{i}) = E(\prod_{j=1}^N_rep \theta(f(Xrep{j}) - f(Xrep{i})))
        # where \theta(f(Xrep{j}) - f(Xrep{i})) is zero if Xrep(j) < Xrep(i)
        sampled_f = rng.multivariate_normal(new_mean, new_cov, n_samples)
        p_min = (np.bincount(np.argmin(sampled_f, axis=0)) + 1) / (n_rep_points + float(n_samples))
        inf_gain[cand_ind - n_rep_points] = entropy(p_min)

    return inf_gain
