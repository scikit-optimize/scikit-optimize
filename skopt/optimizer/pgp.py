"""Gaussian process-based parallel minimization algorithms."""

from sklearn.utils import check_random_state

from .base import base_minimize
from ..space import check_dimension
from ..space import Categorical
from ..space import Space
from ..utils import cook_estimator
from ..utils import normalize_dimensions


def pgp_minimize(func, dimensions, base_estimator=None,
                 n_calls=100, n_random_starts=10,
                 acq_func="EIMCMC", acq_optimizer="auto", x0=None, y0=None,
                 random_state=None, verbose=False, callback=None,
                 n_points=10000, n_restarts_optimizer=5, xi=0.01, kappa=1.96,
                 noise="gaussian", n_jobs=1):
    """Parallel Bayesian optimization using Gaussian Processes.

    TODO:
    1. Currently just copying gp_minimize code

    2. Q: Where to add MCMC parameters (mcmc_iter, mcmc_pending)?
    Not sure if `n_jobs` can be used.

    3. Doc string

    """
    # Check params
    rng = check_random_state(random_state)
    transformed_dims = normalize_dimensions(dimensions)
    space = Space(transformed_dims)
    base_estimator = cook_estimator("GP", space=space, random_state=rng,
                                    noise=noise)

    return base_minimize(
        func, dimensions, base_estimator=base_estimator,
        acq_func=acq_func,
        xi=xi, kappa=kappa, acq_optimizer=acq_optimizer, n_calls=n_calls,
        n_points=n_points, n_random_starts=n_random_starts,
        n_restarts_optimizer=n_restarts_optimizer,
        x0=x0, y0=y0, random_state=random_state, verbose=verbose,
        callback=callback, n_jobs=n_jobs)
