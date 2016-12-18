import warnings

import numpy as np

from scipy.optimize import fmin_l_bfgs_b

from sklearn.base import clone
from sklearn.base import is_regressor
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import check_random_state

from ..acquisition import _gaussian_acquisition
from ..acquisition import gaussian_acquisition_1D
from ..space import Space
from ..utils import create_result


class Optimizer(object):
    """Run bayesian optimisation loop.

    An `Optimizer` represents the steps of a bayesian optimisation loop. To
    use it you need to provide your own loop mechanism. The various
    optimisers provided by `skopt` use this class under the hood.

    Use this class directly if you want to control the iterations of your
    bayesian optimisation loop.

    Parameters
    ----------
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

    * `base_estimator` [sklearn regressor]:
        Should inherit from `sklearn.base.RegressorMixin`.
        In addition, should have an optional `return_std` argument,
        which returns `std(Y | x)`` along with `E[Y | x]`.

    * `n_random_starts` [int, default=10]:
        Number of evaluations of `func` with random initialization points
        before approximating the `func` with `base_estimator`.

    * `acq_func` [string, default=`"EI"`]:
        Function to minimize over the posterior distribution. Can be either

        - `"LCB"` for lower confidence bound,
        - `"EI"` for negative expected improvement,
        - `"PI"` for negative probability of improvement.

    * `acq_optimizer` [string, `"auto"`, `"sampling"` or `"lbfgs"`,
       default=`"auto"`]:
        Method to minimize the acquistion function. The fit model
        is updated with the optimal value obtained by optimizing `acq_func`
        with `acq_optimizer`.

        - If set to `"sampling"`, then `acq_func` is optimized by computing
          `acq_func` at `n_points` sampled randomly.
        - If set to `"lbfgs"`, then `acq_func` is optimized by
              - Sampling `n_restarts_optimizer` points randomly.
              - `"lbfgs"` is run for 20 iterations with these points as initial
                points to find local minima.
              - The optimal of these local minima is used to update the prior.
        - If set to `"auto"`, then it is set to `"lbfgs"`` if
          all the search dimensions are Real (continuous). It defaults to
          `"sampling"` for all other cases.

    * `random_state` [int, RandomState instance, or None (default)]:
        Set random state to something other than None for reproducible
        results.

    * `n_points` [int, default=500]:
        Number of points to sample to determine the next "best" point.
        Useless if acq_optimizer is set to `"lbfgs"`.

    * `n_restarts_optimizer` [int, default=5]:
        The number of restarts of the optimizer when `acq_optimizer`
        is `"lbfgs"`.

    * `xi` [float, default=0.01]:
        Controls how much improvement one wants over the previous best
        values. Used when the acquisition is either `"EI"` or `"PI"`.

    * `kappa` [float, default=1.96]:
        Controls how much of the variance in the predicted values should be
        taken into account. If set to be very high, then we are favouring
        exploration over exploitation and vice versa.
        Used when the acquisition is `"LCB"`.

    * `n_jobs` [int, default=1]
        Number of cores to run in parallel while running the lbfgs
        optimizations over the acquisition function. Valid only when
        `acq_optimizer` is set to "lbfgs."
        Defaults to 1 core. If `n_jobs=-1`, then number of jobs is set
        to number of cores.
    """
    def __init__(self, dimensions, base_estimator,
                 n_random_starts=10, acq_func="EI", acq_optimizer="auto",
                 random_state=None, n_points=10000, n_restarts_optimizer=5,
                 xi=0.01, kappa=1.96, n_jobs=1):
        super(Optimizer, self).__init__()
        # Arguments that are just stored not checked
        self.acq_func = acq_func
        self.rng = check_random_state(random_state)
        self.kappa = kappa
        self.models = []
        self.n_points = n_points
        self.n_restarts_optimizer = n_restarts_optimizer
        self.space = Space(dimensions)
        self.Xi = []
        self.xi = xi
        self.yi = []

        self._check_arguments(dimensions, base_estimator,
                              n_random_starts,
                              acq_func, acq_optimizer,
                              n_jobs)

        self.parallel = Parallel(n_jobs=n_jobs)

    def _check_arguments(self, dimensions, base_estimator,
                         n_random_starts,
                         acq_func, acq_optimizer,
                         n_jobs):
        """Check arguments for sanity."""
        if not is_regressor(base_estimator):
            raise ValueError(
                "%s has to be a regressor." % base_estimator)
        self.base_estimator = base_estimator

        if n_random_starts < 0:
            raise ValueError(
                "Expected `n_random_starts` >= 0, got %d" % n_random_starts)
        self.n_random_starts = n_random_starts

        self.acq_optimizer = acq_optimizer
        if acq_optimizer == "auto":
            warnings.warn("The 'auto' option for the acq_optimizer will be "
                          "removed in 0.4.")
            self.acq_optimizer = "lbfgs"
        elif self.acq_optimizer not in ["lbfgs", "sampling"]:
            raise ValueError(
                "Expected acq_optimizer to be 'lbfgs', 'sampling' or 'auto', "
                "got %s" % acq_optimizer)

    def ask(self):
        """Suggest next point."""
        if len(self.Xi) < self.n_random_starts:
            return self.space.rvs(random_state=self.rng)[0]

        else:
            if not self.models:
                raise ValueError("Random evaluations exhausted and no "
                                 "model has been fit.")
            # return point computed from last call to tell()
            return self._next_x

    def tell(self, x, y, fit_model=True):
        """Record an observation of the objective function.

        Provide values of the objective function at points suggested by `ask()`
        or other points. By default a new model will be fit to all
        observations. The new model is used to suggest the next point at
        which to evluate the objective. This point can be retrieved by calling
        `ask()`.

        To add several observations as a batch without fitting a new model
        set `fit_model` to False.

        * `x` [list]:
            Point at which objective was evaluated.
        * `y` [scalar]:
            Value of objective at `x`.
        * `fit_model` [bool, default=True]
            Fit a model to observed evaluations of the objective.
        """
        self.Xi.append(x)
        self.yi.append(y)

        if fit_model:
            transformed_bounds = np.array(self.space.transformed_bounds)
            est = clone(self.base_estimator)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(self.space.transform(self.Xi), self.yi)

            self.models.append(est)

            # Find the minimum of the acquisition function by randomly sampling
            # points from the space
            if self.acq_optimizer == "sampling":
                X = self.space.transform(self.space.rvs(
                    n_samples=self.n_points, random_state=self.rng))
                values = _gaussian_acquisition(
                    X=X, model=est,  y_opt=np.min(self.yi),
                    acq_func=self.acq_func, xi=self.xi, kappa=self.kappa)
                next_x = X[np.argmin(values)]

            # Use BFGS to find the mimimum of the acquisition function, the
            # minimization starts from `n_restarts_optimizer` different points
            # and the best minimum is used
            elif self.acq_optimizer == "lbfgs":
                best = np.inf
                x0 = self.space.transform(
                    self.space.rvs(n_samples=self.n_restarts_optimizer,
                                   random_state=self.rng))

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    jobs = (delayed(fmin_l_bfgs_b)(
                        gaussian_acquisition_1D, x,
                        args=(est, np.min(self.yi), self.acq_func,
                              self.xi, self.kappa),
                        bounds=self.space.transformed_bounds,
                        approx_grad=False,
                        maxiter=20) for x in x0)
                    results = self.parallel(jobs)

                cand_xs = np.array([r[0] for r in results])
                cand_acqs = np.array([r[1] for r in results])
                best_ind = np.argmin(cand_acqs)
                a = cand_acqs[best_ind]

                if a < best:
                    next_x, best = cand_xs[best_ind], a

            # lbfgs should handle this but just in case there are
            # precision errors.
            next_x = np.clip(
                next_x, transformed_bounds[:, 0], transformed_bounds[:, 1])
            # XXX why the need for [0] at the end?
            self._next_x = self.space.inverse_transform(
                next_x.reshape((1, -1)))[0]

        #return create_result(self.Xi, self.yi, self.space, self.rng,
        #                     self.specs)

    def run(self, func, n_iter=1):
        """Execute ask() + tell() `n_iter` times"""
        for _ in range(n_iter):
            x = self.ask()
            self.tell(x, func(x))
