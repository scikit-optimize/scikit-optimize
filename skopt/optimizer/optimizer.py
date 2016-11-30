import copy
import inspect
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
    """docstring for Optimizer."""
    def __init__(self, func, dimensions, base_estimator,
                 n_random_starts=10,
                 acq_func="EI", acq_optimizer="auto",
                 random_state=None, verbose=False,
                 n_points=10000, n_restarts_optimizer=5,
                 xi=0.01, kappa=1.96, n_jobs=1):
        super(Optimizer, self).__init__()
        self._check_arguments(func, dimensions, base_estimator,
                              n_random_starts,
                              acq_func, acq_optimizer,
                              verbose,
                              n_restarts_optimizer,
                              n_jobs)
        self.specs = {"args": copy.copy(inspect.currentframe().f_locals),
                      "function": inspect.currentframe().f_code.co_name}
        # Check params
        self.rng = check_random_state(random_state)
        self.space = Space(dimensions)
        self.Xi = []
        self.yi = []
        self.models = []
        self.n_points = n_points
        self.kappa = kappa
        # XXX change to a name that is less similar to Xi
        self.xi = xi
        self.parallel = Parallel(n_jobs=n_jobs)
        self.acq_func = acq_func

    def _check_arguments(self, func, dimensions, base_estimator,
                         n_random_starts,
                         acq_func, acq_optimizer,
                         verbose,
                         n_restarts_optimizer,
                         n_jobs):
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
            transformed_bounds = np.array(self.space.transformed_bounds)
            est = clone(self.base_estimator)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(self.space.transform(self.Xi), self.yi)

            self.models.append(est)

            if self.acq_optimizer == "sampling":
                X = self.space.transform(self.space.rvs(
                    n_samples=self.n_points, random_state=self.rng))
                values = _gaussian_acquisition(
                    X=X, model=est,  y_opt=np.min(self.yi),
                    acq_func=self.acq_func, xi=self.xi, kappa=self.kappa)
                next_x = X[np.argmin(values)]

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
            next_x = self.space.inverse_transform(next_x.reshape((1, -1)))[0]
            return next_x

    def tell(self, x, y):
        """Record an observation of the objective function."""
        self.Xi.append(x)
        self.yi.append(y)
        return create_result(self.Xi, self.yi, self.space, self.rng,
                             self.specs)

    def run(self):
        """Execute ask() + tell() `n_iter` times"""
        pass
