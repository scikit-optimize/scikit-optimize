import warnings
from collections import Iterable
from numbers import Number

import numpy as np

from scipy.optimize import fmin_l_bfgs_b

from sklearn.base import clone
from sklearn.base import is_regressor
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import check_random_state

from ..acquisition import _gaussian_acquisition
from ..acquisition import gaussian_acquisition_1D
from ..space import Categorical
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
        In addition the `predict` method, should have an optional `return_std`
        argument, which returns `std(Y | x)`` along with `E[Y | x]`.

    * `n_random_starts` [int, default=10]:
        Number of evaluations of `func` with random initialization points
        before approximating the `func` with `base_estimator`. While random
        points are being suggested no model will be fit to the observations.

    * `acq_func` [string, default=`"EI"`]:
        Function to minimize over the posterior distribution. Can be either

        - `"LCB"` for lower confidence bound.
        - `"EI"` for negative expected improvement.
        - `"PI"` for negative probability of improvement.
        - `"gp_hedge"` Probabilistically choose one of the above three
          acquisition functions at every iteration.
            - The gains `g_i` are initialized to zero.
            - At every iteration,
                - Each acquisition function is optimised independently to
                  propose an candidate point `X_i`.
                - Out of all these candidate points, the next point `X_best` is
                  chosen by $softmax(\eta g_i)$
                - After fitting the surrogate model with `(X_best, y_best)`,
                  the gains are updated such that $g_i -= \mu(X_i)$

    * `acq_optimizer` [string, `"sampling"`, `"lbfgs"`, or `"auto"`, default=`"lbfgs"`]:
        Method to minimize the acquistion function. The fit model
        is updated with the optimal value obtained by optimizing `acq_func`
        with `acq_optimizer`.

        - If set to `"auto"`, then `acq_optimizer` will be chosen based on
          `base_estimator`, i.e., if it is any tree-based regressor, then
          `acq_optimizer` is set to `"sampling"`, otherwise it is set to
          `"lbfgs"`.
        - If set to `"sampling"`, then `acq_func` is optimized by computing
          `acq_func` at `n_points` randomly sampled points.
        - If set to `"lbfgs"`, then `acq_func` is optimized by
              - Sampling `n_restarts_optimizer` points randomly.
              - `"lbfgs"` is run for 20 iterations with these points as initial
                points to find local minima.
              - The optimal of these local minima is used to update the prior.

    * `random_state` [int, RandomState instance, or None (default)]:
        Set random state to something other than None for reproducible
        results.

    * `acq_func_kwargs` [dict]:
        Additional arguments to be passed to the acquistion function.

    * `acq_optimizer_kwargs` [dict]:
        Additional arguments to be passed to the acquistion optimizer.

    Attributes
    ----------
    * `Xi` [list]:
        Points at which objective has been evaluated.
    * `yi` [scalar]:
        Values of objective at corresponding points in `Xi`.
    * `models` [list]:
        Regression models used to fit observations and compute acquisition
        function.
    * `space`
        An instance of `skopt.space.Space`. Stores parameter search space used
        to sample points, bounds, and type of parameters.

    """
    def __init__(self, dimensions, base_estimator,
                 n_random_starts=10, acq_func="gp_hedge",
                 acq_optimizer="lbfgs",
                 random_state=None, acq_func_kwargs=None,
                 acq_optimizer_kwargs=None):
        # Arguments that are just stored not checked
        self.acq_func = acq_func
        self.rng = check_random_state(random_state)
        self.acq_func_kwargs = acq_func_kwargs

        if self.acq_func == "gp_hedge":
            self.cand_acq_funcs_ = ["EI", "LCB", "PI"]
            self.gains_ = np.zeros(3)
        else:
            self.cand_acq_funcs_ = [self.acq_func]

        if acq_func_kwargs is None:
            acq_func_kwargs = dict()
        self.eta = acq_func_kwargs.get("eta", 1.0)

        if acq_optimizer_kwargs is None:
            acq_optimizer_kwargs = dict()

        self.n_points = acq_optimizer_kwargs.get("n_points", 10000)
        self.n_restarts_optimizer = acq_optimizer_kwargs.get(
            "n_restarts_optimizer", 5)
        n_jobs = acq_optimizer_kwargs.get("n_jobs", 1)

        self.space = Space(dimensions)
        self.models = []
        self.Xi = []
        self.yi = []

        self._cat_inds = []
        self._non_cat_inds = []
        for ind, dim in enumerate(self.space.dimensions):
            if isinstance(dim, Categorical):
                self._cat_inds.append(ind)
            else:
                self._non_cat_inds.append(ind)
        self._check_arguments(base_estimator, n_random_starts, acq_optimizer)

        self.n_jobs = n_jobs

    def _check_arguments(self, base_estimator, n_random_starts, acq_optimizer):
        """Check arguments for sanity."""
        if not is_regressor(base_estimator):
            raise ValueError(
                "%s has to be a regressor." % base_estimator)
        self.base_estimator = base_estimator

        if n_random_starts < 0:
            raise ValueError(
                "Expected `n_random_starts` >= 0, got %d" % n_random_starts)
        self._n_random_starts = n_random_starts

        if acq_optimizer == "auto":
            acq_optimizer = "lbfgs"

        self.acq_optimizer = acq_optimizer
        if self.acq_optimizer not in ["lbfgs", "sampling"]:
            raise ValueError(
                "Expected acq_optimizer to be 'lbfgs' or 'sampling', "
                "got %s" % acq_optimizer)

    def ask(self):
        """Suggest next point at which to evaluate the objective.

        Returns a random point for the first `n_random_starts` calls, after
        that `base_estimator` is used to determine the next point.
        """
        if self._n_random_starts > 0:
            self._n_random_starts -= 1
            # this will not make a copy of `self.rng` and hence keep advancing
            # our random state.
            return self.space.rvs(random_state=self.rng)[0]

        else:
            if not self.models:
                raise RuntimeError("Random evaluations exhausted and no "
                                   "model has been fit.")

            next_x = self._next_x
            min_delta_x = np.min([self.space.distance(next_x, xi)
                                  for xi in self.Xi])
            if np.allclose(min_delta_x, 0.):
                warnings.warn("The objective has been evaluated "
                              "at this point before.")

            # return point computed from last call to tell()
            return next_x

    def tell(self, x, y, fit=True):
        """Record an observation (or several) of the objective function.

        Provide values of the objective function at points suggested by `ask()`
        or other points. By default a new model will be fit to all
        observations. The new model is used to suggest the next point at
        which to evluate the objective. This point can be retrieved by calling
        `ask()`.

        To add observations without fitting a new model set `fit` to False.

        To add multiple observations in a batch pass a list-of-lists for `x`
        and a list of scalars for `y`.

        Parameters
        ----------
        * `x` [list or list-of-lists]:
            Point at which objective was evaluated.
        * `y` [scalar or list]:
            Value of objective at `x`.
        * `fit` [bool, default=True]
            Fit a model to observed evaluations of the objective. A model will
            only be fitted after `n_random_starts` points have been queried
            irrespective of the value of `fit`.
        """
        # if y isn't a scalar it means we have been handed a batch of points
        if (isinstance(y, Iterable) and all(isinstance(point, Iterable)
                                            for point in x)):
            if not np.all([p in self.space for p in x]):
                raise ValueError("Not all points are within the bounds of"
                                 " the space.")
            self.Xi.extend(x)
            self.yi.extend(y)

        elif isinstance(x, Iterable) and isinstance(y, Number):
            if x not in self.space:
                raise ValueError("Point (%s) is not within the bounds of"
                                 " the space (%s)."
                                 % (x, self.space.bounds))
            self.Xi.append(x)
            self.yi.append(y)

        else:
            raise ValueError("Type of arguments `x` (%s) and `y` (%s) "
                             "not compatible." % (type(x), type(y)))

        if fit and self._n_random_starts == 0:
            transformed_bounds = np.array(self.space.transformed_bounds)
            est = clone(self.base_estimator)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(self.space.transform(self.Xi), self.yi)

            if hasattr(self, "next_xs_") and self.acq_func == "gp_hedge":
                self.gains_ -= est.predict(np.vstack(self.next_xs_))
            self.models.append(est)

            X = self.space.transform(self.space.rvs(
                n_samples=self.n_points, random_state=self.rng))
            self.next_xs_ = []
            for cand_acq_func in self.cand_acq_funcs_:
                values = _gaussian_acquisition(
                    X=X, model=est, y_opt=np.min(self.yi),
                    acq_func=cand_acq_func,
                    acq_func_kwargs=self.acq_func_kwargs)
                # Find the minimum of the acquisition function by randomly
                # sampling points from the space
                if self.acq_optimizer == "sampling":
                    next_x = X[np.argmin(values)]

                # Use BFGS to find the mimimum of the acquisition function, the
                # minimization starts from `n_restarts_optimizer` different
                # points and the best minimum is used
                elif self.acq_optimizer == "lbfgs":
                    x0 = X[np.argsort(values)[:self.n_restarts_optimizer]]

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        results = Parallel(n_jobs=self.n_jobs)(
                            delayed(fmin_l_bfgs_b)(
                                gaussian_acquisition_1D, x,
                                args=(est, np.min(self.yi), cand_acq_func,
                                      self.acq_func_kwargs),
                                bounds=self.space.transformed_bounds,
                                approx_grad=False,
                                maxiter=20)
                            for x in x0)

                    cand_xs = np.array([r[0] for r in results])
                    cand_acqs = np.array([r[1] for r in results])
                    next_x = cand_xs[np.argmin(cand_acqs)]

                # lbfgs should handle this but just in case there are
                # precision errors.
                if not self.space.is_categorical:
                    next_x = np.clip(
                        next_x, transformed_bounds[:, 0],
                        transformed_bounds[:, 1])
                self.next_xs_.append(next_x)

            if self.acq_func == "gp_hedge":
                logits = np.array(self.gains_)
                logits -= np.max(logits)
                exp_logits = np.exp(self.eta * logits)
                probs = exp_logits / np.sum(exp_logits)
                next_x = self.next_xs_[np.argmax(np.random.multinomial(1,
                                                                       probs))]
            else:
                next_x = self.next_xs_[0]

            # note the need for [0] at the end
            self._next_x = self.space.inverse_transform(
                next_x.reshape((1, -1)))[0]

        # Pack results
        return create_result(self.Xi, self.yi, self.space, self.rng,
                             models=self.models)

    def run(self, func, n_iter=1):
        """Execute ask() + tell() `n_iter` times"""
        for _ in range(n_iter):
            x = self.ask()
            self.tell(x, func(x))

        return create_result(self.Xi, self.yi, self.space, self.rng,
                             models=self.models)
