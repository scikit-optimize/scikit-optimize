import warnings
from collections import Iterable
from numbers import Number
import copy

import numpy as np

from scipy.optimize import fmin_l_bfgs_b

from sklearn.base import clone
from sklearn.base import is_regressor
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import check_random_state

from ..acquisition import _gaussian_acquisition
from ..acquisition import gaussian_acquisition_1D
from ..learning import RandomForestRegressor, ExtraTreesRegressor
from ..learning import GradientBoostingQuantileRegressor
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

    * `acq_optimizer` [string, `"sampling"` or `"lbfgs"`, default=`"lbfgs"`]:
        Method to minimize the acquistion function. The fit model
        is updated with the optimal value obtained by optimizing `acq_func`
        with `acq_optimizer`.

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
        self.acq_optimizer_kwargs = acq_optimizer_kwargs

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

        # The cache of responses of `ask` method for n_points not None.
        # This ensures that multiple calls to `ask` with n_points set
        # return same sets of points.
        # The cache is reset to {} at every call to `tell`.
        self.cache_ = {}

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

        if (isinstance(base_estimator, (ExtraTreesRegressor,
                                        RandomForestRegressor,
                                        GradientBoostingQuantileRegressor))
            and not acq_optimizer == "sampling"):
            raise ValueError("The tree-based regressor {0} should run with "
                             "acq_optimizer='sampling'".format(type(base_estimator)))

        if acq_optimizer not in ["lbfgs", "sampling"]:
            raise ValueError("Expected acq_optimizer to be 'lbfgs' or "
                             "'sampling', got {0}".format(acq_optimizer))
        self.acq_optimizer = acq_optimizer

    def copy(self, random_state=None):
        """Create a shallow copy of an instance of the optimizer.

        Parameters
        ----------
        * `random_state` [int, RandomState instance, or None (default)]:
        Set random state of the copy of optimizer.
        """

        optimizer = Optimizer(
            dimensions=self.space.dimensions,
            base_estimator=self.base_estimator,
            n_random_starts=self._n_random_starts,
            acq_func=self.acq_func,
            acq_optimizer=self.acq_optimizer,
            acq_func_kwargs=self.acq_func_kwargs,
            acq_optimizer_kwargs=self.acq_optimizer_kwargs,
            random_state=random_state,
        )

        if hasattr(self, "gains_"):
            optimizer.gains_ = np.copy(self.gains_)

        if self.Xi:
            optimizer.tell(self.Xi, self.yi)

        return optimizer

    def ask(self, n_points=None, strategy="cl_min"):
        """Query point or multiple points at which objective should be evaluated.

        * `n_points` [int or None, default=None]:
            Number of points returned by the ask method.
            If the value is None, a single point to evaluate is returned.
            Otherwise a list of points to evaluate is returned of size
            n_points. This is useful if you can evaluate your objective in
            parallel, and thus obtain more objective function evaluations per
            unit of time.

        * `strategy` [string, default=`"cl_min"`]:
            Method to use to sample multiple points (see also `n_points`
            description). This parameter is ignored if n_points = None.
            Supported options are `"cl_min"`, `"cl_mean"` or `"cl_max"`.

            - If set to `"cl_min"`, then constant liar strtategy is used
               with lie objective value being minimum of observed objective
               values. `"cl_mean"` and `"cl_max"` means mean and max of values
               respectively. For details on this strategy see:

               https://hal.archives-ouvertes.fr/hal-00732512/document

               With this strategy a copy of optimizer is created, which is
               then asked for a point, and the point is told to the copy of
               optimizer with some fake objective (lie), the next point is
               asked from copy, it is also told to the copy with fake
               objective and so on. The type of lie defines different
               flavours of `cl_x` strategies.

        """
        if n_points is None:
            return self._ask()

        supported_strategies = ["cl_min", "cl_mean", "cl_max"]

        if not (isinstance(n_points, int) and n_points > 0):
            raise ValueError(
                "n_points should be int > 0, got " + str(n_points)
            )

        if strategy not in supported_strategies:
            raise ValueError(
                "Expected parallel_strategy to be one of " +
                str(supported_strategies) + ", " + "got %s" % strategy
            )

        # Caching the result with n_points not None. If some new parameters
        # are provided to the ask, the cache_ is not used.
        if (n_points, strategy) in self.cache_:
            return self.cache_[(n_points, strategy)]

        # Copy of the optimizer is made in order to manage the
        # deletion of points with "lie" objective (the copy of
        # oiptimizer is simply discarded)
        opt = self.copy()

        X = []
        for i in range(n_points):
            x = opt.ask()
            X.append(x)
            if strategy == "cl_min":
                y_lie = np.min(opt.yi) if opt.yi else 0.0  # CL-min lie
            elif strategy == "cl_mean":
                y_lie = np.mean(opt.yi) if opt.yi else 0.0  # CL-max lie
            else:
                y_lie = np.max(opt.yi) if opt.yi else 0.0  # CL-max lie
            opt.tell(x, y_lie)  # lie to the optimizer
            self._n_random_starts = max(0, self._n_random_starts - 1)

        self.cache_ = {(n_points, strategy): X}  # cache_ the result

        return X

    def _ask(self):
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
            min_delta_x = min([self.space.distance(next_x, xi)
                               for xi in self.Xi])
            if abs(min_delta_x) <= 1e-8:
                warnings.warn("The objective has been evaluated "
                              "at this point before.")

            # return point computed from last call to tell()
            return next_x

    def tell(self, x, y, fit=True):
        """Record an observation (or several) of the objective function.

        Provide values of the objective function at points suggested by `ask()`
        or other points. By default a new model will be fit to all
        observations. The new model is used to suggest the next point at
        which to evaluate the objective. This point can be retrieved by calling
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

        self.cache_ = {}  # optimizer learned somethnig new - discard the cache_

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
                next_x = self.next_xs_[np.argmax(self.rng.multinomial(1,
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
