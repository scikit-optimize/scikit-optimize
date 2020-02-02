'''
TODO BEFORE CONSIDERING IF THIS IS WORTH MERGING

- Decide how to manage retrocompatibility. Retrocompatibility is a huge issue
on this PR because:
    1) it adds dask[distributed] as a dependency, which in turns have a lot of
    dependencies, which could create conflicts, which users want to avoid.
    2) it removes joblib. For some users this is transparent, but it would
    break the workflow of people using BayesSearchCV within a joblib context
    manager to change the backend (for e.g, dask).
    3) Is is also complicated because the PR changes significantly the logic
    within BayesSearchCV internals, and (although it makes it simpler and
    closer to sklearn BaseSearchCV) a specific development is yet to be made
    to support both backends. For now it only works with dask.
Because of 1), dask[distributed] should be added as an extra dependency first,
and only made mandatory if joblib.Parallel is completely dropped from
BayesSearchCV.
For 2), an extra argument "backend" should be added to let the user choose
between joblib and dask (taking two values "joblib" or "dask", first defaulting
to "joblib" and as long as "dask" is an extra dependency). A planning should be
prepared to deprecate joblib, add dask as a mandatory dependency, make joblib
non-default, and finally remove it.
3) is yet to be investigated, it is not clear wether what would be better:
either maintaining side by side the new and the old version of BayesSearchCV or
trying to make joblib work within the current logic.

- Compared to the old (synchronous) version, it will be harder to enforce
determinism with this one whenever `n_jobs` > 1. Indeed, depending on the load
of the workers, tasks can be sometimes slower, sometimes faster, and the
optimizer might not be told the same sequences of points. I don't think this
should be an issue outside of testing (given most estimators rely on random
initializations anyway).

- there are a few TODOs to solve in the code. Especially, the output of the
function _call_and_get_cfg must be structured for a better readability.

'''

import warnings
from collections import defaultdict
from functools import partial
from operator import itemgetter

import numpy as np
from scipy.stats import rankdata

from sklearn.base import is_classifier, clone
from joblib import cpu_count
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils import check_random_state
from sklearn.utils.fixes import MaskedArray
from sklearn.utils.validation import indexable, check_is_fitted
from sklearn.model_selection._validation import check_cv
from sklearn.model_selection._validation import _fit_and_score
try:
    from sklearn.metrics import check_scoring
except ImportError:
    from sklearn.metrics.scorer import check_scoring

from dask.distributed import Client, as_completed

from . import Optimizer
from .utils import point_asdict, dimensions_aslist, eval_callbacks
from .space import check_dimension
from .callbacks import check_callback


# TODO: this return format is awful, there is a lot of informations of
# different nature in a tuple that is too long. The manipulations that follow
# are hard to read. Need for a more structured output.
def _call_and_get_cfg(f, cfg, *args, **kwargs):
    return (*f(*args, **kwargs), *cfg)


class _ManageBackend:

    def __init__(self, client=None, **default_kwargs):
        self.client = client
        self.default_kwargs = default_kwargs

    def __enter__(self):
        self.managed = self.client is None
        if self.managed:
            self.client = Client(**self.default_kwargs)
        return self.client

    def __exit__(self, type, value, traceback):
        if self.managed:
            self.client.close()


class BayesSearchCV(BaseSearchCV):
    """Bayesian optimization over hyper parameters.

    BayesSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.

    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by n_iter.

    Parameters are presented as a list of skopt.space.Dimension objects.

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each search point.
        This object is assumed to implement the scikit-learn estimator api.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    search_spaces : dict, list of dict or list of tuple containing
        (dict, int) or (dict, dict).
        One of these cases:
        1. dictionary, where keys are parameter names (strings)
        and values are skopt.space.Dimension instances (Real, Integer
        or Categorical) or any other valid value that defines skopt
        dimension (see skopt.Optimizer docs). Represents search space
        over parameters of the provided estimator.
        2. list of dictionaries: a list of dictionaries, where every
        dictionary fits the description given in case 1 above.
        If a list of dictionary objects is given, then the search is
        performed sequentially for every parameter space with maximum
        number of evaluations set to self.n_iter and number of initial
        points set to self.n_initial_points.
        3. list of (dict, dict): an extension of case 2 above,
        where first element of every tuple is a dictionary representing
        some search subspace, similarly as in case 2, and second element
        can override global parameters for specific subspace. Currently
        support overriding `n_iter`, `n_initial_points` and
        `n_points`.
        4. list of (dict, int > 0): this case is deprecated.
        It is equivalent to case 3 where the second dictionary contains only
        the key `n_iter` with the value set to this integer.

    n_iter : int, default=50
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution. Consider increasing
        ``n_points`` if you want to try more parameter settings in
        parallel.

    optimizer_kwargs : dict, optional
        Dict of arguments passed to :class:`Optimizer`.  For example,
        ``{'base_estimator': 'RF'}`` would use a Random Forest surrogate
        instead of the default Gaussian Process.

    scoring : string, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    n_jobs : int, default=1
        Number of jobs to run in parallel. At maximum there are
        ``n_points`` times ``cv`` jobs available during each iteration.

    n_points : int, default=1
        Number of parameter settings to sample in parallel. If this does
        not align with ``n_iter``, the last iteration will sample less
        points. See also :func:`~Optimizer.ask`

    n_initial_points : int, optional
        If None, will be set to `n_points` for all search spaces. Else, must be
        >= n_points for all search spaces. This parameter could be leveraged
        to exploit the asynchronous backend performances and ensure workers do
        not ever starve (by setting n_initial_points slightly greater than
        n_points).)

    client: dask.distributed.Client, optional
        By default, BayesSearchCV will use dask.distributed backend with a
        LocalCluster. If another client is passed it will be used instead. It
        will override the parameter `n_jobs` (using instead the configuration
        of the client).

    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this RandomizedSearchCV instance after fitting.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    random_state : int or RandomState
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : boolean, default=False
        If ``'True'``, the ``cv_results_`` attribute will include training
        scores.

    Examples
    --------

    >>> from skopt import BayesSearchCV
    >>> # parameter ranges are specified by one of below
    >>> from skopt.space import Real, Categorical, Integer
    >>>
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> X, y = load_iris(True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     train_size=0.75,
    ...                                                     random_state=0)
    >>>
    >>> # log-uniform: understand as search over p = exp(x) by varying x
    >>> opt = BayesSearchCV(
    ...     SVC(),
    ...     {
    ...         'C': Real(1e-6, 1e+6, prior='log-uniform'),
    ...         'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    ...         'degree': Integer(1,8),
    ...         'kernel': Categorical(['linear', 'poly', 'rbf']),
    ...     },
    ...     n_iter=32
    ... )
    >>>
    >>> # executes bayesian optimization
    >>> _ = opt.fit(X_train, y_train)
    >>>
    >>> # model can be saved, used for predictions or scoring
    >>> print(opt.score(X_test, y_test))
    0.973...

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |        0.8        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |        0.9        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |        0.7        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.8, 0.9, 0.7],
            'split1_test_score'  : [0.82, 0.5, 0.7],
            'mean_test_score'    : [0.81, 0.7, 0.7],
            'std_test_score'     : [0.02, 0.2, 0.],
            'rank_test_score'    : [3, 1, 1],
            'split0_train_score' : [0.8, 0.9, 0.7],
            'split1_train_score' : [0.82, 0.5, 0.7],
            'mean_train_score'   : [0.81, 0.7, 0.7],
            'std_train_score'    : [0.03, 0.03, 0.04],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
            'params' : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }

        NOTE that the key ``'params'`` is used to store a list of parameter
        settings dict for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if refit=False.

    best_score_ : float
        Score of best_estimator on the left out data.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

    scorer_ : function
        Scorer function used on the held out data to choose the best
        parameters for the model.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting(and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    --------
    :class:`GridSearchCV`:
        Does exhaustive search over a grid of parameters.

    """

    def __init__(self, estimator, search_spaces, optimizer_kwargs=None,
                 n_iter=50, scoring=None, fit_params=None,
                 n_initial_points=None, n_points=1, n_jobs=1, client=None,
                 iid=True, refit=True, cv=None, verbose=0, random_state=None,
                 error_score='raise', return_train_score=False):

        self.search_spaces = search_spaces
        self.n_iter = n_iter
        self.n_initial_points = n_initial_points
        self.n_points = n_points
        self.random_state = random_state
        self.optimizer_kwargs = optimizer_kwargs
        self.client = client
        self._check_search_space(self.search_spaces)
        # Temporary fix for compatibility with sklearn 0.20 and 0.21
        # See scikit-optimize#762
        # To be consistent with sklearn 0.21+, fit_params should be deprecated
        # in the constructor and be passed in ``fit``.
        self.fit_params = fit_params

        super(BayesSearchCV, self).__init__(
             estimator=estimator, scoring=scoring,
             n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
             pre_dispatch=None, error_score=error_score,
             return_train_score=return_train_score)

    def _check_search_space(self, search_space):
        """Checks whether the search space argument is correct"""

        if len(search_space) == 0:
            raise ValueError(
                "The search_spaces parameter should contain at least one"
                "non-empty search space, got %s" % search_space
            )

        # check if space is a single dict, convert to list if so
        if isinstance(search_space, dict):
            search_space = [search_space]

        # check if the structure of the space is proper
        if isinstance(search_space, list):
            # convert to just a list of dicts
            dicts_only = []

            # 1. check the case when a tuple of space, n_iter is provided
            for elem in search_space:
                if isinstance(elem, tuple):
                    if len(elem) != 2:
                        raise ValueError(
                            "All tuples in list of search spaces should have"
                            "length 2, and contain (dict, dict), got %s" % elem
                        )
                    subspace, param_override = elem

                    if isinstance(param_override, int):
                        warnings.warn(
                            "Overriding n_iter for a particular search pass "
                            "with an integer passed within a tuple is "
                            "deprecated. Instead of this integer, you should "
                            "pass a dictionary with the key 'n_iter' and the "
                            "integer should be the value", DeprecationWarning)
                        param_override = dict(n_iter=param_override)

                    if (not isinstance(param_override, dict)):
                        raise ValueError(
                            "Parameter overriding in search space should be"
                            "given as a dict, got %s in tuple %s " %
                            (param_override, elem)
                        )

                    # save subspaces here for further checking
                    dicts_only.append(subspace)
                elif isinstance(elem, dict):
                    dicts_only.append(elem)
                else:
                    raise TypeError(
                        "A search space should be provided as a dict or"
                        "tuple (dict, dict), got %s" % elem)

            # 2. check all the dicts for correctness of contents
            for subspace in dicts_only:
                for k, v in subspace.items():
                    check_dimension(v)
        else:
            raise TypeError(
                "Search space should be provided as a dict or list of dict,"
                "got %s" % search_space)

    def _init_search_spaces(self):
        # check if space is a single dict, convert to list if so
        search_spaces = self.search_spaces
        if isinstance(search_spaces, dict):
            search_spaces = [search_spaces]

        search_spaces_, n_iters, n_initial_points, n_points = [], [], [], []
        for space in search_spaces:
            space, params = space if isinstance(space, tuple) else (space, {})
            if isinstance(params, int):  # deprecated
                params = dict(n_iter=params)
            n_iter = params.get('n_iter', self.n_iter)
            points = params.get('n_points', self.n_points)
            n_initial = params.get('n_initial', self.n_initial_points)
            if n_initial is None:
                n_initial = points
            if n_initial < points:
                raise ValueError("Number of initial points must be at least "
                                 "equal to n_points")
            search_spaces_.append(space)
            n_initial_points.append(n_initial)
            n_iters.append(n_iter + n_initial)
            n_points.append(points)

        return search_spaces_, n_initial_points, n_iters, n_points

    # copied for compatibility with 0.19 sklearn from 0.18 BaseSearchCV
    @property
    def best_score_(self):
        check_is_fitted(self, 'cv_results_')
        return self.cv_results_['mean_test_score'][self.best_index_]

    # copied for compatibility with 0.19 sklearn from 0.18 BaseSearchCV
    @property
    def best_params_(self):
        check_is_fitted(self, 'cv_results_')
        return self.cv_results_['params'][self.best_index_]

    def _make_optimizers(self, search_spaces):
        """Instantiate skopt Optimizer class.

        Parameters
        ----------
        search_spaces : dict
            Represents parameter search spaces. The keys are parameter
            names (strings) and values are skopt.space.Dimension instances,
            one of Real, Integer or Categorical.
        """

        if self.optimizer_kwargs is None:
            self.optimizer_kwargs_ = {}
        else:
            self.optimizer_kwargs_ = dict(self.optimizer_kwargs)
        random_state = check_random_state(self.random_state)
        if random_state is not None:
            self.optimizer_kwargs_['random_state'] = random_state

        # Instantiate optimizers for all the search spaces.
        optimizers = []
        for search_space in search_spaces:
            if isinstance(search_space, tuple):
                search_space = search_space[0]

            # TODO: different optimizer_kwargs per search space ?
            kwargs = self.optimizer_kwargs_.copy()
            kwargs['dimensions'] = dimensions_aslist(search_space)
            optimizers.append(Optimizer(**kwargs))

        # will save the states of the optimizers
        # TODO: do not attach optimizers to the object by default. Make it
        # configurable. It might not be useful to the user.
        self.optimizers_ = optimizers

    def _ask_new_tasks(self, point_idx, space_idx, space, n_points, X, y,
                       base_estimator, n_splits, point_logs, cv_iter):
        '''
        Generate new points from the optimizer and add them to the queue of
        points to evaluate.
        '''
        params = self.optimizers_[space_idx].ask(n_points=n_points)

        # convert parameters to python native types
        # ???: why ?
        params = [[np.array(v).item() for v in p] for p in params]

        new_tasks = []

        for point_idx, param in enumerate(params, point_idx + 1):

            point_logs[point_idx] = [None] * n_splits
            p_dict = point_asdict(space, param)
            for split_i, (train, test) in cv_iter:

                args = (base_estimator, X, y, self.scorer_, train, test,
                        self.verbose, p_dict)

                cfg = (space_idx, point_idx, split_i, param)

                new_tasks.append((_fit_and_score, cfg, *args))

        return zip(*new_tasks), point_idx

    def _compute(self, client, base_estimator, n_initial_points, n_iters,
                 n_points, X, y, n_splits, search_spaces, cv_iter, callbacks):
        """Generate points asynchronously and log the results.
        """

        optimizers = self.optimizers_
        point_logs = {}  # save the scores for each candidate and each split
        point_idx = -1  # index of points

        _fit_and_score_kwargs = dict(
            fit_params=self.fit_params,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True, return_parameters=True,
            error_score=self.error_score)

        # initialize initial points for each search space
        for space_idx, space in enumerate(search_spaces):
            dispatched, point_idx = self._ask_new_tasks(
                point_idx, space_idx, space, n_initial_points[space_idx], X, y,
                base_estimator, n_splits, point_logs, cv_iter)
            dispatched = client.map(_call_and_get_cfg, *dispatched, pure=False,
                                    **_fit_and_score_kwargs)
            dispatched = as_completed(dispatched, with_results=True)

        # now get asynchronously the result and create new tasks from them
        # save the params and the mean scores for each search space
        space_arriving_results = [([], []) for space in search_spaces]
        n_iters_queued = n_initial_points
        while n_iters_queued < n_iters:
            res = dispatched.next()[1]
            (space_idx, point_idx_out, split_idx, param) = res[-4:]
            point_log = point_logs[point_idx_out]
            point_log[split_idx] = res
            # if all scores for all splits for all points of a batch
            # are available we can compute the means, tell the
            # optimizer and ask for a new batch.
            scores = [res[-9] for res in point_log if res is not None]
            if len(scores) == n_splits:
                mean_score = np.mean(scores)
                params, scores = space_arriving_results[space_idx]
                params.append(param)
                scores.append(-mean_score)
                remaining_iter = n_iters[space_idx] - n_iters_queued[space_idx]
                if len(scores) == n_points[space_idx] and remaining_iter > 0:
                    space_arriving_results[space_idx] = ([], [])
                    optim_result = optimizers[space_idx].tell(params, scores)
                    if eval_callbacks(callbacks, optim_result):
                        # TODO: we should also cancel the futures that are
                        # still dispatched for this search_space.
                        n_iters_queued[space_idx] = n_iters[space_idx]
                        continue
                    n_points_ = min(n_points[space_idx], remaining_iter)
                    point_idx = self._ask_new_tasks(
                        point_idx, space_idx, search_spaces[space_idx],
                        n_points_, X, y, base_estimator, n_splits, point_logs,
                        cv_iter)
                    dispatched.update(client.map(
                        _call_and_get_cfg, *point_idx[0], pure=False,
                        **_fit_and_score_kwargs))
                    point_idx = point_idx[1]
                    n_iters_queued[space_idx] += n_points_

        for res in dispatched:
            res = res[1]
            (space_idx, point_idx_out, split_idx, param) = res[-4:]
            point_logs[point_idx_out][split_idx] = res

        return [res for point in point_logs.values() for res in point]

    @property
    def total_iterations(self):
        """
        Count total iterations that will be taken to explore
        all subspaces with `fit` method.

        Returns
        -------
        max_iter: int, total number of iterations to explore
        """
        total_iter = 0
        for elem in self.search_spaces:
            n_iter = self.n_iter
            if isinstance(elem, tuple):
                n_iter = elem[1]
                if isinstance(n_iter, dict):  # are other cases are deprecated
                    n_iter = elem[1]['n_iter']
            total_iter += n_iter
        return total_iter

    def _run_search(self, x):
        # ???: surely this could be removed ?
        pass

    def fit(self, X, y=None, groups=None, callback=None):
        """Run fit on the estimator with randomly drawn parameters.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples] or [n_samples, n_output]
            Target relative to X for classification or regression (class
            labels should be integers or strings).

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        callback: [callable, list of callables, optional]
            If callable then `callback(res)` is called after each parameter
            combination tested. If list of callables, then each callable in
            the list is called.
        """

        search_spaces, n_initial_points, n_iters, n_points = \
            self._init_search_spaces()

        callbacks = check_callback(callback)

        self._make_optimizers(search_spaces)

        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))
        self.multimetric_ = False

        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)
        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)
        n_candidates = sum(n_iters)
        if self.verbose > 0:
            print("Fitting {0} folds for each of {1} candidates, totalling"
                  " {2} fits".format(n_splits, n_candidates,
                                     n_candidates * n_splits))
        n_jobs = self.n_jobs
        # account for case n_jobs < 0
        if n_jobs < 0:
            n_jobs = max(1, cpu_count() + n_jobs + 1)

        base_estimator = clone(self.estimator)

        with _ManageBackend(self.client, n_workers=n_jobs) as client:
            # XXX: should we also scatter all the constant parameters of
            # _call_and_get_cfg ?
            cv_iter = [(idx, client.scatter([train, test]))
                       for idx, (train, test)
                       in enumerate(cv.split(X, y, groups))]
            X_, y_, groups = client.scatter([X, y, groups])
            out = self._compute(
                client, base_estimator, n_initial_points, n_iters, n_points,
                X_, y_, n_splits, search_spaces, cv_iter, callbacks)

        # sort by candidate and then by split.
        out = map(itemgetter(slice(None, -4)),
                  sorted(out, key=itemgetter(-3, -2)))

        # All the following is copy/pasted from sklearn. Beware to
        # retrocompatibility.
        # If one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            (train_scores, test_scores, test_sample_counts,
             fit_time, score_time, parameters) = zip(*out)
        else:
            (test_scores, test_sample_counts,
             fit_time, score_time, parameters) = zip(*out)

        candidate_params = parameters[::n_splits]
        n_candidates_fitted = len(candidate_params)
        print("%d candidates were fitted successfully" % n_candidates_fitted)

        results = dict()

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            array = np.array(array, dtype=np.float64).reshape(
                             n_candidates_fitted, n_splits)
            if splits:
                for split_i in range(n_splits):
                    results["split%d_%s"
                            % (split_i, key_name)] = array[:, split_i]

            array_means = np.average(array, axis=1, weights=weights)
            results['mean_%s' % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(np.average((array -
                                             array_means[:, np.newaxis]) ** 2,
                                            axis=1, weights=weights))
            results['std_%s' % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method='min'), dtype=np.int32)

        # Computed the (weighted) mean and std for test scores alone
        # NOTE test_sample counts (weights) remain the same for all candidates
        test_sample_counts = np.array(test_sample_counts[:n_splits],
                                      dtype=np.int)

        _store('test_score', test_scores, splits=True, rank=True,
               weights=test_sample_counts if self.iid else None)
        if self.return_train_score:
            _store('train_score', train_scores, splits=True)
        _store('fit_time', fit_time)
        _store('score_time', score_time)

        best_index = np.flatnonzero(results["rank_test_score"] == 1)[0]
        best_parameters = candidate_params[best_index]

        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(
            MaskedArray,
            np.empty(n_candidates_fitted,),
            mask=True,
            dtype=object))

        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)

        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        self.cv_results_ = results
        self.best_index_ = best_index
        self.n_splits_ = n_splits

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = clone(base_estimator).set_params(
                **best_parameters)
            if y is not None:
                best_estimator.fit(X, y, **(self.fit_params or {}))
            else:
                best_estimator.fit(X, **(self.fit_params or {}))
            self.best_estimator_ = best_estimator

        return self
