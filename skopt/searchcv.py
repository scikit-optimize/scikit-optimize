from collections import defaultdict

import numpy as np

from sklearn.base import clone
from sklearn.externals.joblib import Parallel, delayed, cpu_count
import sklearn.model_selection._search as sk_model_sel

from skopt import Optimizer
from skopt.utils import point_asdict, dimensions_aslist

class BayesSearchCV(sk_model_sel.BaseSearchCV):
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

    search_spaces : list of dict or tuple
        Either list  of dictionary objects or list of tuples of
        (dict, int > 0). The keys of every dictionary are parameter
        names (strings) and values are skopt.space.Dimension instances
        (Real, Integer or Categorical) which represents search space
        for particular parameter.
        If a list of dictionary objects is given, then the search is
        performed sequentially for every parameter space with maximum
        number of evaluations set to self.n_iter. Alternatively, if
        a list of (dict, int > 0) is given, the search is done for
        every search space for number of iterations given as a second
        element of tuple.

    n_iter : int, default=128
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    surrogate : string or skopt surrogate, default='auto'
        Surrogate to use for optimization of score of estimator.
        By default skopt.learning.GaussianProcessRegressor() is used.

    scoring : string, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    n_jobs : int, default=1
        Number of jobs to run in parallel.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

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

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

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

    return_train_score : boolean, default=True
        If ``'False'``, the ``cv_results_`` attribute will not include training
        scores.

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
                 n_iter=256, scoring=None, fit_params=None, n_jobs=1,
                 iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', random_state=None,
                 error_score='raise', return_train_score=True):
        self.search_spaces = search_spaces
        self.n_iter = n_iter
        self.random_state = random_state

        if optimizer_kwargs is None:
            self.optimizer_kwargs = {}
        else:
            self.optimizer_kwargs = optimizer_kwargs

        # this dict is used in order to keep track of skopt Optimizer
        # instances for different search spaces. `str(space)` is used as key
        # as `space` is a dict, which is unhashable. however, str(space)
        # is fixed and unique if space is not changed.
        self.optimizer_ = {}
        self.cv_results_ = defaultdict(list)

        self.best_index_ = None

        super(BayesSearchCV, self).__init__(
             estimator=estimator, scoring=scoring, fit_params=fit_params,
             n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
             pre_dispatch=pre_dispatch, error_score=error_score,
             return_train_score=return_train_score)

    def _fit_best_model(self, X, y):
        """Fit the estimator copy with best parameters found to the
        provided data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output],
            Target relative to X for classification or regression.

        Returns
        -------
        self
        """
        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.set_params(**self.best_params_)
        self.best_estimator_.fit(X, y)
        return self

    def _make_optimizer(self, params_space):
        """Instantiate skopt Optimizer class.

        Parameters
        ----------
        params_space : dict
            Represents parameter search space. The keys are parameter
            names (strings) and values are skopt.space.Dimension instances,
            one of Real, Integer or Categorical.

        Returns
        -------
        optimizer: Instance of the `Optimizer` class used for for search
            in some parameter space.

        """

        kwargs = self.optimizer_kwargs.copy()
        kwargs['dimensions'] = dimensions_aslist(params_space)
        optimizer = Optimizer(**kwargs)

        return optimizer


    def step(self, X, y, param_space, groups=None, n_jobs=1):
        """Generates n_jobs parameters and evaluates corresponding
        estimators in parallel.

        Having a separate function for a single step for search allows to
        save easily checkpoints for the parameter search and restore from
        possible failures.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels) as integers or strings.

        params_space : dict
            Represents parameter search space. The keys are parameter
            names (strings) and values are skopt.space.Dimension instances,
            one of Real, Integer or Categorical.

        params : list
            Parameter values as list. The order of parameters in the list
            is given by sorted(params_space.keys()).

        Returns
        -------
        params_dict: dictionary with parameter values.
        """

        # convert n_jobst to int > 0 if necessary
        if n_jobs < 0:
            n_jobs = max(1, cpu_count() + n_jobs + 1)

        # use the cached optimizer for particular parameter space
        key = str(param_space)
        if key not in self.optimizer_:
            self.optimizer_[key] = self._make_optimizer(param_space)
        optimizer = self.optimizer_[key]

        # get parameter values to evaluate
        params = optimizer.ask(n_points=n_jobs)
        params_dict = [point_asdict(param_space, p) for p in params]

        # self.cv_results_ is reset at every call to _fit, keep current
        all_cv_results = self.cv_results_

        # record performances with different points
        refit = self.refit
        self.refit = False # do not fit yet - will be fit later
        self._fit(X, y, groups, params_dict)
        self.refit = refit

        # merge existing and new cv_results_
        for k in self.cv_results_:
            all_cv_results[k].extend(self.cv_results_[k])

        self.cv_results_ = all_cv_results
        self.best_index_ = np.argmax(self.cv_results_['mean_test_score'])

        # feed the point and objective back into optimizer
        local_results = self.cv_results_['mean_test_score'][-len(params):]

        # optimizer minimizes objective, hence provide negative score
        optimizer.tell(params, [-score for score in local_results])

        # fit the best model if necessary
        if self.refit:
            self._fit_best_model(X, y)

    def fit(self, X, y=None, groups=None):
        """Run fit on the estimator with randomly drawn parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output]
            Target relative to X for classification or regression;

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        """

        # check if the list of parameter spaces is provided. If not, then
        # only step in manual mode can be used.

        if self.search_spaces is None:
            raise ValueError(
                "None search space is only supported with manual usage of "
                "`step` method. Please provide list of search spaces "
                " at initialization stage in order to be able to use"
                "`fit` method."
            )

        n_jobs = self.n_jobs

        # account for case n_jobs < 0
        if n_jobs < 0:
            n_jobs = max(1, cpu_count() + n_jobs + 1)

        for elem in self.search_spaces:

            # if tuple: (dict: search space, int: n_iter)
            if isinstance(elem, tuple):
                parameter_search_space, n_iter = elem
            # if dict: represents search space
            elif isinstance(elem, dict):
                parameter_search_space, n_iter = elem, self.n_iter
            else:
                raise ValueError(
                    "Unsupported type of parameter space. "
                    "Expected tuple (dict, int) or dict, got %s " + elem)

            # do the optimization for particular search space
            while n_iter > 0:
                # when n_iter < n_jobs points left for evaluation
                n_jobs_adjusted = min(n_iter, self.n_jobs)

                self.step(
                    X, y, parameter_search_space,
                    groups=groups, n_jobs=n_jobs_adjusted
                )
                n_iter -= n_jobs
