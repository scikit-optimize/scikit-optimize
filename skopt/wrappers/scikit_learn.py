from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor, ExtraTreesRegressor

from sklearn.base import clone
from sklearn.externals.joblib import Parallel, delayed, cpu_count
import sklearn.model_selection._search as skms

import numpy as np
from collections import *


class SkoptSearchCV(skms.BaseSearchCV):
    """Bayesian optimization over hyper parameters.

    SkoptSearchCV implements a "fit" and a "score" method.
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
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
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

    def __init__(self, estimator, search_spaces, surrogate="auto", n_iter=128, scoring=None,
                 fit_params=None, n_jobs=1, iid=True, refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', random_state=None,
                 error_score='raise', return_train_score=True):
        self.search_spaces = search_spaces
        self.n_iter = n_iter
        self.random_state = random_state
        self.surrogate = surrogate

        # this dict is used in order to keep track of skopt Optimizer
        # instances for different search spaces (str(space) is used as key)
        self.optimizer = {}
        self.cv_results_ = defaultdict(list)

        self.best_index_ = None

        super(SkoptSearchCV, self).__init__(
             estimator=estimator, scoring=scoring, fit_params=fit_params,
             n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
             pre_dispatch=pre_dispatch, error_score=error_score,
             return_train_score=return_train_score)

    def _fit_best_model(self, X, y):
        """Fits the estimator copy with best parameters found to the
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
        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_).fit(X, y)
        return self

    def _make_optimizer(self, params_space):
        """Method to instantiate skopt Optimizer class.

        Parameters
        ----------
        params_space : dict
            Represents parameter search space. The keys are parameter
            names (strings) and values are skopt.space.Dimension instances,
            one of Real, Integer or Categorical.

        y : array-like, shape = [n_samples] or [n_samples, n_output],
            Target relative to X for classification or regression.

        Returns
        -------
        optimizer: Instance of the `Optimizer` class used for for search
            in some parameter space.

        """
        # convert search space from dict to list
        dimensions = [params_space[k] for k in sorted(params_space.keys())]

        if self.surrogate == "auto":
            surrogate = GaussianProcessRegressor()
        else:
            surrogate = self.surrogate

        optimizer = Optimizer(dimensions, surrogate, acq_optimizer='sampling')
        return optimizer

    def _skopt_to_dict(self, params_space, params):
        """Converts list of parameter values into the dictionary with
        keys as parameter names and corresponding values of parameter.

        Parameters
        ----------
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
        params_dict = {k: v for k,v in zip(sorted(params_space.keys()), params)}
        return params_dict

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

        # check parameters; taken from BaseSearchCV.
        cv = skms.check_cv(self.cv, y, classifier=skms.is_classifier(self.estimator))
        self.scorer_ = skms.check_scoring(self.estimator, scoring=self.scoring)

        # use the cached optimizer for particular parameter space
        key = str(param_space)
        if key not in self.optimizer:
            self.optimizer[key] = self._make_optimizer(param_space)
        optimizer = self.optimizer[key]

        # get parameter values to evaluate
        params_list = optimizer.ask(n_points=n_jobs)

        # run the evaluation
        cv_iter = list(cv.split(X, y, groups))
        out = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose,
            pre_dispatch=self.pre_dispatch
        )(delayed(skms._fit_and_score)(clone(self.estimator), X, y, self.scorer_,
                                  train, test, self.verbose, self._skopt_to_dict(param_space, params),
                                  fit_params=self.fit_params,
                                  return_train_score=self.return_train_score,
                                  return_n_test_samples=True,
                                  return_times=True, return_parameters=True,
                                  error_score=self.error_score)
          for params in params_list
          for train, test in cv_iter)

        # if one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            (train_scores, test_scores, test_sample_counts,
             fit_time, score_time, parameters) = zip(*out)
        else:
            (test_scores, test_sample_counts,
             fit_time, score_time, parameters) = zip(*out)

        splits = int(len(train_scores) / n_jobs)

        # this is used later in order to record results of
        # evaluation in cv_results_
        out_and_name = [
            ('test_score', test_scores),
            ('test_sample_count', test_sample_counts),
            ('fit_time', fit_time),
            ('score_time', score_time),
        ]

        if self.return_train_score:
            out_and_name = out_and_name + [('train_score', train_scores)]

        # record all results in cv_results_
        for i, params in enumerate(params_list):
            # a slice that corresponds to results with different
            # validation splits for some particular parameter values
            I = slice(i*splits, (i+1)*splits, 1)

            # record results for particular parameters point
            self.cv_results_['params'].append(self._skopt_to_dict(param_space, params))
            for name, result in out_and_name:
                for split, data in enumerate(result[I]):
                    self.cv_results_['split'+str(split)+"_"+name].append(data)
                self.cv_results_['mean_' + name].append(np.mean(result[I]))
                self.cv_results_['std_' + name].append(np.std(result[I]))

            # update index of best parameters if necessary
            score = np.mean(test_scores[I])
            if self.best_index_ is None or score > self.best_score_:
                self.best_index_ = len(self.cv_results_['params'])-1

            # feed the point and objective back into optimizer
            optimizer.tell(params, -score)

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
                psp, n_iter = elem
            # if dict: represents search space
            elif isinstance(elem, dict):
                psp, n_iter = elem, self.n_iter
            else:
                raise ValueError("Unsupported type of parameter space. "
                                 "Expected tuple or dict, got " + str(elem))

            # do the optimization for particular search space
            while n_iter:
                n_iter -= n_jobs
                self.step(X, y, psp, groups=groups, n_jobs=self.n_jobs)

if __name__ == "__main__":
    from skopt.space import Real, Categorical, Integer
    from skopt.wrappers import SkoptSearchCV

    from sklearn.datasets import load_iris
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)

    opt = SkoptSearchCV(
        SVC(),
        [{
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 8),
            'kernel': Categorical(['linear', 'poly', 'rbf']),
        }],
        n_jobs=2, n_iter=32, verbose=2
    )

    opt.fit(X_train, y_train)
    print(opt.score(X_test, y_test))

