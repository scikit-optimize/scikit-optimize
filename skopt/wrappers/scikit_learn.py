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
    It is highly recommended to use continuous distributions for continuous
    parameters.


    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    n_iter : int, default=128
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

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

    def __init__(self, estimator, param_distributions, surrogate="default", n_iter=128, scoring=None,
                 fit_params=None, n_jobs=1, iid=True, refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', random_state=None,
                 error_score='raise', return_train_score=True):
        self.param_space = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        self.surrogate = surrogate

        self.optimizer = {}
        self.cv_results_ = defaultdict(list)

        self.best_index_ = None

        super(SkoptSearchCV, self).__init__(
             estimator=estimator, scoring=scoring, fit_params=fit_params,
             n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
             pre_dispatch=pre_dispatch, error_score=error_score,
             return_train_score=return_train_score)

    def _fit_best_model(self, X, y):
        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_).fit(X, y)

    def _make_optimizer(self, params_space):
        dimensions = [params_space[k] for k in sorted(params_space.keys())]

        if self.surrogate == "default":
            surrogate = GaussianProcessRegressor()
        else:
            surrogate = self.surrogate

        return Optimizer(dimensions, surrogate, acq_optimizer='sampling')

    def _skopt_to_dict(self, params_space, params):
        params_dict = {k: v for k,v in zip(sorted(params_space.keys()), params)}
        return params_dict

    def step(self, X, y, param_space, groups=None, n_jobs=1):
        """
        Having a separate function for a single step for search allows to
        save easily checkpoints for the parameter search and restore from
        possible failures. This provides additional flexibility with
        stopping criterion of the search.

        :param X:
        :param y:
        :param param_space:
        :param groups:
        :return:
        """

        # account for case n_jobs < 0
        if n_jobs < 0:
            n_jobs = max(1, cpu_count() + n_jobs + 1)

        cv = skms.check_cv(self.cv, y, classifier=skms.is_classifier(self.estimator))
        self.scorer_ = skms.check_scoring(self.estimator, scoring=self.scoring)

        key = str(param_space)

        if key not in self.optimizer:
            self.optimizer[key] =  self._make_optimizer(param_space)

        optimizer = self.optimizer[key]

        params_list = optimizer.ask(n_points=n_jobs)

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

        for i, params in enumerate(params_list):
            I = slice(i*splits, (i+1)*splits, 1)

            score = np.mean(test_scores[I])
            score_std = np.std(test_scores[I])

            self.cv_results_['params'].append(self._skopt_to_dict(param_space, params))
            self.cv_results_['mean_test_score'].append(score)
            self.cv_results_['std_test_score'].append(score_std)

            if self.best_index_ is None or score > self.best_score_:
                self.best_index_ = len(self.cv_results_['params'])-1

            optimizer.tell(params, -score)

        if self.refit:
            self._fit_best_model(X, y)


    def fit(self, X, y=None, groups=None):
        """Run fit on the estimator with randomly drawn parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        """

        n_jobs = self.n_jobs

        # account for case n_jobs < 0
        if n_jobs < 0:
            n_jobs = max(1, cpu_count() + n_jobs + 1)

        for elem in self.param_space:

            # if tuple
            if isinstance(elem, tuple):
                psp, n_iter = elem
            elif isinstance(elem, dict):
                psp, n_iter = elem, self.n_iter
            else:
                raise ValueError("Unsupported type of parameter space. "
                                 "Expected tuple or dict, got " + str(elem))

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
        n_jobs=1, n_iter=32,
    )

    opt.fit(X_train, y_train)
    print(opt.score(X_test, y_test))