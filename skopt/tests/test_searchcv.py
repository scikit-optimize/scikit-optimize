"""Test scikit-optimize based implementation of hyperparameter
search with interface similar to those of GridSearchCV
"""

import pytest

from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_equal
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV


@pytest.mark.parametrize("surrogate", ['gp', None])
@pytest.mark.parametrize("n_jobs", [1, -1])  # test sequential and parallel
def test_searchcv_runs(surrogate, n_jobs):
    """
    Test whether the cross validation search wrapper around sklearn
    models runs properly with available surrogates and with single
    or multiple workers.

    Parameters
    ----------

    * `surrogate` [str or None]:
        A class of the scikit-optimize surrogate used. None means
        to use default surrogate.

    * `n_jobs` [int]:
        Number of parallel processes to use for computations.

    """

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=0
    )

    # None search space is only supported when only `step` function is used
    assert_raises(ValueError, BayesSearchCV(SVC(), None).fit, (X, y))

    # check if invalid dimensions are raising errors
    with pytest.raises(ValueError):
        BayesSearchCV(SVC(), {'C': '1 ... 100.0'})

    with pytest.raises(TypeError):
        BayesSearchCV(SVC(), ['C', (1.0, 1)])

    # create an instance of a surrogate if it is not a string
    if surrogate is not None:
        optimizer_kwargs = {'base_estimator': surrogate}
    else:
        optimizer_kwargs = None

    opt = BayesSearchCV(
        SVC(),
        {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 8),
            'kernel': Categorical(['linear', 'poly', 'rbf']),
        },
        n_jobs=n_jobs, n_iter=11,
        optimizer_kwargs=optimizer_kwargs
    )

    opt.fit(X_train, y_train)

    # this normally does not hold only if something is wrong
    # with the optimizaiton procedure as such
    assert_greater(opt.score(X_test, y_test), 0.9)


def test_searchcv_runs_multiple_subspaces():
    """
    Test whether the BayesSearchCV runs without exceptions when
    multiple subspaces are given.
    """

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=0
    )

    # used to try different model classes
    pipe = Pipeline([
        ('model', SVC())
    ])

    # single categorical value of 'model' parameter sets the model class
    lin_search = {
        'model': Categorical([LinearSVC()]),
        'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
    }

    # example subspace with short - hand notation
    dtc_search = {
        'model': [DecisionTreeClassifier()],
        'model__max_depth': (1, 32),
        'model__min_samples_split': (1e-3, 1.0, 'log-uniform'),
    }

    # mixed short - hand and full notations
    svc_search = {
        'model': Categorical([SVC()]),
        'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
        'model__gamma': (1e-6, 1e+1, 'log-uniform'),
        'model__degree': Integer(1, 8),
        'model__kernel': ['linear', 'poly', 'rbf'],
    }

    opt = BayesSearchCV(
        pipe,
        [(lin_search, 1), (dtc_search, 1), svc_search],
        n_iter=2
    )

    # run the search over subspaces
    opt.fit(X_train, y_train)

    # test if all subspaces are explored
    total_evaluations = len(opt.cv_results_['mean_test_score'])
    assert total_evaluations == 1+1+2, "Not all spaces were explored!"


def test_searchcv_iterations():
    """
    Test whether the `monitor` callbacks are called in BayesSearchCV
    on space exploration with proper values of total number of
    iterations to explore and current number of iterations.
    """

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=0
    )

    # used to try different model classes
    pipe = Pipeline([
        ('model', SVC())
    ])

    # single categorical value of 'model' parameter sets the model class
    lin_search = {
        'model': Categorical([LinearSVC()]),
        'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
    }

    # example subspace with short - hand notation
    dtc_search = {
        'model': [DecisionTreeClassifier()],
        'model__max_depth': (1, 32),
        'model__min_samples_split': (1e-3, 1.0, 'log-uniform'),
    }

    # mixed short - hand and full notations
    svc_search = {
        'model': Categorical([SVC()]),
        'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
        'model__gamma': (1e-6, 1e+1, 'log-uniform'),
        'model__degree': Integer(1, 8),
        'model__kernel': ['linear', 'poly', 'rbf'],
    }

    # a few values that should persist among function calls
    persistent_values = {
        'calls_counter': 0
    }

    def on_step(optim_result):
        persistent_values['calls_counter'] += 1
        iter = optim_result.searchcv_iter
        assert iter == persistent_values['calls_counter']


    opt = BayesSearchCV(
        pipe,
        [(lin_search, 1), (dtc_search, 1), svc_search],
        n_iter=2,
    )

    # run the search over subspaces
    opt.fit(X_train, y_train, callback=on_step)

    # test if for all iterations function is called
    total_evaluations = len(opt.cv_results_['mean_test_score'])
    assert persistent_values['calls_counter'] == total_evaluations
    assert persistent_values['calls_counter'] == opt.total_iterations


def test_searchcv_fit_count():
    """
    Test whether BayesSearchCV does not do any unnecessary
    fitting.
    """

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=0
    )

    # class used to record the number of fittings
    persistent_values = {'fits_counter':0}
    n_iter = 2

    class MySVC(LinearSVC):
        def __init__(self, C=0.1):
            LinearSVC.__init__(self, C=C)

        def fit(self, X, y, sample_weight=None):
            persistent_values['fits_counter'] += 1
            LinearSVC.fit(self, X, y, sample_weight)

    model = BayesSearchCV(
        estimator=MySVC(),
        search_spaces={
            'C': (0.1, 1.0)
        },
        n_iter=n_iter,
        refit=True
    )

    model.fit(X_train, y_train)

    # 3 cross - validation folds fits for every iteration
    # + one final fit
    assert persistent_values['fits_counter']==n_iter*3 + 1

