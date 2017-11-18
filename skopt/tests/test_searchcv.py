"""Test scikit-optimize based implementation of hyperparameter
search with interface similar to those of GridSearchCV
"""

import pytest

from sklearn.utils.testing import assert_greater, assert_equal
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

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

    # check if empty search space is raising errors
    with pytest.raises(ValueError):
        BayesSearchCV(SVC(), {})

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

    dtc_search = {
        'model': Categorical([DecisionTreeClassifier()]),
        'model__max_depth': Integer(1, 32),
        'model__min_samples_split': Real(1e-3, 1.0, prior='log-uniform'),
    }

    svc_search = {
        'model': Categorical([SVC()]),
        'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
        'model__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
        'model__degree': Integer(1, 8),
        'model__kernel': Categorical(['linear', 'poly', 'rbf']),
    }

    opt = BayesSearchCV(
        pipe,
        [(lin_search, 1), (dtc_search, 1), svc_search],
        n_iter=2
    )

    opt.fit(X_train, y_train)

    # test if all subspaces are explored
    total_evaluations = len(opt.cv_results_['mean_test_score'])
    assert total_evaluations == 1+1+2, "Not all spaces were explored!"


def test_searchcv_sklearn_compatibility():
    """
    Test whether the BayesSearchCV is compatible with base sklearn methods
    such as clone, set_params, get_params.
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

    dtc_search = {
        'model': Categorical([DecisionTreeClassifier()]),
        'model__max_depth': Integer(1, 32),
        'model__min_samples_split': Real(1e-3, 1.0, prior='log-uniform'),
    }

    svc_search = {
        'model': Categorical([SVC()]),
        'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
        'model__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
        'model__degree': Integer(1, 8),
        'model__kernel': Categorical(['linear', 'poly', 'rbf']),
    }

    opt = BayesSearchCV(
        pipe,
        [(lin_search, 1), svc_search],
        n_iter=2
    )

    opt_clone = clone(opt)

    params, params_clone = opt.get_params(), opt_clone.get_params()
    assert params.keys() == params_clone.keys()

    for param, param_clone in zip(params.items(), params_clone.items()):
        assert param[0] == param_clone[0]
        assert isinstance(param[1], type(param_clone[1]))

    opt.set_params(search_spaces=[(dtc_search, 1)])

    opt.fit(X_train, y_train)
    opt_clone.fit(X_train, y_train)

    total_evaluations = len(opt.cv_results_['mean_test_score'])
    total_evaluations_clone = len(opt_clone.cv_results_['mean_test_score'])

    # test if expected number of subspaces is explored
    assert total_evaluations == 1
    assert total_evaluations_clone == 1+2


def test_searchcv_reproducibility():
    """
    Test whether results of BayesSearchCV can be reproduced with a fixed
    random state.
    """

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=0
    )

    random_state = 42

    opt = BayesSearchCV(
        SVC(random_state=random_state),
        {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 8),
            'kernel': Categorical(['linear', 'poly', 'rbf']),
        },
        n_iter=11, random_state=random_state
    )

    opt.fit(X_train, y_train)
    best_estimator = opt.best_estimator_

    opt2 = clone(opt).fit(X_train, y_train)
    best_estimator2 = opt2.best_estimator_

    for attr in ('C', 'gamma', 'degree', 'kernel'):
        assert_equal(getattr(best_estimator, attr),
                     getattr(best_estimator2, attr))
