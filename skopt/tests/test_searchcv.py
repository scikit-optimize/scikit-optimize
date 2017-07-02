"""Test scikit-optimize based implementation of hyperparameter
search with interface similar to those of GridSearchCV
"""

import pytest

from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_greater
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

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

    opt = BayesSearchCV(
        SVC(),
        [
            ({
                'C': Real(1e-6, 1e+6, prior='log-uniform'),
                'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            },10),
            {
                'degree': Integer(1, 8),
                'kernel': Categorical(['linear', 'poly', 'rbf']),
            }
        ],
        n_iter=10
    )

    opt.fit(X_train, y_train)

