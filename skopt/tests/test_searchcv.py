"""This script contains set of functions that test scikit-optimize
based implementation of parameter search with interface similar to
those of GridSearchCV
"""

from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_greater
import pytest

# Extract available surrogates, so that new ones are used automatically
from test_optimizer import ESTIMATOR_STRINGS

# None means to use default surrogate
available_surrogates = [ESTIMATOR_STRINGS[0], None]


@pytest.mark.parametrize("surrogate", available_surrogates)
@pytest.mark.parametrize("n_jobs", [1, -1])  # test sequential and parallel
def test_searchcv_runs(surrogate, n_jobs):
    """
    Tests whether the cross validation search wrapper around sklearn
    models runs properly with available surrogates and with single
    or multiple workers.

    Parameters
    ----------

    * `surrogate` [scikit-optimize surrogate class]:
        A class of the scikit-optimize surrogate used.

    * `n_jobs` [int]:
        Number of parallel processes to use for computations.

    """
    from skopt.space import Real, Categorical, Integer
    from skopt import BayesSearchCV

    from sklearn.datasets import load_iris
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split

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
        [{
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 8),
            'kernel': Categorical(['linear', 'poly', 'rbf']),
        }],
        n_jobs=n_jobs, n_iter=11,
        optimizer_kwargs=optimizer_kwargs
    )

    opt.fit(X_train, y_train)

    # this normally does not hold only if something is wrong
    # with the optimizaiton procedure as such
    assert_greater(opt.score(X_test, y_test), 0.9)
