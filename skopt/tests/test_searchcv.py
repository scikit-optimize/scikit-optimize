"""This script contains set of functions that test parallel optimization with
skopt, where constant liar parallelization strategy is used.
"""


from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_greater

from skopt.space import Real
from skopt import Optimizer
from skopt.benchmarks import branin
import skopt.learning as sol

from scipy.spatial.distance import pdist
import pytest

# Extract available surrogates, so that new ones are used automatically
available_surrogates = [
    getattr(sol, name) for name in sol.__all__
    if "GradientBoostingQuantileRegressor" not in name
]

# include the "auto" surrogate to test
available_surrogates += ["auto"]

@pytest.mark.parametrize("surrogate", available_surrogates)  # test with all available surrogates
@pytest.mark.parametrize("n_jobs", [1,-1])  # test in sequential and parallel
def test_constant_liar_runs(surrogate, n_jobs):
    """
    Tests whether the cross validation search wrapper around sklearn
     runs properly during the randominitialization phase and beyond

    Parameters
    ----------

    * `surrogate` [scikit-optimize surrogate class]:
        A class of the scikit-optimize surrogate used.

    * `n_jobs` [int]:
        Number of parallel processes to use for computations.

    """
    from skopt.space import Real, Categorical, Integer
    from skopt.wrappers import SkoptSearchCV

    from sklearn.datasets import load_iris
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)

    # None search space is only supported when `step` function is called manually
    assert_raises(ValueError, SkoptSearchCV(SVC(), None).fit, (X, y))

    # create an instance of a surrogate if it is not a string
    surrogate_input = surrogate if isinstance(surrogate, str) else surrogate()

    opt = SkoptSearchCV(
        SVC(),
        [{
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 8),
            'kernel': Categorical(['linear', 'poly', 'rbf']),
        }],
        n_jobs=n_jobs, n_iter=64,
        surrogate=surrogate_input
    )

    opt.fit(X_train, y_train)

    # this normally does not hold only if something is wrong
    # with the optimizaiton procedure as such
    assert_greater(opt.score(X_test, y_test), 0.9)
