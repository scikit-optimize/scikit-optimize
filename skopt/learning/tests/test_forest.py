import numpy as np
from functools import partial
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal

from skopt.learning import RandomForestRegressor
from skopt.learning import ExtraTreesRegressor


def check_variance_toy_data(Regressor):
    # Split into [2, 3, 4] and [100, 103, 106]
    X = [[2.0, 1.], [3.0, 1.0], [4., 1.0],
         [109.0, 1.0], [110.0, 1.0], [111., 1.]]
    y = [2, 3, 4, 100, 103, 106]

    reg = Regressor(max_depth=1, random_state=1)
    reg.fit(X, y)

    pred, var = reg.predict(X, return_std=True)
    assert_array_equal(pred, [3, 3, 3, 103, 103, 103])
    assert_array_almost_equal(
        var, np.sqrt([0.666667, 0.666667, 0.666667, 6.0, 6.0, 6.0]))


def test_variance_toy_data():
    """Test that `return_std` behaves expected on toy data."""
    for Regressor in [partial(RandomForestRegressor, bootstrap=False),
                      ExtraTreesRegressor]:
        yield check_variance_toy_data, Regressor


def check_variance_no_split(Regressor):
    rng = np.random.RandomState(0)
    X = np.ones((1000, 1))
    y = rng.normal(size=(1000,))

    reg = Regressor(random_state=0, max_depth=3)
    reg.fit(X, y)

    pred, std = reg.predict(X, return_std=True)
    assert_array_almost_equal([np.std(y)] * 1000, std)
    assert_array_almost_equal([np.mean(y)] * 1000, pred)


def test_variance_no_split():
    """
    Test that `return_std` behaves expected on a tree with one node.

    The decision tree should not produce a split, because there is
    no information gain which enables us to verify the mean and
    standard deviation.
    """
    for Regressor in [partial(RandomForestRegressor, bootstrap=False),
                      ExtraTreesRegressor]:
        yield check_variance_no_split, Regressor


def test_min_variance():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(1000, 1))
    y = np.ones(1000)
    rf = RandomForestRegressor(min_variance=0.1)
    rf.fit(X, y)
    mean, std = rf.predict(X, return_std=True)
    assert_array_almost_equal(mean, y)
    assert_array_almost_equal(std, np.sqrt(0.1*np.ones(1000)))
