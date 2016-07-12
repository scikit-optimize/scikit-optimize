import numpy as np
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_raises

from skopt.learning import DecisionTreeRegressor


def test_variance_toy_data():
    """Test that return_std behaves expected on toy data."""

    # Split into [2, 3, 4] and [100, 103, 106]
    X = [[2.0, 1.], [3.0, 1.0], [4., 1.0], [9.0, 1.0], [10.0, 1.0], [11., 1.]]
    y = [2, 3, 4, 100, 103, 106]
    dt = DecisionTreeRegressor(max_depth=1, random_state=0)
    dt.fit(X, y)
    pred, var = dt.predict(X, return_std=True)
    assert_array_equal(pred, [3, 3, 3, 103, 103, 103])
    assert_array_almost_equal(
        var, np.sqrt([0.666667, 0.666667, 0.666667, 6.0, 6.0, 6.0]))


def test_variance_no_split():
    """
    Test that return_std behaves expected on a tree with one node.

    The decision tree should not produce a split, because there is
    no information gain which enables us to verify the mean and
    standard deviation.
    """
    rng = np.random.RandomState(0)
    X = np.ones((1000, 1))
    y = rng.normal(size=(1000,))

    # Max depth parameter is irrelevant.
    dt = DecisionTreeRegressor(random_state=0, max_depth=3)
    dt.fit(X, y)
    pred, std = dt.predict(X, return_std=True)
    assert_array_almost_equal([np.std(y)] * 1000, std)
    assert_array_almost_equal([np.mean(y)] * 1000, pred)
