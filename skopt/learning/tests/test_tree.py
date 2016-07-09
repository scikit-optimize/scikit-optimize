from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal

from skopt.learning import DecisionTreeRegressor


def test_variance_toy_data():
    # Split into [2, 3, 4] and [100, 103, 106]
    X = [[2.0, 1.], [3.0, 1.0], [4., 1.0], [9.0, 1.0], [10.0, 1.0], [11., 1.]]
    y = [2, 3, 4, 100, 103, 106]
    dt = DecisionTreeRegressor(max_depth=1)
    dt.fit(X, y)
    pred, var = dt.predict(X, return_variance=True)
    assert_array_equal(pred, [3, 3, 3, 103, 103, 103])
    assert_array_almost_equal(
        var, [0.666667, 0.666667, 0.666667, 6.0, 6.0, 6.0])
