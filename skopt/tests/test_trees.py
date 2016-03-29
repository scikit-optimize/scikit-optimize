import numpy as np

from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_less

from skopt.benchmarks import bench1
from skopt.benchmarks import branin
from skopt.gbt import GradientBoostingQuantileRegressor
from skopt.trees import gbt_minimize
from skopt.trees import _expected_improvement
from skopt.utils import extract_bounds


def test_ei_fixed_surrogate():
    # Use a surrogate model that always returns -1, 0, 1
    class ConstSurrogate:
        def predict(self, X):
            return np.tile([[-1.],[0.],[1.]], (1, X.shape[0]))

    ei = _expected_improvement(np.asarray([10., 10.]),
                               ConstSurrogate(),
                               -0.5)

    assert_almost_equal(ei, -0.1977966)


def test_ei_api():
    # check that it works with a vector as well
    class ConstSurrogate:
        def predict(self, X):
            return np.tile([[-1.],[0.],[1.]], (1, X.shape[0]))

    ei = _expected_improvement(np.array([[10., 10.],
                                         [10., 10.],
                                         [10., 10.]]),
                               ConstSurrogate(),
                               -0.5)

    assert_array_almost_equal(ei, [-0.1977966] * 3)


def test_no_iterations():
    result = gbt_minimize(branin, [[-5, 10], [0, 15]],
                          maxiter=0, random_state=1)

    assert_almost_equal(result.fun, branin(result.x))
    assert_equal(len(result.models), 0)
    assert_array_equal(result.x_iters.shape, (1, 2))


def test_one_iteration():
    result = gbt_minimize(branin, [[-5, 10], [0, 15]],
                          maxiter=1, random_state=1)

    assert_equal(len(result.models), 1)
    assert_array_equal(result.x_iters.shape, (2, 2))
    assert_array_equal(result.x, result.x_iters[np.argmin(result.func_vals)])
    assert_almost_equal(result.fun, branin(result.x))


def test_parabola():
    # find the minimum of a parabola

    result = gbt_minimize(bench1, [[-10, 10]],
                          maxiter=5, random_state=1)

    # 5 iterations and one random point
    assert_array_equal(result.x_iters.shape, (5+1, 1))
    assert_less(result.fun, 0.001)


def test_branin():
    result = gbt_minimize(branin, [[-5, 10], [0, 15]],
                          maxiter=100, random_state=1)
