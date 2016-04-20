import numpy as np

from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_less
from sklearn.utils.testing import assert_raise_message

from skopt.benchmarks import bench1
from skopt.benchmarks import bench2
from skopt.benchmarks import bench3
from skopt.benchmarks import branin
from skopt.benchmarks import hart6
from skopt.gbrt import GradientBoostingQuantileRegressor
from skopt.gbrt_opt import gbrt_minimize
from skopt.gbrt_opt import _expected_improvement
from skopt.utils import extract_bounds


class ConstSurrogate:
    def predict(self, X):
        return np.tile([-1., 0.,1.], (X.shape[0], 1))

def test_ei_fixed_surrogate():
    # Uses a surrogate model that always returns -1, 0, 1
    ei = _expected_improvement(np.asarray([10., 10.]),
                               ConstSurrogate(),
                               -0.5,
                               xi=0.)

    assert_almost_equal(ei, -0.1977966)


def test_ei_api():
    # check that it works with a vector as well
    ei = _expected_improvement(np.array([[10., 10.],
                                         [10., 10.],
                                         [10., 10.],
                                         [10., 10.]]),
                               ConstSurrogate(),
                               -0.5,
                               xi=0.)

    assert_array_almost_equal(ei, [-0.1977966] * 4)


def test_no_iterations():
    assert_raise_message(ValueError, "at least one iteration",
                         gbrt_minimize,
                         branin, [[-5, 10], [0, 15]], maxiter=0, random_state=1)

    assert_raise_message(ValueError, "at least one starting point",
                         gbrt_minimize,
                         branin, [[-5, 10], [0, 15]], n_start=0, maxiter=2,
                         random_state=1)


def test_one_iteration():
    result = gbrt_minimize(branin, [[-5, 10], [0, 15]],
                           maxiter=1, random_state=1)

    assert_equal(len(result.models), 0)
    assert_array_equal(result.x_iters.shape, (1, 2))
    assert_array_equal(result.func_vals.shape, (1,))
    assert_array_equal(result.x, result.x_iters[np.argmin(result.func_vals)])
    assert_almost_equal(result.fun, branin(result.x))


def test_seven_iterations():
    result = gbrt_minimize(branin, [[-5, 10], [0, 15]],
                           n_start=3, maxiter=7, random_state=1)

    assert_equal(len(result.models), 4)
    assert_array_equal(result.x_iters.shape, (7, 2))
    assert_array_equal(result.func_vals.shape, (7,))
    assert_array_equal(result.x, result.x_iters[np.argmin(result.func_vals)])
    assert_almost_equal(result.fun, branin(result.x))


def check_minimize(func, y_opt, bounds, margin, maxiter):
    r = gbrt_minimize(func, bounds, maxiter=maxiter, random_state=1)
    assert_less(r.fun, y_opt + margin)


def test_gbrt_minimize():
    yield (check_minimize, bench1, 0., [[-2, 2]], 0.05, 75)
    yield (check_minimize, bench2, -5, [[-6, 6]], 0.05, 75)
    yield (check_minimize, bench3, -0.9, [[-2, 2]], 0.05, 75)
    yield (check_minimize, branin, 0.39, [[-5, 10], [0, 15]],
           0.1, 100)
    yield (check_minimize, hart6, -3.32, np.tile((0, 1), (6, 1)),
           1.0, 200)
