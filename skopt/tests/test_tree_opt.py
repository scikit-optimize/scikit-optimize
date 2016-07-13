from functools import partial

import numpy as np

from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_less
from sklearn.utils.testing import assert_raise_message

from skopt.benchmarks import bench1
from skopt.benchmarks import bench2
from skopt.benchmarks import bench3
from skopt.benchmarks import branin
from skopt.benchmarks import hart6
from skopt.tree_opt import gbrt_minimize
from skopt.tree_opt import forest_minimize


def check_no_iterations(minimizer):
    assert_raise_message(ValueError, "at least one iteration",
                         minimizer,
                         branin, [(-5.0, 10.0), (0.0, 15.0)], maxiter=0,
                         random_state=1)

    assert_raise_message(ValueError, "at least one starting point",
                         minimizer,
                         branin, [(-5.0, 10.0), (0.0, 15.0)], n_start=0,
                         maxiter=2, random_state=1)


def test_no_iterations():
    yield (check_no_iterations, gbrt_minimize)
    yield (check_no_iterations, partial(forest_minimize, base_estimator='et'))
    yield (check_no_iterations, partial(forest_minimize, base_estimator='dt'))
    yield (check_no_iterations, forest_minimize)


def test_one_iteration():
    result = gbrt_minimize(branin, [(-5.0, 10.0), (0.0, 15.0)],
                           maxiter=1, random_state=1)

    assert_equal(len(result.models), 0)
    assert_array_equal(result.x_iters.shape, (1, 2))
    assert_array_equal(result.func_vals.shape, (1,))
    assert_array_equal(result.x, result.x_iters[np.argmin(result.func_vals)])
    assert_almost_equal(result.fun, branin(result.x))


def test_seven_iterations():
    result = gbrt_minimize(branin, [(-5.0, 10.0), (0.0, 15.0)],
                           n_start=3, maxiter=7, random_state=1)

    assert_equal(len(result.models), 4)
    assert_array_equal(result.x_iters.shape, (7, 2))
    assert_array_equal(result.func_vals.shape, (7,))
    assert_array_equal(result.x, result.x_iters[np.argmin(result.func_vals)])
    assert_almost_equal(result.fun, branin(result.x))


def test_forest_minimize_api():
    # invalid string value
    assert_raise_message(ValueError, "Valid values for base_estimator parameter",
                         forest_minimize, lambda x: 0., [],
                         base_estimator='abc')

    # not a string nor a Regressor instance
    assert_raise_message(ValueError, "The base_estimator parameter has to either",
                         forest_minimize, lambda x: 0., [],
                         base_estimator=42)


def check_minimize(minimizer, func, y_opt, dimensions, margin, maxiter):
    success = 0
    N = 3
    for n in range(1, N + 1):
        r = minimizer(func, dimensions, maxiter=maxiter, random_state=n)
        if r.fun <= y_opt + margin:
            success += 1

    assert_less(N * 0.5, success)


def test_tree_based_minimize():
    for minimizer in (partial(forest_minimize, base_estimator='dt'),
                      partial(forest_minimize, base_estimator='et'),
                      forest_minimize, gbrt_minimize):
        yield (check_minimize, minimizer, bench1, 0., [(-2.0, 2.0)], 0.05, 75)
        yield (check_minimize, minimizer, bench2, -5, [(-6.0, 6.0)], 0.05, 75)
        yield (check_minimize, minimizer, bench3, -0.9, [(-2.0, 2.0)], 0.05, 75)
        yield (check_minimize, minimizer, branin, 0.39,
               [(-5.0, 10.0), (0.0, 15.0)], 0.1, 100)
        yield (check_minimize, minimizer, hart6, -3.32,
               np.tile((0.0, 1.0), (6, 1)), 1.0, 200)
