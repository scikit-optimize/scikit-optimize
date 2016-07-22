from functools import partial

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_less
from sklearn.utils.testing import assert_raise_message

from skopt.benchmarks import bench1
from skopt.benchmarks import bench2
from skopt.benchmarks import bench3
from skopt.benchmarks import bench4
from skopt.benchmarks import branin
from skopt.benchmarks import hart6
from skopt.learning import RandomForestRegressor
from skopt.tree_opt import gbrt_minimize
from skopt.tree_opt import forest_minimize


MINIMIZERS = (partial(forest_minimize, base_estimator='dt'),
              partial(forest_minimize, base_estimator='et'),
              partial(forest_minimize, base_estimator='rf'),
              gbrt_minimize)
SLOW_MINIMIZERS = (
    partial(forest_minimize, acq_min="paramils",
            base_estimator=RandomForestRegressor(random_state=0)),)

def check_no_iterations(minimizer):
    assert_raise_message(ValueError, "Expected n_calls > 0",
                         minimizer,
                         branin, [(-5.0, 10.0), (0.0, 15.0)], n_calls=0,
                         random_state=1)

    assert_raise_message(ValueError, "Expected n_random_starts > 0",
                         minimizer,
                         branin, [(-5.0, 10.0), (0.0, 15.0)],
                         n_random_starts=0,
                         random_state=1)


def test_no_iterations():
    for minimizer in MINIMIZERS:
        yield (check_no_iterations, minimizer)


def test_one_iteration():
    for minimizer in MINIMIZERS + SLOW_MINIMIZERS:
        assert_raise_message(ValueError,
                             "Expected n_calls >= 10",
                             minimizer, branin, [(-5.0, 10.0), (0.0, 15.0)],
                             n_calls=1, random_state=1)


def test_seven_iterations():
    for minimizer in MINIMIZERS + SLOW_MINIMIZERS:
        result = minimizer(branin, [(-5.0, 10.0), (0.0, 15.0)],
                           n_random_starts=3, n_calls=7,
                           random_state=1)

        assert_equal(len(result.models), 4)
        assert_equal(len(result.x_iters), 7)
        assert_equal(len(result.x_iters[0]), 2)
        assert_array_equal(result.func_vals.shape, (7,))
        assert_array_equal(result.x, result.x_iters[np.argmin(result.func_vals)])
        assert_almost_equal(result.fun, branin(result.x))
        assert(isinstance(result.x_iters, list))
        assert(isinstance(result.x_iters[0], list))
        assert(isinstance(result.x, list))


def test_forest_minimize_api():
    # invalid string value
    assert_raise_message(ValueError,
                         "Valid values for the base_estimator parameter",
                         forest_minimize, lambda x: 0., [],
                         base_estimator='abc')

    # not a string nor a regressor
    assert_raise_message(ValueError,
                         "The base_estimator parameter has to either",
                         forest_minimize, lambda x: 0., [],
                         base_estimator=42)

    assert_raise_message(ValueError,
                         "The base_estimator parameter has to either",
                         forest_minimize, lambda x: 0., [],
                         base_estimator=DecisionTreeClassifier())


def check_minimize(minimizer, func, y_opt, dimensions, margin, n_calls):
    # The result depends on random sampling so run the test several
    # times and pass if majority of tests converge. Any single instance
    # converging might just be luck.
    success = 0
    N = 3
    for n in range(1, N + 1):
        r = minimizer(
            func, dimensions, n_calls=n_calls, random_state=n)
        if r.fun <= y_opt + margin:
            success += 1

    assert_less(N * 0.5, success)


def test_tree_based_minimize():
    for minimizer in MINIMIZERS:
        yield (check_minimize, minimizer, bench1, 0., [(-2.0, 2.0)], 0.05, 25)
        yield (check_minimize, minimizer, bench2, -5, [(-6.0, 6.0)], 0.05, 60)
        yield (check_minimize, minimizer, bench3, -0.9, [(-2.0, 2.0)], 0.05, 25)
        yield (check_minimize, minimizer, bench4, 0.0,
               [("-2", "-1", "0", "1", "2")], 0.05, 10)
        yield (check_minimize, minimizer, branin, 0.39,
               [(-5.0, 10.0), (0.0, 15.0)], 0.1, 100)
        yield (check_minimize, minimizer, hart6, -3.32,
               np.tile((0.0, 1.0), (6, 1)), 1.0, 25)


def test_slow_minimize():
    for minimizer in SLOW_MINIMIZERS:
        yield (check_minimize, minimizer, bench1, 0., [(-2.0, 2.0)], 0.05, 25)
        yield (check_minimize, minimizer, bench2, -5, [(-6.0, 6.0)], 0.05, 50)
        yield (check_minimize, minimizer, bench3, -0.9, [(-2.0, 2.0)], 0.05, 50)
        yield (check_minimize, minimizer, branin, 0.39,
               [(-5.0, 10.0), (0.0, 15.0)], 0.1, 50)
        yield (check_minimize, minimizer, hart6, -3.32,
               np.tile((0.0, 1.0), (6, 1)), 1.0, 50)
