from functools import partial

import numpy as np
from itertools import product

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_less
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raises

from skopt.benchmarks import branin
from skopt.benchmarks import bench4
from skopt.dummy_opt import dummy_minimize
from skopt.gp_opt import gp_minimize
from skopt.space import Space
from skopt.tree_opt import forest_minimize
from skopt.tree_opt import gbrt_minimize


# dummy_minimize does not support same parameters so
# treated separately
MINIMIZERS = [gp_minimize,]
ACQUISITION = ["LCB", "PI", "EI"]

for est, acq in product(["dt", "et", "rf"], ACQUISITION):
    MINIMIZERS.append(
        partial(forest_minimize, base_estimator=est, acq=acq))
for acq in ACQUISITION:
    MINIMIZERS.append(partial(gbrt_minimize, acq=acq))


def check_minimizer_api(result, n_models):
    assert(isinstance(result.space, Space))
    assert_equal(len(result.models), n_models)
    assert_equal(len(result.x_iters), 7)
    assert_array_equal(result.func_vals.shape, (7,))

    assert(isinstance(result.x, list))
    assert_equal(len(result.x), 2)

    assert(isinstance(result.x_iters, list))
    for n in range(7):
        assert(isinstance(result.x_iters[n], list))
        assert_equal(len(result.x_iters[n]), 2)

        assert(isinstance(result.func_vals[n], float))

    assert_array_equal(result.x, result.x_iters[np.argmin(result.func_vals)])
    assert_almost_equal(result.fun, branin(result.x))


def check_minimizer_bounds(result):
    # no values should be below or above the bounds
    eps = 10e-9  # check for assert_array_less OR equal
    assert_array_less(result.x_iters, np.tile([10+eps, 15+eps], (7, 1)))
    assert_array_less(np.tile([-5-eps, 0-eps], (7, 1)), result.x_iters)


def test_minimizer_api():
    # dummy_minimize is special as it does not support all parameters
    # and does not fit any models
    result = dummy_minimize(branin, [(-5.0, 10.0), (0.0, 15.0)],
                            n_calls=7, random_state=1)

    yield (check_minimizer_api, result, 0)
    yield (check_minimizer_bounds, result)
    assert_raises(ValueError, dummy_minimize, lambda x: x, [[-5, 10]])

    n_calls = 7
    n_random_starts = 3
    n_models = n_calls - n_random_starts
    for minimizer in MINIMIZERS:
        result = minimizer(branin, [(-5.0, 10.0), (0.0, 15.0)],
                           n_random_starts=n_random_starts,
                           n_calls=n_calls,
                           random_state=1)

        yield (check_minimizer_api, result, n_models)
        yield (check_minimizer_bounds, result)
        assert_raises(ValueError, minimizer, lambda x: x, [[-5, 10]])


def test_init_vals():
    n_random_starts = 5
    optimizers = [
        dummy_minimize,
        partial(gp_minimize, n_random_starts=n_random_starts),
        partial(forest_minimize, n_random_starts=n_random_starts),
        partial(gbrt_minimize, n_random_starts=n_random_starts)
    ]
    space = [(-5.0, 10.0), (0.0, 15.0)]
    x0 = [[1, 2], [3, 4], [5, 6]]
    n_calls = 10
    for optimizer in optimizers:
        yield (check_init_vals, optimizer, branin, space, x0, n_calls)

    space = [("-2", "-1", "0", "1", "2")]
    x0 = [["0"], ["1"], ["2"]]
    n_calls = 10
    for optimizer in optimizers:
        yield (check_init_vals, optimizer, bench4, space, x0, n_calls)


def check_init_vals(optimizer, func, space, x0, n_calls):
    y0 = list(map(func, x0))
    # testing whether the provided points with their evaluations
    # are taken into account
    res = optimizer(
        func, space, x0=x0, y0=y0,
        random_state=0, n_calls=n_calls)
    assert_array_equal(res.x_iters[0:len(x0)], x0)
    assert_array_equal(res.func_vals[0:len(y0)], y0)
    assert_equal(len(res.x_iters), len(x0) + n_calls)
    assert_equal(len(res.func_vals), len(x0) + n_calls)

    # testing whether the provided points are taken into account
    res = optimizer(
        func, space, x0=x0,
        random_state=0, n_calls=n_calls)
    assert_array_equal(res.x_iters[0:len(x0)], x0)
    assert_array_equal(res.func_vals[0:len(y0)], y0)
    assert_equal(len(res.x_iters), n_calls)
    assert_equal(len(res.func_vals), n_calls)

    # testing whether providing a single point instead of a list
    # of points works correctly
    res = optimizer(
        func, space, x0=x0[0],
        random_state=0, n_calls=n_calls)
    assert_array_equal(res.x_iters[0], x0[0])
    assert_array_equal(res.func_vals[0], y0[0])
    assert_equal(len(res.x_iters), n_calls)
    assert_equal(len(res.func_vals), n_calls)

    # testing whether providing a single point and its evaluation
    # instead of a list of points and their evaluations works correctly
    res = optimizer(
        func, space, x0=x0[0], y0=y0[0],
        random_state=0, n_calls=n_calls)
    assert_array_equal(res.x_iters[0], x0[0])
    assert_array_equal(res.func_vals[0], y0[0])
    assert_equal(len(res.x_iters), 1 + n_calls)
    assert_equal(len(res.func_vals), 1 + n_calls)

    # testing whether it correctly raises an exception when
    # the number of input points and the number of evaluations differ
    assert_raises(ValueError, dummy_minimize, func,
                  space, x0=x0, y0=[1])
