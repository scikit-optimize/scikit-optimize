from sklearn.utils.testing import assert_array_equal, assert_equal
from sklearn.utils.testing import assert_raises

from skopt import dummy_minimize, gp_minimize, forest_minimize, gbrt_minimize

from skopt.benchmarks import branin, bench4

from functools import partial


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
    x0 = [["1"], ["3"], ["5"]]
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
