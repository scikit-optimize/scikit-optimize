from functools import partial
from itertools import product

import numpy as np
from scipy.optimize import OptimizeResult

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_less
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raise_message
from sklearn.utils.testing import assert_raises

from skopt import dummy_minimize
from skopt import gp_minimize
from skopt import forest_minimize
from skopt import gbrt_minimize
from skopt.benchmarks import branin
from skopt.benchmarks import bench4
from skopt.space import Space


# dummy_minimize does not support same parameters so
# treated separately
MINIMIZERS = [gp_minimize]
ACQUISITION = ["LCB", "PI", "EI"]


for est, acq in product(["ET", "RF"], ACQUISITION):
    MINIMIZERS.append(
        partial(forest_minimize, base_estimator=est, acq_func=acq))
for acq in ACQUISITION:
    MINIMIZERS.append(partial(gbrt_minimize, acq_func=acq))


def check_minimizer_api(result, n_models=None):
    assert(isinstance(result.space, Space))

    if n_models is not None:
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

    assert(isinstance(result.specs, dict))
    assert("args" in result.specs)
    assert("function" in result.specs)


def check_minimizer_bounds(result):
    # no values should be below or above the bounds
    eps = 10e-9  # check for assert_array_less OR equal
    assert_array_less(result.x_iters, np.tile([10+eps, 15+eps], (7, 1)))
    assert_array_less(np.tile([-5-eps, 0-eps], (7, 1)), result.x_iters)


def check_result_callable(res):
    """
    Check that the result instance is set right at every callable call.
    """
    assert(isinstance(res, OptimizeResult))
    assert_equal(len(res.x_iters), len(res.func_vals))
    assert_equal(np.min(res.func_vals), res.fun)


def test_minimizer_api():
    # dummy_minimize is special as it does not support all parameters
    # and does not fit any models
    call_single = lambda res: res.x
    call_list = [call_single, check_result_callable]

    for verbose, call in product([True, False], [call_single, call_list]):
        result = dummy_minimize(branin, [(-5.0, 10.0), (0.0, 15.0)],
                                n_calls=7, random_state=1,
                                verbose=verbose, callback=call)

        assert(result.models is None)
        yield (check_minimizer_api, result)
        yield (check_minimizer_bounds, result)
        assert_raise_message(ValueError,
                             "return a scalar",
                             dummy_minimize, lambda x: x, [[-5, 10]])

        n_calls = 7
        n_random_starts = 3
        n_models = n_calls - n_random_starts

        for minimizer in MINIMIZERS:
            result = minimizer(branin, [(-5.0, 10.0), (0.0, 15.0)],
                               n_random_starts=n_random_starts,
                               n_calls=n_calls,
                               random_state=1,
                               verbose=verbose, callback=call)

            yield (check_minimizer_api, result, n_models)
            yield (check_minimizer_bounds, result)
            assert_raise_message(ValueError,
                                 "return a scalar",
                                 minimizer, lambda x: x, [[-5, 10]])


def test_init_vals():
    space = [(-5.0, 10.0), (0.0, 15.0)]
    x0 = [[1, 2], [3, 4], [5, 6]]
    n_calls = 10

    for n_random_starts in [0, 5]:
        optimizers = [
            dummy_minimize,
            partial(gp_minimize, n_random_starts=n_random_starts),
            partial(forest_minimize, n_random_starts=n_random_starts),
            partial(gbrt_minimize, n_random_starts=n_random_starts)
        ]
        for optimizer in optimizers:
            yield (check_init_vals, optimizer, branin, space, x0, n_calls)


def test_categorical_init_vals():
    n_random_starts = 5
    optimizers = [
        dummy_minimize,
        partial(gp_minimize, n_random_starts=n_random_starts),
        partial(forest_minimize, n_random_starts=n_random_starts),
        partial(gbrt_minimize, n_random_starts=n_random_starts)
    ]
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


def test_invalid_n_calls_arguments():
    for minimizer in MINIMIZERS:
        assert_raise_message(ValueError,
                             "Expected `n_calls` > 0",
                             minimizer,
                             branin, [(-5.0, 10.0), (0.0, 15.0)], n_calls=0,
                             random_state=1)

        assert_raise_message(ValueError,
                             "set `n_random_starts` > 0, or provide `x0`",
                             minimizer,
                             branin, [(-5.0, 10.0), (0.0, 15.0)],
                             n_random_starts=0,
                             random_state=1)

        # n_calls >= n_random_starts
        assert_raise_message(ValueError,
                             "Expected `n_calls` >= 10",
                             minimizer, branin, [(-5.0, 10.0), (0.0, 15.0)],
                             n_calls=1, n_random_starts=10, random_state=1)

        # n_calls >= n_random_starts + len(x0)
        assert_raise_message(ValueError,
                             "Expected `n_calls` >= 10",
                             minimizer, branin, [(-5.0, 10.0), (0.0, 15.0)],
                             n_calls=1, x0=[[-1, 2], [-3, 3], [2, 5]],
                             random_state=1, n_random_starts=7)

        # n_calls >= n_random_starts when x0 and y0 are provided.
        assert_raise_message(ValueError,
                             "Expected `n_calls` >= 7",
                             minimizer, branin, [(-5.0, 10.0), (0.0, 15.0)],
                             n_calls=1, x0=[[-1, 2], [-3, 3], [2, 5]],
                             y0=[2.0, 3.0, 5.0],
                             random_state=1, n_random_starts=7)
