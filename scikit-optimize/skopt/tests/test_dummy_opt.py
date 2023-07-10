import pytest
from skopt import dummy_minimize
from skopt.benchmarks import bench1
from skopt.benchmarks import bench2
from skopt.benchmarks import bench3


def check_minimize(func, y_opt, dimensions, margin, n_calls):
    r = dummy_minimize(func, dimensions, n_calls=n_calls, random_state=1)
    assert r.fun < y_opt + margin


@pytest.mark.slow_test
def test_dummy_minimize():
    check_minimize(bench1, 0., [(-2.0, 2.0)], 0.05, 100)
    check_minimize(bench2, -5, [(-6.0, 6.0)], 0.05, 100)
    check_minimize(bench3, -0.9, [(-2.0, 2.0)], 0.05, 100)


@pytest.mark.fast_test
def test_dummy_categorical_integer():
    def f(params):
        return 0

    dims = [[1]]
    res = dummy_minimize(f, dims, n_calls=1, random_state=1)
    assert res.x_iters[0][0] == dims[0][0]
