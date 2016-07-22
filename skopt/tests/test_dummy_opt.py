import numpy as np

from sklearn.utils.testing import assert_less

from skopt import dummy_minimize
from skopt.benchmarks import bench1
from skopt.benchmarks import bench2
from skopt.benchmarks import bench3
from skopt.benchmarks import branin
from skopt.benchmarks import hart6


def check_minimize(func, y_opt, dimensions, margin, n_calls):
    r = dummy_minimize(func, dimensions, n_calls=n_calls, random_state=1)
    assert_less(r.fun, y_opt + margin)


def test_dummy_minimize():
    yield (check_minimize, bench1, 0., [(-2.0, 2.0)], 0.05, 10000)
    yield (check_minimize, bench2, -5, [(-6.0, 6.0)], 0.05, 10000)
    yield (check_minimize, bench3, -0.9, [(-2.0, 2.0)], 0.05, 10000)
    yield (check_minimize, branin, 0.39, [(-5.0, 10.0), (0.0, 15.0)], 0.1, 10000)
    yield (check_minimize, hart6, -3.32, np.tile((0.0, 1.0), (6, 1)), 0.5, 10000)
