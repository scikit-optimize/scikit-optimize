import numpy as np
from itertools import product

from sklearn.utils.testing import assert_less

from skopt import gp_minimize
from skopt.benchmarks import bench1
from skopt.benchmarks import bench2
from skopt.benchmarks import bench3
from skopt.benchmarks import bench4
from skopt.benchmarks import branin
from skopt.benchmarks import hart6


def check_minimize(func, y_opt, bounds, acq_optimizer, acq_func,
                   margin, n_calls):
    r = gp_minimize(func, bounds, acq_optimizer=acq_optimizer,
                    acq_func=acq_func,
                    n_calls=n_calls, random_state=1)
    assert_less(r.fun, y_opt + margin)


def test_gp_minimize():
    for search, acq in product(["sampling", "lbfgs"], ["LCB", "EI"]):
        yield (check_minimize, bench1, 0., [(-2.0, 2.0)], search, acq, 0.05, 50)
        yield (check_minimize, bench2, -5, [(-6.0, 6.0)], search, acq, 0.05, 75)
        yield (check_minimize, bench3, -0.9, [(-2.0, 2.0)], search, acq, 0.05, 50)
        yield (check_minimize, bench4, 0.0,
               [("-2", "-1", "0", "1", "2")], search, acq, 0.05, 10)
        yield (check_minimize, branin, 0.39, [(-5.0, 10), (0.0, 15.)],
               search, acq, 0.15, 100)
        yield (check_minimize, hart6, -3.32, np.tile((0., 1.), (6, 1)),
               search, acq, 1.0, 100)
