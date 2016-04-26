import numpy as np
from itertools import product

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_less
from sklearn.utils.testing import assert_less
from sklearn.utils.testing import assert_raises

from skopt.gp_opt import gp_minimize
from skopt.benchmarks import bench1
from skopt.benchmarks import bench2
from skopt.benchmarks import bench3
from skopt.benchmarks import branin
from skopt.benchmarks import hart6


def check_minimize(func, y_opt, bounds, search, acq, margin, maxiter):
    r = gp_minimize(func, bounds, search=search, acq=acq,
                    maxiter=maxiter, random_state=1)
    assert_less(r.fun, y_opt + margin)


def test_gp_minimize():
    for search, acq in product(["sampling", "lbfgs"], ["LCB", "PI", "EI"]):
        yield (check_minimize, bench1, 0., [[-2, 2]], search, acq, 0.05, 75)
        yield (check_minimize, bench2, -5, [[-6, 6]], search, acq, 0.05, 75)
        yield (check_minimize, bench3, -0.9, [[-2, 2]], search, acq, 0.05, 75)
        yield (check_minimize, branin, 0.39, [[-5, 10], [0, 15]],
               search, acq, 0.1, 100)
        yield (check_minimize, hart6, -3.32, np.tile((0, 1), (6, 1)),
               search, acq, 1.0, 150)


def test_api():
    res = gp_minimize(branin, [[-5, 10], [0, 15]], random_state=0, maxiter=20)
    assert_array_equal(res.x.shape, (2,))
    assert_array_equal(res.x_iters.shape, (20, 2))
    assert_array_equal(res.func_vals.shape, (20,))
    assert_array_less(res.x_iters, np.tile([10, 15], (20, 1)))
    assert_array_less(np.tile([-5, 0], (20, 1)), res.x_iters)

    assert_raises(ValueError, gp_minimize, lambda x: x, [[-5, 10]])
