import numpy as np
from sklearn.utils.testing import assert_less

from skopt.benchmarks import bench1 as bench1_1D
from skopt.benchmarks import bench2 as bench2_1D
from skopt.benchmarks import bench3 as bench3_1D
from skopt.benchmarks import branin as branin_1D
from skopt.benchmarks import hart6 as hart6_1D

from skopt.solvers import paramils_smac


# Wrappers that can act on 2-D data
bench1 = lambda x: np.array([bench1_1D(x_) for x_ in x])
bench2 = lambda x: np.array([bench2_1D(x_) for x_ in x])
bench3 = lambda x: np.array([bench3_1D(x_) for x_ in x])
branin = lambda x: np.array([branin_1D(x_) for x_ in x])
hart6 = lambda x: np.array([hart6_1D(x_) for x_ in x])

def check_minimize(func_2D, bounds, n_init, func_1D, y_opt, tol):
    bounds = np.asarray(bounds)
    upper_bounds = bounds[:, 1]
    lower_bounds = bounds[:, 0]
    n_dims = len(upper_bounds)

    for n in range(3):
        rng = np.random.RandomState(n)
        x_init = rng.rand(n_init, n_dims)
        x_init *= upper_bounds - lower_bounds
        x_init += lower_bounds
        r = paramils_smac(func_2D, x_init, bounds, random_state=rng)
        assert_less(func_1D(r), y_opt + tol)

def test_minimize():
    check_minimize(bench1, [(-2, 2),], 3, bench1_1D, 0, 1e-3)
    check_minimize(bench2, [(-6.0, 6.0),], 3, bench2_1D, -5, 1e-2)
    check_minimize(bench3, [(-2.0, 2.0),], 3, bench3_1D, -0.9, 1e-2)
    check_minimize(
        branin, [(-5.0, 10.0), (0.0, 15.0)], 10, branin_1D, 0.39, 0.06)
    check_minimize(
        hart6, np.tile((0.0, 1.0), (6, 1)), 3, hart6_1D, -3.32, 0.05)
