from itertools import product

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_less
import pytest

from skopt import gp_minimize
from skopt.benchmarks import bench1
from skopt.benchmarks import bench2
from skopt.benchmarks import bench3
from skopt.benchmarks import bench4
from skopt.benchmarks import branin
from skopt.learning import GaussianProcessRegressor


def check_minimize(func, y_opt, bounds, acq_optimizer, acq_func,
                   margin, n_calls):
    r = gp_minimize(func, bounds, acq_optimizer=acq_optimizer,
                    acq_func=acq_func,
                    n_calls=n_calls, random_state=1,
                    noise=1e-10)
    assert_less(r.fun, y_opt + margin)


SEARCH_AND_ACQ = list(product(["sampling", "lbfgs"], ["LCB", "EI"]))


@pytest.mark.parametrize("search, acq", SEARCH_AND_ACQ)
def test_gp_minimize_bench1(search, acq):
    check_minimize(bench1, 0.,
                   [(-2.0, 2.0)], search, acq, 0.05, 50)


@pytest.mark.parametrize("search, acq", SEARCH_AND_ACQ)
def test_gp_minimize_bench2(search, acq):
    check_minimize(bench2, -5,
                   [(-6.0, 6.0)], search, acq, 0.05, 75)


@pytest.mark.parametrize("search, acq", SEARCH_AND_ACQ)
def test_gp_minimize_bench3(search, acq):
    check_minimize(bench3, -0.9,
                   [(-2.0, 2.0)], search, acq, 0.05, 50)


@pytest.mark.parametrize("search, acq", SEARCH_AND_ACQ)
def test_gp_minimize_bench4(search, acq):
    check_minimize(bench4, 0.0,
                   [("-2", "-1", "0", "1", "2")], search, acq, 0.05, 10)


def test_n_jobs():
    r_single = gp_minimize(bench3, [(-2.0, 2.0)], acq_optimizer="lbfgs",
                           acq_func="EI", n_calls=2, n_random_starts=1,
                           random_state=1, noise=1e-10)
    r_double = gp_minimize(bench3, [(-2.0, 2.0)], acq_optimizer="lbfgs",
                           acq_func="EI", n_calls=2, n_random_starts=1,
                           random_state=1, noise=1e-10, n_jobs=2)
    assert_array_equal(r_single.x_iters, r_double.x_iters)


def test_gpr_default():
    """Smoke test that gp_minimize does not fail for default values."""
    gpr = GaussianProcessRegressor()
    res = gp_minimize(
        branin, ((-5.0, 10.0), (0.0, 15.0)), n_random_starts=1, n_calls=2)
