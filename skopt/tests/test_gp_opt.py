from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_less
import pytest

from skopt import gp_minimize
from skopt.benchmarks import bench1
from skopt.benchmarks import bench2
from skopt.benchmarks import bench3
from skopt.benchmarks import bench4
from skopt.benchmarks import branin
from skopt.utils import cook_estimator


def check_minimize(func, y_opt, bounds, acq_optimizer, acq_func,
                   margin, n_calls, n_random_starts=10):
    r = gp_minimize(func, bounds, acq_optimizer=acq_optimizer,
                    acq_func=acq_func, n_random_starts=n_random_starts,
                    n_calls=n_calls, random_state=1,
                    noise=1e-10)
    assert_less(r.fun, y_opt + margin)


SEARCH = ["sampling", "lbfgs"]
ACQUISITION = ["LCB", "EI"]


@pytest.mark.slow_test
@pytest.mark.parametrize("search", SEARCH)
@pytest.mark.parametrize("acq", ACQUISITION)
def test_gp_minimize_bench1(search, acq):
    check_minimize(bench1, 0.,
                   [(-2.0, 2.0)], search, acq, 0.05, 20)


@pytest.mark.slow_test
@pytest.mark.parametrize("search", SEARCH)
@pytest.mark.parametrize("acq", ACQUISITION)
def test_gp_minimize_bench2(search, acq):
    check_minimize(bench2, -5,
                   [(-6.0, 6.0)], search, acq, 0.05, 20)


@pytest.mark.slow_test
@pytest.mark.parametrize("search", SEARCH)
@pytest.mark.parametrize("acq", ACQUISITION)
def test_gp_minimize_bench3(search, acq):
    check_minimize(bench3, -0.9,
                   [(-2.0, 2.0)], search, acq, 0.05, 20)


@pytest.mark.fast_test
@pytest.mark.parametrize("search", ["sampling"])
@pytest.mark.parametrize("acq", ACQUISITION)
def test_gp_minimize_bench4(search, acq):
    # this particular random_state picks "2" twice so we can make an extra
    # call to the objective without repeating options
    check_minimize(bench4, 0.0,
                   [("-2", "-1", "0", "1", "2")], search, acq, 1.05, 6, 2)


@pytest.mark.fast_test
def test_n_jobs():
    r_single = gp_minimize(bench3, [(-2.0, 2.0)], acq_optimizer="lbfgs",
                           acq_func="EI", n_calls=2, n_random_starts=1,
                           random_state=1, noise=1e-10)
    r_double = gp_minimize(bench3, [(-2.0, 2.0)], acq_optimizer="lbfgs",
                           acq_func="EI", n_calls=2, n_random_starts=1,
                           random_state=1, noise=1e-10, n_jobs=2)
    assert_array_equal(r_single.x_iters, r_double.x_iters)


@pytest.mark.fast_test
def test_gpr_default():
    """Smoke test that gp_minimize does not fail for default values."""
    gp_minimize(branin, ((-5.0, 10.0), (0.0, 15.0)), n_random_starts=1,
                n_calls=2)


@pytest.mark.fast_test
def test_use_given_estimator():
    """ Test that gp_minimize does not use default estimator if one is passed
    in explicitly. """
    domain = [(1.0, 2.0), (3.0, 4.0)]
    noise_correct = 1e+5
    noise_fake = 1e-10
    estimator = cook_estimator("GP", domain, noise=noise_correct)
    res = gp_minimize(branin, domain, n_calls=1, n_random_starts=1,
                      base_estimator=estimator, noise=noise_fake)

    assert res['models'][-1].noise == noise_correct


@pytest.mark.fast_test
def test_use_given_estimator_with_model_history():
    """ Test that gp_minimize does not use default estimator if one is passed
    in explicitly. """
    domain = [(1.0, 2.0), (3.0, 4.0)]
    noise_correct = 1e+5
    noise_fake = 1e-10
    estimator = cook_estimator("GP", domain, noise=noise_correct)
    res = gp_minimize(branin, domain, n_calls=1, n_random_starts=1,
                      base_estimator=estimator, noise=noise_fake, model_history=1)
    assert len(res['models']) == 1
    assert res['models'][-1].noise == noise_correct
