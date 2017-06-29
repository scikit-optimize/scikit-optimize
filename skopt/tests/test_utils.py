import pytest
import tempfile

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_true

from skopt import gp_minimize
from skopt import load
from skopt import dump
from skopt import expected_minimum
from skopt.benchmarks import bench1
from skopt.benchmarks import bench3
from skopt.learning import ExtraTreesRegressor
from skopt.optimizer import Optimizer
from skopt.utils import point_asdict, point_aslist, dimensions_aslist


def check_optimization_results_equality(res_1, res_2):
    # Check if the results objects have the same keys
    assert_equal(sorted(res_1.keys()), sorted(res_2.keys()))
    # Shallow check of the main optimization results
    assert_array_equal(res_1.x, res_2.x)
    assert_array_equal(res_1.x_iters, res_2.x_iters)
    assert_array_equal(res_1.fun, res_2.fun)
    assert_array_equal(res_1.func_vals, res_2.func_vals)


@pytest.mark.fast_test
def test_dump_and_load():
    res = gp_minimize(bench3,
                      [(-2.0, 2.0)],
                      x0=[0.],
                      acq_func="LCB",
                      n_calls=2,
                      n_random_starts=0,
                      random_state=1)

    # Test normal dumping and loading
    with tempfile.TemporaryFile() as f:
        dump(res, f)
        res_loaded = load(f)
    check_optimization_results_equality(res, res_loaded)
    assert_true("func" in res_loaded.specs["args"])

    # Test dumping without objective function
    with tempfile.TemporaryFile() as f:
        dump(res, f, store_objective=False)
        res_loaded = load(f)
    check_optimization_results_equality(res, res_loaded)
    assert_true(not ("func" in res_loaded.specs["args"]))

    # Delete the objective function and dump the modified object
    del res.specs["args"]["func"]
    with tempfile.TemporaryFile() as f:
        dump(res, f, store_objective=False)
        res_loaded = load(f)
    check_optimization_results_equality(res, res_loaded)
    assert_true(not ("func" in res_loaded.specs["args"]))


@pytest.mark.fast_test
def test_dump_and_load_optimizer():
    base_estimator = ExtraTreesRegressor(random_state=2)
    opt = Optimizer([(-2.0, 2.0)], base_estimator, n_random_starts=1,
                    acq_optimizer="sampling")

    opt.run(bench1, n_iter=3)

    with tempfile.TemporaryFile() as f:
        dump(opt, f)
        load(f)


@pytest.mark.fast_test
def test_expected_minimum():
    res = gp_minimize(bench3,
                      [(-2.0, 2.0)],
                      x0=[0.],
                      noise=1e-8,
                      n_calls=20,
                      random_state=1)

    x_min, f_min = expected_minimum(res, random_state=1)
    x_min2, f_min2 = expected_minimum(res, random_state=1)

    assert f_min <= res.fun  # true since noise ~= 0.0
    assert x_min == x_min2
    assert f_min == f_min2


@pytest.mark.fast_test
def test_dict_list_space_representation():
    """
    Tests whether the conversion of the dictionary and list representation
    of a point from a search space works properly.
    """

    chef_space = {
        'Cooking time': (0, 1200),  # in minutes
        'Main ingredient': [
            'cheese', 'cherimoya', 'chicken', 'chard', 'chocolate', 'chicory'
        ],
        'Secondary ingredient': [
            'love', 'passion', 'dedication'
        ],
        'Cooking temperature': (-273.16, 10000.0)  # in Celsius
    }

    opt = Optimizer(dimensions=dimensions_aslist(chef_space))
    point = opt.ask()

    # check if the back transformed point and original one are equivalent
    assert_equal(
        point,
        point_aslist(chef_space, point_asdict(chef_space, point))
    )
