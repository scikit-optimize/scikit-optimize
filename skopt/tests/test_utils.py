import tempfile

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_true

from skopt import gp_minimize
from skopt import load
from skopt import dump
from skopt.benchmarks import bench3


def check_optimization_results_equality(res_1, res_2):
    # Check if the results objects have the same keys
    assert_equal(sorted(res_1.keys()), sorted(res_2.keys()))
    # Shallow check of the main optimization results
    assert_array_equal(res_1.x, res_2.x)
    assert_array_equal(res_1.x_iters, res_2.x_iters)
    assert_array_equal(res_1.fun, res_2.fun)
    assert_array_equal(res_1.func_vals, res_2.func_vals)


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
