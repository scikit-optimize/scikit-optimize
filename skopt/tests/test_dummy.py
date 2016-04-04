import numpy as np

from skopt import dummy_minimize
from skopt.tests.test_gp_opt import branin

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_less


def test_dummy_search_api():
    res = dummy_minimize(
        branin, [[-5, 10], [0, 15]], random_state=0, maxiter=100)
    assert_array_equal(res.x.shape, (2,))
    assert_array_equal(res.x_iters.shape, (100, 2))
    assert_array_equal(res.func_vals.shape, (100,))
    assert_array_less(res.x_iters, np.tile([10, 15], (100, 1)))
    assert_array_less(np.tile([-5, 0], (100, 1)), res.x_iters)
