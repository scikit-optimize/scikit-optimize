from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_less
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal

from skopt.benchmarks import branin
from skopt.tests.test_tree_opt import MINIMIZERS as TREE_MINIMIZERS

MINIMIZERS = []
MINIMIZERS += TREE_MINIMIZERS


def check_minimizer_api(result):
    assert_equal(len(result.models), 4)
    assert_array_equal(result.x_iters.shape, (7, 2))
    assert_array_equal(result.func_vals.shape, (7,))
    assert_array_equal(result.x, result.x_iters[np.argmin(result.func_vals)])
    assert_almost_equal(result.fun, branin(result.x))


def check_minimizer_bounds(result):
    assert_array_less(result.x_iters, np.tile([10, 15], (7, 1)))
    assert_array_less(np.tile([-5, 0], (7, 1)), result.x_iters)


def test_minimizer_api():
    for minimizer in MINIMIZERS:
        result = minimizer(branin, [(-5.0, 10.0), (0.0, 15.0)],
                           n_random_starts=3, n_calls=7,
                           random_state=1)

        yield (check_minimizer_api, result)
