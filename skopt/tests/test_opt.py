from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_raises

from skopt import dummy_minimize, gp_minimize, forest_minimize, gbrt_minimize

from skopt.benchmarks import branin


def test_init_vals():
    optimizers = [
        dummy_minimize,
        gp_minimize,
        forest_minimize,
        gbrt_minimize
    ]
    x0 = [[1, 2], [3, 4], [5, 6]]
    y0 = map(branin, x0)
    for optimizer in optimizers:
        res = optimizer(
            branin, [(-5.0, 10.0), (0.0, 15.0)], x0=x0, y0=y0,
            random_state=0, maxiter=100)
        assert_array_equal(res.x_iters[0:len(x0)], x0)
        assert_array_equal(res.func_vals[0:len(y0)], y0)

        res = optimizer(
            branin, [(-5.0, 10.0), (0.0, 15.0)], x0=x0,
            random_state=0, maxiter=100)
        assert_array_equal(res.x_iters[0:len(x0)], x0)
        assert_array_equal(res.func_vals[0:len(y0)], y0)

        assert_raises(ValueError, dummy_minimize, branin,
                      [(-5.0, 10.0), (0.0, 15.0)], x0=x0, y0=[1])
