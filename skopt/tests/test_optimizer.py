from sklearn.utils.testing import assert_equal

from skopt.benchmarks import bench1
from skopt.learning import ExtraTreesRegressor
from skopt.optimizer import Optimizer


def test_multiple_asks():
    # calling ask() multiple times without a tell() inbetween should
    # be a "no op"
    base_estimator = ExtraTreesRegressor(random_state=2)
    opt = Optimizer([(-2.0, 2.0)], base_estimator, n_random_starts=1,
                    acq_optimizer="sampling")

    opt.run(bench1, n_iter=3)
    # tell() computes the next point ready for the next call to ask()
    # hence there are three after three iterations
    assert_equal(len(opt.models), 3)
    opt.ask()
    assert_equal(len(opt.models), 3)
    assert_equal(opt.ask(), opt.ask())
