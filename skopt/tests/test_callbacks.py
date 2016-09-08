from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_less

from skopt.benchmarks import bench1
from skopt.callbacks import TimerCallback
from skopt.optimizer.dummy_opt import dummy_minimize

def test_timer_callback():
    callback = TimerCallback()
    res = dummy_minimize(bench1, [(-1.0, 1.0)], callback=callback, n_calls=10)
    assert_equal(len(callback.iter_time), 10)
    assert_less(0.0, sum(callback.iter_time))
