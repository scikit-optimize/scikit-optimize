import pytest

import numpy as np
import os
from collections import namedtuple

from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_less

from ProcessOptimizer import dummy_minimize
from ProcessOptimizer import gp_minimize
from ProcessOptimizer.benchmarks import bench1
from ProcessOptimizer.benchmarks import bench3
from ProcessOptimizer.callbacks import TimerCallback
from ProcessOptimizer.callbacks import DeltaYStopper
from ProcessOptimizer.callbacks import DeadlineStopper
from ProcessOptimizer.callbacks import CheckpointSaver

from ProcessOptimizer.utils import load

@pytest.mark.fast_test
def test_timer_callback():
    callback = TimerCallback()
    dummy_minimize(bench1, [(-1.0, 1.0)], callback=callback, n_calls=10)
    assert_equal(len(callback.iter_time), 10)
    assert_less(0.0, sum(callback.iter_time))


@pytest.mark.fast_test
def test_deltay_stopper():
    deltay = DeltaYStopper(0.2, 3)

    Result = namedtuple('Result', ['func_vals'])

    assert deltay(Result([0, 1, 2, 3, 4, 0.1, 0.19]))
    assert not deltay(Result([0, 1, 2, 3, 4, 0.1]))
    assert deltay(Result([0, 1])) is None


@pytest.mark.fast_test
def test_deadline_stopper():
    deadline = DeadlineStopper(0.0001)
    gp_minimize(bench3, [(-1.0, 1.0)], callback=deadline, n_calls=10, random_state=1)
    assert len(deadline.iter_time) == 1
    assert np.sum(deadline.iter_time) > deadline.total_time

    deadline = DeadlineStopper(60)
    gp_minimize(bench3, [(-1.0, 1.0)], callback=deadline, n_calls=10, random_state=1)
    assert len(deadline.iter_time) == 10
    assert np.sum(deadline.iter_time) < deadline.total_time


@pytest.mark.fast_test
def test_checkpoint_saver():
    checkpoint_path = "./test_checkpoint.pkl"

    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)

    checkpoint_saver = CheckpointSaver(checkpoint_path, compress=9)
    result = dummy_minimize(bench1,
        [(-1.0, 1.0)],
        callback=checkpoint_saver,
        n_calls=10)

    assert os.path.exists(checkpoint_path)
    assert load(checkpoint_path).x == result.x

    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)
