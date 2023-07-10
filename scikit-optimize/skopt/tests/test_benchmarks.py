import numpy as np
import pytest

from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal

from skopt.benchmarks import branin
from skopt.benchmarks import hart6


@pytest.mark.fast_test
def test_branin():
    xstars = np.asarray([(-np.pi, 12.275), (+np.pi, 2.275), (9.42478, 2.475)])
    f_at_xstars = np.asarray([branin(xstar) for xstar in xstars])
    branin_min = np.array([0.397887] * xstars.shape[0])
    assert_array_almost_equal(f_at_xstars, branin_min)


@pytest.mark.fast_test
def test_hartmann6():
    assert_almost_equal(hart6((0.20169, 0.15001, 0.476874,
                               0.275332, 0.311652, 0.6573)),
                        -3.32237, decimal=5)
