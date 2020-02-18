import numpy as np
import pytest

from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal

from skopt.benchmarks import hart4, hart3
from skopt.benchmarks import Sumpow
from skopt.benchmarks import Branin, Ackley, Sumquares, Forrester, Hartmann6


@pytest.mark.fast_test
def test_branin():
    branin = Branin()
    xstars = np.asarray([(-np.pi, 12.275), (+np.pi, 2.275), (9.42478, 2.475)])
    f_at_xstars = np.asarray([branin(xstar) for xstar in xstars])
    branin_min = np.array([0.397887] * xstars.shape[0])
    assert_array_almost_equal(f_at_xstars, branin_min)
    b = Branin(scaled=False)
    xstars = b.minimum_pos
    f_at_xstars = np.asarray([b(xstar) for xstar in xstars])
    branin_min = np.array([0.397887] * xstars.shape[0])
    assert_array_almost_equal(f_at_xstars, branin_min)
    b = Branin(scaled=True)
    f_at_xstars = np.asarray([b(xstar) for xstar in xstars])
    assert_array_almost_equal(f_at_xstars, b.minimum)


@pytest.mark.fast_test
def test_hartmann6():
    hart6 = Hartmann6()
    assert_almost_equal(hart6((0.20169, 0.15001, 0.476874,
                               0.275332, 0.311652, 0.6573)),
                        -3.32237, decimal=5)
    assert_almost_equal(hart6(hart6.minimum_pos[0]),
                        hart6.minimum, decimal=5)


@pytest.mark.fast_test
def test_hartmann3():
    assert_almost_equal(hart3((0.114614, 0.555649, 0.852547)),
                        -3.86278, decimal=5)


@pytest.mark.fast_test
def test_hartmann4():
    assert_almost_equal(hart4((0.20169, 0.15001, 0.476874,
                               0.275332)),
                        -3.6475056362745373, decimal=5)


@pytest.mark.fast_test
def test_sumpow():
    sumpow = Sumpow(10)
    assert_almost_equal(sumpow((0, )*10),
                        0, decimal=5)
    assert_almost_equal(sumpow(sumpow.minimum_pos[0]),
                        sumpow.minimum, decimal=5)
    sumpow = Sumpow(2)
    assert_almost_equal(sumpow((0, )*2),
                        0, decimal=5)
    sumpow = Sumpow(5)
    assert_almost_equal(sumpow([-1., -0.5, 0., 0.5, 1.]),
                        2.15625, decimal=5)


@pytest.mark.fast_test
def test_ackley():
    ackley = Ackley(n_dim=10)
    assert_almost_equal(ackley((0, )*10),
                        0, decimal=5)
    assert_almost_equal(ackley(ackley.minimum_pos[0]),
                        ackley.minimum, decimal=5)
    ackley = Ackley(n_dim=2)
    assert_almost_equal(ackley((0, )*2),
                        0, decimal=5)


@pytest.mark.fast_test
def test_sumquares():
    sumquares = Sumquares(n_dim=10)
    assert_almost_equal(sumquares((0, )*10),
                        0, decimal=5)
    assert_almost_equal(sumquares(sumquares.minimum_pos[0]),
                        sumquares.minimum, decimal=5)
    sumquares = Sumquares(n_dim=2)
    assert_almost_equal(sumquares((0, )*2),
                        0, decimal=5)
    sumquares = Sumquares(n_dim=1)
    assert_almost_equal(sumquares([2]), 4.)


@pytest.mark.fast_test
def test_forrester():
    forrester = Forrester()
    assert_almost_equal(forrester(forrester.minimum_pos[0]),
                        forrester.minimum, decimal=5)

