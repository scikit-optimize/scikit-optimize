import pytest
import numbers
import numpy as np
import os
import yaml
from tempfile import NamedTemporaryFile

from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from scipy import spatial
from skopt import Optimizer
from skopt.space import Space
from skopt.space import Real
from skopt.space import Integer
from skopt.space import Categorical
from skopt.space import check_dimension as space_check_dimension
from skopt.sampler.sobol import _bit_lo0, _bit_hi1
from skopt.sampler.halton import _van_der_corput_samples, _create_primes
from skopt.sampler import Hammersly, Halton, Lhs, Sobol, Grid
from skopt.sampler import InitialPointGenerator
from skopt.sampler.grid import _create_uniform_grid_include_border
from skopt.sampler.grid import _create_uniform_grid_exclude_border
from skopt.sampler.grid import _quadrature_combine
from skopt.sampler.grid import _create_uniform_grid_only_border
from skopt.utils import cook_initial_point_generator


LHS_TYPE = ["classic", "centered"]
CRITERION = ["maximin", "ratio", "correlation", None]
SAMPLER = ["lhs", "halton", "sobol", "hammersly", "grid"]


@pytest.mark.fast_test
def test_lhs_centered():
    lhs = Lhs(lhs_type="centered")
    samples = lhs.generate([(0., 1.), ] * 3, 3)
    assert_almost_equal(np.sum(samples), 4.5)


@pytest.mark.parametrize("samlper", SAMPLER)
def test_sampler(samlper):
    s = cook_initial_point_generator(samlper)
    samples = s.generate([(0., 1.), ] * 2, 200)
    assert len(samples) == 200
    assert len(samples[0]) == 2
    assert isinstance(s, InitialPointGenerator)

    samples = s.generate([("a", "b", "c")], 3)
    assert samples[0][0] in ["a", "b", "c"]

    samples = s.generate([("a", "b", "c"), (0, 1)], 1)
    assert samples[0][0] in ["a", "b", "c"]
    assert samples[0][1] in [0, 1]

    samples = s.generate([("a", "b", "c"), (0, 1)], 3)
    assert samples[0][0] in ["a", "b", "c"]
    assert samples[0][1] in [0, 1]


@pytest.mark.parametrize("lhs_type", LHS_TYPE)
@pytest.mark.parametrize("criterion", CRITERION)
def test_lhs_criterion(lhs_type, criterion):
    lhs = Lhs(lhs_type=lhs_type, criterion=criterion, iterations=100)
    samples = lhs.generate([(0., 1.), ] * 2, 200)
    assert len(samples) == 200
    assert len(samples[0]) == 2
    samples = lhs.generate([("a", "b", "c")], 3)
    assert samples[0][0] in ["a", "b", "c"]

    samples = lhs.generate([("a", "b", "c"), (0, 1)], 1)
    assert samples[0][0] in ["a", "b", "c"]
    assert samples[0][1] in [0, 1]

    samples = lhs.generate([("a", "b", "c"), (0, 1)], 3)
    assert samples[0][0] in ["a", "b", "c"]
    assert samples[0][1] in [0, 1]


def test_lhs_pdist():
    n_dim = 2
    n_samples = 20
    lhs = Lhs()

    h = lhs._lhs_normalized(n_dim, n_samples, 0)
    d_classic = spatial.distance.pdist(np.array(h), 'euclidean')
    lhs = Lhs(criterion="maximin", iterations=100)
    h = lhs.generate([(0., 1.), ] * n_dim, n_samples, random_state=0)
    d = spatial.distance.pdist(np.array(h), 'euclidean')
    assert np.min(d) > np.min(d_classic)


@pytest.mark.parametrize("criterion", CRITERION)
def test_lhs_random_state(criterion):
    n_dim = 2
    n_samples = 20
    lhs = Lhs()

    h = lhs._lhs_normalized(n_dim, n_samples, 0)
    h2 = lhs._lhs_normalized(n_dim, n_samples, 0)
    assert_array_equal(h, h2)
    lhs = Lhs(criterion=criterion, iterations=100)
    h = lhs.generate([(0., 1.), ] * n_dim, n_samples, random_state=0)
    h2 = lhs.generate([(0., 1.), ] * n_dim, n_samples, random_state=0)
    assert_array_equal(h, h2)


@pytest.mark.fast_test
def test_bit():
    X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    res = [2, 1, 3, 1, 2, 1, 4, 1, 2, 1]
    for i in range(len(X)):
        assert _bit_lo0(X[i]) == res[i]

    X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    res = [1, 2, 2, 3, 3, 3, 3, 4, 4, 4]
    for i in range(len(X)):
        assert _bit_hi1(X[i]) == res[i]


@pytest.mark.fast_test
def test_sobol():
    sobol = Sobol()
    x, seed = sobol._sobol(3, 1)
    assert_array_equal(x, [0.5, 0.5, 0.5])
    x, seed = sobol._sobol(3, 2)
    assert_array_equal(x, [0.75, 0.25, 0.75])
    x, seed = sobol._sobol(3, 3)
    assert_array_equal(x, [0.25, 0.75, 0.25])
    x, seed = sobol._sobol(3, 4)
    assert_array_equal(x, [0.375, 0.375, 0.625])
    x, seed = sobol._sobol(3, 5)
    assert_array_equal(x, [0.875, 0.875, 0.125])
    x, seed = sobol._sobol(3, 6)
    assert_array_equal(x, [0.625, 0.125, 0.375])


@pytest.mark.fast_test
def test_generate():
    sobol = Sobol(min_skip=1, max_skip=1)
    x = sobol.generate([(0., 1.), ] * 3, 3)
    x = np.array(x)
    assert_array_equal(x[0, :], [0.5, 0.5, 0.5])
    assert_array_equal(x[1, :], [0.75, 0.25, 0.75])
    assert_array_equal(x[2, :], [0.25, 0.75, 0.25])

    sobol.set_params(max_skip=2)
    assert sobol.max_skip == 2
    assert isinstance(sobol, InitialPointGenerator)


@pytest.mark.fast_test
def test_van_der_corput():
    x = _van_der_corput_samples(range(11), number_base=10)
    y = [0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9, 0.01, 0.11]
    assert_array_equal(x, y)

    x = _van_der_corput_samples(range(8), number_base=2)
    y = [0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625]
    assert_array_equal(x, y)


@pytest.mark.fast_test
def test_halton():
    h = Halton()
    x = h.generate([(0., 1.), ] * 2, 3)
    y = np.array([[0.125, 0.625, 0.375], [0.4444, 0.7778, 0.2222]]).T
    assert_array_almost_equal(x, y, 1e-3)

    h = Halton()
    x = h.generate([(0., 1.), ] * 2, 4)
    y = np.array([[0.125, 0.625, 0.375, 0.875],
                  [0.4444, 0.7778, 0.2222, 0.5556]]).T
    assert_array_almost_equal(x, y, 1e-3)

    samples = h.generate([(0., 1.), ] * 2, 200)
    assert len(samples) == 200
    assert len(samples[0]) == 2


@pytest.mark.fast_test
def test_hammersly():
    h = Hammersly()
    x = h.generate([(0., 1.), ] * 2, 3)
    y = np.array([[0.75, 0.125, 0.625], [0.25, 0.5, 0.75]]).T
    assert_almost_equal(x, y)
    x = h.generate([(0., 1.), ] * 2, 4)
    y = np.array([[0.75, 0.125, 0.625, 0.375], [0.2, 0.4, 0.6, 0.8]]).T
    assert_almost_equal(x, y)

    samples = h.generate([(0., 1.), ] * 2, 200)
    assert len(samples) == 200
    assert len(samples[0]) == 2


@pytest.mark.fast_test
def test_primes():

    x = _create_primes(1)
    assert_array_equal(x, [])
    x = _create_primes(2)
    assert_array_equal(x, [2])
    x = _create_primes(3)
    assert_array_equal(x, [2, 3])
    x = _create_primes(20)
    assert_array_equal(x, [2, 3, 5, 7, 11, 13, 17, 19])


@pytest.mark.fast_test
def test_quadrature_combine():
    a = [1, 2]
    b = [[4, 4], [5, 6]]
    x = [[1, 4, 4], [1, 5, 6], [2, 4, 4], [2, 5, 6]]
    x_test = _quadrature_combine([a, b])
    assert_array_equal(x_test, x)


@pytest.mark.fast_test
def test_uniform_grid():
    x = _create_uniform_grid_exclude_border(1, 2)
    assert_array_equal(x, [[1./3.], [2./3.]])
    x = _create_uniform_grid_include_border(1, 2)
    assert_array_equal(x, [[0.], [1.]])
    x = _create_uniform_grid_only_border(1, 2)
    assert_array_equal(x, [[0.], [1.]])

    x = _create_uniform_grid_exclude_border(1, 3)
    assert_array_equal(x, [[1./4.], [2./4.], [3./4.]])
    x = _create_uniform_grid_include_border(1, 3)
    assert_array_equal(x, [[0./2.], [1./2.], [2./2.]])
    x = _create_uniform_grid_only_border(1, 3)
    assert_array_equal(x, [[0./2.], [1./2.], [2./2.]])

    x = _create_uniform_grid_exclude_border(1, 5)
    assert_array_equal(x, [[1./6.], [2./6.], [3./6.], [4./6.], [5./6.]])
    x = _create_uniform_grid_include_border(1, 5)
    assert_array_equal(x, [[0./4.], [1./4.], [2./4.], [3./4.], [4./4.]])
    x = _create_uniform_grid_only_border(1, 5)
    assert_array_equal(x, [[0./4.], [1./4.], [2./4.], [3./4.], [4./4.]])

    x = _create_uniform_grid_exclude_border(2, 2)
    assert_array_equal(x, [[1. / 3., 1./3.], [1. / 3., 2. / 3.],
                           [2. / 3., 1. / 3.], [2. / 3., 2. / 3.]])
    x = _create_uniform_grid_include_border(2, 2)
    assert_array_equal(x, [[0., 0.], [0., 1.],
                           [1., 0.], [1., 1.]])
    x = _create_uniform_grid_only_border(2, 3)
    assert_array_equal(x, [[0., 0.], [0., 0.5],
                           [0., 1.], [1., 0.],
                           [1., 0.5], [1., 1.]])

    assert_raises(AssertionError, _create_uniform_grid_exclude_border, 1, 0)
    assert_raises(AssertionError, _create_uniform_grid_exclude_border, 0, 1)
    assert_raises(AssertionError, _create_uniform_grid_include_border, 1, 0)
    assert_raises(AssertionError, _create_uniform_grid_include_border, 0, 1)
    assert_raises(AssertionError, _create_uniform_grid_only_border, 1, 1)
    assert_raises(AssertionError, _create_uniform_grid_only_border, 0, 2)


@pytest.mark.fast_test
def test_grid():
    grid = Grid()
    samples = grid.generate([(0., 1.), ] * 2, 200)
    assert len(samples) == 200
    assert len(samples[0]) == 2

    grid = Grid(border="include")
    samples = grid.generate([(0., 1.), ] * 2, 200)
    assert len(samples) == 200
    assert len(samples[0]) == 2

    grid = Grid(use_full_layout=False)
    samples = grid.generate([(0., 1.), ] * 2, 200)
    assert len(samples) == 200
    assert len(samples[0]) == 2

    grid = Grid(use_full_layout=True, append_border="include")
    samples = grid.generate([(0., 1.), ] * 2, 200)
    assert len(samples) == 200
    assert len(samples[0]) == 2
