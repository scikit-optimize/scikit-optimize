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
from numpy.testing import assert_raises_regex

from skopt import Optimizer
from skopt.space import Space
from skopt.space import Real
from skopt.space import Integer
from skopt.space import Categorical
from skopt.space import check_dimension as space_check_dimension
from skopt.samples.sobol import _bit_lo0, _bit_hi1
from skopt.samples.halton import _van_der_corput_samples, _create_primes
from skopt.samples import Hammersly, Halton, Lhs, Sobol


@pytest.mark.fast_test
def test_lhs_type():
    lhs = Lhs(lhs_type="classic")
    samples = lhs.generate(2, 200)
    assert len(samples) == 200
    assert len(samples[0]) == 2
    lhs = Lhs(lhs_type="centered")
    samples = lhs.generate(3, 3)
    assert_almost_equal(np.sum(samples), 4.5)


def test_lhs_criterion():
    for criterion in ["maximin", "ratio", "correlation"]:
        lhs = Lhs(criterion=criterion, iterations=100)
        samples = lhs.generate(2, 200)
        assert len(samples) == 200
        assert len(samples[0]) == 2


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
    x = sobol.generate(3, 3)
    assert_array_equal(x[0, :], [0.5, 0.5, 0.5])
    assert_array_equal(x[1, :], [0.75, 0.25, 0.75])
    assert_array_equal(x[2, :], [0.25, 0.75, 0.25])


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
    x = h.generate(2, 3)
    y = np.array([[0.125, 0.625, 0.375], [0.4444, 0.7778, 0.2222]]).T
    assert_array_almost_equal(x, y, 1e-3)

    h = Halton()
    x = h.generate(2, 4)
    y = np.array([[0.125, 0.625, 0.375, 0.875],
                  [0.4444, 0.7778, 0.2222, 0.5556]]).T
    assert_array_almost_equal(x, y, 1e-3)

    samples = h.generate(2, 200)
    assert len(samples) == 200
    assert len(samples[0]) == 2


@pytest.mark.fast_test
def test_hammersly():
    h = Hammersly()
    x = h.generate(2, 3)
    y = np.array([[0.75, 0.125, 0.625], [0.25, 0.5, 0.75]]).T
    assert_almost_equal(x, y)
    x = h.generate(2, 4)
    y = np.array([[0.75, 0.125, 0.625, 0.375], [0.2, 0.4, 0.6, 0.8]]).T
    assert_almost_equal(x, y)

    samples = h.generate(2, 200)
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
