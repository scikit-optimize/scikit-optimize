import pytest
import numbers
import numpy as np
from numpy.testing import assert_raises
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises_regex
from skopt.space import LogN, Normalize


@pytest.mark.fast_test
def test_logn2_integer():

    transformer = LogN(2)
    for X in range(2, 31):
        X_orig = transformer.inverse_transform(transformer.transform(X))
        assert_array_equal(int(np.round(X_orig)), X)

@pytest.mark.fast_test
def test_logn10_integer():

    transformer = LogN(2)
    for X in range(2, 31):
        X_orig = transformer.inverse_transform(transformer.transform(X))
        assert_array_equal(int(np.round(X_orig)), X)


@pytest.mark.fast_test
def test_normalize_integer():
    transformer = Normalize(1, 20, is_int=True)
    assert transformer.transform(19.8) == 1.0
    assert transformer.transform(20.2) == 1.0
    assert transformer.transform(1.2) == 0.0
    assert transformer.transform(0.9) == 0.0
    assert_raises(ValueError, transformer.transform, 20.6)
    assert_raises(ValueError, transformer.transform, 0.4)

    assert transformer.inverse_transform(0.99) == 20
    assert transformer.inverse_transform(0.01) == 1
    assert_raises(ValueError, transformer.inverse_transform, 1. + 1e-8)
    assert_raises(ValueError, transformer.transform, 0. - 1e-8)

@pytest.mark.fast_test
def test_normalize():
    transformer = Normalize(1, 20, is_int=False)
    assert transformer.transform(20.) == 1.0
    assert transformer.transform(1.) == 0.0
    assert_raises(ValueError, transformer.transform, 20. + 1e-7)
    assert_raises(ValueError, transformer.transform, 1.0 - 1e-7)
    assert_raises(ValueError, transformer.inverse_transform, 1. + 1e-8)
    assert_raises(ValueError, transformer.transform, 0. - 1e-8)
