import pytest
import numbers
import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises_regex
from skopt.space import LogN


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
