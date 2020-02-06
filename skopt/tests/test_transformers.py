import pytest
import numbers
import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises_regex
from skopt.space import LogN
from skopt.space.transformers import StringEncoder, IntegerEncoder, Identity


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
def test_integer_encoder():

    transformer = IntegerEncoder()
    X = [1, 5, 9]
    transformer.fit(X)
    assert_array_equal(transformer.transform(X), [0, 1, 2])
    assert_array_equal(transformer.inverse_transform([0, 1, 2]), X)

    transformer = IntegerEncoder(X)
    assert_array_equal(transformer.transform(X), [0, 1, 2])
    assert_array_equal(transformer.inverse_transform([0, 1, 2]), X)

    X = ["a", "b", "c"]
    transformer.fit(X)
    assert_array_equal(transformer.transform(X), [0, 1, 2])
    assert_array_equal(transformer.inverse_transform([0, 1, 2]), X)

    transformer = IntegerEncoder(X)
    assert_array_equal(transformer.transform(X), [0, 1, 2])
    assert_array_equal(transformer.inverse_transform([0, 1, 2]), X)


@pytest.mark.fast_test
def test_string_encoder():

    transformer = StringEncoder()
    X = [1, 5, 9]
    transformer.fit(X)
    assert_array_equal(transformer.transform(X), ['1', '5', '9'])
    assert_array_equal(transformer.inverse_transform(['1', '5', '9']), X)

    X = ['a', True, 1]
    transformer.fit(X)
    assert_array_equal(transformer.transform(X), ['a', 'True', '1'])
    assert_array_equal(transformer.inverse_transform(['a', 'True', '1']), X)

    X = ["a", "b", "c"]
    transformer.fit(X)
    assert_array_equal(transformer.transform(X), X)
    assert_array_equal(transformer.inverse_transform(X), X)


@pytest.mark.fast_test
def test_identity_encoder():

    transformer = Identity()
    X = [1, 5, 9]
    transformer.fit(X)
    assert_array_equal(transformer.transform(X), X)
    assert_array_equal(transformer.inverse_transform(X), X)

    X = ['a', True, 1]
    transformer.fit(X)
    assert_array_equal(transformer.transform(X), X)
    assert_array_equal(transformer.inverse_transform(X), X)

    X = ["a", "b", "c"]
    transformer.fit(X)
    assert_array_equal(transformer.transform(X), X)
    assert_array_equal(transformer.inverse_transform(X), X)
