import numpy as np

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_true

from skopt.parameters import Real


def check_distribution(dist, random_val):
    assert_equal(dist.rvs(random_state=1), random_val)
    assert_equal(dist.rvs(random_state=1), random_val)

def test_distributions():
    r = Real(1.0, 4.0)
    yield (check_distribution, r, 2.668088018810296)
    yield (check_distribution, r, 2.668088018810296)