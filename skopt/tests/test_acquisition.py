import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_raises

from skopt.acquisition import gaussian_acquisition

class ConstSurrogate:
    def predict(self, X, return_std=True):
        return np.zeros(X.shape[0]), np.ones(X.shape[0])

def test_acquisition_ei_correctness():
    # check that it works with a vector as well
    ei = gaussian_acquisition(np.array([[10., 10.],
                                        [10., 10.],
                                        [10., 10.],
                                        [10., 10.]]),
                              ConstSurrogate(),
                              -0.5,
                              xi=0., method="EI")

    assert_array_almost_equal(ei, [-0.1977966] * 4)

def test_acquisition_api():
    rng = np.random.RandomState(0)
    X = rng.randn(10, 2)
    y = rng.randn(10)
    gpr = GaussianProcessRegressor()
    gpr.fit(X, y)

    assert_array_equal(gaussian_acquisition(X, gpr).shape, 10)
    assert_raises(ValueError, gaussian_acquisition, rng.rand(10), gpr)
