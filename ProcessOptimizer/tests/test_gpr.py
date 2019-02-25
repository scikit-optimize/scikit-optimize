import pytest

import numpy as np
from ProcessOptimizer.learning import GaussianProcessRegressor


@pytest.mark.fast_test
def test_gpr_uses_noise():
    """ Test that gpr is using WhiteKernel"""

    X = np.random.normal(size=[100, 2])
    Y = np.random.normal(size=[100])

    g_gaussian = GaussianProcessRegressor(noise='gaussian')
    g_gaussian.fit(X, Y)
    m, sigma = g_gaussian.predict(X[0:1], return_cov=True)
    assert sigma > 0
