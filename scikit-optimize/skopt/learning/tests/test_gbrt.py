import numpy as np
import pytest

from scipy import stats

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_almost_equal

from skopt.learning import GradientBoostingQuantileRegressor


def truth(X):
    return 0.5 * np.sin(1.75*X[:, 0])


@pytest.mark.fast_test
def test_gbrt_gaussian():
    # estimate quantiles of the normal distribution
    rng = np.random.RandomState(1)
    N = 10000
    X = np.ones((N, 1))
    y = rng.normal(size=N)

    rgr = GradientBoostingQuantileRegressor()
    rgr.fit(X, y)

    estimates = rgr.predict(X, return_quantiles=True)
    assert_almost_equal(stats.norm.ppf(rgr.quantiles),
                        np.mean(estimates, axis=0),
                        decimal=2)


@pytest.mark.fast_test
def test_gbrt_base_estimator():
    rng = np.random.RandomState(1)
    N = 10000
    X = np.ones((N, 1))
    y = rng.normal(size=N)

    base = RandomForestRegressor()
    rgr = GradientBoostingQuantileRegressor(base_estimator=base)
    with pytest.raises(ValueError):
        # 'type GradientBoostingRegressor',
        rgr.fit(X, y)

    base = GradientBoostingRegressor()
    rgr = GradientBoostingQuantileRegressor(base_estimator=base)
    with pytest.raises(ValueError):
        # 'quantile loss'
        rgr.fit(X, y)

    base = GradientBoostingRegressor(loss='quantile', n_estimators=20)
    rgr = GradientBoostingQuantileRegressor(base_estimator=base)
    rgr.fit(X, y)

    estimates = rgr.predict(X, return_quantiles=True)
    assert_almost_equal(stats.norm.ppf(rgr.quantiles),
                        np.mean(estimates, axis=0),
                        decimal=2)


@pytest.mark.fast_test
def test_gbrt_with_std():
    # simple test of the interface
    rng = np.random.RandomState(1)
    X = rng.uniform(0, 5, 500)[:, np.newaxis]

    noise_level = 0.5
    y = truth(X) + rng.normal(0, noise_level, len(X))
    X_ = np.linspace(0, 5, 1000)[:, np.newaxis]

    model = GradientBoostingQuantileRegressor()
    model.fit(X, y)

    # three quantiles, so three numbers per sample
    assert_array_equal(model.predict(X_, return_quantiles=True).shape,
                       (len(X_), 3))
    # "traditional" API which returns one number per sample, in this case
    # just the median/mean
    assert_array_equal(model.predict(X_).shape, (len(X_)))

    l, c, h = model.predict(X_, return_quantiles=True).T
    assert_equal(l.shape, c.shape)
    assert_equal(c.shape, h.shape)
    assert_equal(l.shape[0], X_.shape[0])

    mean, std = model.predict(X_, return_std=True)
    assert_array_equal(mean, c)
    assert_array_equal(std, (h - l) / 2.0)


@pytest.mark.fast_test
def test_gbrt_in_parallel():
    # check estimate quantiles with parallel
    rng = np.random.RandomState(1)
    N = 10000
    X = np.ones((N, 1))
    y = rng.normal(size=N)

    rgr = GradientBoostingQuantileRegressor(
        n_jobs=1, random_state=np.random.RandomState(1))
    rgr.fit(X, y)
    estimates = rgr.predict(X)

    rgr.set_params(n_jobs=2, random_state=np.random.RandomState(1))
    rgr.fit(X, y)
    estimates_parallel = rgr.predict(X)

    assert_array_equal(estimates, estimates_parallel)
