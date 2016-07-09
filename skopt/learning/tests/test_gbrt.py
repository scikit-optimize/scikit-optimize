import numpy as np
from scipy import stats

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.utils import check_random_state
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raise_message
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_almost_equal

from skopt.learning import GradientBoostingQuantileRegressor


def truth(X):
    return 0.5 * np.sin(1.75*X[:, 0])


def constant_noise(X):
    return np.ones_like(X)


def sample_noise(X, std=0.2, noise=constant_noise,
                 random_state=None):
    """Uncertainty inherent to the process

    The regressor should try and model this.
    """
    rng = check_random_state(random_state)
    return np.array([rng.normal(0, std*noise(x)) for x in X])


def test_gbrt_gaussian():
    # estimate quantiles of the normal distribution
    rng = np.random.RandomState(1)
    N = 10000
    X = np.ones((N, 1))
    y = rng.normal(size=N)

    rgr = GradientBoostingQuantileRegressor()
    rgr.fit(X, y)

    estimates = rgr.predict(X)
    assert_almost_equal(stats.norm.ppf(rgr.quantiles),
                        np.mean(estimates, axis=0),
                        decimal=2)


def test_gbrt_base_estimator():
    rng = np.random.RandomState(1)
    N = 10000
    X = np.ones((N, 1))
    y = rng.normal(size=N)

    base = RandomForestRegressor()
    rgr = GradientBoostingQuantileRegressor(base_estimator=base)
    assert_raise_message(ValueError, 'type GradientBoostingRegressor',
                         rgr.fit, X, y)

    base = GradientBoostingRegressor()
    rgr = GradientBoostingQuantileRegressor(base_estimator=base)
    assert_raise_message(ValueError, 'quantile loss', rgr.fit, X, y)

    base = GradientBoostingRegressor(loss='quantile', n_estimators=20)
    rgr = GradientBoostingQuantileRegressor(base_estimator=base)
    rgr.fit(X, y)

    estimates = rgr.predict(X)
    assert_almost_equal(stats.norm.ppf(rgr.quantiles),
                        np.mean(estimates, axis=0),
                        decimal=2)


def test_gbrt_with_std():
    # simple test of the interface
    rng = np.random.RandomState(1)
    X = rng.uniform(0, 5, 500)[:, np.newaxis]

    noise_level = 0.5
    y = truth(X) + sample_noise(X, noise_level, random_state=rng)

    X_ = np.linspace(0, 5, 1000)[:, np.newaxis]

    model = GradientBoostingQuantileRegressor()
    model.fit(X, y)

    assert_array_equal(model.predict(X_).shape, (len(X_), 3))

    l, c, h = model.predict(X_).T
    assert_equal(l.shape, c.shape, h.shape)
    assert_equal(l.shape[0], X_.shape[0])

    mean, std = model.predict(X_, return_std=True)
    assert_array_equal(mean, c)
    assert_array_equal(std, (h - l) / 2.0)
