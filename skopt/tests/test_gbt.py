import numpy as np

from sklearn.utils.testing import assert_equal

from skopt.gbt import GradientBoostingRegressorWithStd


rng = np.random.RandomState(324)

def truth(X):
    return 0.5 * np.sin(1.75*X[:, 0])

def constant_noise(X):
    return np.ones_like(X)

def sample_noise(X, std=0.2, noise=constant_noise):
    """Uncertainty inherent to the process

    The regressor should try and model this.
    """
    return np.array([rng.normal(0, std*noise(x)) for x in X])

def test_gbt_with_std():
    rng = np.random.RandomState(1)
    # simple test of interface
    X = rng.uniform(0, 5, 500)[:, np.newaxis]

    noise_level = 0.5
    y = truth(X) + sample_noise(X, noise_level)

    X_ = np.linspace(0, 5, 1000)[:, np.newaxis]

    model = GradientBoostingRegressorWithStd(alpha=0.68)
    model.fit(X, y)

    p, s = model.predict(X_)
    assert_equal(p.shape, s.shape)
    assert_equal(p.shape[0], X_.shape[0])
