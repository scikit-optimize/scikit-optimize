import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import check_random_state


class GBTQuantiles(BaseEstimator, RegressorMixin):
    def __init__(self, quantiles=[0.16, 0.5, 0.84], random_state=None):
        self.quantiles = quantiles
        self.random_state = random_state

    def fit(self, X, y):
        """Fit one regressor for each quantile"""
        rng = check_random_state(self.random_state)
        self.regressors_ = [GradientBoostingRegressor(loss='quantile',
                                                      alpha=a,
                                                      random_state=rng)
                            for a in self.quantiles]
        for rgr in self.regressors_:
            rgr.fit(X, y)

    def predict(self, X):
        """Predictions for each quantile"""
        return np.vstack([rgr.predict(X) for rgr in self.regressors_])
