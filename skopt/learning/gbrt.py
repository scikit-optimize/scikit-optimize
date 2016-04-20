import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import check_random_state


class GradientBoostingQuantileRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, quantiles=[0.16, 0.5, 0.84], random_state=None):
        """Predict several quantiles with one estimator

        This is a wrapper around `GradientBoostingRegressor`'s quantile
        regression that allows you to predict several `quantiles` in
        one go.

        Parameters
        ----------
        quantiles : array-like, optional
            Quantiles to predict. By default the 16, 50 and 84%
            quantiles are predicted.

        random-state : int, RandomState instance, or None (default)
            Set random state to something other than None for reproducible
            results.
        """
        self.quantiles = quantiles
        self.random_state = random_state

    def fit(self, X, y):
        """Fit one regressor for each quantile.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values (real numbers in regression)
        """
        rng = check_random_state(self.random_state)
        self.regressors_ = [GradientBoostingRegressor(loss='quantile',
                                                      alpha=a,
                                                      random_state=rng)
                            for a in self.quantiles]
        for rgr in self.regressors_:
            rgr.fit(X, y)

        return self

    def predict(self, X):
        """Predictions for each quantile."""
        return np.asarray([rgr.predict(X) for rgr in self.regressors_]).T
