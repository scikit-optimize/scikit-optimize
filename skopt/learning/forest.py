import numpy as np
from sklearn.ensemble import RandomForestRegressor as sk_RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor as sk_ExtraTreesRegressor


class RandomForestRegressor(sk_RandomForestRegressor):
    """
    RandomForestRegressor that supports `return_std`.
    """
    def predict(self, X, return_std=False):
        """
        Predict continuous output for X.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            Input data.

        * `return_std` [bool, default False]:
            Whether or not to return the standard deviation.

        Returns
        -------
        * `predictions` [array-like, shape=(n_samples,)]:
            Predicted values for X. If criterion is set to "mse",
            then `predictions[i] ~= mean(y | X[i])`.

        * `std` [array-like, shape=(n_samples,)]:
            Standard deviation of `y` at `X`. If criterion
            is set to "mse", then `std[i] ~= std(y | X[i])`.
        """
        mean = super(RandomForestRegressor, self).predict(X)

        if return_std:
            if self.criterion != "mse":
                raise ValueError(
                    "Expected impurity to be 'mse', got %s instead"
                    % self.criterion)

            # This derives std(y | x) as described in 4.3.2 of arXiv:1211.0906
            std = np.zeros(len(X))

            for tree in self.estimators_:
                var_tree = tree.tree_.impurity[tree.apply(X)]
                mean_tree = tree.predict(X)
                std += var_tree + mean_tree ** 2

            std /= len(self.estimators_)
            std -= mean ** 2.0
            std[std < 0.0] = 0.0
            std = std ** 0.5

            return mean, std

        return mean


class ExtraTreesRegressor(sk_ExtraTreesRegressor):
    """
    ExtraTreesRegressor that supports `return_std`.
    """
    def predict(self, X, return_std=False):
        """
        Predict continuous output for X.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            Input data.

        * `return_std` [bool, default False]:
            Whether or not to return the standard deviation.

        Returns
        -------
        * `predictions` [array-like, shape=(n_samples,)]:
            Predicted values for X. If criterion is set to "mse",
            then `predictions[i] ~= mean(y | X[i])`.

        * `std` [array-like, shape=(n_samples,)]:
            Standard deviation of `y` at `X`. If criterion
            is set to "mse", then `std[i] ~= std(y | X[i])`.
        """
        mean = super(ExtraTreesRegressor, self).predict(X)

        if return_std:
            if self.criterion != "mse":
                raise ValueError(
                    "Expected impurity to be 'mse', got %s instead"
                    % self.criterion)

            # This derives std(y | x) as described in 4.3.2 of arXiv:1211.0906
            std = np.zeros(len(X))

            for tree in self.estimators_:
                var_tree = tree.tree_.impurity[tree.apply(X)]
                mean_tree = tree.predict(X)
                std += var_tree + mean_tree ** 2

            std /= len(self.estimators_)
            std -= mean ** 2.0
            std[std < 0.0] = 0.0
            std = std ** 0.5

            return mean, std

        return mean
