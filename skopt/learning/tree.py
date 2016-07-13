import numpy as np
from sklearn.tree import DecisionTreeRegressor as sk_DecisionTreeRegressor


class DecisionTreeRegressor(sk_DecisionTreeRegressor):
    """
    DecisionTreeRegressor that supports `return_std`.
    """
    def predict(self, X, return_std=False):
        """
        Predict continuous output for `X`.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            Input data.

        * `return_std` [bool, default=False]:
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
        mean = super(DecisionTreeRegressor, self).predict(X)

        if return_std:
            if self.criterion != "mse":
                raise ValueError(
                    "Expected impurity to be 'mse', got %s instead"
                    % self.criterion)

            var = self.tree_.impurity[self.apply(X)]
            var[var < 0.01] = 0.01
            return mean, var**0.5

        return mean
