import numpy as np
from sklearn.tree import DecisionTreeRegressor as sk_DecisionTreeRegressor


class DecisionTreeRegressor(sk_DecisionTreeRegressor):
    """
    DecisionTreeRegressor that supports `return_std`.
    """
    def __init__(self, criterion='mse', splitter='best', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features=None,
                 random_state=None, max_leaf_nodes=None, presort=False,
                 min_variance=0.0):
        self.min_variance = min_variance
        super(DecisionTreeRegressor, self).__init__(
            criterion=criterion, splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features, max_leaf_nodes=max_leaf_nodes,
            presort=presort, random_state=random_state)

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
            var[var < self.min_variance] = self.min_variance
            return mean, var**0.5

        return mean
