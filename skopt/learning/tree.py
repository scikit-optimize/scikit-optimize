import numpy as np
from sklearn.utils import check_array
from sklearn.tree import DecisionTreeRegressor as sklearn_DecisionTreeRegressor


class DecisionTreeRegressor(sklearn_DecisionTreeRegressor):
    """
    DecisionTreeRegressor that can return the variance of predictions.
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
            this is the mean of the target variables in the leaf node that each
            data point ends up in.

        * `std` [array-like, shape=(n_samples,)]:
            Standard deviation of the predicted values for X. If criterion
            is set to "mse", this is the std of the target variables in the leaf
            node that each data point ends up in.
        """
        predictions = super(DecisionTreeRegressor, self).predict(X)
        if return_std:
            if self.criterion != "mse":
                raise ValueError(
                    "Expected impurity to be 'mse', got %s instead"
                    % self.criterion)
            leaf_node = self.apply(X)
            return predictions, np.sqrt(self.tree_.impurity[leaf_node])
        return predictions
