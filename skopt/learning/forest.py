import numpy as np
from sklearn.ensemble import RandomForestRegressor as sk_RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor as sk_ExtraTreesRegressor


def _return_std(X, trees, predictions):
    """
    Returns ``std(Y | X)``.

    Can be calculated by E[Var(Y | Tree)] + Var(E[Y | Tree]) where
    P(Tree) is ``(1 / len(trees))``

    Parameters
    ----------
    * `X` [array-like, shape=(n_samples, n_features)]:
        Input data.

    * `trees` [list, shape=(n_estimators,)]
        List of fit sklearn trees as obtained from the ``estimators_``
        attribute of a fit RandomForestRegressor or ExtraTreesRegressor.

    * `predictions` [array-like, shape=(n_samples,)]
        Prediction of each data point as returned by RandomForestRegressor
        or ExtraTreesRegressor.

    Returns
    -------
    * `std` [array-like, shape=(n_samples,)]:
        Standard deviation of `y` at `X`. If criterion
        is set to "mse", then `std[i] ~= std(y | X[i])`.
    """
    # This derives std(y | x) as described in 4.3.2 of arXiv:1211.0906
    std = np.zeros(len(X))

    for tree in trees:
        var_tree = tree.tree_.impurity[tree.apply(X)]
        mean_tree = tree.predict(X)
        std += var_tree + mean_tree ** 2

    std /= len(trees)
    std -= predictions ** 2.0
    std[std < 0.0] = 0.0
    std = std ** 0.5
    return std


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
            return mean, _return_std(X, self.estimators_, mean)
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
            return mean, _return_std(X, self.estimators_, mean)

        return mean
