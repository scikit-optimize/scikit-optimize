from sklearn.utils import check_array
from sklearn.tree import DecisionTreeRegressor as DTRegressor


class DecisionTreeRegressor(DTRegressor):
    """
    DecisionTreeRegressor that can return the variance of predictions.
    """
    def predict(self, X, return_variance=False):
        predictions = super(DecisionTreeRegressor, self).predict(X)
        if return_variance:
            leaf_node = self.apply(X)
            return predictions, self.tree_.impurity[leaf_node]
        return predictions
