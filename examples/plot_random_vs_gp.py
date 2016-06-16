"""
==========================================
Tying a sklearn estimator with gp_minimize
==========================================

This example shows how to chain gp_minimize with RandomForestClassifier
in this case.
"""
print(__doc__)

import numpy as np

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from skopt.gp_opt import gp_minimize


digits = load_digits()
X, y = digits.data, digits.target
rfc = RandomForestClassifier(random_state=10)

def compute_mean_validation_score(forest_params):
    max_depth, max_features, mss, msl = forest_params

    rfc.set_params(
        max_depth=max_depth, max_features=max_features,
        min_samples_split=mss, min_samples_leaf=msl)

    return -np.mean(cross_val_score(rfc, X, y, cv=3, n_jobs=-1))

# Bounds inspired by
# http://scikit-learn.org/dev/auto_examples/model_selection/randomized_search.html#example-model-selection-randomized-search-py
dimensions = [(3, 50), (1, 12), (1, 12), (1, 12)]
best_dummy_scores = np.zeros((5, 100))
best_gp_scores = np.zeros((5, 100))

gp_model = gp_minimize(compute_mean_validation_score,
                       dimensions,
                       maxiter=100,
                       random_state=0,
                       n_start=1)

print("Best score obtained = %0.4f for parameters %s" % (-gp_model.fun,
                                                         gp_model.x))
