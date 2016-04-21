"""
==========================================
Tying a sklearn estimator with gp_minimize
==========================================

This example shows how to chain gp_minimize with RandomForestClassifier
in this case.
"""
print(__doc__)
from time import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint as sp_randint

from sklearn.base import clone
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from skopt.gp_opt import gp_minimize

digits = load_digits()
X, y = digits.data, digits.target
rfc = RandomForestClassifier(random_state=10)

def compute_mean_validation_score(forest_params):
    # Hack to allow integer parameters, since the parameters
    # sampled internally are uniform in a given range.
    forest_params = [int(param) for param in forest_params]
    max_depth, max_features, mss, msl = forest_params

    rfc.set_params(
        max_depth=max_depth, max_features=max_features,
        min_samples_split=mss, min_samples_leaf=msl)
    return -np.mean(cross_val_score(rfc, X, y, cv=3, n_jobs=-1))

# Bounds inspired by
# http://scikit-learn.org/dev/auto_examples/model_selection/randomized_search.html#example-model-selection-randomized-search-py
bounds = [(3, 50), (1, 12), (1, 12), (1, 12)]
best_dummy_scores = np.zeros((5, 100))
best_gp_scores = np.zeros((5, 100))

print("Doing a gp-based search for the best random forest hyperparameter.")
t = time()
gp_model = gp_minimize(
    compute_mean_validation_score, bounds, maxiter=100,
    random_state=0,
    n_start=1
    )
print(time() - t)
print("Best score obtained, %0.4f" % -gp_model.fun)
