from time import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from skopt.dummy_opt import dummy_minimize
from skopt.gp_opt import gp_minimize

digits = load_digits()
X, y = digits.data, digits.target
rfc = RandomForestClassifier(random_state=10)

def compute_mean_validation_score(forest_params):
    forest_params = [int(param) for param in forest_params]
    max_depth, max_features, mss, msl = forest_params

    params = {
        'max_depth': [max_depth], 'max_features': [max_features],
        'min_samples_split': [mss], 'min_samples_leaf': [msl]}
    gscv = GridSearchCV(rfc, params, n_jobs=-1)
    gscv.fit(X, y)
    return -gscv.best_score_

# Bounds inspired by
# http://scikit-learn.org/dev/auto_examples/model_selection/randomized_search.html#example-model-selection-randomized-search-py
bounds = [(3, 50), (1, 12), (1, 12), (1, 12)]

print("Doing a random search for the best random forest hyperparameter.")
t = time()
dummy_model = dummy_minimize(
    compute_mean_validation_score, bounds, maxiter=100, random_state=0)
print(time() - t)
print("Best score obtained, %0.4f" % -dummy_model.fun)

print("Doing a gp-based search for the best random forest hyperparameter.")
t = time()
gp_model = gp_minimize(
    compute_mean_validation_score, bounds, maxiter=100, random_state=0,
    n_start=1
    )
print(time() - t)
print("Best score obtained, %0.4f" % -gp_model.fun)

best_dummy_scores = [-np.min(dummy_model.func_vals[:i]) for i in range(1, 101)]
best_gp_scores = [-np.min(gp_model.func_vals[:i]) for i in range(1, 101)]

plt.title("Best score obtained at every iteration")
plt.plot(range(1, 101), best_dummy_scores, label="Dummy search")
plt.plot(range(1, 101), best_gp_scores, label="GP search")
plt.legend(loc="best")
plt.xlabel("Number of iterations.")
plt.ylabel("Mean accuracy score")
plt.ylim([0.885, 0.920])
plt.show()
