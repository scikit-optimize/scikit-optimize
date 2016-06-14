from math import sqrt
from time import time

import numpy as np
import matplotlib.pyplot as plt

from skopt.benchmarks import branin
from skopt.dummy_opt import dummy_minimize
from skopt.gp_opt import gp_minimize

dimensions = [(-5.0, 10.0), (0.0, 15.0)]
best_dummy_scores = np.zeros((5, 200))
best_gp_scores = np.zeros((5, 200))
n_iterations = range(1, 201)

for random_state in range(5):
    print("Doing a random search for the minimum.")
    t = time()
    dummy_model = dummy_minimize(
        branin, dimensions, maxiter=200, random_state=random_state)
    print(time() - t)
    print("Best score obtained, %0.4f" % dummy_model.fun)

    print("Doing a gp-based search for the minimum")
    t = time()
    gp_model = gp_minimize(
        branin, dimensions, maxiter=200, random_state=random_state, n_start=1)
    print(time() - t)
    print("Best score obtained, %0.4f" % gp_model.fun)

    for j in range(1, 201):
        best_dummy_scores[random_state, j-1] = np.min(
            dummy_model.func_vals[:j])
        best_gp_scores[random_state, j-1] = np.min(
            gp_model.func_vals[:j])

mean_dummy_scores = np.mean(best_dummy_scores, axis=0)
mean_gp_scores = np.mean(best_gp_scores, axis=0)
err_dummy_scores = np.std(best_dummy_scores, axis=0) / sqrt(10)
err_gp_scores = np.std(best_gp_scores, axis=0) / sqrt(10)

print("Mean minimum value obtained after 200 iterations by dummy search "
      "across 5 random states.")
print("%0.4f" % mean_dummy_scores[-1])
print("Mean minimum value obtained after 200 iterations by gp-based search "
      "across 5 random states.")
print("%0.4f" % mean_gp_scores[-1])

plt.title("Minimum obtained at every iteration for branin")
plt.plot(n_iterations, mean_dummy_scores, label="Dummy search", color='red')
plt.plot(n_iterations, mean_gp_scores, label="GP search", color='green')
plt.fill_between(
    n_iterations, mean_dummy_scores - err_dummy_scores,
    mean_dummy_scores + err_dummy_scores, color='red', alpha=0.3)
plt.fill_between(
    n_iterations, mean_gp_scores - err_gp_scores,
    mean_gp_scores + err_gp_scores, color='green', alpha=0.3)
plt.legend(loc="best")
plt.xlabel("Number of iterations.")
plt.ylabel("Optimal value.")
plt.ylim([0, 5])
plt.show()
