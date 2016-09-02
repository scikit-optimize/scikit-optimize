import numpy as np

from skopt.benchmarks import branin
from skopt import gp_minimize

bounds = [(-5.0, 10.0), (0.0, 15.0)]
n_calls = 200

# Mean over ten calls to the optimization procdeure.
branin_results = []
for random_state in range(10):
    res = gp_minimize(
        branin, bounds, n_random_starts=1, random_state=random_state,
        n_calls=n_calls)
    branin_results.append(res)

optimal_values = [result.fun for result in branin_results]
mean_optimum = np.mean(optimal_values)
std = np.std(optimal_values)
best = np.min(optimal_values)
print(mean_optimum)
print(std)
print(best)
