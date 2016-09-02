import numpy as np

from skopt.benchmarks import hart6
from skopt import gp_minimize

bounds = np.tile((0., 1.), (6, 1))
n_calls = 200

hart_results = []
# Mean over ten calls to the optimization procdeure.
for random_state in range(10):
    res = gp_minimize(hart6, bounds, n_random_starts=1,
        random_state=random_state, verbose=True, n_calls=n_calls)
    hart_results.append(res)

optimal_values = [result.fun for result in hart_results]
mean_optimum = np.mean(optimal_values)
std = np.std(optimal_values)
best = np.min(optimal_values)
print(mean_optimum)
print(std)
print(best)
