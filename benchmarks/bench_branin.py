from functools import partial

import numpy as np

from skopt.benchmarks import branin
from skopt import gp_minimize, forest_minimize, gbrt_minimize

bounds = [(-5.0, 10.0), (0.0, 15.0)]
n_calls = 200
optimizers = [("gp_minimize", gp_minimize), ("forest_minimize", forest_minimize),
              ("gbrt_minimize", gbrt_minimize)]

for name, optimizer in optimizers:
    print(name)
    results = []
    for random_state in range(10):
        print(random_state)
        res = optimizer(branin, bounds, random_state=random_state, n_calls=n_calls)
        results.append(res)

    optimal_values = [result.fun for result in results]
    mean_optimum = np.mean(optimal_values)
    std = np.std(optimal_values)
    best = np.min(optimal_values)
    print("Mean optimum: " + str(mean_optimum))
    print("Std of optimal values" + str(std))
    print("Best optima:" + str(best))
