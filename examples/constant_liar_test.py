from skopt.space import Real
from skopt import Optimizer
from skopt.learning import ExtraTreesRegressor
import numpy as np

from multiprocessing.pool import ThreadPool
pool = ThreadPool()

n_steps = 24
n_repeats = 13

results = {}

# test different numbers of parallel workers
for n_jobs in [1, 4, 8]:
    history = []

    for rep in range(n_repeats):

        optimizer = Optimizer(
            base_estimator=ExtraTreesRegressor(),
            dimensions=[Real(-3.0, 3.0) for i in range(10)],
            acq_optimizer='sampling'
        )

        min_y_for_steps = []
        for i in range(n_steps):
            x = optimizer.ask(n_jobs)
            y = pool.map(lambda x: np.sum(np.array(x) ** 2), x)
            optimizer.tell(x, y)

            min_y_for_steps.append(min(y))
            print min(min_y_for_steps)
        history.append(min_y_for_steps)

    results[n_jobs] = history

# average results for multiple repeats and visualize them
for k in results:
    history = results[k]
    history = np.array(history)
    history = np.minimum.accumulate(history, axis = 1)
    history = np.mean(history, axis=0)
    results[k] = history

import matplotlib.pyplot as plt

for k in results:
    plt.scatter(range(len(results[k])), results[k], label="n_jobs = " + str(k), c=np.random.rand(3))

plt.legend()
plt.grid()
plt.show()