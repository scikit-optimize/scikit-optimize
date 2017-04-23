"""
Example for parallel optimization with skopt.
The points to evaluate in parallel are selected according to the "constant lie" approach.

"""

import numpy as np
from multiprocessing.pool import ThreadPool
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor
from skopt import Optimizer

# ThreadPool is used for parallel computations
pool = ThreadPool()
optimizer = Optimizer(
    base_estimator=GaussianProcessRegressor(),
    dimensions=[Real(-3.0, 3.0) for i in range(10)]
)

# configure number of threads to be used in parallel, and overall # of computations
n_points, n_steps, Y = 4, 20, []

for i in range(n_steps):
    x = optimizer.ask(n_points)
    # evaluate n_points in parallel
    y = pool.map(lambda x: np.sum(np.array(x) ** 2), x)
    # tell points and corresponding objectives to the optimizer
    optimizer.tell(x, y)
    # keep objectives history
    Y.extend(y)
    print min(Y)
