import numpy as np
from multiprocessing.pool import ThreadPool
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor
from skopt import Optimizer

pool = ThreadPool()
optimizer = Optimizer(
    base_estimator=GaussianProcessRegressor(),
    dimensions=[Real(-3.0, 3.0) for i in range(10)]
)

n_jobs, n_steps, Y = 4, 20, []

for i in range(n_steps):
    x = optimizer.ask(n_jobs)
    y = pool.map(lambda x: np.sum(np.array(x) ** 2), x)
    optimizer.tell(x, y)

    Y.extend(y) # keep objectives history here
    print min(Y)
