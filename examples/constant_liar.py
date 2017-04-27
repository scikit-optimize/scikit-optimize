"""
Example for parallel optimization with skopt.
The points to evaluate in parallel are selected according to the "constant lie" approach.

"""

import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor
from skopt import Optimizer

optimizer = Optimizer(
    base_estimator=GaussianProcessRegressor(),
    dimensions=[Real(-3.0, 3.0) for i in range(10)]
)

# objective function to minimze
def objective(x):
    return np.sum(np.array(x) ** 2)

# configure number of threads to be used in parallel, and overall # of computations
n_points, n_steps, Y = 4, 20, []

for i in range(n_steps):
    x = optimizer.ask(n_points)
    # evaluate n_points in parallel
    y = Parallel()(delayed(objective)(v) for v in x)
    # tell points and corresponding objectives to the optimizer
    optimizer.tell(x, y)
    # keep objectives history
    Y.extend(y)
    print min(Y)
