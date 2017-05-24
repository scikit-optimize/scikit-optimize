"""This script contains set of functions that test parallel optimization with
skopt, where constant liar parallelization strategy is used.
"""

import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from skopt.space import Real
from skopt.learning import ExtraTreesRegressor
from skopt import Optimizer
from skopt.benchmarks import branin

def test_constant_liar_runs():
    optimizer = Optimizer(
        base_estimator=ExtraTreesRegressor(),
        dimensions=[Real(-5.0, 10.0),Real(0.0, 15.0)],
        acq_optimizer='sampling'
    )

    # 13 points to evaluate - 10 random, 3 using some strategy.
    # It is like this for speed of test - suggestions are welcome.
    n_points, n_steps, Y = 2, 13, []


    for i in range(n_steps):
        strategy = optimizer.par_strats[i % len(optimizer.par_strats)]
        x = optimizer.ask(n_points, strategy)
        optimizer.tell(x, [branin(v) for v in x])