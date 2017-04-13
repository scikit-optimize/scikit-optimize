from copy import deepcopy
import numpy as np

class ParallelCLOptimizer():
    """
   "Constant Liar" parallel optimization strategy wrapper around instance of Optimizer class.
   See https://hal.archives-ouvertes.fr/hal-00732512/document for more details.
   Parameters
   ----------
   * `optimizer`: scikit-learn Optimizer class instance.
   """
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.rng = self.optimizer.rng # keep random generator in order to not generate same points

    def ask(self, n_jobs=1):
        opt = deepcopy(self.optimizer)
        opt.rng = self.rng # set random generator to not generate same points
        x = []
        for i in range(n_jobs):
            x.append(opt.ask())
            y_lie = np.min(opt.yi) if len(opt.yi) > 0 else 0.0 # CL-min lie
            opt.tell(x[-1], y_lie) # lie to the optimizer
        return x

    def tell(self, x, y):
        # decrease _n_random_starts only when new points are provided
        self.optimizer._n_random_starts = max(0, self.optimizer._n_random_starts-1)
        self.optimizer.tell(x, y)

if __name__ == "__main__":
    from multiprocessing.pool import ThreadPool
    from skopt.space import Real
    from skopt.learning import ExtraTreesRegressor
    from skopt import Optimizer

    pool = ThreadPool()

    obj = lambda x: np.sum(np.array(x) ** 2)
    dimensions = [Real(-3.0, 3.0) for i in range(10)]

    optimizer = Optimizer(base_estimator=ExtraTreesRegressor(), dimensions=dimensions, acq_optimizer='sampling')
    parallel_optimizer = ParallelCLOptimizer(optimizer)

    n_jobs = 4
    n_steps = 20
    Y_all = []

    for i in range(n_steps):
        x = parallel_optimizer.ask(min(n_jobs, n_steps))
        y = pool.map(obj, x)
        parallel_optimizer.tell(x, y)

        Y_all.append(min(y))
        print min(Y_all)
