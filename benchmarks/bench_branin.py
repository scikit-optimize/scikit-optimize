import numpy as np
import argparse

from skopt.benchmarks import Branin
from skopt import gp_minimize
from skopt import forest_minimize
from skopt import gbrt_minimize
from skopt import dummy_minimize
from skopt.callbacks import EarlyStopper

class DeltaStopper(EarlyStopper):
    def __init__(self, minimum_pos, eps):
        super(EarlyStopper, self).__init__()
        self.minimum_pos = minimum_pos
        self.eps = eps

    def _criterion(self, result):
        return np.any(np.all(np.abs(self.minimum_pos - result.x) < self.eps,
                             axis=1))


class VerboseCallback(object):

    def __init__(self, n_calls, n_run):
        self.iter_no = 1
        self.n_calls = n_calls
        self.n_run = n_run

    def __call__(self, res):
        curr_y = res.func_vals[-1]
        curr_min = res.fun
        if curr_min == curr_y:
            print("run %d - call %d/%d - obtained new min: %.4f " %
                  (self.n_run, self.iter_no, self.n_calls, curr_y))
            print("Parameters: %s" % str(res.x_iters[-1]))

        self.iter_no += 1


def run(n_calls=1000, n_runs=10, acq_optimizer="lbfgs"):
    branin = Branin(scaled=False, noise_level=0.02)
    bounds = branin.dimensions
    minimum_pos = branin.minimum_pos
    func_min = np.min(branin.minimum)
    eps = np.ones(len(bounds))
    for i in range(len(bounds)):
        eps[i] = np.abs(bounds[i][1] - bounds[i][0]) / 1e3
    stopper = DeltaStopper(minimum_pos, eps)

    optimizers = [("dummy_minimize", dummy_minimize),
                  ("gp_minimize", gp_minimize),
                  ("forest_minimize", forest_minimize),
                  ("gbrt_minimize", gbrt_minimize)]

    for name, optimizer in optimizers:
        print(name)
        results = []
        min_func_calls = []
        min_xiters_calls = []
        time_ = 0.0

        for random_state in range(n_runs):
            verbose = VerboseCallback(n_calls, random_state)
            if name == "gp_minimize":
                res = optimizer(
                    branin, bounds, random_state=random_state, n_calls=n_calls,
                    noise=1e-10, acq_optimizer=acq_optimizer, callback=[stopper, verbose],
                    n_jobs=-1)
            elif name == "dummy_minimize" or name == "forest_minimize":
                res = optimizer(
                    branin, bounds, random_state=random_state, n_calls=n_calls,
                    callback=[stopper, verbose])
            else:
                res = optimizer(
                    branin, bounds, random_state=random_state, n_calls=n_calls,
                    callback=[stopper],
                    acq_optimizer=acq_optimizer)
            results.append(res)
            func_vals = np.round(res.func_vals, 3)
            best_x_pos = np.zeros(len(minimum_pos))
            for i in range(len(minimum_pos)):
                best_x_pos[i] = np.argmin(np.sum(np.abs(minimum_pos[i] - res.x_iters), axis=1))
            if np.any(np.all(np.abs(minimum_pos - res.x) < eps, axis=1)):
                min_func_calls.append(np.argmin(func_vals) + 1)
                min_xiters_calls.append(np.min(best_x_pos) + 1)
            else:
                min_func_calls.append(len(func_vals))
                min_xiters_calls.append(len(func_vals))

        optimal_values = [result.fun for result in results]
        mean_optimum = np.mean(optimal_values)
        std = np.std(optimal_values)
        best = np.min(optimal_values)
        print("Mean optimum: %.4f (difference to true optimum %.4f)" % (mean_optimum, mean_optimum - func_min))
        print("Std of optimal values: %.4f" % (std))
        print("Best optima: %.4f, Difference to true optima: %.4f" %(best, best - func_min))

        mean_fcalls = np.mean(min_func_calls)
        std_fcalls = np.std(min_func_calls)
        best_fcalls = np.min(min_func_calls)
        print("Mean func_calls to reach min: " + str(mean_fcalls))
        print("Std func_calls to reach min: " + str(std_fcalls))
        print("Fastest no of func_calls to reach min: " + str(best_fcalls))

        mean_xcalls = np.mean(min_xiters_calls)
        std_xcalls = np.std(min_xiters_calls)
        best_xcalls = np.min(min_xiters_calls)
        print("Mean func_calls to reach min pos: " + str(mean_xcalls))
        print("Std func_calls to reach min pos: " + str(std_xcalls))
        print("Fastest no of func_calls to reach min pos: " + str(best_xcalls))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_calls', nargs="?", default=200, type=int, help="Number of function calls.")
    parser.add_argument(
        '--n_runs', nargs="?", default=5, type=int, help="Number of runs.")
    parser.add_argument(
        '--acq_optimizer', nargs="?", default="lbfgs", type=str,
        help="Acquistion optimizer.")
    args = parser.parse_args()
    run(args.n_calls, args.n_runs, args.acq_optimizer)
