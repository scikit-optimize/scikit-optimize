import numpy as np

from skopt.benchmarks import hart6
from skopt import gp_minimize, forest_minimize, gbrt_minimize

def run(n_calls=50, n_runs=5, acq_optimizer="lbfgs"):
    bounds = np.tile((0., 1.), (6, 1))
    optimizers = [("gp_minimize", gp_minimize),]
                #   ("forest_minimize", forest_minimize),
                #   ("gbrt_minimize", gbrt_minimize)]

    for name, optimizer in optimizers:
        print(name)
        results = []
        min_func_calls = []

        for random_state in range(n_runs):
            print(random_state)
            if name == "gp_minimize":
                res = optimizer(
                    hart6, bounds, random_state=random_state, n_calls=n_calls,
                    noise=1e-10, n_jobs=-1, acq_optimizer=acq_optimizer)
            else:
                res = optimizer(
                    hart6, bounds, random_state=random_state, n_calls=n_calls)
            results.append(res)
            min_func_calls.append(np.argmin(res.func_vals) + 1)

        optimal_values = [result.fun for result in results]
        mean_optimum = np.mean(optimal_values)
        std = np.std(optimal_values)
        best = np.min(optimal_values)
        print("Mean optimum: " + str(mean_optimum))
        print("Std of optimal values" + str(std))
        print("Best optima:" + str(best))

        mean_fcalls = np.mean(min_func_calls)
        std_fcalls = np.std(min_func_calls)
        best_fcalls = np.min(min_func_calls)
        print("Mean func_calls to reach min: " + str(mean_fcalls))
        print("Std func_calls to reach min: " + str(std_fcalls))
        print("Fastest no of func_calls to reach min: " + str(best_fcalls))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_calls', nargs="?", default=50, type=int, help="Number of function calls.")
    parser.add_argument(
        '--n_runs', nargs="?", default=5, type=int, help="Number of runs.")
    parser.add_argument(
        '--acq_optimizer', nargs="?", default="lbfgs", type=str,
        help="Acquistion optimizer.")
    args = parser.parse_args()
    run(args.n_calls, args.n_runs, args.acq_optimizer)
