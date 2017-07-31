from types import ModuleType
from joblib import Parallel, delayed
from itertools import groupby
import numpy as np


def single_rep_evaluate(solver, problem, n_calls=64, seed=0):
    np.random.seed(seed)
    fnc = problem()
    res = solver(fnc, fnc.space, n_calls=n_calls)

    # below is used to reduce the size of the result
    res.models = None
    res.space = None
    res.random_state = None
    res.specs = None

    return res


def parallel_evaluate(solvers, task_subset=None, n_reps=32,
                      joblib_kwargs=None, eval_kwargs=None):

    if joblib_kwargs is None:
        joblib_kwargs = {}
    if eval_kwargs is None:
        eval_kwargs = {}

    if task_subset is None:
        from tracks import all_tracks
        task_subset = all_tracks

    if isinstance(solvers, list):
        list_of_solvers = solvers
    else:
        list_of_solvers = [solvers]

    list_of_tasks = []

    if isinstance(task_subset, list):
        for task in task_subset:
            if isinstance(task, ModuleType):
                # task is a track
                list_of_tasks.extend(task.problems)
            else:
                # task is a particular problem
                list_of_tasks.append(task)
    elif isinstance(task_subset, ModuleType):
        list_of_tasks.extend(task_subset.problems)
    else:
        list_of_tasks = [task_subset]

    all_tasks = [(p, s)
        for s in list_of_solvers
        for p in list_of_tasks
        for _ in range(n_reps)]

    results = Parallel(**joblib_kwargs)(
        delayed(single_rep_evaluate)(s, p, seed=i, **eval_kwargs)
        for i, (p, s) in enumerate(all_tasks)
    )

    pr = zip(all_tasks, results)

    results = {k: list(v) for k, v in groupby(pr, lambda x: x[0][1].__name__)}

    for solver, solver_res in results.items():
        results[solver] = {
            k: list(v)
            for k, v in groupby(list(solver_res), lambda x: x[0][0].__module__)
        }

    for solver, solver_res in results.items():
        for track, track_res in solver_res.items():
            solver_res[track] = {
                k: [e[1] for e in v]
                for k, v in groupby(list(track_res), lambda x: x[0][0].__name__)
            }

    return results


def plot_results(results):
    from skopt.plots import plot_convergence
    import matplotlib.pyplot as plt

    egsolver = list(results.keys())[0]

    for t in results[egsolver]:
        for p in results[egsolver][t]:
            ax = plot_convergence(*((s, results[s][t][p]) for s in results.keys()))
            ax.set_title(p)
            plt.show()


def calculate_metrics(results):
    import bootstrapped.bootstrap as bs
    import bootstrapped.stats_functions as bs_stats

    stat_dict = {}

    for s in results:
        for t in results[s]:
            for p in results[s][t]:
                if not p in stat_dict:
                    stat_dict[p] = {}
                opts = np.array([result.fun for result in results[s][t][p]])
                stats = bs.bootstrap(opts, stat_func=bs_stats.mean, num_iterations=1000000)
                l, m, u = stats.lower_bound, stats.value, stats.upper_bound
                stat_dict[p][s] = "%s<%s<%s" % tuple(round(v, 3) for v in (l, m, u))

    from pandas import DataFrame
    # https://stackoverflow.com/questions/19258772/write-2d-dictionary-into-a-dataframe-or-tab-delimited-file-using-python
    df = DataFrame(stat_dict, index=list(results.keys()))
    return df
