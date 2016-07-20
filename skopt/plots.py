"""Plotting functions."""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from scipy.optimize import OptimizeResult


def plot_convergence(*args, true_minimum=None, yscale=None):
    """Plot one or several convergence traces.

    Parameters
    ----------
    * `args[i]` [`OptimizeResult`, list of `OptimizeResult`, or tuple]:
        The result(s) for which to plot the convergence trace.

        - if `OptimizeResult`, then draw the corresponding single trace;
        - if list of `OptimizeResult`, then draw the corresponding convergence
          traces in light, along with the average convergence trace;
        - if tuple, then `args[i][0]` should be a string label and `args[i][1]`
          an `OptimizeResult` or a list of `OptimizeResult`.
    """
    colors = cm.viridis(np.linspace(0.25, 1.0, len(args)))

    for results, color in zip(args, colors):
        if isinstance(results, tuple):
            name, results = results
        else:
            name = None

        if isinstance(results, OptimizeResult):
            n_calls = len(results.x_iters)
            mins = [np.min(results.func_vals[:i])
                    for i in range(1, n_calls + 1)]
            plt.plot(range(n_calls), mins, c=color,
                     marker=".", markersize=12, lw=2, label=name)

        elif isinstance(results, list):
            n_calls = len(results[0].x_iters)
            mins = [[np.min(r.func_vals[:i])
                     for i in range(1, n_calls + 1)] for r in results]

            for m in mins:
                plt.plot(range(n_calls), m, c=color, alpha=0.2)

            plt.plot(range(n_calls), np.mean(mins, axis=0), c=color,
                     marker=".", markersize=12, lw=2, label=name)

    if true_minimum:
        plt.axhline(true_minimum, linestyle="--",
                    color="r", lw=1,
                    label="True minimum")

    plt.title("Convergence plot")
    plt.xlabel("Number of calls $n$")
    plt.ylabel(r"$\min f(x)$ after $n$ calls")
    plt.grid()

    if yscale is not None:
        plt.yscale(yscale)

    if true_minimum or name:
        plt.legend(loc="best")

    plt.show()
