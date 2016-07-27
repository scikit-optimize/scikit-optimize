"""Plotting functions."""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import MaxNLocator

from scipy.interpolate import griddata
from scipy.optimize import OptimizeResult


def plot_convergence(*args, **kwargs):
    """Plot one or several convergence traces.

    Parameters
    ----------
    * `args[i]` [`OptimizeResult`, list of `OptimizeResult`, or tuple]:
        The result(s) for which to plot the convergence trace.

        - if `OptimizeResult`, then draw the corresponding single trace;
        - if list of `OptimizeResult`, then draw the corresponding convergence
          traces in transparency, along with the average convergence trace;
        - if tuple, then `args[i][0]` should be a string label and `args[i][1]`
          an `OptimizeResult` or a list of `OptimizeResult`.

    * `ax` [`Axes`, optional]:
        The matplotlib axes on which to draw the plot, or `None` to create
        a new one.

    * `true_minimum` [float, optional]:
        The true minimum value of the function, if known.

    * `yscale` [None or string, optional]:
        The scale for the y-axis.

    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    # <3 legacy python
    ax = kwargs.get("ax", None)
    true_minimum = kwargs.get("true_minimum", None)
    yscale = kwargs.get("yscale", None)

    if ax is None:
        ax = plt.gca()

    ax.set_title("Convergence plot")
    ax.set_xlabel("Number of calls $n$")
    ax.set_ylabel(r"$\min f(x)$ after $n$ calls")
    ax.grid()

    if yscale is not None:
        ax.set_yscale(yscale)

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
            ax.plot(range(n_calls), mins, c=color,
                    marker=".", markersize=12, lw=2, label=name)

        elif isinstance(results, list):
            n_calls = len(results[0].x_iters)
            mins = [[np.min(r.func_vals[:i])
                     for i in range(1, n_calls + 1)] for r in results]

            for m in mins:
                ax.plot(range(n_calls), m, c=color, alpha=0.2)

            ax.plot(range(n_calls), np.mean(mins, axis=0), c=color,
                    marker=".", markersize=12, lw=2, label=name)

    if true_minimum:
        ax.axhline(true_minimum, linestyle="--",
                   color="r", lw=1,
                   label="True minimum")

    if true_minimum or name:
        ax.legend(loc="best")

    return ax


def _format_scatter_plot_axes(ax, space):
    # Deal with formatting of the axes
    for i in range(space.n_dims):
        for j in range(space.n_dims):
            ax_ = ax[i, j]

            if j > i:
                ax_.axis("off")

            # adjust bounds for every off-diagonal axis
            if i != j:
                ax[i, j].set_ylim(*space.dimensions[i].bounds)
                ax[i, j].set_xlim(*space.dimensions[j].bounds)

            ax_.xaxis.set_major_locator(MaxNLocator(6, prune='both'))
            ax_.yaxis.set_major_locator(MaxNLocator(6, prune='both'))

            if i < space.n_dims - 1:
                ax_.set_xticklabels([])
            # bottom row
            else:
                [l.set_rotation(45) for l in ax_.get_xticklabels()]

            if j > 0:
                ax_.set_yticklabels([])

    return ax


def plot_objective_function(result, levels=10):
    """Pairwise scatter plot of objective function

    Pairwise scatter plots are shown on the off-diagonal for each
    dimension of the search space. A red point indicates the minimum.

    Note: the objective function contours are obtained by interpolating
    between samples. The surrogate model is not used.

    Note: search spaces that contain `Categorical` dimensions are
    currently not supported by this function.

    Parameters
    ----------
    * `result` [`OptimizeResult`]
        The result for which to create the scatter plot matrix.

    * `levels` [int, default=10]
        Number of levels to draw on the contour plot, passed directly
        to `plt.contour()`

    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    space = result.space
    samples = np.asarray(result.x_iters)
    order = range(samples.shape[0])
    rvs = space.rvs(n_samples=10)

    fig, ax = plt.subplots(space.n_dims, space.n_dims, figsize=(8, 8))

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.1, wspace=0.1)

    for i in range(result.space.n_dims):
        for j in range(result.space.n_dims):
            if i == j:
                bounds = space.dimensions[i].bounds
                xi = np.linspace(bounds[0], bounds[1], 40)
                values = []

                for x in xi:
                    rvs_ = np.array(rvs)
                    rvs_[:, i] = x
                    values.append(np.mean(result.models[-1].predict(rvs_)))

                ax[i, i].plot(xi, values)
                ax[i, i].axvline(result.x[i], linestyle="--", color="r", lw=1)

            # lower triangle
            elif i > j:
                # define grid
                # XXX use linspace(*args, 100) after python2 support ends
                bounds = space.dimensions[j].bounds
                xi = np.linspace(bounds[0], bounds[1], 40)
                bounds = space.dimensions[i].bounds
                yi = np.linspace(bounds[0], bounds[1], 40)

                zi = []
                for x_ in xi:
                    row = []
                    for y_ in yi:
                        rvs_ = np.array(rvs)
                        rvs_[:, (i, j)] = (x_, y_)
                        row.append(np.mean(result.models[-1].predict(rvs_)))
                    zi.append(row)

                ax[i, j].contour(xi, yi, zi, levels, linewidths=0.5, colors='k')
                ax[i, j].contourf(xi, yi, zi, levels, cmap='viridis_r')
                ax[i, j].scatter(samples[:, j], samples[:, i], c='k', s=10, lw=0.)
                ax[i, j].scatter(result.x[j], result.x[i], c=['r'], s=20, lw=0.)

    return _format_scatter_plot_axes(ax, space)


def plot_sampling_order(result, bins=20):
    """Visualize order in which points where sampled

    Pairwise scatter plots are shown on the off-diagonal for each
    dimension of the search space. The order in which samples were
    evaluated is as the colour of each point. The diagonal shows a
    histogram of sampled values for each dimension. A red point
    indicates the minimum.

    Note: search spaces that contain `Categorical` dimensions are
    currently not supported by this function.

    Parameters
    ----------
    * `result` [`OptimizeResult`]
        The result for which to create the scatter plot matrix.

    * `bins` [int, bins=20]:
        Number of bins to use for histograms on the diagonal.

    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    space = result.space
    samples = np.asarray(result.x_iters)
    order = range(samples.shape[0])
    fig, ax = plt.subplots(space.n_dims, space.n_dims, figsize=(8, 8))

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.1, wspace=0.1)

    for i in range(result.space.n_dims):
        for j in range(result.space.n_dims):
            if i == j:
                ax[i, i].hist(samples[:, j], bins=bins)

            # lower triangle
            elif i > j:
                ax[i, j].scatter(samples[:, j], samples[:, i], c=order,
                                 s=40, lw=0., cmap='viridis')
                ax[i, j].scatter(result.x[j], result.x[i],
                                 c=['r'], s=20, lw=0.)

    return _format_scatter_plot_axes(ax, space)
