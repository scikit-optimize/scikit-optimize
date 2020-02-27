"""Plotting functions."""
import sys
import numpy as np
from itertools import count
from functools import partial
from scipy.optimize import OptimizeResult

from skopt import expected_minimum, expected_minimum_random_sampling
from .space import Categorical
from collections import Counter

# For plot tests, matplotlib must be set to headless mode early
if 'pytest' in sys.modules:
    import matplotlib

    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import LogLocator
from matplotlib.ticker import MaxNLocator, FuncFormatter  # noqa: E402


def plot_convergence(*args, **kwargs):
    """Plot one or several convergence traces.

    Parameters
    ----------
    args[i] :  `OptimizeResult`, list of `OptimizeResult`, or tuple
        The result(s) for which to plot the convergence trace.

        - if `OptimizeResult`, then draw the corresponding single trace;
        - if list of `OptimizeResult`, then draw the corresponding convergence
          traces in transparency, along with the average convergence trace;
        - if tuple, then `args[i][0]` should be a string label and `args[i][1]`
          an `OptimizeResult` or a list of `OptimizeResult`.

    ax : `Axes`, optional
        The matplotlib axes on which to draw the plot, or `None` to create
        a new one.

    true_minimum : float, optional
        The true minimum value of the function, if known.

    yscale : None or string, optional
        The scale for the y-axis.

    Returns
    -------
    ax : `Axes`
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
            ax.plot(range(1, n_calls + 1), mins, c=color,
                    marker=".", markersize=12, lw=2, label=name)

        elif isinstance(results, list):
            n_calls = len(results[0].x_iters)
            iterations = range(1, n_calls + 1)
            mins = [[np.min(r.func_vals[:i]) for i in iterations]
                    for r in results]

            for m in mins:
                ax.plot(iterations, m, c=color, alpha=0.2)

            ax.plot(iterations, np.mean(mins, axis=0), c=color,
                    marker=".", markersize=12, lw=2, label=name)

    if true_minimum:
        ax.axhline(true_minimum, linestyle="--",
                   color="r", lw=1,
                   label="True minimum")

    if true_minimum or name:
        ax.legend(loc="best")

    return ax


def plot_regret(*args, **kwargs):
    """Plot one or several cumulative regret traces.

    Parameters
    ----------
    args[i] : `OptimizeResult`, list of `OptimizeResult`, or tuple
        The result(s) for which to plot the cumulative regret trace.

        - if `OptimizeResult`, then draw the corresponding single trace;
        - if list of `OptimizeResult`, then draw the corresponding cumulative
            regret traces in transparency, along with the average cumulative
            regret trace;
        - if tuple, then `args[i][0]` should be a string label and `args[i][1]`
          an `OptimizeResult` or a list of `OptimizeResult`.

    ax : Axes`, optional
        The matplotlib axes on which to draw the plot, or `None` to create
        a new one.

    true_minimum : float, optional
        The true minimum value of the function, if known.

    yscale : None or string, optional
        The scale for the y-axis.

    Returns
    -------
    ax : `Axes`
        The matplotlib axes.
    """
    # <3 legacy python
    ax = kwargs.get("ax", None)
    true_minimum = kwargs.get("true_minimum", None)
    yscale = kwargs.get("yscale", None)

    if ax is None:
        ax = plt.gca()

    ax.set_title("Cumulative regret plot")
    ax.set_xlabel("Number of calls $n$")
    ax.set_ylabel(r"$\sum_{i=0}^n(f(x_i) - optimum)$ after $n$ calls")
    ax.grid()

    if yscale is not None:
        ax.set_yscale(yscale)

    colors = cm.viridis(np.linspace(0.25, 1.0, len(args)))

    if true_minimum is None:
        results = []
        for res in args:
            if isinstance(res, tuple):
                res = res[1]

            if isinstance(res, OptimizeResult):
                results.append(res)
            elif isinstance(res, list):
                results.extend(res)
        true_minimum = np.min([np.min(r.func_vals) for r in results])

    for results, color in zip(args, colors):
        if isinstance(results, tuple):
            name, results = results
        else:
            name = None

        if isinstance(results, OptimizeResult):
            n_calls = len(results.x_iters)
            regrets = [np.sum(results.func_vals[:i] - true_minimum)
                       for i in range(1, n_calls + 1)]
            ax.plot(range(1, n_calls + 1), regrets, c=color,
                    marker=".", markersize=12, lw=2, label=name)

        elif isinstance(results, list):
            n_calls = len(results[0].x_iters)
            iterations = range(1, n_calls + 1)
            regrets = [[np.sum(r.func_vals[:i] - true_minimum) for i in
                        iterations] for r in results]

            for cr in regrets:
                ax.plot(iterations, cr, c=color, alpha=0.2)

            ax.plot(iterations, np.mean(regrets, axis=0), c=color,
                    marker=".", markersize=12, lw=2, label=name)

    if name:
        ax.legend(loc="best")

    return ax


def _format_scatter_plot_axes(ax, space, ylabel, plot_dims,
                              dimensions=None):
    # Work out min, max of y axis for the diagonal so we can adjust
    # them all to the same value
    diagonal_ylim = _get_ylim_diagonal(ax)
    diagonal_ylim = tuple(diagonal_ylim)

    # Number of search-space dimensions we are using.
    n_dims = len(plot_dims)

    if dimensions is None:
        dimensions = ["$X_{%i}$" % i if d.name is None else d.name
                      for i, d in plot_dims]
    # Axes for categorical dimensions are really integers; we have to
    # label them with the category names
    iscat = [isinstance(dim[1], Categorical) for dim in plot_dims]

    # Deal with formatting of the axes
    for i in range(n_dims):  # rows
        for j in range(n_dims):  # columns
            ax_ = ax[i, j]
            index_i, dim_i = plot_dims[i]
            index_j, dim_j = plot_dims[j]
            if j > i:
                ax_.axis("off")
            elif i > j:  # off-diagonal plots
                # plots on the diagonal are special, like Texas. They have
                # their own range so do not mess with them.
                if not iscat[i]:  # bounds not meaningful for categoricals
                    ax_.set_ylim(*dim_i.bounds)
                if iscat[j]:
                    # partial() avoids creating closures in a loop
                    ax_.xaxis.set_major_formatter(FuncFormatter(
                        partial(_cat_format, dim_j)))
                else:
                    ax_.set_xlim(*dim_j.bounds)
                if j == 0:  # only leftmost column (0) gets y labels
                    ax_.set_ylabel(dimensions[i])
                    if iscat[i]:  # Set category labels for left column
                        ax_.yaxis.set_major_formatter(FuncFormatter(
                            partial(_cat_format, dim_i)))
                else:
                    ax_.set_yticklabels([])

                # for all rows except ...
                if i < n_dims - 1:
                    ax_.set_xticklabels([])
                # ... the bottom row
                else:
                    [l.set_rotation(45) for l in ax_.get_xticklabels()]
                    ax_.set_xlabel(dimensions[j])

                # configure plot for linear vs log-scale
                if dim_j.prior == 'log-uniform':
                    ax_.set_xscale('log')
                else:
                    ax_.xaxis.set_major_locator(MaxNLocator(6, prune='both',
                                                            integer=iscat[j]))

                if dim_i.prior == 'log-uniform':
                    ax_.set_yscale('log')
                else:
                    ax_.yaxis.set_major_locator(MaxNLocator(6, prune='both',
                                                            integer=iscat[i]))

            else:  # diagonal plots
                ax_.set_ylim(*diagonal_ylim)
                ax_.yaxis.tick_right()
                ax_.yaxis.set_label_position('right')
                ax_.yaxis.set_ticks_position('both')
                ax_.set_ylabel(ylabel)

                ax_.xaxis.tick_top()
                ax_.xaxis.set_label_position('top')
                ax_.set_xlabel(dimensions[j])

                if dim_i.prior == 'log-uniform':
                    ax_.set_xscale('log')
                else:
                    ax_.xaxis.set_major_locator(MaxNLocator(6, prune='both',
                                                            integer=iscat[i]))
                    if iscat[i]:
                        ax_.xaxis.set_major_formatter(FuncFormatter(
                            partial(_cat_format, dim_i)))

    return ax


def partial_dependence(space, model, i, j=None, sample_points=None,
                       n_samples=250, n_points=40, x_eval=None):
    """Calculate the partial dependence for dimensions `i` and `j` with
    respect to the objective value, as approximated by `model`.

    The partial dependence plot shows how the value of the dimensions
    `i` and `j` influence the `model` predictions after "averaging out"
    the influence of all other dimensions.

    When `x_eval` is not `None`, the given values are used instead of
    random samples. In this case, `n_samples` will be ignored.

    Parameters
    ----------
    space : `Space`
        The parameter space over which the minimization was performed.

    model
        Surrogate model for the objective function.

    i : int
        The first dimension for which to calculate the partial dependence.

    j : int, default=None
        The second dimension for which to calculate the partial dependence.
        To calculate the 1D partial dependence on `i` alone set `j=None`.

    sample_points : np.array, shape=(n_points, n_dims), default=None
        Only used when `x_eval=None`, i.e in case partial dependence should
        be calculated.
        Randomly sampled and transformed points to use when averaging
        the model function at each of the `n_points` when using partial
        dependence.

    n_samples : int, default=100
        Number of random samples to use for averaging the model function
        at each of the `n_points` when using partial dependence. Only used
        when `sample_points=None` and `x_eval=None`.

    n_points : int, default=40
        Number of points at which to evaluate the partial dependence
        along each dimension `i` and `j`.

    x_eval : list, default=None
        `x_eval` is a list of parameter values or None. In case `x_eval`
        is not None, the parsed dependence will be calculated using these
        values.
        Otherwise, random selected samples will be used.

    Returns
    -------
    For 1D partial dependence:

    xi : np.array
        The points at which the partial dependence was evaluated.

    yi : np.array
        The value of the model at each point `xi`.

    For 2D partial dependence:

    xi : np.array, shape=n_points
        The points at which the partial dependence was evaluated.
    yi : np.array, shape=n_points
        The points at which the partial dependence was evaluated.
    zi : np.array, shape=(n_points, n_points)
        The value of the model at each point `(xi, yi)`.

    For Categorical variables, the `xi` (and `yi` for 2D) returned are
    the indices of the variable in `Dimension.categories`.
    """
    # If we haven't parsed an x_eval list we use random sampled values instead
    if x_eval is None and sample_points is None:
        sample_points = space.transform(space.rvs(n_samples=n_samples))
    elif sample_points is None:
        sample_points = space.transform([x_eval])

    if j is None:
        return partial_dependence_1D(space, model, i,
                                     sample_points, n_points)
    else:
        return partial_dependence_2D(space, model, i, j,
                                     sample_points, n_points)


def plot_objective(result, levels=10, n_points=40, n_samples=250, size=2,
                   zscale='linear', dimensions=None, sample_source='random',
                   minimum='result', n_minimum_search=None, plot_dims=None):
    """Plot a 2-d matrix with so-called Partial Dependence plots
    of the objective function. This shows the influence of each
    search-space dimension on the objective function.

    This uses the last fitted model for estimating the objective function.

    The diagonal shows the effect of a single dimension on the
    objective function, while the plots below the diagonal show
    the effect on the objective function when varying two dimensions.

    The Partial Dependence is calculated by averaging the objective value
    for a number of random samples in the search-space,
    while keeping one or two dimensions fixed at regular intervals. This
    averages out the effect of varying the other dimensions and shows
    the influence of one or two dimensions on the objective function.

    Also shown are small black dots for the points that were sampled
    during optimization.

    A red star indicates per default the best observed minimum, but
    this can be changed by changing argument ´minimum´.

    .. note::
          The Partial Dependence plot is only an estimation of the surrogate
          model which in turn is only an estimation of the true objective
          function that has been optimized. This means the plots show
          an "estimate of an estimate" and may therefore be quite imprecise,
          especially if few samples have been collected during the
          optimization
          (e.g. less than 100-200 samples), and in regions of the search-space
          that have been sparsely sampled (e.g. regions away from the optimum).
          This means that the plots may change each time you run the
          optimization and they should not be considered completely reliable.
          These compromises are necessary because we cannot evaluate the
          expensive objective function in order to plot it, so we have to use
          the cheaper surrogate model to plot its contour. And in order to
          show search-spaces with 3 dimensions or more in a 2-dimensional
          plot,
          we further need to map those dimensions to only 2-dimensions using
          the Partial Dependence, which also causes distortions in the plots.

    Parameters
    ----------
    result : `OptimizeResult`
        The optimization results from calling e.g. `gp_minimize()`.

    levels : int, default=10
        Number of levels to draw on the contour plot, passed directly
        to `plt.contour()`.

    n_points : int, default=40
        Number of points at which to evaluate the partial dependence
        along each dimension.

    n_samples : int, default=250
        Number of samples to use for averaging the model function
        at each of the `n_points` when `sample_method` is set to 'random'.

    size : float, default=2
        Height (in inches) of each facet.

    zscale : str, default='linear'
        Scale to use for the z axis of the contour plots. Either 'linear'
        or 'log'.

    dimensions : list of str, default=None
        Labels of the dimension
        variables. `None` defaults to `space.dimensions[i].name`, or
        if also `None` to `['X_0', 'X_1', ..]`.

    plot_dims : list(str), list(int), default=None
        List of names or indices for search-space dimensions to be
        used in the plot.
        If `None` then use all dimensions from the search-space.

    sample_source : str or list of floats, default='random'
        Defines to samples generation to use for averaging the model function
        at each of the `n_points`.

        A partial dependence plot is only generated, when `sample_source`
        is set to 'random' and `n_samples` is sufficient.

        `sample_source` can also be a list of
        floats, which is then used for averaging.

        Valid strings:

            - 'random' - `n_samples` random samples will used

            - 'result' - Use only the best observed parameters

            - 'expected_minimum' - Parameters that gives the best
                  minimum Calculated using scipy's minimize method.
                  This method currently does not work with categorical values.

            - 'expected_minimum_random' - Parameters that gives the
                  best minimum when using naive random sampling.
                  Works with categorical values.

    minimum : str or list of floats, default = 'result'
        Defines the values for the red points in the plots.
        Valid strings:

            - 'result' - Use best observed parameters

            - 'expected_minimum' - Parameters that gives the best
                  minimum Calculated using scipy's minimize method.
                  This method currently does not work with categorical values.

            - 'expected_minimum_random' - Parameters that gives the
                  best minimum when using naive random sampling.
                  Works with categorical values

    n_minimum_search : int, default = None
        Determines how many points should be evaluated
        to find the minimum when using 'expected_minimum' or
        'expected_minimum_random'. Parameter is used when
        `sample_source` and/or `minimum` is set to
        'expected_minimum' or 'expected_minimum_random'.

    Returns
    -------
    ax : `Matplotlib.Axes`
        A 2-d matrix of Axes-objects with the sub-plots.
    """
    # Here we define the values for which to plot the red dot (2d plot) and
    # the red dotted line (1d plot).
    # These same values will be used for evaluating the plots when
    # calculating dependence. (Unless partial
    # dependence is to be used instead).
    space = result.space
    if space.n_dims == 1:
        raise ValueError("plot_objective needs at least two"
                         "variables. Found only one.")
    # Get the relevant search-space dimensions.
    if plot_dims is None:
        # Get all dimensions.
        plot_dims = []
        for row in range(space.n_dims):
            plot_dims.append((row, space.dimensions[row]))
    else:
        plot_dims = space[plot_dims]
    # Number of search-space dimensions we are using.
    n_dims = len(plot_dims)
    if dimensions is not None:
        assert len(dimensions) == n_dims
    x_vals = _evaluate_min_params(result, minimum, n_minimum_search)
    if sample_source == "random":
        x_eval = None
        samples = space.transform(space.rvs(n_samples=n_samples))
    else:
        x_eval = _evaluate_min_params(result, sample_source,
                                      n_minimum_search)
        samples = space.transform([x_eval])
    x_samples, minimum, _ = _map_categories(space, result.x_iters, x_vals)

    if zscale == 'log':
        locator = LogLocator()
    elif zscale == 'linear':
        locator = None
    else:
        raise ValueError("Valid values for zscale are 'linear' and 'log',"
                         " not '%s'." % zscale)

    fig, ax = plt.subplots(n_dims, n_dims,
                           figsize=(size * n_dims, size * n_dims))

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.1, wspace=0.1)

    for i in range(n_dims):
        for j in range(n_dims):
            if i == j:
                index, dim = plot_dims[i]
                xi, yi = partial_dependence_1D(space, result.models[-1],
                                               index,
                                               samples=samples,
                                               n_points=n_points)

                ax[i, i].plot(xi, yi)
                ax[i, i].axvline(minimum[i], linestyle="--", color="r", lw=1)

            # lower triangle
            elif i > j:
                index1, dim1 = plot_dims[i]
                index2, dim2 = plot_dims[j]
                xi, yi, zi = partial_dependence_2D(space, result.models[-1],
                                                   index1, index2,
                                                   samples, n_points)
                ax[i, j].contourf(xi, yi, zi, levels,
                                  locator=locator, cmap='viridis_r')
                ax[i, j].scatter(x_samples[:, index2], x_samples[:, index1],
                                 c='k', s=10, lw=0.)
                ax[i, j].scatter(minimum[index2], minimum[index1],
                                 c=['r'], s=100, lw=0., marker='*')
    ylabel = "Partial dependence"

    # Make various adjustments to the plots.
    return _format_scatter_plot_axes(ax, space, ylabel=ylabel,
                                     plot_dims=plot_dims,
                                     dimensions=dimensions)


def plot_evaluations(result, bins=20, dimensions=None,
                     plot_dims=None):
    """Visualize the order in which points were sampled during optimization.

    This creates a 2-d matrix plot where the diagonal plots are histograms
    that show the distribution of samples for each search-space dimension.

    The plots below the diagonal are scatter-plots of the samples for
    all combinations of search-space dimensions.

    The order in which samples
    were evaluated is encoded in each point's color.

    A red star shows the best found parameters.

    Parameters
    ----------
    result : `OptimizeResult`
        The optimization results from calling e.g. `gp_minimize()`.

    bins : int, bins=20
        Number of bins to use for histograms on the diagonal.

    dimensions : list of str, default=None
        Labels of the dimension
        variables. `None` defaults to `space.dimensions[i].name`, or
        if also `None` to `['X_0', 'X_1', ..]`.

    plot_dims : list(str), list(int), default=None
        List of names or indices for search-space dimensions to be
        used in the plot.
        If `None` then use all dimensions from the search-space.

    Returns
    -------
    ax : `Matplotlib.Axes`
        A 2-d matrix of Axes-objects with the sub-plots.
    """
    space = result.space
    # Convert categoricals to integers, so we can ensure consistent ordering.
    # Assign indices to categories in the order they appear in the Dimension.
    # Matplotlib's categorical plotting functions are only present in v 2.1+,
    # and may order categoricals differently in different plots anyway.
    samples, minimum, iscat = _map_categories(space, result.x_iters, result.x)
    order = range(samples.shape[0])

    if plot_dims is None:
        # Get all dimensions.
        plot_dims = []
        for row in range(space.n_dims):
            plot_dims.append((row, space.dimensions[row]))
    else:
        plot_dims = space[plot_dims]
    # Number of search-space dimensions we are using.
    n_dims = len(plot_dims)
    if dimensions is not None:
        assert len(dimensions) == n_dims

    fig, ax = plt.subplots(n_dims, n_dims,
                           figsize=(2 * n_dims, 2 * n_dims))

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.1, wspace=0.1)

    for i in range(n_dims):
        for j in range(n_dims):
            if i == j:
                index, dim = plot_dims[i]
                if iscat[j]:
                    bins_ = len(dim.categories)
                elif dim.prior == 'log-uniform':
                    low, high = space.bounds[index]
                    bins_ = np.logspace(np.log10(low), np.log10(high), bins)
                else:
                    bins_ = bins
                ax[i, i].hist(
                    samples[:, index], bins=bins_,
                    range=None if iscat[j] else dim.bounds)

            # lower triangle
            elif i > j:
                index_i, dim_i = plot_dims[i]
                index_j, dim_j = plot_dims[j]
                ax[i, j].scatter(samples[:, index_j], samples[:, index_i],
                                 c=order, s=40, lw=0., cmap='viridis')
                ax[i, j].scatter(minimum[index_j], minimum[index_i],
                                 c=['r'], s=100, lw=0., marker='*')

    # Make various adjustments to the plots.
    return _format_scatter_plot_axes(ax, space, ylabel="Number of samples",
                                     plot_dims=plot_dims,
                                     dimensions=dimensions)


def _get_ylim_diagonal(ax):
    """Get the min / max of the ylim for all diagonal plots.
    This is used in _adjust_fig() so the ylim is the same
    for all diagonal plots.

    Parameters
    ----------
    ax : `Matplotlib.Axes`
        2-dimensional matrix with Matplotlib Axes objects.

    Returns
    -------
    ylim_diagonal : list(int)
        The common min and max ylim for the diagonal plots.
    """

    # Number of search-space dimensions used in this plot.
    n_dims = len(ax)

    # Get ylim for all diagonal plots.
    ylim = [ax[row, row].get_ylim() for row in range(n_dims)]

    # Separate into two lists with low and high ylim.
    ylim_lo, ylim_hi = zip(*ylim)

    # Min and max ylim for all diagonal plots.
    ylim_min = np.min(ylim_lo)
    ylim_max = np.max(ylim_hi)

    # The common ylim for the diagonal plots.
    ylim_diagonal = [ylim_min, ylim_max]

    return ylim_diagonal


def partial_dependence_1D(space, model, i, samples,
                          n_points=40):
    """
    Calculate the partial dependence for a single dimension.

    This uses the given model to calculate the average objective value
    for all the samples, where the given dimension is fixed at
    regular intervals between its bounds.

    This shows how the given dimension affects the objective value
    when the influence of all other dimensions are averaged out.

    Parameters
    ----------
    space : `Space`
        The parameter space over which the minimization was performed.

    model
        Surrogate model for the objective function.

    i : int
        The dimension for which to calculate the partial dependence.

    samples : np.array, shape=(n_points, n_dims)
        Randomly sampled and transformed points to use when averaging
        the model function at each of the `n_points` when using partial
        dependence.

    n_points : int, default=40
        Number of points at which to evaluate the partial dependence
        along each dimension `i`.

    Returns
    -------
    xi : np.array
        The points at which the partial dependence was evaluated.

    yi : np.array
        The average value of the modelled objective function at
        each point `xi`.
    """
    # The idea is to step through one dimension, evaluating the model with
    # that dimension fixed and averaging either over random values or over
    # the given ones in x_val in all other dimensions.
    # (Or step through 2 dimensions when i and j are given.)
    # Categorical dimensions make this interesting, because they are one-
    # hot-encoded, so there is a one-to-many mapping of input dimensions
    # to transformed (model) dimensions.

    # dim_locs[i] is the (column index of the) start of dim i in
    # sample_points.
    # This is usefull when we are using one hot encoding, i.e using
    # categorical values
    dim_locs = np.cumsum([0] + [d.transformed_size for d in space.dimensions])

    def _calc(x):
        """
        Helper-function to calculate the average predicted
        objective value for the given model, when setting
        the index'th dimension of the search-space to the value x,
        and then averaging over all samples.
        """
        rvs_ = np.array(samples)  # copy
        # We replace the values in the dimension that we want to keep
        # fixed
        rvs_[:, dim_locs[i]:dim_locs[i + 1]] = x
        # In case of `x_eval=None` rvs conists of random samples.
        # Calculating the mean of these samples is how partial dependence
        # is implemented.
        return np.mean(model.predict(rvs_))
    xi, xi_transformed = _evenly_sample(space.dimensions[i], n_points)
    # Calculate the partial dependence for all the points.
    yi = [_calc(x) for x in xi_transformed]

    return xi, yi


def partial_dependence_2D(space, model, i, j, samples,
                          n_points=40):
    """
    Calculate the partial dependence for two dimensions in the search-space.

    This uses the given model to calculate the average objective value
    for all the samples, where the given dimensions are fixed at
    regular intervals between their bounds.

    This shows how the given dimensions affect the objective value
    when the influence of all other dimensions are averaged out.

    Parameters
    ----------
    space : `Space`
        The parameter space over which the minimization was performed.

    model
        Surrogate model for the objective function.

    i : int
        The first dimension for which to calculate the partial dependence.

    j : int
        The second dimension for which to calculate the partial dependence.

    samples : np.array, shape=(n_points, n_dims)
        Randomly sampled and transformed points to use when averaging
        the model function at each of the `n_points` when using partial
        dependence.

    n_points : int, default=40
        Number of points at which to evaluate the partial dependence
        along each dimension `i` and `j`.

    Returns
    -------
    xi : np.array, shape=n_points
        The points at which the partial dependence was evaluated.

    yi : np.array, shape=n_points
        The points at which the partial dependence was evaluated.

    zi : np.array, shape=(n_points, n_points)
        The average value of the objective function at each point `(xi, yi)`.
    """
    # The idea is to step through one dimension, evaluating the model with
    # that dimension fixed and averaging either over random values or over
    # the given ones in x_val in all other dimensions.
    # (Or step through 2 dimensions when i and j are given.)
    # Categorical dimensions make this interesting, because they are one-
    # hot-encoded, so there is a one-to-many mapping of input dimensions
    # to transformed (model) dimensions.

    # dim_locs[i] is the (column index of the) start of dim i in
    # sample_points.
    # This is usefull when we are using one hot encoding, i.e using
    # categorical values
    dim_locs = np.cumsum([0] + [d.transformed_size for d in space.dimensions])

    def _calc(x, y):
        """
        Helper-function to calculate the average predicted
        objective value for the given model, when setting
        the index1'th dimension of the search-space to the value x
        and setting the index2'th dimension to the value y,
        and then averaging over all samples.
        """
        rvs_ = np.array(samples)  # copy
        rvs_[:, dim_locs[j]:dim_locs[j + 1]] = x
        rvs_[:, dim_locs[i]:dim_locs[i + 1]] = y
        return np.mean(model.predict(rvs_))

    xi, xi_transformed = _evenly_sample(space.dimensions[j], n_points)
    yi, yi_transformed = _evenly_sample(space.dimensions[i], n_points)
    # Calculate the partial dependence for all combinations of these points.
    zi = [[_calc(x, y) for x in xi_transformed] for y in yi_transformed]

    # Convert list-of-list to a numpy array.
    zi = np.array(zi)

    return xi, yi, zi


def plot_objective_2D(result, dimension_identifier1, dimension_identifier2,
                      n_points=40, n_samples=250, levels=10, zscale='linear',
                      sample_source='random',
                      minimum='result', n_minimum_search=None, ax=None):
    """
    Create and return a Matplotlib figure and axes with a landscape
    contour-plot of the last fitted model of the search-space,
    overlaid with all the samples from the optimization results,
    for the two given dimensions of the search-space.

    This is similar to `plot_objective()` but only for 2 dimensions
    whose doc-string also has a more extensive explanation.

    Parameters
    ----------
    result : `OptimizeResult`
        The optimization results e.g. from calling `gp_minimize()`.

    dimension_identifier1 : str or int
        Name or index of a dimension in the search-space.

    dimension_identifier2 : str or int
        Name or index of a dimension in the search-space.

    n_samples : int, default=250
        Number of random samples used for estimating the contour-plot
        of the objective function.

    n_points : int, default=40
        Number of points along each dimension where the partial dependence
        is evaluated when generating the contour-plots.

    levels : int, default=10
        Number of levels to draw on the contour plot.

    zscale : str, default='linear'
        Scale to use for the z axis of the contour plots.
        Either 'log' or linear for all other choices.

    ax : `Matplotlib.Axes`, default: None
        When set, everything is plotted inside this axis.

    Returns
    -------
    ax : `Matplotlib.Axes`
        The Matplotlib Figure-object.
        For example, you can save the plot by calling
        `fig.savefig('file.png')`
    """

    # Get the search-space instance from the optimization results.
    space = result.space
    x_vals = _evaluate_min_params(result, minimum, n_minimum_search)
    if sample_source == "random":
        x_eval = None
        samples = space.transform(space.rvs(n_samples=n_samples))
    else:
        x_eval = _evaluate_min_params(result, sample_source,
                                      n_minimum_search)
        samples = space.transform([x_eval])
    x_samples, x_minimum, _ = _map_categories(space, result.x_iters, x_vals)
    # Get the dimension-object, its index in the search-space, and its name.
    index1, dimension1 = space[dimension_identifier1]
    index2, dimension2 = space[dimension_identifier2]

    # Get the samples from the optimization-log for the relevant dimensions.
    # samples1 = get_samples_dimension(result=result, index=index1)
    samples1 = x_samples[:, index1]
    samples2 = x_samples[:, index2]
    # samples2 = get_samples_dimension(result=result, index=index2)

    # Get the best-found samples for the relevant dimensions.
    best_sample1 = x_minimum[index1]
    best_sample2 = x_minimum[index2]

    # Get the last fitted model for the search-space.
    last_model = result.models[-1]

    # Estimate the objective function for these sampled points
    # using the last fitted model for the search-space.
    xi, yi, zi = partial_dependence_2D(space, last_model, index1, index2,
                                       samples, n_points=n_points)

    if ax is None:
        ax = plt.gca()

    # Scale for the z-axis of the contour-plot. Either Log or Linear (None).
    locator = LogLocator() if zscale == 'log' else None

    # Plot the contour-landscape for the objective function.
    ax.contourf(xi, yi, zi, levels, locator=locator, cmap='viridis_r')

    # Plot all the parameters that were sampled during optimization.
    # These are plotted as small black dots.
    ax.scatter(samples2, samples1, c='black', s=10, linewidths=1)

    # Plot the best parameters that were sampled during optimization.
    # These are plotted as a big red star.
    ax.scatter(best_sample2, best_sample1,
               c='red', s=50, linewidths=1, marker='*')

    # Use the dimension-names as the labels for the plot-axes.
    ax.set_xlabel(dimension2.name)
    ax.set_ylabel(dimension1.name)

    # Use log-scale on the x-axis?
    if dimension1.prior == 'log-uniform':
        ax.set_xscale('log')

    # Use log-scale on the y-axis?
    if dimension2.prior == 'log-uniform':
        ax.set_yscale('log')

    return ax


def plot_histogram(result, dimension_identifier, bins=20, rotate_labels=0, ax=None):
    """
    Create and return a Matplotlib figure with a histogram
    of the samples from the optimization results,
    for a given dimension of the search-space.

    Parameters
    ----------
    result : `OptimizeResult`
        The optimization results e.g. from calling `gp_minimize()`.

    dimension_identifier : str or int
        Name or index of a dimension in the search-space.

    bins : int, bins=20
        Number of bins in the histogram.

    rotate_labels : int, rotate_labels=0
        Degree to rotate category-names on the x-axis.
        Only used for Categorical dimensions.

    Returns
    -------
    ax : `Matplotlib.Axes`
        The Matplotlib Axes-object.
    """

    # Get the search-space instance from the optimization results.
    space = result.space

    # Get the dimension-object.
    index, dimension = space[dimension_identifier]

    # Get the samples from the optimization-log for that particular dimension.
    samples = [x[index] for x in result.x_iters]

    if ax is None:
        ax = plt.gca()

    if isinstance(dimension, Categorical):
        # When the search-space dimension is Categorical, it means
        # that the possible values are strings. Matplotlib's histogram
        # does not support this, so we have to make a bar-plot instead.

        # NOTE: This only shows the categories that are in the samples.
        # So if a category was not sampled, it will not be shown here.

        # Count the number of occurrences of the string-categories.
        counter = Counter(samples)

        # The counter returns a dict where the keys are the category-names
        # and the values are the number of occurrences for each category.
        names = list(counter.keys())
        counts = list(counter.values())

        # Although Matplotlib's docs indicate that the bar() function
        # can take a list of strings for the x-axis, it doesn't appear to work.
        # So we hack it by creating a list of integers and setting the
        # tick-labels with the category-names instead.
        x = np.arange(len(counts))

        # Plot using bars.
        ax.bar(x, counts, tick_label=names)

        # Adjust the rotation of the category-names on the x-axis.
        ax.set_xticklabels(labels=names, rotation=rotate_labels)
    else:
        # Otherwise the search-space Dimension is either integer or float,
        # in which case the histogram can be plotted more easily.
        if dimension.prior == 'log-uniform':
            # Map the number of bins to a log-space for the dimension bounds.
            bins_mapped = np.logspace(*np.log10(dimension.bounds), bins)
        else:
            # Use the original number of bins.
            bins_mapped = bins
        # Plot the histogram.
        ax.hist(samples, bins=bins_mapped, range=dimension.bounds)

        # Use log-scale on the x-axis?
        if dimension.prior == 'log-uniform':
            ax.set_xscale('log')

    # Set the labels.
    ax.set_xlabel(dimension.name)
    ax.set_ylabel('Sample Count')

    return ax


def _map_categories(space, points, minimum):
    """
    Map categorical values to integers in a set of points.

    Returns
    -------
    mapped_points : np.array, shape=points.shape
        A copy of `points` with categoricals replaced with their indices in
        the corresponding `Dimension`.

    mapped_minimum : np.array, shape (space.n_dims,)
        A copy of `minimum` with categoricals replaced with their indices in
        the corresponding `Dimension`.

    iscat : np.array, shape (space.n_dims,)
       Boolean array indicating whether dimension `i` in the `space` is
       categorical.
    """
    points = np.asarray(points, dtype=object)  # Allow slicing, preserve cats
    iscat = np.repeat(False, space.n_dims)
    min_ = np.zeros(space.n_dims)
    pts_ = np.zeros(points.shape)
    for i, dim in enumerate(space.dimensions):
        if isinstance(dim, Categorical):
            iscat[i] = True
            catmap = dict(zip(dim.categories, count()))
            pts_[:, i] = [catmap[cat] for cat in points[:, i]]
            min_[i] = catmap[minimum[i]]
        else:
            pts_[:, i] = points[:, i]
            min_[i] = minimum[i]
    return pts_, min_, iscat


def _evenly_sample(dim, n_points):
    """Return `n_points` evenly spaced points from a Dimension.

    Parameters
    ----------
    dim : `Dimension`
        The Dimension to sample from.  Can be categorical; evenly-spaced
        category indices are chosen in order without replacement (result
        may be smaller than `n_points`).

    n_points : int
        The number of points to sample from `dim`.

    Returns
    -------
    xi : np.array
        The sampled points in the Dimension.  For Categorical
        dimensions, returns the index of the value in
        `dim.categories`.

    xi_transformed : np.array
        The transformed values of `xi`, for feeding to a model.
    """
    cats = np.array(getattr(dim, 'categories', []), dtype=object)
    if len(cats):  # Sample categoricals while maintaining order
        xi = np.linspace(0, len(cats) - 1, min(len(cats), n_points),
                         dtype=int)
        xi_transformed = dim.transform(cats[xi])
    else:
        bounds = dim.bounds
        # XXX use linspace(*bounds, n_points) after python2 support ends
        xi = np.linspace(bounds[0], bounds[1], n_points)
        xi_transformed = dim.transform(xi)
    return xi, xi_transformed


def _cat_format(dimension, x, _):
    """Categorical axis tick formatter function.  Returns the name of category
    `x` in `dimension`.  Used with `matplotlib.ticker.FuncFormatter`."""
    return str(dimension.categories[int(x)])


def _evaluate_min_params(result, params='result',
                         n_minimum_search=None,
                         random_state=None):
    """Returns the minimum based on `params`"""
    x_vals = None
    space = result.space
    if isinstance(params, str):
        if params == 'result':
            # Using the best observed result
            x_vals = result.x
        elif params == 'expected_minimum':
            if result.space.is_partly_categorical:
                # space is also categorical
                raise ValueError('expected_minimum does not support any'
                                 'categorical values')
            # Do a gradient based minimum search using scipys own minimizer
            if n_minimum_search:
                # If a value for
                # expected_minimum_samples has been parsed
                x_vals, _ = expected_minimum(
                    result,
                    n_random_starts=n_minimum_search,
                    random_state=random_state)
            else:  # Use standard of 20 random starting points
                x_vals, _ = expected_minimum(result,
                                             n_random_starts=20,
                                             random_state=random_state)
        elif params == 'expected_minimum_random':
            # Do a minimum search by evaluating the function with
            # n_samples sample values
            if n_minimum_search is not None:
                # If a value for
                # n_minimum_samples has been parsed
                x_vals, _ = expected_minimum_random_sampling(
                    result,
                    n_random_starts=n_minimum_search,
                    random_state=random_state)
            else:
                # Use standard of 10^n_parameters. Note this
                # becomes very slow for many parameters
                n_minimum_search = 10 ** len(result.x)
                if n_minimum_search > 100000:
                    n_minimum_search = 100000
                x_vals, _ = expected_minimum_random_sampling(
                    result,
                    n_random_starts=n_minimum_search,
                    random_state=random_state)
        else:
            raise ValueError('Argument ´eval_min_params´ must be a valid'
                             'string (´result´)')
    elif isinstance(params, list):
        assert len(params) == len(result.x), 'Argument' \
            '´eval_min_params´ of type list must have same length as' \
            'number of features'
        # Using defined x_values
        x_vals = params
    else:
        raise ValueError('Argument ´eval_min_params´ must'
                         'be a string or a list')
    return x_vals
