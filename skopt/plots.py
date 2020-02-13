"""Plotting functions."""
import sys
import numpy as np
from itertools import count
from functools import partial
from scipy.optimize import OptimizeResult

from skopt import expected_minimum, expected_minimum_random_sampling
from .space import Categorical

# For plot tests, matplotlib must be set to headless mode early
if 'pytest' in sys.modules:
    import matplotlib

    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import LogLocator
from matplotlib.ticker import MaxNLocator, FuncFormatter  # noqa: E402

from skopt.space import Categorical
from skopt.utils import get_samples_dimension
from collections import Counter


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


def _get_ylim_diagonal(ax):
    """Get the min / max of the ylim for all diagonal plots.
    This is used in _adjust_fig() so the ylim is the same
    for all diagonal plots.

    Parameters
    ----------
    * `ax` [`Matplotlib.Axes`]:
        2-dimensional matrix with Matplotlib Axes objects.

    Returns
    -------
    * `ylim_diagonal` [list(int)]
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


def _adjust_fig(fig, ax, space, ylabel, dimensions):
    """
    Process and adjust a 2-dimensional plot-matrix in various ways,
    by writing axis-labels, etc.
    
    This is used by plot_objective() and plot_evaluations().
    
    Parameters
    ----------
    * `fig` [`Matplotlib.Figure`]:
        Figure-object for the plots.

    * `ax` [`Matplotlib.Axes`]:
        2-dimensional matrix with Matplotlib Axes objects.

    * `space` [`Space`]:
        Search-space object.

    * `ylabel` [`str`]:
        String to be printed on the top-left diagonal plot
        e.g. 'Sample Count'.

    * `dimensions` [`list(Dimension)`]:
        List of `Dimension` objects used in the plots.

    Returns
    -------
    * Nothing.
    """

    # Adjust spacing of the figure.
    # This looks bad on some outputs so it has been disabled for now.
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
    #                     hspace=0.1, wspace=0.1)

    # Get min/max ylim for the diagonal plots, used to normalize their y-axis.
    ylim_diagonal = _get_ylim_diagonal(ax=ax)

    # The following for-loops process the sub-plots inside the 2-d matrix.
    # This could perhaps also be implemented using other Python tricks,
    # but these for-loops are probably much easier to understand.
    # Similarly, they have been separated into several for-loops to make
    # them easier to understand and modify.

    # Number of search-space dimensions used in this plot.
    n_dims = len(dimensions)

    # Process the plots on the diagonal.
    for row in range(n_dims):
        # Get the search-space dimension for this row.
        dim = dimensions[row]

        # Reference to the diagonal plot for this row.
        a = ax[row, row]

        # Write the dimension-name as a label on top of the diagonal plot.
        a.xaxis.set_label_position('top')
        a.set_xlabel(dim.name)

        # Set the x-axis limits to correspond to the search-space bounds.
        a.set_xlim(dim.bounds)

        # Use a common limit for the y-axis on all diagonal plots.
        a.set_ylim(ylim_diagonal)

        # Use log-scale on the x-axis?
        if dim.prior == 'log-uniform':
            a.set_xscale('log')

    # Process the plots below the diagonal.
    for row in range(n_dims):
        # Get the search-space dimension for this row.
        dim_row = dimensions[row]

        # Only iterate until the diagonal.
        for col in range(row):
            # Get the search-space dimension for this column.
            dim_col = dimensions[col]

            # Reference to the plot for this row and column.
            a = ax[row, col]

            # Plot a grid.
            a.grid(True)

            # Set the plot-limits to correspond to the search-space bounds.
            a.set_xlim(dim_col.bounds)
            a.set_ylim(dim_row.bounds)

            # Use log-scale on the x-axis?
            if dim_col.prior == 'log-uniform':
                a.set_xscale('log')

            # Use log-scale on the y-axis?
            if dim_row.prior == 'log-uniform':
                a.set_yscale('log')

    # Turn off all plots to the upper-right of the diagonal.
    for row in range(n_dims):
        for col in range(row+1, n_dims):
            ax[row, col].axis("off")

    # Set the designated ylabel for the top-left plot.
    row = col = 0
    ax[row, col].set_ylabel(ylabel)

    # Set the dimension-names for the left-most column.
    col = 0
    for row in range(1, n_dims):
        ax[row, col].set_ylabel(dimensions[row].name)

    # Set the dimension-names for the bottom row.
    row = n_dims - 1
    for col in range(0, n_dims):
        ax[row, col].set_xlabel(dimensions[col].name)

    # Remove the y-tick labels for all plots except the left-most column.
    for row in range(n_dims):
        for col in range(1, n_dims):
            ax[row, col].set_yticklabels([])

    # Remove the x-tick labels for all plots except the bottom row.
    for row in range(n_dims-1):
        for col in range(n_dims):
            ax[row, col].set_xticklabels([])


def _map_bins(bins, bounds, prior):
    """
    For use when plotting histograms.
    Maps the number of bins to a log-scale between the bounds, if necessary.

    When `x_eval` is not `None`, the given values are used instead of
    random samples. In this case, `n_samples` will be ignored.

    Parameters
    ----------
    * `bins` [int]
        Number of bins in the histogram.

    * `bounds` [(int, int)]
        Tuple or list with lower- and upper-bounds for a search-space dimension.

    * `prior` [str or None]
        If 'log-uniform' then use log-scaling for the bins,
        otherwise use the original number of bins.

    Returns
    -------
    * `bins_mapped`: [int or np.array(int)]:
         Number of bins for a histogram if no mapping,
         or a log-scaled array of bin-points if mapping is needed.
    """

    if prior == 'log-uniform':
        # Map the number of bins to a log-space for the dimension bounds.
        bounds_log = np.log10(bounds)
        bins_mapped = np.logspace(bounds_log[0], bounds_log[1], bins)

        # Note that Python 3.X supports the following, but not Python 2.7
        # bins_mapped = np.logspace(*np.log10(bounds), bins)
    else:
        # Use the original number of bins.
        bins_mapped = bins

    return bins_mapped


def partial_dependence_1D(model, dimension, samples, n_points=40):
    """
    Calculate the partial dependence for a single dimension.
    
    This uses the given model to calculate the average objective value
    for all the samples, where the given dimension is fixed at
    regular intervals between its bounds.

    This shows how the given dimension affects the objective value
    when the influence of all other dimensions are averaged out.

    Parameters
    ----------
    * `model`
        Surrogate model for the objective function.

    * `dimension` [Dimension]
        The `Dimension`-object for which to calculate the partial dependence.

    * `samples` [np.array, shape=(n_points, n_dims)]
        Randomly sampled and transformed points to use when averaging
        the model function at each of the `n_points` when using partial
        dependence.

    * `n_points` [int, default=40]
        Number of points along each dimension where the partial dependence
        is evaluated.

    x_eval : list, default=None
        `x_eval` is a list of parameter values or None. In case `x_eval`
        is not None, the parsed dependence will be calculated using these
        values.
        Otherwise, random selected samples will be used.

    Returns
    -------
    * `xi`: [np.array]:
        The points at which the partial dependence was evaluated.

    * `yi`: [np.array]:
        The average value of the modelled objective function at each point `xi`.
    """

    def _calc(x):
        """
        Helper-function to calculate the average predicted
        objective value for the given model, when setting
        the index'th dimension of the search-space to the value x,
        and then averaging over all samples.
        """

        # Copy the samples so we don't destroy the originals.
        samples_copy = np.copy(samples)

        # Set the index'th dimension to x for all samples.
        samples_copy[:, index] = x

        # Calculate the predicted objective value for all samples.
        y_pred = model.predict(samples_copy)

        # The average predicted value for the objective function.
        y_pred_mean = np.mean(y_pred)

        return y_pred_mean

    # Get search-space index for the given dimension.
    index = dimension.index

    # Get the bounds of the dimension.
    bounds = dimension.bounds

    # Generate evenly spaced points between the bounds.
    xi = np.linspace(bounds[0], bounds[1], n_points)

    # Transform the points if necessary.
    xi_transformed = dimension.transform(xi)

    # Calculate the partial dependence for all the points.
    yi = [_calc(x) for x in xi_transformed]

    return xi, yi


def partial_dependence_2D(model, dimension1, dimension2, samples, n_points=40):
    """
    Calculate the partial dependence for two dimensions in the search-space.

    This uses the given model to calculate the average objective value
    for all the samples, where the given dimensions are fixed at
    regular intervals between their bounds.

    This shows how the given dimensions affect the objective value
    when the influence of all other dimensions are averaged out.

    Parameters
    ----------
    * `model`
        Surrogate model for the objective function.

    * `dimension1` [Dimension]
        The first `Dimension`-object for which to calculate the
        partial dependence.

    * `dimension2` [Dimension]
        The second `Dimension`-object for which to calculate the
        partial dependence.

    * `samples` [np.array, shape=(n_points, n_dims)]
        Randomly sampled and transformed points to use when averaging
        the model function at each of the `n_points`.

    * `n_points` [int, default=40]
        Number of points along each dimension where the partial dependence
        is evaluated.

    Returns
    -------
    * `xi`: [np.array, shape=n_points]:
        The points at which the partial dependence was evaluated.

    * `yi`: [np.array, shape=n_points]:
        The points at which the partial dependence was evaluated.

    * `zi`: [np.array, shape=(n_points, n_points)]:
        The average value of the objective function at each point `(xi, yi)`.
    """

    def _calc(x, y):
        """
        Helper-function to calculate the average predicted
        objective value for the given model, when setting
        the index1'th dimension of the search-space to the value x
        and setting the index2'th dimension to the value y,
        and then averaging over all samples.
        """

        # Copy the samples so we don't destroy the originals.
        samples_copy = np.copy(samples)

        # Set the index1'th dimension to x for all samples.
        samples_copy[:, index1] = x

        # Set the index2'th dimension to y for all samples.
        samples_copy[:, index2] = y

        # Calculate the predicted objective value for all samples.
        z_pred = model.predict(samples_copy)

        # The average predicted value for the objective function.
        z_pred_mean = np.mean(z_pred)

        return z_pred_mean

    # Get search-space indices for the dimensions.
    index1 = dimension1.index
    index2 = dimension2.index

    # Get search-space bounds for the dimensions.
    bounds1 = dimension1.bounds
    bounds2 = dimension2.bounds

    # Generate evenly spaced points between the dimension bounds.
    xi = np.linspace(bounds1[0], bounds1[1], n_points)
    yi = np.linspace(bounds2[0], bounds2[1], n_points)

    # Transform the points if necessary.
    xi_transformed = dimension1.transform(xi)
    yi_transformed = dimension2.transform(yi)

    # Calculate the partial dependence for all combinations of these points.
    zi = [[_calc(x, y) for x in xi_transformed] for y in yi_transformed]

    # Convert list-of-list to a numpy array.
    zi = np.array(zi)

    return xi, yi, zi


def plot_evaluations(result, bins=20, dimension_names=None):
    """
    Visualize the order in which points were sampled during optimization.

    This creates a 2-d matrix plot where the diagonal plots are histograms
    that show the distribution of samples for each search-space dimension.

    The plots below the diagonal are scatter-plots of the samples for
    all combinations of search-space dimensions.

    The ordering of the samples are shown as different colour-shades.

    A red star shows the best found parameters.

    NOTE: Search-spaces with `Categorical` dimensions are not supported.

    Parameters
    ----------
    result : `OptimizeResult`
        The optimization results from calling e.g. `gp_minimize()`.

    bins : int, bins=20
        Number of bins to use for histograms on the diagonal.

    dimension_names : list(str)
        List of names for search-space dimensions to be used in the plot.
        You can omit `Categorical` dimensions here as they are not supported. 
        If `None` then use all dimensions from the search-space.

    Returns
    -------
    fig : `Matplotlib.Figure`
        The object for the figure.
        For example, call `fig.savefig('plot.png')` to save the plot.

    ax : `Matplotlib.Axes`
        A 2-d matrix of Axes-objects with the sub-plots.
    """

    # Get the search-space instance from the optimization results.
    space = result.space

    # Get the relevant search-space dimensions.
    if dimension_names is None:
        # Get all dimensions.
        dimensions = space.dimensions
    else:
        # Only get the named dimensions.
        dimensions = space[dimension_names]

    # Ensure there are no categorical dimensions.
    # TODO replace with check_list_types(dimensions, (Integer, Real)) in PR #597
    if any(isinstance(dim, Categorical) for dim in dimensions):
        raise ValueError("Categorical dimension is not supported.")

    # Number of search-space dimensions we are using.
    n_dims = len(dimensions)

    # Create a figure for plotting a 2-d matrix of sub-plots.
    fig, ax = plt.subplots(n_dims, n_dims, figsize=(2 * n_dims, 2 * n_dims))

    # Used to plot colour-shades for the sample-ordering.
    # It is just a range from 0 to the number of samples.
    sample_order = range(len(result.x_iters))

    # For all rows in the 2-d plot matrix.
    for row in range(n_dims):
        # Get the search-space dimension for this row.
        dim_row = dimensions[row]

        # Get the index for the search-space dimension.
        # This is used to lookup that particular dimension in some functions.
        index_row = dim_row.index

        # Get the samples from the optimization-log for this dimension.
        samples_row = get_samples_dimension(result=result, index=index_row)

        # Get the best-found sample for this dimension.
        best_sample_row = result.x[index_row]

        # Search-space boundary for this dimension.
        bounds_row = dim_row.bounds

        # Map the number of bins to a log-space if necessary.
        bins_mapped = _map_bins(bins=bins,
                                bounds=dim_row.bounds,
                                prior=dim_row.prior)

        # Plot a histogram on the diagonal.
        ax[row, row].hist(samples_row, bins=bins_mapped, range=bounds_row)

        # For all columns until the diagonal in the 2-d plot matrix.
        for col in range(row):
            # Get the search-space dimension for this column.
            dim_col = dimensions[col]

            # Get the index for this search-space dimension.
            # This is used to lookup that dimension in some functions.
            index_col = dim_col.index

            # Get the samples from the optimization-log for that dimension.
            samples_col = get_samples_dimension(result=result, index=index_col)

            # Plot all the parameters that were sampled during optimization.
            # These are plotted as small coloured dots, where the colour-shade
            # indicates the time-progression.
            ax[row, col].scatter(samples_col, samples_row,
                                 c=sample_order, s=40, lw=0., cmap='viridis')

            # Get the best-found sample for this dimension.
            best_sample_col = result.x[index_col]

            # Plot the best parameters that were sampled during optimization.
            # These are plotted as a big red star.
            ax[row, col].scatter(best_sample_col, best_sample_row,
                                 c='red', s=100, lw=0., marker='*')

    # Make various adjustments to the plots.
    _adjust_fig(fig=fig, ax=ax, space=space,
                dimensions=dimensions, ylabel="Sample Count")

    return fig, ax


def plot_objective(result, levels=10, n_points=40, n_samples=250,
                   zscale='linear', dimension_names=None):
    """
    Plot a 2-d matrix with so-called Partial Dependence plots
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
    during optimization, and large red stars show the best found points.

    NOTE: The Partial Dependence plot is only an estimation of the surrogate
          model which in turn is only an estimation of the true objective
          function that has been optimized. This means the plots show
          an "estimate of an estimate" and may therefore be quite imprecise,
          especially if few samples have been collected during the optimization
          (e.g. less than 100-200 samples), and in regions of the search-space
          that have been sparsely sampled (e.g. regions away from the optimum).
          This means that the plots may change each time you run the
          optimization and they should not be considered completely reliable.
          These compromises are necessary because we cannot evaluate the
          expensive objective function in order to plot it, so we have to use
          the cheaper surrogate model to plot its contour. And in order to
          show search-spaces with 3 dimensions or more in a 2-dimensional plot,
          we further need to map those dimensions to only 2-dimensions using
          the Partial Dependence, which also causes distortions in the plots.

    NOTE: Search-spaces with `Categorical` dimensions are not supported.

    NOTE: This function can be very slow for dimensions greater than 5.

    Parameters
    ----------
    result : `OptimizeResult`
        The optimization results from calling e.g. `gp_minimize()`.

    levels : int, default=10
        Number of levels to draw on the contour plot, passed directly
        to `plt.contour()`.

    n_points : int, default=40
        Number of points along each dimension where the partial dependence
        is evaluated when generating the contour-plots.

    n_samples : int, default=250
        Number of points along each dimension where the partial dependence
        is evaluated when generating the contour-plots.

    zscale : str, default='linear'
        Scale to use for the z-axis of the contour plots.
        Either 'log' or linear for all other choices.

    dimension_names : list(str), default=None
        List of names for search-space dimensions to be used in the plot.
        You can omit `Categorical` dimensions here as they are not supported.
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
    fig : `Matplotlib.Figure`
        The object for the figure.
        For example, call `fig.savefig('plot.png')` to save the plot.
    
    ax : `Matplotlib.Axes`
        A 2-d matrix of Axes-objects with the sub-plots.
    """

    # Scale for the z-axis of the contour-plot. Either Log or Linear (None).
    locator = LogLocator() if zscale == 'log' else None

    # Get the search-space instance from the optimization results.
    space = result.space

    # Get the relevant search-space dimensions.
    if dimension_names is None:
        # Get all dimensions.
        dimensions = space.dimensions
    else:
        # Only get the named dimensions.
        dimensions = space[dimension_names]

    # Ensure there are no categorical dimensions.
    # TODO replace with check_list_types(dimensions, (Integer, Real)) in PR #597
    if any(isinstance(dim, Categorical) for dim in dimensions):
        raise ValueError("Categorical dimension is not supported.")

    # Number of search-space dimensions we are using.
    n_dims = len(dimensions)

    # Get the last fitted model for the search-space.
    last_model = result.models[-1]

    # Get new random samples from the search-space and transform if necessary.
    new_samples = space.rvs(n_samples=n_samples)
    new_samples = space.transform(new_samples)

    # Create a figure for plotting a 2-d matrix of sub-plots.
    fig, ax = plt.subplots(n_dims, n_dims, figsize=(2*n_dims, 2*n_dims))

    # For all rows in the 2-d plot matrix.
    for row in range(n_dims):
        # Get the search-space dimension for this row.
        dim_row = dimensions[row]

        # Get the index for the search-space dimension.
        # This is used to lookup that particular dimension in some functions.
        index_row = dim_row.index

        # Get the samples from the optimization-log for this dimension.
        samples_row = get_samples_dimension(result=result, index=index_row)

        # Get the best-found sample for this dimension.
        best_sample_row = result.x[index_row]

        # Search-space boundary for this dimension.
        bounds_row = dim_row.bounds

        # Calculate partial dependence for this dimension.
        xi, yi = partial_dependence_1D(model=last_model,
                                       dimension=dim_row,
                                       samples=new_samples,
                                       n_points=n_points)

        # Reference to the plot for the diagonal of this row.
        a = ax[row, row]

        # TODO: There is a problem here if yi is very large, then matplotlib
        # TODO: writes a number above the plot that I don't know how to turn off.
        # Plot the partial dependence for this dimension.
        a.plot(xi, yi)

        # Plot a dashed line for the best-found parameter.
        a.axvline(best_sample_row, linestyle="--", color="red", lw=1)

        # For all columns until the diagonal in the 2-d plot matrix.
        for col in range(row):
            # Get the search-space dimension for this column.
            dim_col = dimensions[col]

            # Get the index for this search-space dimension.
            # This is used to lookup that dimension in some functions.
            index_col = dim_col.index

            # Get the samples from the optimization-log for that dimension.
            samples_col = get_samples_dimension(result=result, index=index_col)

            # Get the best-found sample for this dimension.
            best_sample_col = result.x[index_col]

            # Calculate the partial dependence for these two dimensions.
            # Note that column and row are switched here.
            xi, yi, zi = partial_dependence_2D(model=last_model,
                                               dimension1=dim_col,
                                               dimension2=dim_row,
                                               samples=new_samples,
                                               n_points=n_points)

            # Reference to the plot for this row and column.
            a = ax[row, col]

            # Plot the contour landscape for the objective function.
            a.contourf(xi, yi, zi, levels, locator=locator, cmap='viridis_r')

            # Plot all the parameters that were sampled during optimization.
            # These are plotted as small black dots.
            a.scatter(samples_col, samples_row, c='black', s=10, lw=0.)

            # Plot the best parameters that were sampled during optimization.
            # These are plotted as a big red star.
            a.scatter(best_sample_col, best_sample_row,
                      c='red', s=100, lw=0., marker='*')

    # Make various adjustments to the plots.
    _adjust_fig(fig=fig, ax=ax, space=space,
                dimensions=dimensions, ylabel="Partial Dependence")

    return fig, ax


def plot_objective_2D(result, dimension_name1, dimension_name2,
                      n_points=40, n_samples=250, levels=10, zscale='linear'):
    """
    Create and return a Matplotlib figure and axes with a landscape
    contour-plot of the last fitted model of the search-space,
    overlaid with all the samples from the optimization results,
    for the two given dimensions of the search-space.

    This is similar to `plot_objective()` but only for 2 dimensions
    whose doc-string also has a more extensive explanation.
    
    NOTE: Categorical dimensions are not supported.

    Parameters
    ----------
    * `result` [`OptimizeResult`]
        The optimization results e.g. from calling `gp_minimize()`.

    * `dimension_name1` [str]:
        Name of a dimension in the search-space.

    * `dimension_name2` [str]:
        Name of a dimension in the search-space.

    * `n_samples` [int, default=250]
        Number of random samples used for estimating the contour-plot
        of the objective function.

    * `n_points` [int, default=40]
        Number of points along each dimension where the partial dependence
        is evaluated when generating the contour-plots.

    * `levels` [int, default=10]
        Number of levels to draw on the contour plot.

    * `zscale` [str, default='linear']
        Scale to use for the z axis of the contour plots.
        Either 'log' or linear for all other choices.

    Returns
    -------
    * `fig`: [`Matplotlib.Figure`]:
        The Matplotlib Figure-object.
        For example, you can save the plot by calling `fig.savefig('file.png')` 

    * `ax`: [`Matplotlib.Axes`]:
        The Matplotlib Figure-object.
        For example, you can save the plot by calling `fig.savefig('file.png')` 
    """

    # Get the search-space instance from the optimization results.
    space = result.space

    # Get the dimension-object, its index in the search-space, and its name.
    dimension1 = space[dimension_name1]
    dimension2 = space[dimension_name2]

    # Ensure dimensions are not Categorical.
    # TODO replace with check_list_types(dimensions, (Integer, Real)) in PR #597
    if any(isinstance(dim, Categorical) for dim in [dimension1, dimension2]):
        raise ValueError("Categorical dimension is not supported.")

    # Get the indices for the search-space dimensions.
    index1 = dimension1.index
    index2 = dimension2.index

    # Get the samples from the optimization-log for the relevant dimensions.
    samples1 = get_samples_dimension(result=result, index=index1)
    samples2 = get_samples_dimension(result=result, index=index2)

    # Get the best-found samples for the relevant dimensions.
    best_sample1 = result.x[index1]
    best_sample2 = result.x[index2]

    # Get the last fitted model for the search-space.
    last_model = result.models[-1]

    # Get new random samples from the search-space and transform if necessary.
    new_samples = space.rvs(n_samples=n_samples)
    new_samples = space.transform(new_samples)

    # Estimate the objective function for these sampled points
    # using the last fitted model for the search-space.
    xi, yi, zi = partial_dependence_2D(model=last_model,
                                       dimension1=dimension1,
                                       dimension2=dimension2,
                                       samples=new_samples,
                                       n_points=n_points)

    # Start a new plot.
    fig, ax = plt.subplots(nrows=1, ncols=1)

    # Scale for the z-axis of the contour-plot. Either Log or Linear (None).
    locator = LogLocator() if zscale == 'log' else None

    # Plot the contour-landscape for the objective function.
    ax.contourf(xi, yi, zi, levels, locator=locator, cmap='viridis_r')

    # Plot all the parameters that were sampled during optimization.
    # These are plotted as small black dots.
    ax.scatter(samples1, samples2, c='black', s=10, linewidths=1)

    # Plot the best parameters that were sampled during optimization.
    # These are plotted as a big red star.
    ax.scatter(best_sample1, best_sample2,
               c='red', s=50, linewidths=1, marker='*')

    # Use the dimension-names as the labels for the plot-axes.
    ax.set_xlabel(dimension_name1)
    ax.set_ylabel(dimension_name2)

    # Use log-scale on the x-axis?
    if dimension1.prior == 'log-uniform':
        ax.set_xscale('log')

    # Use log-scale on the y-axis?
    if dimension2.prior == 'log-uniform':
        ax.set_yscale('log')

    return fig, ax


def plot_histogram(result, dimension_name, bins=20, rotate_labels=0):
    """
    Create and return a Matplotlib figure with a histogram
    of the samples from the optimization results,
    for a given dimension of the search-space.

    Parameters
    ----------
    * `result` [`OptimizeResult`]
        The optimization results e.g. from calling `gp_minimize()`.

    * `dimension_name` [str]:
        Name of a dimension in the search-space.

    * `bins` [int, bins=20]:
        Number of bins in the histogram.

    * `rotate_labels` [int, rotate_labels=0]:
        Degree to rotate category-names on the x-axis.
        Only used for Categorical dimensions.

    Returns
    -------
    fig : `Matplotlib.Figure`
        The Matplotlib Figure-object.
        For example, you can save the plot by calling
        `fig.savefig('file.png')`

    ax : `Matplotlib.Axes`
        The Matplotlib Axes-object.
    """

    # Get the search-space instance from the optimization results.
    space = result.space

    # Get the dimension-object.
    dimension = space[dimension_name]

    # Get the samples from the optimization-log for that particular dimension.
    samples = get_samples_dimension(result=result, index=dimension.index)

    # Start a new plot.
    fig, ax = plt.subplots(nrows=1, ncols=1)

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

        # Map the number of bins to a log-space if necessary.
        bins_mapped = _map_bins(bins=bins,
                                bounds=dimension.bounds,
                                prior=dimension.prior)

        # Plot the histogram.
        ax.hist(samples, bins=bins_mapped, range=dimension.bounds)

        # Use log-scale on the x-axis?
        if dimension.prior == 'log-uniform':
            ax.set_xscale('log')

    # Set the labels.
    ax.set_xlabel(dimension_name)
    ax.set_ylabel('Sample Count')

    return fig, ax
