import numpy as np

from sklearn.utils import check_random_state

def _transform_back(x0, lower_bounds, upper_bounds):
    return x0 * (upper_bounds - lower_bounds) + lower_bounds

def _local_search(func, x0, bounds, std, n_neighbors,
                  random_state=None, maxiter=10):
    rng = check_random_state(random_state)
    upper_bounds = bounds[:, 1]
    lower_bounds = bounds[:, 0]

    transformed_x_min = np.asarray(x0)
    actual_x_min = _transform_back(
        transformed_x_min, lower_bounds, upper_bounds)

    fun_min = func(actual_x_min.reshape(1, -1))[0]
    n_dims = len(x0)
    transformed_points = np.zeros((n_neighbors*n_dims, n_dims))

    for i in range(maxiter):

        transformed_points[:] = transformed_x_min

        for dim_ind in range(n_dims):

            # Sample n_neighbors number of points from a univariate
            # gaussian with the previous minimum as the mean
            # and provided standard deviation.
            random_points = np.clip(
                transformed_x_min[dim_ind] + std * rng.randn(n_neighbors),
                0, 1)

            start_ptr = dim_ind * n_neighbors
            end_ptr = start_ptr + n_neighbors
            transformed_points[start_ptr: end_ptr, dim_ind] = random_points

        actual_points = _transform_back(
            transformed_points, lower_bounds, upper_bounds)
        func_values = func(actual_points)
        curr_min_ind = np.argmin(func_values)
        curr_min = func_values[curr_min_ind]
        if curr_min < fun_min:
            transformed_x_min = transformed_points[curr_min_ind]
            actual_x_min = actual_points[curr_min_ind]
            fun_min = curr_min
        else:
            break
    return actual_x_min

def paramils_smac(func, x0, bounds, n_neighbors=20, std=0.6,
                  random_state=None, n_jobs=1):
    """
    Reimplement the local-search technique described in
    http://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf

    Parameters
    ----------
    * `func` [callable]:
        Function to be optimized. Should take an 2-D array of
        (n_samples, n_parameters) as input and return a 1-D array
        of (n_samples,)
        XXX: This function should be one that is easy to evaluate.

    * `x0` [array-like, shape=(n_iters, n_parameters)]:
        A local search is started at every point.

    * `bounds` [array-like, shape=(n_parameters, 2)]:
        Array of bounds for every parameter.

    * `n_neighbors` [int, default=20]:
        In search for the optimal point, for every parameter
        `n_neighbors` number of points are drawn from a univariate gaussian
        with that parameter as mean and `std` as standard deviation.
        Hence there are (n_neighbors * n_params) candidates for the next
        optimal point.

    * `std` [float, default=0.6]:
        See `n_neighbors` for description.

    * `random_state` [int, RandomState instance, or None (default)]:
        Set random state to something other than None for reproducible
        results.
    """
    rng = check_random_state(random_state)
    x0 = np.asarray(x0)

    # XXX: Assume user is intelligent enough to attend correct bounds.
    # Also prevents useless re-checks when this function is called
    # in the *_minimize functions.
    bounds = np.asarray(bounds)
    upper_bounds = bounds[:, 1]
    lower_bounds = bounds[:, 0]
    x0 = (x0 - lower_bounds) / (upper_bounds - lower_bounds)
    x_mins = []
    for x0_local in x0:
        curr_min = _local_search(
            func, x0_local, bounds, std, n_neighbors, random_state)
        x_mins.append(curr_min)
    x_mins = np.asarray(x_mins)
    y_min = func(x_mins)
    return x_mins[np.argmin(y_min)]
