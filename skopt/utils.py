import numpy as np

def extract_bounds(bounds):
    """
    Extract lower bounds and upper bounds from a list of (lower, upper)
    tuples.
    """
    lower_bounds, upper_bounds = zip(*bounds)
    return np.asarray(lower_bounds), np.asarray(upper_bounds)
