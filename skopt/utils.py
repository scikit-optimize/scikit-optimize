"""Developer utitilies."""

import numpy as np


def extract_bounds(bounds):
    """Extract lower and upper bounds from a list of (lower, upper) tuples."""
    lb, ub = zip(*bounds)
    return np.asarray(lb), np.asarray(ub)
