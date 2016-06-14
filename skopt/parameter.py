from collections import Sequence
import numbers

import numpy as np

from scipy.stats.distributions import randint
from scipy.stats.distributions import rv_discrete
from scipy.stats.distributions import uniform

from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state
from sklearn.utils.fixes import sp_version


class Identity(object):
    """Identity transform."""

    def fit(self, values):
        return self

    def transform(self, values):
        return values

    def inverse_transform(self, values):
        return values


class Log10(object):
    """Base 10 logarithm transform."""

    def fit(self, values):
        return self

    def transform(self, values):
        return np.log10(values)

    def inverse_transform(self, values):
        return 10**np.asarray(values)



class CategoricalEncoder(object):
    """
    OneHotEncoder of scikit-learn that can handle categorical
    variables.
    """
    def __init__(self):
        """Convert labeled categories into one-hot encoded features"""
        self._lb = LabelBinarizer()

    def fit(self, values):
        """
        Fit a list or array of categories.

        Parameters
        ----------
        * `values` [array-like]:
            List of categories.
        """
        self._lb.fit(values)
        self.n_classes = len(self._lb.classes_)
        return self

    def transform(self, values):
        """
        Transform an array of categories to a one-hot encoded representation.

        Parameters
        ----------
        * `values` [array-like]:
            List of categories.
        """
        return self._lb.transform(values).ravel()

    def inverse_transform(self, values):
        """
        Transform points from a warped space.
        """
        values = np.reshape(values, (-1, self.n_classes))
        return self._lb.inverse_transform(values)


class Distribution:
    def rvs(self, n_samples=1, random_state=None):
        """
        Randomly sample points from the original space

        Parameters
        ----------
        * `n_samples` [int or None]:
            The number of samples to be drawn.

        * `random_state` [int, RandomState instance, or None (default)]:
            Set random state to something other than None for reproducible
            results.
        """
        # Assume that `_rvs` samples in the warped space
        rng = check_random_state(random_state)
        random_vals = self._rvs.rvs(size=n_samples, random_state=rng)
        return self.inverse_transform(random_vals)

    def transform(self, random_vals):
        """
        Transform points to a warped space.
        """
        return self.transformer.transform(random_vals)

    def inverse_transform(self, random_vals):
        """
        Transform points from a warped space.
        """
        return self.transformer.inverse_transform(random_vals)


class Real(Distribution):
    def __init__(self, low, high, prior="uniform"):
        """Search space dimension that can take on any real value.

        Parameters
        ----------
        * `low` [float]:
            Lower bound of the parameter. (Inclusive)

        * `high` [float]:
            Upper bound of the parameter. (Exclusive)

        * `prior` ["uniform" or "log-uniform", default='uniform']:
            Distribution to use when sampling random points for this parameter.
            If uniform, points are sampled uniformly between the lower and
            upper bounds.
            If log-uniform, points are sampled uniformly between log10(lower)
            and log10(upper bounds)
        """
        self._low = low
        self._high = high
        self.prior = prior

        if prior == "uniform":
            self._rvs = uniform(self._low, self._high - self._low)
            self.transformer = Identity()

        elif prior == "log-uniform":
            self._rvs = uniform(
                np.log10(self._low),
                np.log10(self._high - self._low))
            self.transformer = Log10()

        else:
            raise ValueError(
                "Prior should be either 'uniform' or 'log-uniform', "
                "got '%s'." % self._rvs)


class Integer(Distribution):
    def __init__(self, low, high):
        """Search space dimension that can take on integer values.

        Parameters
        ----------
        * `low` [float]:
            Lower bound of the parameter. Inclusive

        * `high` [float]:
            Upper bound of the parameter. Inclusive
        """
        self._low = low
        self._high = high
        self._rvs = randint(self._low, self._high + 1)
        self.transformer = Identity()


class Categorical(Distribution):
    def __init__(self, *categories, prior=None):
        """Search space dimension that can take on categorical values.

        Parameters
        ----------
        *categories :
            sequence of possible categories

        * `prior` [array-like, shape=(categories,), default=None]:
            Prior probabilities for each category. By default all categories
            are equally likely.
        """
        self.categories = np.asarray(categories)
        self.transformer = CategoricalEncoder()
        self.transformer.fit(self.categories)
        if prior is None:
            prior = np.tile(1. / len(self.categories), len(self.categories))
        self._rvs = rv_discrete(values=(range(len(self.categories)), prior))

    def rvs(self, n_samples=None, random_state=None):
        choices = self._rvs.rvs(size=n_samples, random_state=random_state)
        return self.categories[choices]


def _check_grid(grid):
    grid = list(grid)

    for i, dist in enumerate(grid):
        if isinstance(dist, Distribution):
            pass

        elif (len(dist) == 3 and
              isinstance(dist[0], numbers.Real) and
              isinstance(dist[2], str)):
            grid[i] = Real(*dist)

        elif len(dist) > 2 or isinstance(dist[0], str):
            grid[i] = Categorical(*dist)

        elif isinstance(dist[0], numbers.Integral):
            grid[i] = Integer(*dist)

        elif isinstance(dist[0], numbers.Real):
            grid[i] = Real(*dist)

    return grid


def sample_points(grid, n_points=1, random_state=None):
    """Sample points from the provided grid.

    Parameters
    ----------
    * `grid` [array-like, shape=(n_parameters,)]:
        Each parameter of the grid can be a

        1. (upper_bound, lower_bound) tuple.
        2. (upper_bound, lower_bound, "prior") tuple.
        3. Instance of a Distribution object
        4. list of categories.

    * `n_points`: int
        Number of parameters to be sampled from the grid.

    * `random_state` [int, RandomState instance, or None (default)]:
        Set random state to something other than None for reproducible
        results.

    Returns
    -------
    * `sampled_points`: [array-like,]
       Points sampled from the grid.
    """
    grid = _check_grid(grid)
    rng = check_random_state(random_state)

    for n in range(n_points):
        params = []

        for dist in grid:
            if sp_version < (0, 16):
                params.append(dist.rvs())
            else:
                params.append(dist.rvs(random_state=rng))

        yield tuple(params)
