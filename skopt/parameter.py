from collections import Sequence
from numbers import Integral
from numbers import Number
from numbers import Real

from scipy.stats.distributions import randint
from scipy.stats.distributions import uniform

from scipy.stats._distn_infrastructure import rv_frozen
from sklearn.utils import check_random_state
from sklearn.utils.fixes import sp_version


class Distribution:
    def transform(self, value):
        """Transform `value` from original space into warped space."""
        return value
    def inverse_transform(self, value):
        """Transform `value` from warped into original space."""
        return value


class Uniform(Distribution):
    def __init__(self, low, high):
        self._low = low
        self._high = high
        self._rvs = uniform(self._low, self._high)

    def bounds(self):
        return (self._low, self._high)

    def rvs(self, n_samples=None, random_state=None):
        return self._rvs.rvs(size=n_samples, random_state=random_state)


class Integer(Distribution):
    def __init__(self, low, high):
        self._low = low
        self._high = high
        self._rvs = randint(self._low, self._high)

    def bounds(self):
        return (self._low, self._high)

    def rvs(self, n_samples=None, random_state=None):
        return self._rvs.rvs(size=n_samples, random_state=random_state)


class Categorical(Distribution):
    def __init__(self, *categories):
        self.categories = categories
        self._rvs = randint(0, len(self.categories))

    def rvs(self, n_samples=None, random_state=None):
        choices = self._rvs.rvs(size=n_samples, random_state=random_state)
        if n_samples is None:
            return self.categories[choices]
        else:
            return [self.categories[choice] for choice in choices]


def sample_points(grid, n_points=1, random_state=None):
    # XXX how to detect [(1,2), (3., 5.)] and convert it to
    # XXX [[(1,2), (3., 5.)]] to support sub-grids
    if (isinstance(grid[0], Distribution) or
        (isinstance(grid[0], Sequence) and isinstance(grid[0][0], (Number, str)))):
        grid = [grid]

    # create a copy of the grid that we can modify without interfering with
    # the caller's copy
    grid_ = []
    for sub_grid in grid:
        sub_grid_ = list(sub_grid)
        grid_.append(sub_grid_)

        for i, dist in enumerate(sub_grid_):
            if isinstance(dist, Distribution):
                pass

            elif len(dist) > 2 or isinstance(dist[0], str):
                sub_grid_[i] = Categorical(*dist)

            # important to check for Integral first as int(3) is
            # also a Real but not the other way around
            elif isinstance(dist[0], Integral):
                sub_grid_[i] = Integer(*dist)
            elif isinstance(dist[0], Real):
                sub_grid_[i] = Uniform(*dist)

    rng = check_random_state(random_state)

    for n in range(n_points):
        sub_grid = grid_[rng.randint(0, len(grid_))]

        params = []
        for dist in sub_grid:
            if sp_version < (0, 16):
                params.append(dist.rvs())
            else:
                params.append(dist.rvs(random_state=rng))
        yield tuple(params)
