from collections import Mapping
from numbers import Integral
from numbers import Real

from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats.distributions import randint
from scipy.stats.distributions import uniform

from sklearn.utils import check_random_state
from sklearn.utils.fixes import sp_version


class Categorical:
    def __init__(self, *args):
        self.values = args

    def rvs(self, random_state=None):
        rng = check_random_state(random_state)
        return self.values[rng.randint(len(self.values))]

Discrete = randint
Continous = uniform


def points(grid, random_state=None):
    if isinstance(grid, Mapping):
        grid = [grid]

    for sub_grid in grid:
        for k, v in sub_grid.items():
            if isinstance(v, (Categorical, rv_frozen)):
                pass

            elif len(v) > 2 or isinstance(v[0], str):
                sub_grid[k] = Categorical(*v)

            # important to check for Integral first as int(3) is
            # also a Real but not the other way around
            elif isinstance(v[0], Integral):
                sub_grid[k] = randint(*v)
            elif isinstance(v[0], Real):
                sub_grid[k] = uniform(*v)

    rng = check_random_state(random_state)

    while True:
        sub_grid = rng.choice(grid)
        items = sorted(sub_grid.items())

        params = dict()
        for k, v in items:
            if sp_version < (0, 16):
                params[k] = v.rvs()
            else:
                params[k] = v.rvs(random_state=rng)
        yield params
