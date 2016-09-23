from sklearn.utils import check_random_state

from .space import Space


class Optimize:
    """Sequential model-based optimisation, one step at a time.

    Minimize the objective function using `optimizer` one step
    at a time. This gives the caller full control over the number
    of iterations and how to evaluate the objective.
    """
    def __init__(self, optimizer, dimensions, random_state=None):
        self.optimizer = optimizer
        self.rng = check_random_state(random_state)
        self.space = Space(dimensions)

        self.X_ = []
        self.y_ = []

    def suggest(self):
        """Suggest the next point at which to evaluate the objective."""
        if self.X_:
            return self.X_[-1]

        else:
            return self.space.rvs(random_state=self.rng)[0]

    def report(self, point, value):
        """Record the `value` of the objective at `point`."""
        self.X_.append(point)
        self.y_.append(value)

        self.fit()

    def fit(self):
        """Fit surrogate model to observations from the objective."""
        res = self.optimizer(self._func, self.space.dimensions,
                             n_calls=len(self.X_),
                             x0=self.X_, y0=self.y_, random_state=self.rng)

    def _func(self, point):
        self.X_.append(point)
        return -1
