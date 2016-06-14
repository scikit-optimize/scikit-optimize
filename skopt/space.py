import numbers
import numpy as np

from scipy.stats.distributions import randint
from scipy.stats.distributions import rv_discrete
from scipy.stats.distributions import uniform

from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state
from sklearn.utils.fixes import sp_version


class _Identity:
    """Identity transform."""

    def fit(self, values):
        return self

    def transform(self, values):
        return values

    def inverse_transform(self, values):
        return values


class _Log10:
    """Base 10 logarithm transform."""

    def fit(self, values):
        return self

    def transform(self, values):
        return np.log10(values)

    def inverse_transform(self, values):
        return 10 ** np.asarray(values)


class _CategoricalEncoder:
    """OneHotEncoder that can handle categorical variables."""

    def __init__(self):
        """Convert labeled categories into one-hot encoded features."""
        self._lb = LabelBinarizer()

    def fit(self, values):
        """Fit a list or array of categories.

        Parameters
        ----------
        * `values` [array-like]:
            List of categories.
        """
        self._lb.fit(values)
        self.n_classes = len(self._lb.classes_)
        return self

    def transform(self, values):
        """Transform an array of categories to a one-hot encoded representation.

        Parameters
        ----------
        * `values` [array-like]:
            List of categories.
        """
        return self._lb.transform(values).ravel()

    def inverse_transform(self, values):
        """Transform points from a warped space."""
        values = np.reshape(values, (-1, self.n_classes))
        return self._lb.inverse_transform(values)


class Dimension:
    """Base class for search space dimensions."""

    def rvs(self, n_samples=1, random_state=None):
        """Draw random samples.

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
        samples = self._rvs.rvs(size=n_samples, random_state=rng)
        return self.inverse_transform(samples)

    def transform(self, random_vals):
        """Transform points to a warped space."""
        return self.transformer.transform(random_vals)

    def inverse_transform(self, random_vals):
        """Transform points from a warped space."""
        return self.transformer.inverse_transform(random_vals)


class Real(Dimension):
    def __init__(self, low, high, prior="uniform"):
        """Search space dimension that can take on any real value.

        Parameters
        ----------
        * `low` [float]:
            Lower bound (inclusive).

        * `high` [float]:
            Upper bound (exclusive).

        * `prior` ["uniform" or "log-uniform", default="uniform"]:
            Distribution to use when sampling random points for this dimension.
            - If `"uniform"`, points are sampled uniformly between the lower
              and upper bounds.
            - If `"log-uniform"`, points are sampled uniformly between
              `log10(lower)` and `log10(upper)`.`
        """
        self._low = low
        self._high = high
        self.prior = prior

        if prior == "uniform":
            self._rvs = uniform(self._low, self._high - self._low)
            self.transformer = _Identity()

        elif prior == "log-uniform":
            self._rvs = uniform(
                np.log10(self._low),
                np.log10(self._high - self._low))
            self.transformer = _Log10()

        else:
            raise ValueError(
                "Prior should be either 'uniform' or 'log-uniform', "
                "got '%s'." % self._rvs)


class Integer(Dimension):
    def __init__(self, low, high):
        """Search space dimension that can take on integer values.

        Parameters
        ----------
        * `low` [float]:
            Lower bound (inclusive).

        * `high` [float]:
            Upper bound (inclusive).
        """
        self._low = low
        self._high = high
        self._rvs = randint(self._low, self._high + 1)
        self.transformer = _Identity()


class Categorical(Dimension):
    def __init__(self, *categories, prior=None):
        """Search space dimension that can take on categorical values.

        Parameters
        ----------
        * `categories` [list, shape=(n_categories,)]:
            Sequence of possible categories.

        * `prior` [list, shape=(categories,), default=None]:
            Prior probabilities for each category. By default all categories
            are equally likely.
        """
        self.categories = np.asarray(categories)
        self.transformer = _CategoricalEncoder()
        self.transformer.fit(self.categories)

        if prior is None:
            prior = np.tile(1. / len(self.categories), len(self.categories))

        self._rvs = rv_discrete(values=(range(len(self.categories)), prior))

    def rvs(self, n_samples=None, random_state=None):
        choices = self._rvs.rvs(size=n_samples, random_state=random_state)
        return self.categories[choices]


class Space:
    def __init__(self, dimensions):
        """Initialize a search space from given specifications.

        Parameters
        ----------
        * `dimensions` [list, shape=(n_dims,)]:
            List of search space dimensions.
            Each search dimension can be defined either as

            - a `(upper_bound, lower_bound)` tuple (for `Real` or `Integer`
              dimensions),
            - a `(upper_bound, lower_bound, "prior")` tuple (for `Real`
              dimensions),
            - an instance of a `Dimension` object (for `Real`, `Integer` or
              `Categorical` dimensions), or
            - as a list of categories (for `Categorical` dimensions).
        """
        space = []

        for dim in dimensions:
            if isinstance(dim, Dimension):
                space.append(dim)

            elif (len(dim) == 3 and
                  isinstance(dim[0], numbers.Real) and
                  isinstance(dim[2], str)):
                space.append(Real(*dim))

            elif len(dim) > 2 or isinstance(dim[0], str):
                space.append(Categorical(*dim))

            elif isinstance(dim[0], numbers.Integral):
                space.append(Integer(*dim))

            elif isinstance(dim[0], numbers.Real):
                space.append(Real(*dim))

            else:
                raise ValueError("Invalid grid component (got %s)." % dim)

        self.space_ = space

    def rvs(self, n_samples=1, random_state=None):
        """Draw random samples.

        Parameters
        ----------
        * `n_samples` [int, default=1]:
            Number of samples to be drawn from the space.

        * `random_state` [int, RandomState instance, or None (default)]:
            Set random state to something other than None for reproducible
            results.

        Returns
        -------
        * `points`: [array-like, shape=(n_points, n_dims)]
           Points sampled from the space.
        """
        rng = check_random_state(random_state)

        # Draw
        points = []

        for dim in self.space_:
            if sp_version < (0, 16):
                points.append(dim.rvs(n_samples=n_samples))
            else:
                points.append(dim.rvs(n_samples=n_samples, random_state=rng))

        # Tranpose
        points_tr = []

        for i in range(n_samples):
            p = []
            for j in range(len(self.space_)):
                p.append(points[j][i])

            points_tr.append(p)

        return points_tr

    def transform(self, X):
        pass

    def inverse_transform(self, Xt):
        pass
