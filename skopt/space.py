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

    def fit(self, X):
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, Xt):
        return Xt


class _Log10:
    """Base 10 logarithm transform."""

    def fit(self, X):
        return self

    def transform(self, X):
        return np.log10(np.asarray(X, dtype=np.float))

    def inverse_transform(self, Xt):
        return 10.0 ** np.asarray(Xt, dtype=np.float)


class _CategoricalEncoder:
    """OneHotEncoder that can handle categorical variables."""

    def __init__(self):
        """Convert labeled categories into one-hot encoded features."""
        self._lb = LabelBinarizer()

    def fit(self, X):
        """Fit a list or array of categories.

        Parameters
        ----------
        * `X` [array-like, shape=(n_categories,)]:
            List of categories.
        """
        self.mapping_ = {v: i for i, v in enumerate(X)}
        self.inverse_mapping_ = {i: v for v, i in self.mapping_.items()}
        self._lb.fit([self.mapping_[v] for v in X])
        self.n_classes = len(self._lb.classes_)

        return self

    def transform(self, X):
        """Transform an array of categories to a one-hot encoded representation.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples,)]:
            List of categories.

        Returns
        -------
        * `Xt` [array-like, shape=(n_samples, n_categories)]:
            The one-hot encoded categories.
        """
        return self._lb.transform([self.mapping_[v] for v in X])

    def inverse_transform(self, Xt):
        """Inverse transform one-hot encoded categories back to their original
           representation.

        Parameters
        ----------
        * `Xt` [array-like, shape=(n_samples, n_categories)]:
            One-hot encoded categories.

        Returns
        -------
        * `X` [array-like, shape=(n_samples,)]:
            The original categories.
        """
        Xt = np.asarray(Xt)
        return [
            self.inverse_mapping_[i] for i in self._lb.inverse_transform(Xt)
        ]


class Dimension(object):
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
        rng = check_random_state(random_state)
        samples = self._rvs.rvs(size=n_samples, random_state=rng)
        return self.inverse_transform(samples)

    def transform(self, X):
        """Transform samples form the original space to a warped space."""
        return self.transformer.transform(X)

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back into the
           original space.
        """
        return self.transformer.inverse_transform(Xt)

    @property
    def size(self):
        return 1

    @property
    def transformed_size(self):
        return 1

    @property
    def bounds(self):
        raise NotImplementedError

    @property
    def transformed_bounds(self):
        raise NotImplementedError


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
                np.log10(self._high) - np.log10(self._low))
            self.transformer = _Log10()

        else:
            raise ValueError(
                "Prior should be either 'uniform' or 'log-uniform', "
                "got '%s'." % self._rvs)

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back into the
           orignal space.
        """
        return super(Real, self).inverse_transform(Xt).astype(np.float)

    @property
    def bounds(self):
        return (self._low, self._high)

    @property
    def transformed_bounds(self):
        if self.prior == "uniform":
            return (self._low, self._high)

        else:  # self.prior == "log-uniform"
            return (np.log10(self._low), np.log10(self._high))


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

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back into the
           orignal space.
        """
        # The concatenation of all transformed dimensions makes Xt to be
        # of type float, hence the required cast back to int.
        return super(Integer, self).inverse_transform(Xt).astype(np.int)

    @property
    def bounds(self):
        return (self._low, self._high)

    @property
    def transformed_bounds(self):
        return (self._low, self._high)


class Categorical(Dimension):
    def __init__(self, categories, prior=None):
        """Search space dimension that can take on categorical values.

        Parameters
        ----------
        * `categories` [list, shape=(n_categories,)]:
            Sequence of possible categories.

        * `prior` [list, shape=(categories,), default=None]:
            Prior probabilities for each category. By default all categories
            are equally likely.
        """
        self.categories = categories
        self.transformer = _CategoricalEncoder()
        self.transformer.fit(self.categories)

        if prior is None:
            prior = np.tile(1. / len(self.categories), len(self.categories))

        # XXX check that sum(prior) == 1
        self._rvs = rv_discrete(values=(range(len(self.categories)), prior))

    def rvs(self, n_samples=None, random_state=None):
        choices = self._rvs.rvs(size=n_samples, random_state=random_state)

        if isinstance(choices, numbers.Integral):
            return self.categories[choices]
        else:
            return [self.categories[c] for c in choices]

    @property
    def transformed_size(self):
        size = len(self.categories)
        # when len(categories) == 2, CategoricalEncoder outputs a single value
        return size if size != 2 else 1

    @property
    def bounds(self):
        return self.categories

    @property
    def transformed_bounds(self):
        if self.transformed_size == 1:
            return (0.0, 1.0)
        else:
            return [(0.0, 1.0) for i in range(self.transformed_size)]


class Space:
    """Search space."""

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
            - as a list of categories (for `Categorical` dimensions), or
            - an instance of a `Dimension` object (`Real`, `Integer` or
              `Categorical`).
        """
        _dimensions = []

        for dim in dimensions:
            if isinstance(dim, Dimension):
                _dimensions.append(dim)

            elif (len(dim) == 3 and
                  isinstance(dim[0], numbers.Real) and
                  isinstance(dim[2], str)):
                _dimensions.append(Real(*dim))

            elif len(dim) > 2 or isinstance(dim[0], str):
                _dimensions.append(Categorical(dim))

            elif isinstance(dim[0], numbers.Integral):
                _dimensions.append(Integer(*dim))

            elif isinstance(dim[0], numbers.Real):
                _dimensions.append(Real(*dim))

            else:
                raise ValueError("Invalid grid component (got %s)." % dim)

        self.dimensions = _dimensions

    @property
    def is_real(self):
        """
        Returns true if all dimensions are Real
        """
        return all([isinstance(dim, Real) for dim in self.dimensions])

    def rvs(self, n_samples=1, random_state=None):
        """Draw random samples.

        The samples are in the original space. They need to be transformed
        before being passed to a model or minimizer by `space.transform()`.

        Parameters
        ----------
        * `n_samples` [int, default=1]:
            Number of samples to be drawn from the space.

        * `random_state` [int, RandomState instance, or None (default)]:
            Set random state to something other than None for reproducible
            results.

        Returns
        -------
        * `points`: [list of lists, shape=(n_points, n_dims)]
           Points sampled from the space.
        """
        rng = check_random_state(random_state)

        # Draw
        columns = []

        for dim in self.dimensions:
            if sp_version < (0, 16):
                columns.append(dim.rvs(n_samples=n_samples))
            else:
                columns.append(dim.rvs(n_samples=n_samples, random_state=rng))

        # Transpose
        rows = []

        for i in range(n_samples):
            r = []
            for j in range(self.n_dims):
                r.append(columns[j][i])

            rows.append(r)

        return rows

    def transform(self, X):
        """Transform samples from the original space into a warped space.

        Note: this transformation is expected to be used to project samples
              into a suitable space for numerical optimization.

        Parameters
        ----------
        * `X` [list of lists, shape=(n_samples, n_dims)]:
            The samples to transform.

        Returns
        -------
        * `Xt` [array of floats, shape=(n_samples, transformed_n_dims)]
            The transformed samples.
        """
        # Pack by dimension
        columns = []
        for dim in self.dimensions:
            columns.append([])

        for i in range(len(X)):
            for j in range(self.n_dims):
                columns[j].append(X[i][j])

        # Transform
        for j in range(self.n_dims):
            columns[j] = self.dimensions[j].transform(columns[j])

        # Repack as an array
        Xt = np.hstack([np.asarray(c).reshape((len(X), -1)) for c in columns])
        Xt = Xt.astype(np.float)

        return Xt

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back to the
           original space.

        Parameters
        ----------
        * `Xt` [array of floats, shape=(n_samples, transformed_n_dims)]:
            The samples to inverse transform.

        Returns
        -------
        * `X` [list of lists, shape=(n_samples, n_dims)]
            The original samples.
        """
        # Inverse transform
        columns = []
        start = 0

        for j in range(self.n_dims):
            dim = self.dimensions[j]
            offset = dim.transformed_size

            if offset == 1:
                columns.append(dim.inverse_transform(Xt[:, start]))
            else:
                columns.append(
                    dim.inverse_transform(Xt[:, start:start+offset]))

            start += offset

        # Transpose
        rows = []

        for i in range(len(Xt)):
            r = []
            for j in range(self.n_dims):
                r.append(columns[j][i])

            rows.append(r)

        return rows

    @property
    def n_dims(self):
        """The dimensionality of the original space."""
        return len(self.dimensions)

    @property
    def transformed_n_dims(self):
        """The dimensionality of the warped space."""
        return sum([dim.transformed_size for dim in self.dimensions])

    @property
    def bounds(self):
        """The dimension bounds, in the original space."""
        b = []

        for dim in self.dimensions:
            if dim.size == 1:
                b.append(dim.bounds)
            else:
                b.extend(dim.bounds)

        return b

    @property
    def transformed_bounds(self):
        """The dimension bounds, in the warped space."""
        b = []

        for dim in self.dimensions:
            if dim.transformed_size == 1:
                b.append(dim.transformed_bounds)
            else:
                b.extend(dim.transformed_bounds)

        return b


class DictSpace(Space):

    def __init__(self, dimensions):
        self.keys = sorted(list(dimensions.keys()))
        super(DictSpace, self).__init__([dimensions[k] for k in self.keys])

    def rvs(self, n_samples=1, random_state=None):
        return [
            dict(zip(self.keys, sample))
            for sample in super(DictSpace, self).rvs(n_samples, random_state)
        ]

    def transform(self, X):
        return super(DictSpace, self).transform([
            [sample[k] for k in self.keys] for sample in X
        ])

    def inverse_transform(self, Xt):
        return [
            dict(zip(self.keys, sample))
            for sample in super(DictSpace, self).inverse_transform(Xt)
        ]


def make_space(dimensions):
    if isinstance(dimensions, Space):
        return dimensions
    elif type(dimensions) in (list, tuple):
        return Space(dimensions)
    elif type(dimensions) is dict:
        return DictSpace(dimensions)
    else:
        raise ValueError(
            "dimensions must be one of list, tuple, dict, Space, DictSpace")
