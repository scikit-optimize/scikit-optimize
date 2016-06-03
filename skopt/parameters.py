import abc

from scipy.stats.distributions import uniform

from sklearn.preprocessing import MinMaxScaler
from sklearn.base import TransformerMixin

class Identity(TransformerMixin):
    """Identity transform."""
    def fit(self, values):
        return self

    def transform(self, values):
        return values

    def inverse_transform(self, values):
        return values


class Distribution(object):

    def transform(self, values):
        """Transform `values` from original into warped space."""
        return self.transformer

    @abc.abstractmethod
    def inverse_transform(self, values):
        """Transform `values` from warped into original space."""
        return

    @abc.abstractmethod
    def rvs(self, n_samples=None, random_state=None):
        """
        Sample points randomly.

        This should be overriden by every class inheriting from Distribution
        or else there is no point of it being a Distribution.
        """
        return


class Real(Distribution):
    def __init__(self, low, high, transformer='identity'):
        """Search space dimension that can take on any real value.

        Parameters
        ----------
        * `low` [float]:
            Lower bound of the parameter.

        * `high` [float]:
            Upper bound of the parameter.

        * `transformer` [instance of TransformerMixin, default=`'identity'`]:
            Transformer to convert between original and warped search space.
            Parameter values are always transformed before being handed to the
            optimizer.
        """
        self.low = low
        self.high = high

        if transformer == 'identity':
            self.transformer = Identity()
        elif isinstance(transformer, TransformerMixin):
            self.transformer = transformer
        else:
            raise ValueError('%s is not a valid transformer.'% transformer)
        self._rvs = uniform(self.low, self.high)

    def rvs(self, n_samples=None, random_state=None):
        return self._rvs.rvs(size=n_samples, random_state=random_state)

    def bounds(self):
        return self.low, self.high
