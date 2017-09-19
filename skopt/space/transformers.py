from __future__ import division
import numpy as np
from sklearn.preprocessing import LabelBinarizer


# Base class for all 1-D transformers.
class Transformer(object):
    def fit(self, X):
        return self

    def transform(self, X):
        raise NotImplementedError

    def inverse_transform(self, X):
        raise NotImplementedError


class Identity(Transformer):
    """Identity transform."""

    def transform(self, X):
        return X

    def inverse_transform(self, Xt):
        return Xt


class Log10(Transformer):
    """Base 10 logarithm transform."""

    def transform(self, X):
        return np.log10(np.asarray(X, dtype=np.float))

    def inverse_transform(self, Xt):
        return 10.0 ** np.asarray(Xt, dtype=np.float)


class CategoricalEncoder(Transformer):
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


class Normalize(Transformer):
    """
    Scales each dimension into the interval [0, 1].

    Parameters
    ----------
    * `low` [float]:
        Lower bound.

    * `high` [float]:
        Higher bound.

    * `is_int` [bool, default=True]
        Round and cast the return value of `inverse_transform` to integer. Set
        to `True` when applying this transform to integers.
    """
    def __init__(self, low, high, is_int=False):
        self.low = float(low)
        self.high = float(high)
        self.is_int = is_int

    def transform(self, X):
        X = np.asarray(X)
        if np.any(X > self.high + 1e-8):
            raise ValueError("All values should be less than %f" % self.high)
        if np.any(X < self.low - 1e-8):
            raise ValueError("All values should be greater than %f" % self.low)
        return (X - self.low) / (self.high - self.low)

    def inverse_transform(self, X):
        X = np.asarray(X)
        if np.any(X > 1.0):
            raise ValueError("All values should be less than 1.0")
        if np.any(X < 0.0):
            raise ValueError("All values should be greater than 0.0")
        X_orig = X * (self.high - self.low) + self.low
        if self.is_int:
            return np.round(X_orig).astype(np.int)
        return X_orig


class Pipeline(Transformer):
    """
    A lightweight pipeline to chain transformers.

    Parameters
    ----------
    * 'transformers' [list]:
        A list of Transformer instances.
    """
    def __init__(self, transformers):
        self.transformers = list(transformers)
        for transformer in self.transformers:
            if not isinstance(transformer, Transformer):
                raise ValueError(
                    "Provided transformers should be a Transformer "
                    "instance. Got %s" % transformer
                )

    def fit(self, X):
        for transformer in self.transformers:
            transformer.fit(X)
        return self

    def transform(self, X):
        for transformer in self.transformers:
            X = transformer.transform(X)
        return X

    def inverse_transform(self, X):
        for transformer in self.transformers[::-1]:
            X = transformer.inverse_transform(X)
        return X
