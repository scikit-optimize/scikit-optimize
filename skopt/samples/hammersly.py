# -*- coding: utf-8 -*-
""" Inspired by https://github.com/jonathf/chaospy/blob/master/chaospy/
distributions/sampler/sequences/hammersley.py
"""
import numpy as np
from .halton import Halton
from .utils import InitialPointGenerator


class Hammersly(InitialPointGenerator):
    """The Hammersley set is equivalent to the Halton sequence, except for one
    dimension is replaced with a regular grid. It is not recommended to
    generate a Hammersley sequence more than 10 dimension.

    For ``dim == 1`` the sequence falls back to Van Der Corput sequence.

    T-T. Wong, W-S. Luk, and P-A. Heng, "Sampling with Hammersley and Halton
    Points," Journal of Graphics Tools, vol. 2, no. 2, 1997, pp. 9 - 24.

    Parameters
    ----------
    skip : int
        Skip the first ``skip`` samples. If negative, the maximum of
        ``primes`` is used.
    primes : tuple
        The (non-)prime base to calculate values along each axis. If
        empty, growing prime values starting from 2 will be used.
    """
    def __init__(self, skip=-1, primes=()):
        self.skip = skip
        self.primes = primes

    def generate(self, n_dim, n_samples, random_state=None):
        """Creates samples from Hammersly set.

        Parameters
        ----------
        n_dim : int
           The number of dimension
        n_samples : int
            The order of the Hammersley sequence. Defines the number of samples.
        random_state : int, RandomState instance, or None (default)
            Set random state to something other than None for reproducible
            results.
        Returns
        -------
        np.array, shape=(n_dim, n_samples)
            Hammersley set
        """
        halton = Halton(skip=self.skip, primes=self.primes)

        if n_dim == 1:
            return halton.generate(n_dim, n_samples,
                                   random_state=random_state)
        out = np.empty((n_dim, n_samples), dtype=float)
        out[:n_dim - 1] = halton.generate(n_dim - 1, n_samples,
                                          random_state=random_state).T

        out[n_dim - 1] = np.linspace(0, 1, n_samples + 2)[1:-1]
        return out.T
