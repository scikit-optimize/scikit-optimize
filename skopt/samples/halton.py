"""
Inspired by https://github.com/jonathf/chaospy/blob/master/chaospy/
distributions/sampler/sequences/halton.py
"""
import numpy as np
from .base import InitialPointGenerator


class Halton(InitialPointGenerator):
    """Creates `Halton` sequence samples.
    In statistics, Halton sequences are sequences used to generate
    points in space for numerical methods such as Monte Carlo simulations.
    Although these sequences are deterministic, they are of low discrepancy,
    that is, appear to be random
    for many purposes. They were first introduced in 1960 and are an example
    of a quasi-random number sequence. They generalise the one-dimensional
    van der Corput sequences.

    For ``dim == 1`` the sequence falls back to Van Der Corput sequence.

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
        """Creates samples from Halton set.

        Parameters
        ----------
        n_dim : int
           The number of dimension
        n_samples : int
            The order of the Halton sequence. Defines the number of samples.
        random_state : int, RandomState instance, or None (default)
            Set random state to something other than None for reproducible
            results.
        Returns
        -------
        np.array, shape=(n_dim, n_samples)
            Halton set
        """
        primes = list(self.primes)
        if not primes:
            prime_order = 10 * n_dim
            while len(primes) < n_dim:
                primes = _create_primes(prime_order)
                prime_order *= 2
        primes = primes[:n_dim]
        assert len(primes) == n_dim, "not enough primes"

        if self.skip < 0:
            skip = max(primes)
        else:
            skip = self.skip

        out = np.empty((n_dim, n_samples))
        indices = [idx + skip for idx in range(n_samples)]
        for dim_ in range(n_dim):
            out[dim_] = _van_der_corput_samples(
                indices, number_base=primes[dim_])
        return np.transpose(out)


def _van_der_corput_samples(idx, number_base=2):
    """
    Create `Van Der Corput` low discrepancy sequence samples.

    A van der Corput sequence is an example of the simplest one-dimensional
    low-discrepancy sequence over the unit interval; it was first described in
    1935 by the Dutch mathematician J. G. van der Corput. It is constructed by
    reversing the base-n representation of the sequence of natural numbers
    (1, 2, 3, ...).

    In practice, use Halton sequence instead of Van Der Corput, as it is the
    same, but generalized to work in multiple dimensions.

    Parameters
    ----------
    idx (int, numpy.ndarray):
        The index of the sequence. If array is provided, all values in
        array is returned.
    number_base : int
        The numerical base from where to create the samples from.

    Returns
    -------
    float, numpy.ndarray
        Van der Corput samples.
    """
    assert number_base > 1

    idx = np.asarray(idx).flatten() + 1
    out = np.zeros(len(idx), dtype=float)

    base = float(number_base)
    active = np.ones(len(idx), dtype=bool)
    while np.any(active):
        out[active] += (idx[active] % number_base)/base
        idx //= number_base
        base *= number_base
        active = idx > 0
    return out


def _create_primes(threshold):
    """
    Generate prime values using sieve of Eratosthenes method.

    Parameters
    ----------
    threshold : int
        The upper bound for the size of the prime values.

    Returns
    ------
    List
        All primes from 2 and up to ``threshold``.
    """
    if threshold == 2:
        return [2]

    elif threshold < 2:
        return []

    numbers = list(range(3, threshold+1, 2))
    root_of_threshold = threshold ** 0.5
    half = int((threshold+1)/2-1)
    idx = 0
    counter = 3
    while counter <= root_of_threshold:
        if numbers[idx]:
            idy = int((counter*counter-3)/2)
            numbers[idy] = 0
            while idy < half:
                numbers[idy] = 0
                idy += counter
        idx += 1
        counter = 2*idx+3
    return [2] + [number for number in numbers if number]
