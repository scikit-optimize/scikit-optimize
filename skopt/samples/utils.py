import numpy as np
from sklearn.utils import check_random_state
import math


def create_primes(threshold):
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


def w2_discrepancy_fast(D):
    """The vectorized version of wrap-around L2-discrepancy calculation, faster!
    The formula for the Wrap-Around L2-Discrepancy is taken from Eq.5 of (1)
    :math:`WD^2(D) = -(4/3)^K + 1/N^2 \Sigma_{i,j=1}^{N} \
    Pi_{k=1}^K [3/2 - |x_k^1 - x_k^2| * (1 - |x_k^1 - x_k^2|)]`
    The implementation below uses a vector operation of numpy array to avoid the
    nested loop in the more straightforward implementation

    Parameters
    ----------
    D : np.array
     the design matrix

    Returns
    -------
    the wrap-around L2-discrepancy
    """

    n = D.shape[0]      # the number of samples
    k = D.shape[1]      # the number of dimension
    delta = [None] * k
    for i in range(k):
        # loop over dimension to calculate the absolute difference between point
        # in a given dimension, note the vectorized operation
        delta[i] = np.abs(D[:, i] - np.reshape(D[:, i], (len(D[:, i]), 1)))

    product = 1.5 - delta[0] * (1 - delta[0])
    for i in range(1, k):
        product *= (1.5 - delta[i] * (1 - delta[i]))

    w2_disc = -1 * (4.0/3.0)**k + 1/n**2 * np.sum(product)

    return w2_disc


def random_permute_matrix(h, random_state=None):
    rng = check_random_state(random_state)
    h_rand_perm = np.zeros_like(h)
    samples, n = h.shape
    for j in range(n):
        order = rng.permutation(range(samples))
        h_rand_perm[:, j] = h[order, j]
    return h_rand_perm


def _bit_hi1(n):
    """
    Returns the position of the high 1 bit base 2 in an integer.

    Parameters
    ----------
    n : int
        input, should be positive
    """
    bin_repr = np.binary_repr(n)
    most_left_one = bin_repr.find('1')
    if most_left_one == -1:
        return 0
    else:
        return len(bin_repr) - most_left_one


def _bit_lo0(n):
    """
    Returns the position of the low 0 bit base 2 in an integer.

    Parameters
    ----------
    n : int
        input, should be positive

    """
    bin_repr = np.binary_repr(n)
    most_right_zero = bin_repr[::-1].find('0')
    if most_right_zero == -1:
        most_right_zero = len(bin_repr)
    return most_right_zero + 1


def random_shift(dm, random_state=None):
    """Random shifting of a vector
    Randomization of the quasi-MC samples can be achieved in the easiest manner by
    random shift (or the Cranley-Patterson rotation).
    **Reference:**
    (1) C. Lemieux, "Monte Carlo and Quasi-Monte Carlo Sampling," Springer
        Series in Statistics 692, Springer Science+Business Media, New York,
        2009

    Parameters
    ----------
    dm : array, shape(n,d)
        input matrix
    random_state : int, RandomState instance, or None (default)
        Set random state to something other than None for reproducible
        results.

    Returns
    -------
    Randomized Sobol' design matrix
    """
    rng = check_random_state(random_state)
    # Generate random shift matrix from uniform distribution
    shift = np.repeat(rng.rand(1, dm.shape[1]), dm.shape[0], axis=0)
    # Return the shifted Sobol' design
    return (dm + shift) % 1


def calc_num_candidate(n):
    """Calculate the number of candidates from perturbing the current design
    Recommended in the article is the maximum number of pair combination from a
    given column divided by a factor of 5.
    It is also recommended that the number of candidates to be evaluated does
    not exceed 50

    Parameters
    ----------
    n : int
        the number of elements to be permuted
    Returns
    -------
    the number of candidates from perturbing the current design
        column-wise
    """
    pairs = math.factorial(n) / math.factorial(n - 2) / math.factorial(2)
    fac = 5  # The factor recommended in the article

    return min(int(pairs / fac), 50)


def calc_max_inner(n, k):
    """Calculate the maximum number of inner iterations
    :math:`\frac{2 \times n_e \times k}{J}`
    It is recommended that the number of inner iterations does not exceed 100
    Parameters
    ----------
    n : int
        the number of samples in the design
    k : int
        the number of design dimension
    Returns
    -------
    the maximum number of inner iterations/loop
    """
    pairs = math.factorial(n) / math.factorial(n - 2) / math.factorial(2)

    return min(int(2 * pairs * k / calc_num_candidate(n)), 100)


class InitialPointGenerator(object):
    def generate(self, n_dim, n_samples, random_state=None):
        raise NotImplemented
